// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime provider credential snapshots.

use crate::secrets::SecretResolver;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

const MAX_RETAINED_CREDENTIAL_GENERATIONS: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct ProviderCredentialSnapshot {
    pub revision: u64,
    pub child_env: HashMap<String, String>,
}

#[derive(Debug)]
struct ProviderCredentialStateInner {
    current: Arc<ProviderCredentialSnapshot>,
    generations: VecDeque<Arc<SecretResolver>>,
    current_resolver: Option<Arc<SecretResolver>>,
    combined_resolver: Option<Arc<SecretResolver>>,
}

#[derive(Debug, Clone)]
pub struct ProviderCredentialState {
    inner: Arc<RwLock<ProviderCredentialStateInner>>,
}

impl ProviderCredentialState {
    pub fn from_environment(
        revision: u64,
        env: HashMap<String, String>,
        credential_expires_at_ms: HashMap<String, i64>,
    ) -> Self {
        let (child_env, generation_resolver, current_resolver) =
            SecretResolver::from_provider_env_for_current_revision(
                env,
                credential_expires_at_ms,
                revision,
            );
        let snapshot = Arc::new(ProviderCredentialSnapshot {
            revision,
            child_env,
        });
        let generations: VecDeque<_> = generation_resolver.map(Arc::new).into_iter().collect();
        let current_resolver = current_resolver.map(Arc::new);
        let combined_resolver = merge_resolvers(&generations, current_resolver.as_ref());

        Self {
            inner: Arc::new(RwLock::new(ProviderCredentialStateInner {
                current: snapshot,
                generations,
                current_resolver,
                combined_resolver,
            })),
        }
    }

    pub fn snapshot(&self) -> Arc<ProviderCredentialSnapshot> {
        self.inner
            .read()
            .expect("provider credential state poisoned")
            .current
            .clone()
    }

    pub fn resolver(&self) -> Option<Arc<SecretResolver>> {
        self.inner
            .read()
            .expect("provider credential state poisoned")
            .combined_resolver
            .clone()
    }

    /// Return child_env with GCP static config vars resolved to real values.
    ///
    /// The credential pipeline placeholderizes ALL env values, but GCP SDKs
    /// and coding agents read certain vars (project ID, region, metadata host)
    /// at process startup before any HTTP request flows through the proxy.
    /// This method overrides those vars with resolved real values while
    /// keeping secret credentials (like `GCP_ACCESS_TOKEN`) as placeholders.
    ///
    /// Three layers of env var injection:
    /// 1. **Synthetic vars** (`GCE_METADATA_IP`, `METADATA_SERVER_DETECTION`)
    ///    — sandbox-internal config not from user
    ///    input, inserted directly here with real values.
    /// 2. **`gcp::STATIC_CONFIG_KEYS`** — user-provided non-secret config
    ///    (project ID, region, SA email) that was placeholderized by
    ///    `ProviderPlugin::inject_env` → SecretResolver; un-placeholderized
    ///    here so SDKs can read them at startup.
    /// 3. Everything else stays as placeholders for proxy-time resolution.
    pub fn child_env_resolved(&self) -> HashMap<String, String> {
        use openshell_core::gcp;

        let inner = self
            .inner
            .read()
            .expect("provider credential state poisoned");
        let mut env = inner.current.child_env.clone();

        if !env.contains_key("GCE_METADATA_HOST") {
            return env;
        }

        // Synthetic vars: sandbox-internal config that doesn't originate from
        // user input and was never placeholderized.
        env.insert(
            "GCE_METADATA_HOST".to_string(),
            gcp::METADATA_HOST.to_string(),
        );
        // Python's google-auth uses GCE_METADATA_IP for the initial ping
        // that detects whether it's running on GCE. Without this, ADC
        // discovery skips compute engine credentials entirely.
        env.insert(
            "GCE_METADATA_IP".to_string(),
            gcp::METADATA_HOST.to_string(),
        );
        // Node.js gcp-metadata uses METADATA_SERVER_DETECTION to skip the
        // runtime ping that otherwise fails in sandboxed environments.
        env.insert(
            "METADATA_SERVER_DETECTION".to_string(),
            "assume-present".to_string(),
        );

        // Un-placeholderize non-secret config vars so SDKs can read them
        // at process startup before any HTTP flows through the proxy.
        if let Some(ref resolver) = inner.combined_resolver {
            for key in gcp::STATIC_CONFIG_KEYS {
                let placeholder = crate::secrets::placeholder_for_env_key(key);
                if let Some(value) = resolver.resolve_placeholder(&placeholder) {
                    env.insert(key.to_string(), value.to_string());
                }
            }
        }

        env
    }

    /// Return the placeholder for the first available GCP token credential.
    ///
    /// Searches `gcp::TOKEN_ENV_KEYS` in priority order (SA before ADC) and
    /// returns the placeholder string if the credential exists and is not
    /// expired.
    pub fn gcp_token_placeholder(&self) -> Option<String> {
        let resolver = self.resolver()?;
        for key in openshell_core::gcp::TOKEN_ENV_KEYS {
            let placeholder = crate::secrets::placeholder_for_env_key(key);
            if resolver.resolve_placeholder(&placeholder).is_some() {
                return Some(placeholder);
            }
        }
        None
    }

    /// Return the remaining lifetime of a GCP token in seconds.
    ///
    /// Returns 3600 (one hour) when expiry is unknown or the timestamp is
    /// non-positive. This matches the default `expires_in` that the real GCE
    /// metadata server returns.
    pub fn gcp_token_expires_in(&self, placeholder: &str) -> i64 {
        const DEFAULT_EXPIRES_IN: i64 = 3600;
        self.resolver()
            .and_then(|r| r.expires_at_ms_for_placeholder(placeholder))
            .map(|expires_at_ms| {
                if expires_at_ms <= 0 {
                    DEFAULT_EXPIRES_IN
                } else {
                    let now = crate::secrets::current_time_ms();
                    ((expires_at_ms - now) / 1000).max(0)
                }
            })
            .unwrap_or(DEFAULT_EXPIRES_IN)
    }

    pub fn install_environment(
        &self,
        revision: u64,
        env: HashMap<String, String>,
        credential_expires_at_ms: HashMap<String, i64>,
    ) -> usize {
        let (child_env, generation_resolver, current_resolver) =
            SecretResolver::from_provider_env_for_current_revision(
                env,
                credential_expires_at_ms,
                revision,
            );
        let mut inner = self
            .inner
            .write()
            .expect("provider credential state poisoned");

        inner.current = Arc::new(ProviderCredentialSnapshot {
            revision,
            child_env,
        });
        inner.current_resolver = current_resolver.map(Arc::new);

        if let Some(resolver) = generation_resolver {
            inner.generations.push_back(Arc::new(resolver));
            while inner.generations.len() > MAX_RETAINED_CREDENTIAL_GENERATIONS {
                inner.generations.pop_front();
            }
        }
        inner.combined_resolver =
            merge_resolvers(&inner.generations, inner.current_resolver.as_ref());
        inner.current.child_env.len()
    }
}

fn merge_resolvers(
    generations: &VecDeque<Arc<SecretResolver>>,
    current_resolver: Option<&Arc<SecretResolver>>,
) -> Option<Arc<SecretResolver>> {
    SecretResolver::merge(
        generations
            .iter()
            .map(Arc::as_ref)
            .chain(current_resolver.into_iter().map(Arc::as_ref)),
    )
    .map(Arc::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_use_revision_scoped_placeholders() {
        let state = ProviderCredentialState::from_environment(
            10,
            HashMap::from([("GITHUB_TOKEN".to_string(), "old".to_string())]),
            HashMap::new(),
        );
        let first = state.snapshot();
        assert_eq!(
            first.child_env.get("GITHUB_TOKEN").map(String::as_str),
            Some("openshell:resolve:env:v10_GITHUB_TOKEN")
        );

        state.install_environment(
            11,
            HashMap::from([("GITHUB_TOKEN".to_string(), "new".to_string())]),
            HashMap::new(),
        );
        let second = state.snapshot();
        assert_eq!(
            second.child_env.get("GITHUB_TOKEN").map(String::as_str),
            Some("openshell:resolve:env:v11_GITHUB_TOKEN")
        );

        let resolver = state.resolver().expect("resolver");
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:v10_GITHUB_TOKEN"),
            Some("old")
        );
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:v11_GITHUB_TOKEN"),
            Some("new")
        );
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:GITHUB_TOKEN"),
            Some("new")
        );
        assert_eq!(
            resolver.resolve_placeholder("provider-OPENSHELL-RESOLVE-ENV-GITHUB_TOKEN"),
            Some("new")
        );
    }

    #[test]
    fn empty_refresh_removes_current_aliases_but_retains_revisioned_resolver() {
        let state = ProviderCredentialState::from_environment(
            10,
            HashMap::from([("GITHUB_TOKEN".to_string(), "old".to_string())]),
            HashMap::new(),
        );

        state.install_environment(11, HashMap::new(), HashMap::new());

        assert!(state.snapshot().child_env.is_empty());
        let resolver = state.resolver().expect("old resolver retained");
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:v10_GITHUB_TOKEN"),
            Some("old")
        );
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:GITHUB_TOKEN"),
            None
        );
        assert_eq!(
            resolver.resolve_placeholder("provider-OPENSHELL-RESOLVE-ENV-GITHUB_TOKEN"),
            None
        );
    }

    #[test]
    fn expired_retained_generation_does_not_resolve() {
        let now_ms = i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap();
        let state = ProviderCredentialState::from_environment(
            10,
            HashMap::from([("GITHUB_TOKEN".to_string(), "old".to_string())]),
            HashMap::from([("GITHUB_TOKEN".to_string(), now_ms - 1_000)]),
        );

        state.install_environment(
            11,
            HashMap::from([("GITHUB_TOKEN".to_string(), "new".to_string())]),
            HashMap::from([("GITHUB_TOKEN".to_string(), now_ms + 60_000)]),
        );

        let resolver = state.resolver().expect("resolver");
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:v10_GITHUB_TOKEN"),
            None
        );
        assert_eq!(
            resolver.resolve_placeholder("openshell:resolve:env:v11_GITHUB_TOKEN"),
            Some("new")
        );
    }

    #[test]
    fn child_env_resolved_without_gcp_returns_unchanged() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([("GITHUB_TOKEN".to_string(), "ghp_abc".to_string())]),
            HashMap::new(),
        );
        let env = state.child_env_resolved();
        assert_eq!(
            env.get("GITHUB_TOKEN").map(String::as_str),
            Some("openshell:resolve:env:v1_GITHUB_TOKEN"),
            "non-GCP env should remain as placeholder"
        );
        assert!(!env.contains_key("GCE_METADATA_HOST"));
        assert!(!env.contains_key("CLAUDE_CODE_USE_VERTEX"));
    }

    #[test]
    fn child_env_resolved_overrides_gcp_static_vars() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([
                ("GCE_METADATA_HOST".to_string(), "marker".to_string()),
                (
                    "GCP_ADC_ACCESS_TOKEN".to_string(),
                    "ya29.secret".to_string(),
                ),
                ("GCP_PROJECT_ID".to_string(), "my-project".to_string()),
                ("CLOUD_ML_REGION".to_string(), "us-central1".to_string()),
            ]),
            HashMap::new(),
        );
        let env = state.child_env_resolved();

        assert_eq!(
            env.get("GCE_METADATA_HOST").map(String::as_str),
            Some(openshell_core::gcp::METADATA_HOST),
            "GCE_METADATA_HOST should be the real hostname"
        );
        assert!(
            !env.contains_key("CLAUDE_CODE_USE_VERTEX"),
            "inference-specific vars should not be injected"
        );
        assert_eq!(
            env.get("GCP_PROJECT_ID").map(String::as_str),
            Some("my-project"),
            "static config should be resolved to real value"
        );
        assert_eq!(
            env.get("CLOUD_ML_REGION").map(String::as_str),
            Some("us-central1"),
        );

        let token = env.get("GCP_ADC_ACCESS_TOKEN").map(String::as_str).unwrap();
        assert!(
            token.starts_with("openshell:resolve:env:"),
            "GCP_ACCESS_TOKEN must stay as placeholder, got: {token}"
        );
    }

    #[test]
    fn child_env_resolved_handles_missing_config_keys() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([
                ("GCE_METADATA_HOST".to_string(), "marker".to_string()),
                ("GCP_ADC_ACCESS_TOKEN".to_string(), "ya29.tok".to_string()),
            ]),
            HashMap::new(),
        );
        let env = state.child_env_resolved();

        assert_eq!(
            env.get("GCE_METADATA_HOST").map(String::as_str),
            Some(openshell_core::gcp::METADATA_HOST),
        );
        assert!(
            !env.contains_key("GCP_PROJECT_ID")
                || env
                    .get("GCP_PROJECT_ID")
                    .unwrap()
                    .starts_with("openshell:resolve:env:"),
            "missing config key should not be injected with a real value"
        );
    }

    #[test]
    fn gcp_token_placeholder_returns_sa_over_adc() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([
                ("GCP_SA_ACCESS_TOKEN".to_string(), "sa-tok".to_string()),
                ("GCP_ADC_ACCESS_TOKEN".to_string(), "adc-tok".to_string()),
            ]),
            HashMap::new(),
        );
        let placeholder = state.gcp_token_placeholder().expect("should find token");
        assert!(
            placeholder.contains("GCP_SA_ACCESS_TOKEN"),
            "SA token should win over ADC, got: {placeholder}"
        );
    }

    #[test]
    fn gcp_token_placeholder_falls_back_to_adc() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([(
                "GCP_ADC_ACCESS_TOKEN".to_string(),
                "adc-tok".to_string(),
            )]),
            HashMap::new(),
        );
        let placeholder = state.gcp_token_placeholder().expect("should find ADC token");
        assert!(placeholder.contains("GCP_ADC_ACCESS_TOKEN"));
    }

    #[test]
    fn gcp_token_placeholder_returns_none_without_gcp() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([("GITHUB_TOKEN".to_string(), "ghp_abc".to_string())]),
            HashMap::new(),
        );
        assert!(state.gcp_token_placeholder().is_none());
    }

    #[test]
    fn gcp_token_expires_in_defaults_to_3600() {
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([(
                "GCP_ADC_ACCESS_TOKEN".to_string(),
                "adc-tok".to_string(),
            )]),
            HashMap::new(),
        );
        let placeholder = state.gcp_token_placeholder().unwrap();
        let expires_in = state.gcp_token_expires_in(&placeholder);
        assert_eq!(expires_in, 3600, "should default to 3600 when no expiry set");
    }

    #[test]
    fn gcp_token_expires_in_calculates_remaining() {
        let now_ms = i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap();
        let state = ProviderCredentialState::from_environment(
            1,
            HashMap::from([(
                "GCP_ADC_ACCESS_TOKEN".to_string(),
                "adc-tok".to_string(),
            )]),
            HashMap::from([(
                "GCP_ADC_ACCESS_TOKEN".to_string(),
                now_ms + 120_000,
            )]),
        );
        let placeholder = state.gcp_token_placeholder().unwrap();
        let expires_in = state.gcp_token_expires_in(&placeholder);
        assert!(
            (110..=120).contains(&expires_in),
            "expected ~120s remaining, got {expires_in}"
        );
    }
}
