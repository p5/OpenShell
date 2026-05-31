// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared GCP constants for the metadata emulator and provider env injection.
//!
//! Both `openshell-server` (env injection) and `openshell-sandbox` (metadata
//! emulator + env resolution) need the same set of constants. Defining them
//! here gives a single source of truth with compile-time linkage.

/// Hostname served by the GCE metadata emulator inside sandboxes.
pub const METADATA_HOST: &str = "gcp.metadata.openshell.internal";

/// Provider config key → env var alias mappings.
///
/// When `inject_provider_type_env` sees a `gcp` provider with
/// `config.project_id = "foo"`, it inserts each of `PROJECT_ID_ENV_VARS` as
/// `"foo"` into the sandbox env. Same for region and service account email.
pub const PROJECT_ID_ENV_VARS: &[&str] = &[
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
];

pub const REGION_ENV_VARS: &[&str] = &[
    "CLOUD_ML_REGION",
    "GCP_LOCATION",
];

pub const SERVICE_ACCOUNT_EMAIL_ENV_VARS: &[&str] = &["GCP_SERVICE_ACCOUNT_EMAIL"];

/// Non-secret GCP config vars that must be resolved to real values in the
/// child environment. Everything else stays as placeholders for proxy-time
/// resolution.
///
/// This list MUST be the union of all alias arrays above. If you add an
/// alias to `PROJECT_ID_ENV_VARS` or `REGION_ENV_VARS`, add it here too.
pub const STATIC_CONFIG_KEYS: &[&str] = &[
    // project_id aliases
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    // region aliases
    "CLOUD_ML_REGION",
    "GCP_LOCATION",
    // service account email
    "GCP_SERVICE_ACCOUNT_EMAIL",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn static_config_keys_is_exact_union_of_aliases() {
        let static_set: HashSet<&str> = STATIC_CONFIG_KEYS.iter().copied().collect();
        let alias_set: HashSet<&str> = PROJECT_ID_ENV_VARS
            .iter()
            .chain(REGION_ENV_VARS)
            .chain(SERVICE_ACCOUNT_EMAIL_ENV_VARS)
            .copied()
            .collect();
        for key in &alias_set {
            assert!(
                static_set.contains(key),
                "STATIC_CONFIG_KEYS is missing alias {key}"
            );
        }
        for key in &static_set {
            assert!(
                alias_set.contains(key),
                "STATIC_CONFIG_KEYS has stray key {key} not in any alias array"
            );
        }
        assert_eq!(
            STATIC_CONFIG_KEYS.len(),
            alias_set.len(),
            "STATIC_CONFIG_KEYS length mismatch — check for duplicates across alias arrays"
        );
    }

    #[test]
    fn no_duplicates_in_static_config_keys() {
        let mut seen = HashSet::new();
        for key in STATIC_CONFIG_KEYS {
            assert!(seen.insert(key), "duplicate in STATIC_CONFIG_KEYS: {key}");
        }
    }
}
