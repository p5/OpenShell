// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vertex AI route resolution.
//!
//! Resolves provider config + model identifier into a concrete
//! [`RouterResolvedRoute`] for Vertex AI endpoints. Handles two API surfaces:
//! - Anthropic Messages via `rawPredict`
//! - OpenAI-compatible via `/chat/completions`

use openshell_core::inference::{
    VERTEX_AI_PROJECT_ID_KEY, VERTEX_AI_PUBLISHER_KEY, VERTEX_AI_REGION_KEY,
};
use openshell_router::config::ResolvedRoute as RouterResolvedRoute;
use tonic::Status;

/// Infer the Vertex AI publisher segment from a model identifier.
///
/// Currently only the `"anthropic"` result is consumed by
/// `resolve_vertex_ai_route` to select between the native Anthropic
/// Messages API (`rawPredict`) and the OpenAI-compatible endpoint.
/// Non-Anthropic publisher mappings (`meta`, `mistralai`, `ai21`,
/// `deepseek`, `google`) are maintained for forward compatibility
/// and documentation value — all non-Anthropic models route to the
/// same OpenAI-compatible endpoint regardless of publisher.
///
/// Returns `None` for unrecognized models, which causes resolution to
/// fall back to the OpenAI-compatible endpoint
/// (`v1beta1/.../endpoints/openapi`).
fn infer_vertex_publisher(model_id: &str) -> Option<&'static str> {
    if model_id.starts_with("claude-") {
        Some("anthropic")
    } else if model_id.starts_with("gemini-")
        || model_id.starts_with("text-bison-")
        || model_id.starts_with("chat-bison-")
    {
        Some("google")
    } else if model_id.starts_with("llama-") {
        Some("meta")
    } else if model_id.starts_with("mistral-") || model_id.starts_with("codestral-") {
        Some("mistralai")
    } else if model_id.starts_with("jamba-") {
        Some("ai21")
    } else if model_id.starts_with("deepseek-") {
        Some("deepseek")
    } else {
        None
    }
}

/// Return a required Vertex AI config value, or a `FailedPrecondition` status.
fn required_vertex_config<'a>(
    config: &'a std::collections::HashMap<String, String>,
    key: &str,
) -> Result<&'a str, Status> {
    config
        .get(key)
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| {
            Status::failed_precondition(format!("Vertex AI provider requires {key} config"))
        })
}

/// Validate a GCP project ID against the documented format.
///
/// GCP project IDs must be 6–30 characters, start with a lowercase letter,
/// contain only lowercase letters, digits, and hyphens, and not end with a hyphen.
fn validate_gcp_project_id(value: &str) -> Result<(), Status> {
    let valid = value.len() >= 6
        && value.len() <= 30
        && value.starts_with(|c: char| c.is_ascii_lowercase())
        && !value.ends_with('-')
        && value
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if valid {
        Ok(())
    } else {
        Err(Status::invalid_argument(format!(
            "VERTEX_AI_PROJECT_ID has invalid format: {value:?}. \
             GCP project IDs must be 6-30 characters, start with a lowercase letter, \
             contain only lowercase letters, digits, and hyphens, and not end with a hyphen."
        )))
    }
}

/// Validate a GCP region/location value.
///
/// Accepts the special keywords `global`, `us`, and `eu`, plus standard
/// regional patterns like `us-central1`, `europe-west4`, `us-east4-a`.
fn validate_gcp_region(value: &str) -> Result<(), Status> {
    let lower = value.trim().to_ascii_lowercase();
    let valid = matches!(lower.as_str(), "global" | "us" | "eu")
        || (lower
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            && lower.contains('-')
            && !lower.starts_with('-')
            && !lower.ends_with('-'));
    if valid {
        Ok(())
    } else {
        Err(Status::invalid_argument(format!(
            "VERTEX_AI_REGION has invalid format: {value:?}. \
             Expected a GCP region (e.g. us-central1, europe-west4) \
             or one of: global, us, eu."
        )))
    }
}

/// Resolve the Vertex AI API host and normalized location from a configured region.
fn vertex_location_and_host(region: &str) -> (String, String) {
    let location = region.trim().to_ascii_lowercase();
    let host = match location.as_str() {
        "global" => "aiplatform.googleapis.com".to_string(),
        "us" | "eu" => format!("aiplatform.{location}.rep.googleapis.com"),
        _ => format!("{location}-aiplatform.googleapis.com"),
    };
    (location, host)
}

fn validate_vertex_model_id(value: &str) -> Result<(), Status> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Status::invalid_argument("model_id is required"));
    }
    if value != trimmed {
        return Err(Status::invalid_argument(format!(
            "Vertex AI model_id must not include leading or trailing whitespace: {value:?}"
        )));
    }
    if value.contains('/') || value.contains('\\') {
        return Err(Status::invalid_argument(format!(
            "Vertex AI model_id must not contain path separators: {value:?}"
        )));
    }
    if value.chars().any(|c| matches!(c, '?' | '#' | '%')) {
        return Err(Status::invalid_argument(format!(
            "Vertex AI model_id must not contain URL delimiters or percent escapes: {value:?}"
        )));
    }
    if value.contains("..") {
        return Err(Status::invalid_argument(format!(
            "Vertex AI model_id must not contain traversal segments: {value:?}"
        )));
    }
    if value.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(Status::invalid_argument(format!(
            "Vertex AI model_id must not contain whitespace or control characters: {value:?}"
        )));
    }
    Ok(())
}

fn is_allowed_vertex_override_host(host: &str) -> bool {
    matches!(
        host,
        "aiplatform.googleapis.com"
            | "aiplatform.us.rep.googleapis.com"
            | "aiplatform.eu.rep.googleapis.com"
    ) || host.ends_with("-aiplatform.googleapis.com")
}

fn validate_vertex_base_url(value: &str) -> Result<String, Status> {
    let trimmed = value.trim();
    let url = url::Url::parse(trimmed).map_err(|err| {
        Status::invalid_argument(format!("Vertex AI base URL override is invalid: {err}"))
    })?;

    if url.scheme() != "https" {
        return Err(Status::invalid_argument(
            "Vertex AI base URL override must use https".to_string(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Status::invalid_argument(
            "Vertex AI base URL override must not include userinfo".to_string(),
        ));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(Status::invalid_argument(
            "Vertex AI base URL override must not include query or fragment components".to_string(),
        ));
    }
    if let Some(port) = url.port()
        && port != 443
    {
        return Err(Status::invalid_argument(format!(
            "Vertex AI base URL override must use port 443 when an explicit port is set, got {port}"
        )));
    }

    match url.host() {
        Some(url::Host::Domain(host)) if is_allowed_vertex_override_host(host) => {}
        Some(url::Host::Domain(host)) => {
            return Err(Status::invalid_argument(format!(
                "Vertex AI base URL override must target an official Vertex AI hostname, got {host:?}"
            )));
        }
        Some(url::Host::Ipv4(_) | url::Host::Ipv6(_)) => {
            return Err(Status::invalid_argument(format!(
                "Vertex AI base URL override must not use IP literal hosts: {}",
                url.host_str().unwrap_or("<unknown>")
            )));
        }
        None => {
            return Err(Status::invalid_argument(
                "Vertex AI base URL override must include a host".to_string(),
            ));
        }
    }

    Ok(trimmed.to_string())
}

/// Build a [`RouterResolvedRoute`] for Vertex AI without duplicating the 15-field struct.
#[allow(clippy::too_many_arguments)]
fn build_vertex_route(
    route_name: &str,
    endpoint: String,
    model_id: &str,
    api_key: &str,
    protocols: Vec<String>,
    profile: &openshell_core::inference::InferenceProviderProfile,
    model_in_path: bool,
    request_path_override: Option<String>,
) -> RouterResolvedRoute {
    RouterResolvedRoute {
        name: route_name.to_string(),
        endpoint,
        model: model_id.to_string(),
        api_key: api_key.to_string(),
        protocols,
        auth: profile.auth.clone(),
        default_headers: profile
            .default_headers
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect(),
        passthrough_headers: profile
            .passthrough_headers
            .iter()
            .map(|p| (*p).to_string())
            .collect(),
        timeout: openshell_router::config::DEFAULT_ROUTE_TIMEOUT,
        model_in_path,
        request_path_override,
    }
}

/// Resolve a Vertex AI route given provider config, model, and bearer token.
pub(crate) fn resolve_vertex_ai_route(
    config: &std::collections::HashMap<String, String>,
    model_id: &str,
    route_name: &str,
    api_key: &str,
    profile: &openshell_core::inference::InferenceProviderProfile,
) -> Result<RouterResolvedRoute, Status> {
    // Validate model_id early — it appears in URL paths for Anthropic routes
    // and in JSON request bodies for all routes. Rejecting path separators,
    // traversal segments, and control characters up front is defense-in-depth.
    validate_vertex_model_id(model_id)?;

    // Determine if this is an Anthropic model.
    // Explicit VERTEX_AI_PUBLISHER=anthropic overrides inference.
    // All non-Anthropic models route to the OpenAI-compatible endpoint.
    let explicit_publisher = config
        .get(VERTEX_AI_PUBLISHER_KEY)
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty());

    let is_anthropic = explicit_publisher.map_or_else(
        || infer_vertex_publisher(model_id) == Some("anthropic"),
        |p| p.eq_ignore_ascii_case("anthropic"),
    );

    // Escape hatch: caller-supplied full base URL still uses the model-derived
    // protocol and path contract, but only for the OpenAI-compatible Vertex surface.
    // Anthropic-on-Vertex needs model-path shaping and body adaptation that a fully
    // caller-controlled URL cannot safely preserve.
    if let Some(base_url) = config
        .get(profile.base_url_config_keys[0])
        .or_else(|| config.get(profile.base_url_config_keys[1]))
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty())
    {
        if is_anthropic {
            return Err(Status::invalid_argument(
                "Vertex AI base URL overrides are not supported for Anthropic models. \
                 Remove GOOGLE_VERTEX_AI_BASE_URL / VERTEX_AI_BASE_URL and configure \
                 VERTEX_AI_PROJECT_ID + VERTEX_AI_REGION instead."
                    .to_string(),
            ));
        }
        let base_url = validate_vertex_base_url(base_url)?;

        return Ok(build_vertex_route(
            route_name,
            base_url,
            model_id,
            api_key,
            vec!["openai_chat_completions".to_string()],
            profile,
            false,
            Some("/chat/completions".to_string()),
        ));
    }

    let project = required_vertex_config(config, VERTEX_AI_PROJECT_ID_KEY)?;
    validate_gcp_project_id(project)?;
    let region = config
        .get(VERTEX_AI_REGION_KEY)
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty())
        .unwrap_or("us-central1");
    validate_gcp_region(region)?;
    let (location, host) = vertex_location_and_host(region);

    if is_anthropic {
        // Native Anthropic Messages API via rawPredict.
        // model_id is NOT embedded in the endpoint — it is carried in route.model
        // and appended with the suffix by build_provider_url(). The router upgrades
        // `:rawPredict` to `:streamRawPredict` only for streaming proxy calls.
        let endpoint = format!(
            "https://{host}/v1/projects/{project}/locations/{location}/publishers/anthropic/models"
        );
        let protocols = vec!["anthropic_messages".to_string()];
        Ok(build_vertex_route(
            route_name,
            endpoint,
            model_id,
            api_key,
            protocols,
            profile,
            true,
            Some(":rawPredict".to_string()),
        ))
    } else {
        // OpenAI-compatible endpoint for all non-Anthropic models
        // (Gemini, Llama, Mistral, unknown, etc.). Vertex's OpenAI-compatible
        // surface uses `/chat/completions` under the `.../endpoints/openapi`
        // base, so we pin the route to that path instead of appending the
        // router's default `/v1/...` protocol path.
        let endpoint = format!(
            "https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"
        );
        let protocols = vec!["openai_chat_completions".to_string()];
        Ok(build_vertex_route(
            route_name,
            endpoint,
            model_id,
            api_key,
            protocols,
            profile,
            false,
            Some("/chat/completions".to_string()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_vertex_publisher_anthropic() {
        assert_eq!(
            infer_vertex_publisher("claude-3-5-sonnet@20241022"),
            Some("anthropic")
        );
        assert_eq!(infer_vertex_publisher("claude-opus-4"), Some("anthropic"));
    }

    #[test]
    fn infer_vertex_publisher_gemini() {
        assert_eq!(infer_vertex_publisher("gemini-pro"), Some("google"));
        assert_eq!(infer_vertex_publisher("gemini-1.5-flash"), Some("google"));
        assert_eq!(infer_vertex_publisher("text-bison-001"), Some("google"));
        assert_eq!(infer_vertex_publisher("chat-bison-001"), Some("google"));
    }

    #[test]
    fn infer_vertex_publisher_unknown() {
        assert_eq!(infer_vertex_publisher("some-unknown-model"), None);
        assert_eq!(infer_vertex_publisher("gpt-4o"), None);
    }

    #[test]
    fn infer_vertex_publisher_other_publishers() {
        assert_eq!(infer_vertex_publisher("llama-3-70b"), Some("meta"));
        assert_eq!(infer_vertex_publisher("mistral-large"), Some("mistralai"));
        assert_eq!(infer_vertex_publisher("codestral-22b"), Some("mistralai"));
        assert_eq!(infer_vertex_publisher("jamba-1.5-large"), Some("ai21"));
        assert_eq!(infer_vertex_publisher("deepseek-r1"), Some("deepseek"));
    }

    #[test]
    fn validate_gcp_project_id_accepts_valid() {
        assert!(validate_gcp_project_id("my-project").is_ok());
        assert!(validate_gcp_project_id("my-project-123").is_ok());
        assert!(validate_gcp_project_id("abcdef").is_ok());
    }

    #[test]
    fn validate_gcp_project_id_rejects_invalid() {
        assert!(validate_gcp_project_id("").is_err());
        assert!(validate_gcp_project_id("ab").is_err());
        assert!(validate_gcp_project_id("../admin").is_err());
        assert!(validate_gcp_project_id("MY-PROJECT").is_err());
        assert!(validate_gcp_project_id("my-project-").is_err());
        assert!(validate_gcp_project_id("1my-project").is_err());
    }

    #[test]
    fn validate_gcp_region_accepts_valid() {
        assert!(validate_gcp_region("us-central1").is_ok());
        assert!(validate_gcp_region("europe-west4").is_ok());
        assert!(validate_gcp_region("global").is_ok());
        assert!(validate_gcp_region("us").is_ok());
        assert!(validate_gcp_region("eu").is_ok());
        assert!(validate_gcp_region("us-east4-a").is_ok());
    }

    #[test]
    fn validate_gcp_region_rejects_invalid() {
        assert!(validate_gcp_region("").is_err());
        assert!(validate_gcp_region("../../etc").is_err());
        assert!(validate_gcp_region("us central1").is_err());
        assert!(validate_gcp_region("-us-central1").is_err());
        assert!(validate_gcp_region("us-central1-").is_err());
    }

    #[test]
    fn validate_vertex_base_url_rejects_ipv6_literal() {
        let err = validate_vertex_base_url(
            "https://[::1]/v1beta1/projects/p/locations/l/endpoints/openapi",
        )
        .expect_err("IPv6 literals must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("IP literal"));
    }

    #[test]
    fn validate_vertex_base_url_rejects_userinfo() {
        let err =
            validate_vertex_base_url("https://user:pass@us-central1-aiplatform.googleapis.com/v1")
                .expect_err("userinfo must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("userinfo"));
    }

    #[test]
    fn validate_vertex_base_url_rejects_query_string() {
        let err =
            validate_vertex_base_url("https://us-central1-aiplatform.googleapis.com/v1?key=val")
                .expect_err("query string must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("query or fragment"));
    }

    #[test]
    fn validate_vertex_base_url_rejects_fragment() {
        let err =
            validate_vertex_base_url("https://us-central1-aiplatform.googleapis.com/v1#section")
                .expect_err("fragment must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("query or fragment"));
    }

    #[test]
    fn validate_vertex_base_url_rejects_non_443_port() {
        let err = validate_vertex_base_url("https://us-central1-aiplatform.googleapis.com:8443/v1")
            .expect_err("non-443 port must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("443"));
    }

    #[test]
    fn validate_vertex_model_id_rejects_double_dot_traversal() {
        let err = validate_vertex_model_id("model..v2")
            .expect_err("double-dot traversal must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("traversal"));
    }

    #[test]
    fn vertex_location_and_host_regional() {
        let (loc, host) = vertex_location_and_host("us-east1");
        assert_eq!(loc, "us-east1");
        assert_eq!(host, "us-east1-aiplatform.googleapis.com");
    }

    #[test]
    fn vertex_location_and_host_global() {
        let (loc, host) = vertex_location_and_host("global");
        assert_eq!(loc, "global");
        assert_eq!(host, "aiplatform.googleapis.com");
    }

    #[test]
    fn vertex_location_and_host_multiregion() {
        let (loc, host) = vertex_location_and_host("us");
        assert_eq!(loc, "us");
        assert_eq!(host, "aiplatform.us.rep.googleapis.com");
    }
}
