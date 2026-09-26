//! Pluggable **System-1 fast-decision** stage (plans/system1-fast-decisions.md).
//!
//! A System-1 decision engine runs on the Anamanti Core **before** memory recall and
//! the LLM. Given the turn's transcript (plus light context), it either **Resolves** a
//! common intent — which the orchestrator answers via existing tools + `DeviceAction`s,
//! skipping the blocking GraphRAG embedding round-trip *and* the rig+tools full
//! completion — or **Defers** the turn to today's "System-2" path unchanged.
//!
//! Like the LLM backend, the engine is chosen from config and hidden behind one trait
//! ([`DecisionEngine`]), so a local `laya-serve` sidecar or Jev/OpenRouter (both speak
//! the shared `/v1/systemone` contract) are swappable without touching the pipeline.
//!
//! **Status: M0 scaffold.** Only the trait, the no-op [`NoDecision`] default, and the
//! test [`mock::MockDecider`] exist. The `laya-serve`/`jev` HTTP backend and the intent
//! handlers land in M1 (see the plan). The default `backend = "none"` reproduces
//! today's behavior exactly.

pub mod http;
pub mod mock;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

/// The built-in intents the router may resolve when the config leaves `intents` empty.
/// Grows as handlers land (M1: weather; M3: timer). An intent listed here still only
/// resolves if the orchestrator has a handler for it *and* the handler's preconditions
/// hold (e.g. timer needs a parseable duration), otherwise the turn defers to System-2.
pub fn default_intents() -> Vec<String> {
    vec!["weather".to_string(), "timer".to_string()]
}

/// Parse a spoken timer/alarm duration into whole seconds, or `None` when the text has
/// no clear duration (e.g. "cancel my timer", "set a timer") — the caller then defers to
/// System-2, whose timer tool handles cancels and free-form phrasing. Deliberately
/// conservative: it only fires on an unambiguous `<number> <unit>` (repeatable, e.g.
/// "1 hour 30 minutes"). Supports digits and common number words; `a`/`an` = 1,
/// `half` = 0.5, `quarter` = 0.25 of the following unit.
pub fn parse_duration_secs(text: &str) -> Option<u64> {
    fn word_value(tok: &str) -> Option<f64> {
        Some(match tok {
            "one" => 1.0,
            "two" => 2.0,
            "three" => 3.0,
            "four" => 4.0,
            "five" => 5.0,
            "six" => 6.0,
            "seven" => 7.0,
            "eight" => 8.0,
            "nine" => 9.0,
            "ten" => 10.0,
            "eleven" => 11.0,
            "twelve" => 12.0,
            "thirteen" => 13.0,
            "fourteen" => 14.0,
            "fifteen" => 15.0,
            "twenty" => 20.0,
            "thirty" => 30.0,
            "forty" => 40.0,
            "fifty" => 50.0,
            "sixty" => 60.0,
            "ninety" => 90.0,
            "half" => 0.5,
            "quarter" => 0.25,
            _ => return None,
        })
    }
    fn unit_secs(tok: &str) -> Option<f64> {
        Some(match tok {
            "hour" | "hours" | "hr" | "hrs" => 3600.0,
            "minute" | "minutes" | "min" | "mins" => 60.0,
            "second" | "seconds" | "sec" | "secs" => 1.0,
            _ => return None,
        })
    }

    let normalized = text.to_lowercase().replace('-', " ");
    let mut total = 0.0f64;
    let mut current = 0.0f64;
    let mut have_current = false;
    let mut saw_unit = false;

    for raw in normalized.split_whitespace() {
        let tok = raw.trim_matches(|c: char| !c.is_alphanumeric());
        if tok.is_empty() {
            continue;
        }
        if let Ok(n) = tok.parse::<f64>() {
            current += n;
            have_current = true;
        } else if let Some(v) = word_value(tok) {
            current += v;
            have_current = true;
        } else if let Some(u) = unit_secs(tok) {
            // A unit with no explicit number means one ("a minute", "an hour", "half an
            // hour" → the `half` is the number, `an` is ignored, `hour` defaults nothing).
            let n = if have_current { current } else { 1.0 };
            total += n * u;
            current = 0.0;
            have_current = false;
            saw_unit = true;
        }
        // Any other token — the article "a"/"an", "set", "timer", "for", "and", … — is
        // ignored. (Counting "a" as a number would turn "a timer for 10 minutes" into 11.)
    }

    if !saw_unit {
        return None;
    }
    let secs = total.round();
    if secs < 1.0 {
        None
    } else {
        Some(secs as u64)
    }
}

/// The turn state handed to a decision engine — **light context only** (no memory
/// recall; that's exactly the cost System-1 exists to skip).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRequest {
    /// The final STT transcript for this turn.
    pub transcript: String,
    /// A short label for what the display is currently showing (e.g. `"recipe"`), so
    /// screen-relative commands ("scroll down") can route. `None` on an idle display.
    /// Populated in M1 from the turn's `DisplayContext`.
    pub screen: Option<String>,
    /// Recent `(user, assistant)` turns for follow-up disambiguation. Empty for an
    /// ordinary single-shot turn. Populated in M1.
    pub history: Vec<(String, String)>,
    /// The household home location (e.g. `"Austin, Texas"`), from `LiveHomeLocation`.
    /// Location-dependent intents like `weather` are ambiguous without a place, so the
    /// HTTP engine uses this to retry an otherwise-deferred turn with the location folded
    /// into the question. `None` when unset — no retry, the turn just defers.
    pub location: Option<String>,
}

/// A resolved fast intent. M1+ extends this with structured args so the orchestrator
/// can dispatch the matching tool / `DeviceAction`.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    /// The routed intent id, e.g. `"weather"`, `"timer"`.
    pub intent: String,
    /// The engine's calibrated confidence in this resolution (`0.0..=1.0`).
    pub confidence: f64,
}

/// The outcome of a System-1 evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Confident, closed intent — answer it on the fast path (skips recall + LLM).
    Resolve(Resolution),
    /// Ambiguous / open-ended / low-confidence / disabled — fall through to System-2.
    Defer,
}

/// A swappable System-1 decision engine, selected from config like the LLM backend.
/// Implementations are `Send + Sync` so one instance is shared across concurrent turns
/// behind an `Arc`.
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// A short label for logs/settings (e.g. `"none"`, `"laya-serve"`, `"jev"`).
    /// The pipeline skips the whole stage when this is `"none"`.
    fn name(&self) -> &str;

    /// Evaluate the turn. Errors (or low confidence) should surface as `Err`/`Defer`
    /// rather than blocking the turn — the caller always falls through to System-2.
    async fn decide(&self, req: &DecisionRequest) -> Result<Decision>;
}

/// The default no-op engine: always defers to System-2. Selected by `backend = "none"`
/// (the default), so the whole feature is inert until explicitly enabled.
pub struct NoDecision;

#[async_trait]
impl DecisionEngine for NoDecision {
    fn name(&self) -> &str {
        "none"
    }
    async fn decide(&self, _req: &DecisionRequest) -> Result<Decision> {
        Ok(Decision::Defer)
    }
}

/// The default engine (`NoDecision`) as a shared trait object.
pub fn none() -> Arc<dyn DecisionEngine> {
    Arc::new(NoDecision)
}

/// Build a decision engine from a backend label + settings. The single construction
/// point shared by the boot path ([`crate::config::Config::shared_settings`]) and the
/// runtime config-page swap ([`crate::settings::SharedSettings::apply_system1`]), so both
/// agree on what each backend name means. An empty `intents` list falls back to
/// [`default_intents`].
///
/// - `none`/`off`/`""` → [`NoDecision`] (disabled)
/// - `mock` → a deferring [`mock::MockDecider`]
/// - `laya-serve` → local HTTP sidecar (`base_url`)
/// - `jev` → OpenRouter (`base_url` API root + `model` + `api_key`)
/// - `laya-embedded` → not yet implemented (bails)
pub fn build(
    backend: &str,
    base_url: &str,
    model: &str,
    api_key: Option<String>,
    min_confidence: f64,
    intents: Vec<String>,
) -> Result<Arc<dyn DecisionEngine>> {
    use http::HttpDecider;
    let intents = if intents.is_empty() {
        default_intents()
    } else {
        intents
    };
    match backend.to_lowercase().as_str() {
        "none" | "off" | "" => Ok(none()),
        "mock" => Ok(Arc::new(mock::MockDecider::defer())),
        "laya-serve" | "laya_serve" => {
            Ok(Arc::new(HttpDecider::laya_serve(base_url, min_confidence, intents)))
        }
        "jev" => Ok(Arc::new(HttpDecider::jev(
            base_url,
            model,
            api_key,
            min_confidence,
            intents,
        ))),
        other @ ("laya-embedded" | "laya_embedded") => anyhow::bail!(
            "system1.backend `{other}` (in-process candle) is not implemented yet — it is the \
             optional escape hatch. Use `laya-serve` (local sidecar) or `jev` for now."
        ),
        other => anyhow::bail!(
            "unknown system1.backend `{other}` (expected none/mock/laya-serve/jev/laya-embedded)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(t: &str) -> DecisionRequest {
        DecisionRequest {
            transcript: t.to_string(),
            screen: None,
            history: Vec::new(),
            location: None,
        }
    }

    #[tokio::test]
    async fn no_decision_always_defers() {
        let e = NoDecision;
        assert_eq!(e.name(), "none");
        assert_eq!(
            e.decide(&req("show me the weather")).await.unwrap(),
            Decision::Defer
        );
    }

    #[test]
    fn parse_duration_handles_common_phrasings() {
        assert_eq!(parse_duration_secs("set a timer for 10 minutes"), Some(600));
        assert_eq!(parse_duration_secs("10 minute timer"), Some(600));
        assert_eq!(parse_duration_secs("set a timer for 1 hour 30 minutes"), Some(5400));
        assert_eq!(parse_duration_secs("half an hour"), Some(1800));
        assert_eq!(parse_duration_secs("a minute"), Some(60));
        assert_eq!(parse_duration_secs("90 seconds"), Some(90));
        assert_eq!(parse_duration_secs("twenty five minutes"), Some(1500));
        assert_eq!(parse_duration_secs("2 hrs"), Some(7200));
    }

    #[test]
    fn parse_duration_defers_when_unclear() {
        // No unit → defer (the LLM's timer tool handles these).
        assert_eq!(parse_duration_secs("set a timer"), None);
        assert_eq!(parse_duration_secs("cancel my timer"), None);
        assert_eq!(parse_duration_secs("what's the weather"), None);
        assert_eq!(parse_duration_secs(""), None);
    }
}
