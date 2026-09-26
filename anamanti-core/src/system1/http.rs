//! HTTP [`DecisionEngine`] over the shared **`/v1/systemone`** contract, covering both
//! a local **`laya-serve`** sidecar and **Jev on OpenRouter** — they speak the same
//! wire shape (Laya is "Jev-compatible"), so one client serves both by swapping the
//! base URL, optional bearer auth, and optional `model` field.
//!
//! The engine is a **router**: it asks a fixed typed question set (an `intent` choice
//! plus a `needs_full_understanding` noul) in one forward pass and maps the calibrated
//! answers to [`Decision::Resolve`] / [`Decision::Defer`] under a strict confidence
//! floor. It never extracts free-form slots — anything open-ended defers to System-2.
//!
//! Wire shapes (verified live against OpenRouter's `POST /api/v1/systemone`, which is
//! the shared System One contract the local `laya-serve` sidecar also speaks):
//! ```json
//! // request
//! { "model": "typesafe/jev-1.13",              // omitted for laya-serve
//!   "state": { "message": "<transcript>" },
//!   "questions": {
//!     "intent": { "type": "choice", "instructions": "...", "criteria": { "<id>": "<desc>" } },
//!     "needs_full_understanding": { "type": "noul", "instructions": "..." } } }
//! // response — a `choice` answer carries `choice` + `confidence` + `probabilities`;
//! // a `noul` answer carries only its `noul` probability (0..1), no confidence field.
//! { "model": "...", "answers": {
//!     "intent": { "type": "choice", "choice": "<id>", "probabilities": {..}, "confidence": 0.97 },
//!     "needs_full_understanding": { "type": "noul", "noul": 0.03 } },
//!   "usage": {..} }
//! ```

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

use super::{Decision, DecisionEngine, DecisionRequest, Resolution};

/// The `noul` question id used to detect turns that need System-2 reasoning.
const NEEDS_FULL: &str = "needs_full_understanding";

/// Intents that are ambiguous without a place. When one of these is chosen confidently
/// but the turn still defers (the model judged it "needs a location"), the engine retries
/// once with the home location folded into the question — a second fast System One call,
/// still far cheaper than a System-2 turn. Verified live: "what's the weather" defers
/// (noul ~0.94) but "what's the weather in <place>" resolves (noul ~0.3).
const LOCATION_INTENTS: &[&str] = &["weather"];

/// An HTTP System-1 engine speaking `/v1/systemone`.
pub struct HttpDecider {
    client: reqwest::Client,
    /// The fully-qualified endpoint, e.g. `http://127.0.0.1:8000/v1/systemone`.
    url: String,
    /// Model id sent in the body (`typesafe/jev-1.13` for OpenRouter); `None` for
    /// laya-serve, which serves whatever checkpoint it loaded.
    model: Option<String>,
    /// Bearer token (OpenRouter). `None` = no `Authorization` header.
    api_key: Option<String>,
    /// Short label for logs/settings.
    name: String,
    /// Confidence floor: both the intent choice and the `needs_full_understanding`
    /// answer must clear this, or the turn defers.
    min_confidence: f64,
    /// The intent labels this router may resolve (does not include the implicit
    /// `other` escape hatch).
    intents: Vec<String>,
}

impl HttpDecider {
    fn build(
        name: impl Into<String>,
        base_url: &str,
        model: Option<String>,
        api_key: Option<String>,
        min_confidence: f64,
        intents: Vec<String>,
    ) -> Self {
        // A short client timeout keeps a hung/absent sidecar from blocking the turn —
        // on any error the caller simply defers to System-2.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(4))
            .build()
            .unwrap_or_default();
        let url = format!("{}/v1/systemone", base_url.trim_end_matches('/'));
        Self {
            client,
            url,
            model,
            api_key,
            name: name.into(),
            min_confidence,
            intents,
        }
    }

    /// A client pointed at a local `laya-serve` sidecar (no auth, no model pin).
    pub fn laya_serve(base_url: &str, min_confidence: f64, intents: Vec<String>) -> Self {
        Self::build("laya-serve", base_url, None, None, min_confidence, intents)
    }

    /// A client pointed at OpenRouter's Jev decisions endpoint (bearer auth + model id).
    /// `base_url` should be the API root (e.g. `https://openrouter.ai/api`).
    pub fn jev(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
        min_confidence: f64,
        intents: Vec<String>,
    ) -> Self {
        Self::build(
            "jev",
            base_url,
            Some(model.to_string()),
            api_key,
            min_confidence,
            intents,
        )
    }
}

#[async_trait]
impl DecisionEngine for HttpDecider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn decide(&self, req: &DecisionRequest) -> Result<Decision> {
        let body = build_request(&req.transcript, self.model.as_deref(), &self.intents);
        let value = self.post(&body).await?;
        let decision = interpret(&value, &self.intents, self.min_confidence);
        if matches!(decision, Decision::Resolve(_)) {
            return Ok(decision);
        }

        // Escalate once: a confident, location-dependent intent (e.g. `weather`) that
        // deferred usually just lacks a place. Retry with the home location folded into
        // the question text — the model's `needs_full_understanding` noul then drops
        // below the defer line. Two fast System One calls still beat a System-2 turn.
        if let Some(loc) = req.location.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
            if confident_location_intent(&value, &self.intents, self.min_confidence).is_some() {
                let augmented = format!("{} in {}", req.transcript.trim(), loc);
                let body = build_request(&augmented, self.model.as_deref(), &self.intents);
                let value = self.post(&body).await?;
                return Ok(interpret(&value, &self.intents, self.min_confidence));
            }
        }
        Ok(decision)
    }
}

impl HttpDecider {
    /// POST a `/v1/systemone` body and decode the JSON response, erroring on a non-2xx
    /// status so the caller defers to System-2.
    async fn post(&self, body: &Value) -> Result<Value> {
        let mut rb = self.client.post(&self.url).json(body);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await.context("posting to /v1/systemone")?;
        let status = resp.status();
        let value: Value = resp
            .json()
            .await
            .context("decoding /v1/systemone response")?;
        if !status.is_success() {
            anyhow::bail!("/v1/systemone returned {status}: {value}");
        }
        Ok(value)
    }
}

/// Human-readable criteria for the built-in intents (M1: weather; M3 adds the rest).
/// Unknown/custom labels fall back to the label itself as their description.
fn intent_description(intent: &str) -> &str {
    match intent {
        "weather" => "current conditions or the forecast",
        "timer" => "start, cancel, or ask about a timer or alarm",
        "recipe_nav" => "navigate or scroll the recipe already on screen",
        "music" => "play, pause, skip, or change music volume",
        "shopping_add" => "add an item to the shopping list",
        "smalltalk" => "a greeting, thanks, or acknowledgement needing no data",
        _ => "",
    }
}

/// Build the `/v1/systemone` request body: the transcript as state plus the fixed
/// routing question set (an `intent` choice over `intents` + `other`, and a
/// `needs_full_understanding` noul). Pure, so it is unit-tested without a server.
pub fn build_request(transcript: &str, model: Option<&str>, intents: &[String]) -> Value {
    let mut criteria = Map::new();
    for intent in intents {
        let desc = intent_description(intent);
        let desc = if desc.is_empty() { intent.as_str() } else { desc };
        criteria.insert(intent.clone(), json!(desc));
    }
    criteria.insert(
        "other".to_string(),
        json!("anything that needs reasoning, memory, or open-ended understanding"),
    );

    let questions = json!({
        "intent": {
            "type": "choice",
            "instructions": "What does the user want in `message`? Pick the single best fit.",
            "criteria": Value::Object(criteria),
        },
        NEEDS_FULL: {
            "type": "noul",
            "instructions": "Does answering `message` require open-ended reasoning, personal \
                             memory, or details not implied by a simple command?",
        },
    });

    let mut body = Map::new();
    if let Some(m) = model {
        body.insert("model".to_string(), json!(m));
    }
    body.insert("state".to_string(), json!({ "message": transcript }));
    body.insert("questions".to_string(), questions);
    Value::Object(body)
}

/// Map a `/v1/systemone` response to a [`Decision`] under the confidence floor. Pure,
/// so it is unit-tested against canned JSON. Any missing/al­ternate field defers safely.
pub fn interpret(resp: &Value, intents: &[String], min_confidence: f64) -> Decision {
    let answers = match resp.get("answers") {
        Some(a) => a,
        None => return Decision::Defer,
    };

    // The open-ended escape hatch: a high `needs_full_understanding` noul means the
    // turn needs System-2. A `noul` answer carries only the probability (0..1) — no
    // confidence field — so we gate on that probability alone; a missing answer defers.
    let nfu_p_true = answers
        .get(NEEDS_FULL)
        .and_then(|a| a.get("noul"))
        .and_then(Value::as_f64);
    if nfu_p_true.map(|p| p >= 0.5).unwrap_or(true) {
        return Decision::Defer; // needs System-2 (or the answer was missing)
    }

    let intent_ans = match answers.get("intent") {
        Some(a) => a,
        None => return Decision::Defer,
    };
    let label = match intent_ans.get("choice").and_then(Value::as_str) {
        Some(l) => l,
        None => return Decision::Defer,
    };
    if label == "other" || !intents.iter().any(|i| i == label) {
        return Decision::Defer;
    }
    let intent_conf = choice_confidence(Some(intent_ans));

    // Strict gate: the chosen intent must clear the floor.
    if intent_conf >= min_confidence {
        Decision::Resolve(Resolution {
            intent: label.to_string(),
            confidence: intent_conf,
        })
    } else {
        Decision::Defer
    }
}

/// If the response confidently chose a location-dependent intent (in both
/// [`LOCATION_INTENTS`] and this router's `intents`) that nonetheless deferred, return
/// it — the signal to retry with the home location folded into the question. Returns
/// `None` when the choice was open-ended, low-confidence, or not location-dependent.
fn confident_location_intent<'a>(
    resp: &'a Value,
    intents: &[String],
    min_confidence: f64,
) -> Option<&'a str> {
    let intent = resp.get("answers")?.get("intent")?;
    let choice = intent.get("choice").and_then(Value::as_str)?;
    if !LOCATION_INTENTS.contains(&choice) || !intents.iter().any(|i| i == choice) {
        return None;
    }
    if choice_confidence(Some(intent)) >= min_confidence {
        Some(choice)
    } else {
        None
    }
}

/// Read a `choice`/`score` answer's calibrated `confidence` (0.0 when absent). The
/// OpenRouter Jev `/v1/systemone` response names this field `confidence`; a local
/// `laya-serve` build that instead emits `answer_confidence` is also accepted. `noul`
/// answers have no confidence field and are gated on their probability instead.
fn choice_confidence(answer: Option<&Value>) -> f64 {
    answer
        .and_then(|a| a.get("confidence").or_else(|| a.get("answer_confidence")))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intents() -> Vec<String> {
        vec!["weather".to_string(), "timer".to_string()]
    }

    #[test]
    fn build_request_shapes_state_and_questions() {
        let body = build_request("show me the weather", Some("typesafe/jev-1.13"), &intents());
        assert_eq!(body["model"], json!("typesafe/jev-1.13"));
        assert_eq!(body["state"]["message"], json!("show me the weather"));
        assert_eq!(body["questions"]["intent"]["type"], json!("choice"));
        // criteria carries every intent plus the `other` escape hatch.
        let crit = &body["questions"]["intent"]["criteria"];
        assert!(crit.get("weather").is_some());
        assert!(crit.get("timer").is_some());
        assert!(crit.get("other").is_some());
        assert_eq!(body["questions"][NEEDS_FULL]["type"], json!("noul"));
        // laya-serve omits the model.
        let no_model = build_request("hi", None, &intents());
        assert!(no_model.get("model").is_none());
    }

    /// A System One response in the real OpenRouter/laya-serve shape: the `choice`
    /// answer carries `confidence`; the `noul` answer carries only its probability.
    fn resp(intent: &str, intent_conf: f64, nfu_p: f64) -> Value {
        json!({
            "model": "typesafe/jev-1.13",
            "answers": {
                "intent": {
                    "type": "choice", "choice": intent,
                    "probabilities": {}, "confidence": intent_conf,
                },
                "needs_full_understanding": {
                    "type": "noul", "noul": nfu_p,
                },
            },
            "usage": {}
        })
    }

    #[test]
    fn interpret_resolves_confident_closed_intent() {
        let d = interpret(&resp("weather", 0.97, 0.02), &intents(), 0.85);
        match d {
            Decision::Resolve(r) => {
                assert_eq!(r.intent, "weather");
                assert!((r.confidence - 0.97).abs() < 1e-9);
            }
            Decision::Defer => panic!("expected Resolve"),
        }
    }

    #[test]
    fn interpret_defers_on_low_confidence() {
        assert_eq!(
            interpret(&resp("weather", 0.60, 0.02), &intents(), 0.85),
            Decision::Defer
        );
    }

    #[test]
    fn interpret_defers_when_needs_full_understanding() {
        assert_eq!(
            interpret(&resp("weather", 0.97, 0.90), &intents(), 0.85),
            Decision::Defer
        );
    }

    #[test]
    fn interpret_defers_on_other_or_unknown_intent() {
        assert_eq!(
            interpret(&resp("other", 0.99, 0.01), &intents(), 0.85),
            Decision::Defer
        );
        assert_eq!(
            interpret(&resp("philosophy", 0.99, 0.01), &intents(), 0.85),
            Decision::Defer
        );
    }

    #[test]
    fn confident_location_intent_flags_deferred_weather_for_retry() {
        // Confident `weather` that deferred (high noul) → retry candidate.
        let deferred_weather = resp("weather", 0.99, 0.94);
        assert_eq!(
            confident_location_intent(&deferred_weather, &intents(), 0.85),
            Some("weather")
        );
        // A confident but non-location intent (`timer`) is never retried.
        assert_eq!(
            confident_location_intent(&resp("timer", 0.99, 0.9), &intents(), 0.85),
            None
        );
        // Low-confidence or open-ended choices are not retried.
        assert_eq!(
            confident_location_intent(&resp("weather", 0.50, 0.94), &intents(), 0.85),
            None
        );
        assert_eq!(
            confident_location_intent(&resp("other", 0.99, 0.94), &intents(), 0.85),
            None
        );
    }

    #[test]
    fn interpret_accepts_legacy_answer_confidence_from_laya_serve() {
        // A laya-serve build that emits `answer_confidence` instead of `confidence`
        // on the choice answer still resolves.
        let legacy = json!({
            "answers": {
                "intent": { "type": "choice", "choice": "weather",
                            "probabilities": {}, "answer_confidence": 0.96 },
                "needs_full_understanding": { "type": "noul", "noul": 0.02 },
            }
        });
        assert!(matches!(
            interpret(&legacy, &intents(), 0.85),
            Decision::Resolve(_)
        ));
    }

    #[test]
    fn interpret_defers_on_malformed_response() {
        assert_eq!(interpret(&json!({}), &intents(), 0.85), Decision::Defer);
        assert_eq!(
            interpret(&json!({"answers": {}}), &intents(), 0.85),
            Decision::Defer
        );
    }
}
