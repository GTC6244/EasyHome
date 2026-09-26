//! A scripted [`DecisionEngine`] for tests: returns a fixed [`Decision`] regardless of
//! input, so pipeline tests can exercise the Resolve/Defer fork deterministically.

use anyhow::Result;
use async_trait::async_trait;

use super::{Decision, DecisionEngine, DecisionRequest, Resolution};

/// A decision engine that always returns the same scripted decision.
pub struct MockDecider {
    decision: Decision,
}

impl MockDecider {
    /// An engine that always defers (like `NoDecision`, but reports `name() == "mock"`
    /// so the pipeline actually invokes it).
    pub fn defer() -> Self {
        Self {
            decision: Decision::Defer,
        }
    }

    /// An engine that always resolves the given `intent` at `confidence`.
    pub fn resolve(intent: impl Into<String>, confidence: f64) -> Self {
        Self {
            decision: Decision::Resolve(Resolution {
                intent: intent.into(),
                confidence,
            }),
        }
    }
}

#[async_trait]
impl DecisionEngine for MockDecider {
    fn name(&self) -> &str {
        "mock"
    }
    async fn decide(&self, _req: &DecisionRequest) -> Result<Decision> {
        Ok(self.decision.clone())
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
    async fn mock_reports_its_name_and_scripted_decision() {
        let deferring = MockDecider::defer();
        assert_eq!(deferring.name(), "mock");
        assert_eq!(deferring.decide(&req("hi")).await.unwrap(), Decision::Defer);

        let resolving = MockDecider::resolve("weather", 0.97);
        match resolving.decide(&req("weather?")).await.unwrap() {
            Decision::Resolve(r) => {
                assert_eq!(r.intent, "weather");
                assert!((r.confidence - 0.97).abs() < 1e-9);
            }
            Decision::Defer => panic!("expected a Resolve"),
        }
    }
}
