// Raised for the embedded HelixDB engine: its deeply nested generic types exceed
// the default type-layout recursion limit of 128.
#![recursion_limit = "512"]
//! Ambient Smart Display — Mac Mini assistant orchestrator (Plan.MD Phase 4).
//!
//! The "brain": a Wyoming server the Echo Show discovers over mDNS, wiring
//! downstream Whisper (STT) → a pluggable LLM + persistent SQLite memory → Piper
//! (TTS), and streaming the synthesized reply back to the device. See
//! `architecture.md` §2.3 for the component design and `Plan.MD` Phase 4 for the
//! delivery scope.
//!
//! The library exposes every layer so both the `anamanti-core` binary and
//! the integration tests drive the same code.

#[cfg(feature = "aec")]
pub mod aec;
pub mod audio_dump;
pub mod cadora;
pub mod calendar;
pub mod config;
pub mod control;
pub mod directions;
pub mod discovery;
pub mod drive_consent;
pub mod llm;
pub mod memory;
pub mod music;
pub mod notify;
pub mod orchestrator;
pub mod recipe;
pub mod server;
pub mod settings;
pub mod speaker;
pub mod spotify_consent;
pub mod stt;
pub mod system1;
pub mod weather;
pub mod webconfig;
pub mod wyoming;
