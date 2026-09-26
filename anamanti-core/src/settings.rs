//! Runtime-swappable orchestrator settings (Plan.MD Phase 6).
//!
//! Phase 4 bound the LLM backend and Piper voice once, from the environment. Phase
//! 6 makes them **changeable at runtime** from the on-device settings screen: the
//! device sends a project-local Wyoming control frame (see `wyoming::protocol`
//! `ambient-set-settings`) and the orchestrator rebuilds the selected backend and
//! swaps it in place, without a restart.
//!
//! [`SharedSettings`] holds the live [`RuntimeSettings`] behind an `RwLock` so the
//! (cheaply cloneable) [`crate::orchestrator::Pipeline`] takes a fresh snapshot at
//! the start of each turn while a concurrent control request can swap the backend.
//! An [`LlmFactory`] carries the immutable credentials/endpoints needed to
//! (re)build any backend from a `(backend, model)` pair, so a swap never has to
//! re-read the environment.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cadora::{
    CadoraVoiceApi, GroceryController, DEFAULT_BASE_URL as CADORA_DEFAULT_BASE_URL,
};
use crate::calendar::CalendarSource;
use crate::directions::DirectionsProvider;
use crate::llm::anthropic_auth::{AnthropicAuth, AnthropicTokenProvider};
use crate::llm::{
    anthropic::AnthropicBackend, mock::MockLlm, ollama::OllamaBackend, openai::OpenAiBackend,
    LlmBackend,
};
use crate::music::{SpotifyController, SpotifyWebApi};

/// Which implementation drives the local/cloud LLM backends: the hand-rolled HTTP
/// clients, or the rig-core agent framework (selected at runtime with
/// `llm.engine="rig"`; required for the web-search tool).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LlmEngine {
    /// Hand-rolled `ollama.rs` / `anthropic.rs` HTTP clients (always available).
    #[default]
    Native,
    /// rig-core agents (`llm.engine="rig"`; needs the `rig` feature to take
    /// effect — otherwise it transparently falls back to `Native`).
    Rig,
}

impl LlmEngine {
    /// Canonical lowercase label.
    pub fn as_str(self) -> &'static str {
        match self {
            LlmEngine::Native => "native",
            LlmEngine::Rig => "rig",
        }
    }

    /// Parse a label; anything other than `rig` is `Native`.
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            "rig" | "rig-core" | "rigcore" => LlmEngine::Rig,
            _ => LlmEngine::Native,
        }
    }
}

/// Default trailing-silence (ms) the energy VAD waits for after speech before
/// finalizing the STT transcript. A mild reduction from the historical 900 ms to
/// cut end-of-turn latency; A/B-tunable from the device settings screen.
pub const DEFAULT_END_SILENCE_MS: u64 = 700;

/// Default RMS (i16 units) above which an incoming chunk counts as speech rather
/// than room noise. Set above the Echo Show's measured far-field idle noise floor
/// (~100–200 i16), which the original 120 sat *inside* — so every frame read as
/// speech and end-of-speech never fired, stalling the turn. Speech runs ~1000+, so
/// 450 cleanly separates the two. Lower it for a very quiet mic, raise it for a
/// noisy room (A/B-tunable from the device settings screen / config page).
pub const DEFAULT_VOICE_RMS_THRESHOLD: f64 = 450.0;

fn default_end_silence_ms() -> u64 {
    DEFAULT_END_SILENCE_MS
}

fn default_voice_rms_threshold() -> f64 {
    DEFAULT_VOICE_RMS_THRESHOLD
}

fn default_anthropic_auth() -> String {
    AnthropicAuth::ApiKey.as_str().to_string()
}

/// Google Drive photo-slideshow credentials + linkage, owned by the orchestrator.
///
/// The interim idle-screen photo source is Google Drive (`drive.readonly`). The
/// orchestrator is the single source of truth: it runs the one-time OAuth consent
/// (see [`crate::drive_consent`], driven from the config page) and holds the
/// "Desktop app" OAuth **client id/secret** plus the resulting **refresh token**
/// and the chosen **folder ids**. The device pulls this whole bundle over Wyoming
/// (`ambient-get-drive-token`) and mints Drive access tokens on-device — so the
/// tablet APK ships with no baked-in credentials.
///
/// Plaintext secrets (0600 settings file); keep on a trusted machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DriveConfig {
    /// "Desktop app" OAuth client id (loopback consent). `None`/empty = unset.
    #[serde(default)]
    pub client_id: Option<String>,
    /// "Desktop app" OAuth client secret. `None`/empty = unset.
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Long-lived refresh token minted by the consent flow. `None` = not linked.
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Drive folder ids the slideshow reads images from.
    #[serde(default)]
    pub folder_ids: Vec<String>,
    /// OAuth scope granted (informational; `drive.readonly` by default).
    #[serde(default)]
    pub scope: Option<String>,
}

impl DriveConfig {
    fn non_empty(v: &Option<String>) -> bool {
        v.as_deref().is_some_and(|s| !s.is_empty())
    }

    /// True when both OAuth client credentials are present — i.e. the device can
    /// refresh Drive access tokens. Mirrors the old build-time `kGoogleDriveConfigured`.
    pub fn configured(&self) -> bool {
        Self::non_empty(&self.client_id) && Self::non_empty(&self.client_secret)
    }

    /// True when configured *and* a refresh token exists — Drive is fully linked.
    pub fn linked(&self) -> bool {
        self.configured() && Self::non_empty(&self.refresh_token)
    }
}

/// A requested change to the Drive config. Absent fields are left unchanged; a
/// `Some(None)` clears a value, `Some(Some(v))` sets it (empty strings clear).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriveUpdate {
    pub client_id: Option<Option<String>>,
    pub client_secret: Option<Option<String>>,
    pub refresh_token: Option<Option<String>>,
    pub folder_ids: Option<Vec<String>>,
    pub scope: Option<Option<String>>,
}

/// A person who lives in the home — the canonical household directory the
/// orchestrator holds so tools and the LLM prompt have grounded context about who
/// is here and how to reach them. Contact details are PII; they live in the `0600`
/// settings file, so keep it on a trusted machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HouseholdMember {
    /// The person's name as the assistant should address them. Required (an empty
    /// name drops the member on save).
    pub name: String,
    /// Email addresses for this person (any number). Empty = none known.
    #[serde(default)]
    pub emails: Vec<String>,
    /// Phone numbers for this person (any number). Empty = none known.
    #[serde(default)]
    pub phones: Vec<String>,
    /// Optional relationship / role note (e.g. "parent", "kid", "roommate").
    #[serde(default)]
    pub relationship: Option<String>,
}

/// Canonical household + home information the orchestrator provides as context:
/// **where** the home is (grounds an unqualified "here" for weather / nearby
/// questions) and **who** lives in it (names + contact details). The location seeds
/// from the config file's home_location / weather_units at boot and then becomes
/// editable from the config dashboard; the roster is dashboard-only.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Household {
    /// The home's physical location (e.g. "Austin, Texas" or a street address).
    /// `None`/blank = unset (the prompt omits the location line).
    #[serde(default)]
    pub location: Option<String>,
    /// Preferred measurement units ("imperial" / "metric"). `None` = model's choice.
    #[serde(default)]
    pub weather_units: Option<String>,
    /// The people who live in the home.
    #[serde(default)]
    pub members: Vec<HouseholdMember>,
}

impl Household {
    /// The roster member whose name matches `name` (case-insensitive, trimmed), or
    /// `None`. Used to reconcile an identified speaker (speaker_id_plan.md) with the
    /// canonical household directory so the prompt can note who they are.
    pub fn member_matching(&self, name: &str) -> Option<&HouseholdMember> {
        let needle = name.trim();
        if needle.is_empty() {
            return None;
        }
        self.members
            .iter()
            .find(|m| m.name.trim().eq_ignore_ascii_case(needle))
    }

    /// True when nothing is configured — no location, units, or members.
    pub fn is_empty(&self) -> bool {
        self.location.as_deref().is_none_or(str::is_empty)
            && self.weather_units.as_deref().is_none_or(str::is_empty)
            && self.members.is_empty()
    }

    /// A cleaned copy: strings trimmed, blanks dropped, and any member without a
    /// name removed (so a half-filled form row never becomes a nameless entry).
    pub fn sanitized(&self) -> Household {
        let clean_opt = |v: &Option<String>| {
            v.as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let clean_list = |v: &[String]| {
            v.iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        };
        Household {
            location: clean_opt(&self.location),
            weather_units: clean_opt(&self.weather_units),
            members: self
                .members
                .iter()
                .map(|m| HouseholdMember {
                    name: m.name.trim().to_string(),
                    emails: clean_list(&m.emails),
                    phones: clean_list(&m.phones),
                    relationship: clean_opt(&m.relationship),
                })
                .filter(|m| !m.name.is_empty())
                .collect(),
        }
    }
}

/// Spotify voice-control credentials + linkage, owned by the orchestrator.
///
/// Drives the rig-engine `spotify_control` tool (`crate::llm::rig`) over the
/// Spotify Web API (`crate::music::spotify`). The orchestrator holds the Premium
/// account's app **client id/secret** plus the **refresh token** minted by the
/// one-time consent flow ([`crate::spotify_consent`], driven from the config page
/// Music tab). Unlike Drive, this is used by the LLM tool set, so a change rebuilds
/// the backend so the tool is advertised/withdrawn live.
///
/// Plaintext secrets (0600 settings file); keep on a trusted machine. Requires
/// Spotify Premium.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SpotifyConfig {
    /// Spotify app client id. `None`/empty = unset.
    #[serde(default)]
    pub client_id: Option<String>,
    /// Spotify app client secret. `None`/empty = unset.
    #[serde(default)]
    pub client_secret: Option<String>,
    /// Long-lived refresh token minted by consent. `None` = not linked.
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// librespot Connect device to target (default `"Ambient"` when unset).
    #[serde(default)]
    pub device_name: Option<String>,
    /// OAuth scope granted (informational).
    #[serde(default)]
    pub scope: Option<String>,
}

impl SpotifyConfig {
    fn non_empty(v: &Option<String>) -> bool {
        v.as_deref().is_some_and(|s| !s.is_empty())
    }

    /// True when both app credentials are present (consent can run).
    pub fn configured(&self) -> bool {
        Self::non_empty(&self.client_id) && Self::non_empty(&self.client_secret)
    }

    /// True when configured *and* a refresh token exists — the tool can be built.
    pub fn linked(&self) -> bool {
        self.configured() && Self::non_empty(&self.refresh_token)
    }

    /// The device name to target, defaulting to `"Ambient"` (matches librespot).
    pub fn device_label(&self) -> String {
        self.device_name
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("Ambient")
            .to_string()
    }

    /// Build the live controller when fully linked, else `None` (so the
    /// `spotify_control` tool simply isn't advertised).
    pub fn controller(&self) -> Option<Arc<dyn SpotifyController>> {
        if !self.linked() {
            return None;
        }
        Some(Arc::new(SpotifyWebApi::new(
            self.client_id.clone().unwrap_or_default(),
            self.client_secret.clone().unwrap_or_default(),
            self.refresh_token.clone().unwrap_or_default(),
            self.device_label(),
        )))
    }
}

/// Cadora shared-household **shopping list** linkage, owned by the orchestrator.
///
/// Drives the rig-engine `shopping_list_add` tool (`crate::llm::rig`) over Cadora's
/// voice API (`crate::cadora`). The orchestrator holds the durable `vl_…` voice-link
/// token minted by redeeming a spoken 6-digit pairing code (config page → Household
/// tab). Like Spotify, this is used by the LLM tool set, so a change rebuilds the
/// backend so the tool is advertised/withdrawn live.
///
/// Plaintext token (0600 settings file); keep on a trusted machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CadoraConfig {
    /// Cadora app-server base URL. `None`/empty ⇒ the built-in default.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Long-lived `vl_…` voice-link token from the pairing redeem. `None` = not linked.
    #[serde(default)]
    pub link_token: Option<String>,
}

impl CadoraConfig {
    fn non_empty(v: &Option<String>) -> bool {
        v.as_deref().is_some_and(|s| !s.is_empty())
    }

    /// The base URL to target, defaulting to the built-in Cadora server when unset.
    pub fn base_url_or_default(&self) -> String {
        self.base_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(CADORA_DEFAULT_BASE_URL)
            .to_string()
    }

    /// True once a voice-link token exists — the tool can be built.
    pub fn linked(&self) -> bool {
        Self::non_empty(&self.link_token)
    }

    /// Build the live controller when linked, else `None` (so the `shopping_list_add`
    /// tool simply isn't advertised).
    pub fn controller(&self) -> Option<Arc<dyn GroceryController>> {
        if !self.linked() {
            return None;
        }
        Some(Arc::new(CadoraVoiceApi::new(
            self.base_url_or_default(),
            self.link_token.clone().unwrap_or_default(),
        )))
    }
}

/// A requested change to the Spotify config (tri-state per field, like `DriveUpdate`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpotifyUpdate {
    pub client_id: Option<Option<String>>,
    pub client_secret: Option<Option<String>>,
    pub refresh_token: Option<Option<String>>,
    pub device_name: Option<Option<String>>,
    pub scope: Option<Option<String>>,
}

/// A requested change to the Cadora shopping-list config (tri-state per field, like
/// [`SpotifyUpdate`]). Applied by [`SharedSettings::apply_cadora`], which rebuilds the
/// backend so the `shopping_list_add` tool is advertised/withdrawn live.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CadoraUpdate {
    pub base_url: Option<Option<String>>,
    pub link_token: Option<Option<String>>,
}

/// A requested change to the directions tool config. `mapbox_token` is tri-state:
/// `None` = leave unchanged; `Some(None)`/`Some(Some(""))` = clear; `Some(Some(v))` =
/// set. Applied by [`SharedSettings::apply_directions`], which rebuilds the backend so
/// the `directions_lookup` tool is advertised/withdrawn live.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DirectionsUpdate {
    pub mapbox_token: Option<Option<String>>,
}

/// The mutable settings persisted to disk so page/device changes survive a
/// restart. Contains the Tavily key in plaintext, so the file is written with
/// `0600` permissions on unix and should stay on a trusted machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSettings {
    pub engine: String,
    pub web_search: bool,
    pub search_provider: String,
    pub search_api_key: Option<String>,
    pub llm_backend: String,
    pub llm_model: Option<String>,
    /// Runtime-set Anthropic API key, so the cloud backend can be selected without
    /// a restart even if `ANTHROPIC_API_KEY` was absent at boot. Plaintext (0600
    /// file); defaulted (absent) for older files, which then fall back to the env key.
    #[serde(default)]
    pub anthropic_api_key: Option<String>,
    /// Runtime-set OpenAI API key (same rationale as `anthropic_api_key`).
    #[serde(default)]
    pub openai_api_key: Option<String>,
    /// Runtime-set Anthropic subscription OAuth token (config page). Defaulted (absent)
    /// for older files, which then fall back to the env token / token command.
    #[serde(default)]
    pub anthropic_oauth_token: Option<String>,
    /// Runtime-set Mapbox token for the directions tool. Defaulted (absent) for older
    /// files, which then fall back to the `MAPBOX_TOKEN` env seed.
    #[serde(default)]
    pub mapbox_token: Option<String>,
    /// Anthropic auth mode (`apikey`/`subscription`). Defaulted for older files.
    #[serde(default = "default_anthropic_auth")]
    pub anthropic_auth: String,
    pub tts_voice: Option<String>,
    /// End-of-speech trailing silence in ms (VAD). Defaulted for older files.
    #[serde(default = "default_end_silence_ms")]
    pub end_silence_ms: u64,
    /// Speech-vs-noise RMS threshold (VAD). Defaulted for older files.
    #[serde(default = "default_voice_rms_threshold")]
    pub voice_rms_threshold: f64,
    /// Google Drive photo-slideshow credentials + linkage. Defaulted (empty) for
    /// older files.
    #[serde(default)]
    pub drive: DriveConfig,
    /// Canonical household + home information (location, units, people). Defaulted
    /// (empty) for older files; seeded from env at boot when absent.
    #[serde(default)]
    pub household: Household,
    /// Spotify voice-control credentials + linkage. Defaulted (empty) for older
    /// files.
    #[serde(default)]
    pub spotify: SpotifyConfig,
    /// Cadora shopping-list linkage (voice-link token). Defaulted (empty) for older
    /// files.
    #[serde(default)]
    pub cadora: CadoraConfig,
    /// Selected System-1 backend. Defaults to **empty** (not "none") for older files, so
    /// an old persisted file can't clobber a config-file `system1.backend` seed — the
    /// overlay only applies when this is non-empty (a save always writes a real label).
    #[serde(default)]
    pub system1_backend: String,
    #[serde(default = "default_system1_base_url")]
    pub system1_base_url: String,
    #[serde(default = "default_system1_model")]
    pub system1_model: String,
    #[serde(default = "default_system1_min_confidence")]
    pub system1_min_confidence: f64,
    #[serde(default)]
    pub system1_intents: Vec<String>,
    /// Runtime-set OpenRouter API key (config page). Defaulted (absent) for older files,
    /// which then fall back to the `OPENROUTER_API_KEY` env seed.
    #[serde(default)]
    pub openrouter_api_key: Option<String>,
}

fn default_system1_base_url() -> String {
    DEFAULT_SYSTEM1_BASE_URL.to_string()
}
fn default_system1_model() -> String {
    DEFAULT_SYSTEM1_MODEL.to_string()
}
fn default_system1_min_confidence() -> f64 {
    DEFAULT_SYSTEM1_MIN_CONFIDENCE
}

/// Load persisted settings, or `None` if the file is absent/unreadable.
pub fn load_persisted(path: &Path) -> Option<PersistedSettings> {
    let data = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&data) {
        Ok(p) => Some(p),
        Err(e) => {
            log::warn!("ignoring unreadable settings file {}: {e}", path.display());
            None
        }
    }
}

/// Best-effort write of the persisted settings (0600 on unix). Failures are
/// logged, never fatal — a settings change still takes effect in memory.
fn persist(path: &Path, s: &PersistedSettings) {
    let json = match serde_json::to_string_pretty(s) {
        Ok(j) => j,
        Err(e) => {
            log::warn!("could not serialize settings: {e}");
            return;
        }
    };
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        #[cfg(unix)]
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)?
        };
        #[cfg(not(unix))]
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())
    })();
    if let Err(e) = result {
        log::warn!("could not persist settings to {}: {e}", path.display());
    }
}

/// Immutable inputs needed to (re)build any LLM backend on demand. Captured once
/// from the environment/config so a runtime swap never re-reads `env`.
#[derive(Clone)]
pub struct LlmFactory {
    /// Local Ollama / llama.cpp base URL.
    pub ollama_url: String,
    /// Cloud Anthropic API base URL.
    pub anthropic_base_url: String,
    /// Anthropic API key seeded from the environment at boot. This is only the
    /// initial default: the live key lives in [`RuntimeSettings::anthropic_api_key`]
    /// and `apply` overrides a factory clone with it, so a key entered at runtime
    /// (config page) takes effect without a restart. Absent here *and* in the
    /// runtime settings means the cloud backend can't be selected (rejected in-band).
    pub anthropic_api_key: Option<String>,
    /// `max_tokens` for Anthropic replies (kept small for spoken output).
    pub anthropic_max_tokens: u32,
    /// Cloud OpenAI API base URL.
    pub openai_base_url: String,
    /// OpenAI API key seeded from the environment at boot (initial default only;
    /// the live key lives in [`RuntimeSettings::openai_api_key`] — see
    /// `anthropic_api_key` above).
    pub openai_api_key: Option<String>,
    /// `max_completion_tokens` for OpenAI replies (kept small for spoken output).
    pub openai_max_tokens: u32,
    /// Subscription (OAuth) token source for Anthropic, used when the auth mode is
    /// `Subscription`. Shared with the model catalog so both authenticate the same.
    pub anthropic_token: Option<Arc<AnthropicTokenProvider>>,
    /// Live home location shared with every rebuilt backend's `directions_lookup`
    /// tool as its default origin. Held here (not per-backend) so a household-location
    /// edit — [`SharedSettings::apply_household`] calls `set` on this same cell —
    /// changes the directions origin without rebuilding the LLM. Seeded from the
    /// [`Household`] record at boot.
    pub home_location: crate::directions::LiveHomeLocation,
    /// The live Spotify controller for the `spotify_control` tool, or `None` when
    /// Spotify isn't linked. Set from the current [`SpotifyConfig`] on every
    /// (re)build so the tool reflects the latest consent without re-reading env.
    pub spotify: Option<Arc<dyn SpotifyController>>,
    /// The live Cadora controller for the `shopping_list_add` tool, or `None` when the
    /// shopping list isn't linked. Set from the current [`CadoraConfig`] on every
    /// (re)build so the tool reflects the latest linkage.
    pub cadora: Option<Arc<dyn GroceryController>>,
    /// The web-calendar source for the `calendar_lookup` tool, or `None` when no
    /// subscriptions are configured. Prebuilt once from the config file's `calendar`
    /// block so a rebuild never re-parses config; shared into every rebuilt backend.
    pub calendar: Option<Arc<dyn CalendarSource>>,
    /// The routing provider + imperial-units flag for the `directions_lookup` tool,
    /// or `None` when no provider/token is configured. Built from the live Mapbox token
    /// ([`RuntimeSettings::mapbox_token`]) and refreshed on every (re)build — like
    /// `spotify` — so a token entered on the config page advertises the tool live.
    pub directions: Option<(Arc<dyn DirectionsProvider>, bool)>,
    /// The routing provider label (`directions.provider`, default `mapbox`), kept so
    /// `directions` can be rebuilt from a new runtime token.
    pub directions_provider: String,
    /// Whether directions distances are spoken in imperial units (from the household
    /// `weather_units` at boot). Kept alongside `directions_provider` for rebuilds.
    pub directions_imperial: bool,
    /// The forecast provider for the `weather_lookup` tool, or `None` when weather is
    /// disabled. Keyless (Open-Meteo), so present whenever `weather.enabled`; shared
    /// into every rebuilt backend. The same provider also feeds the ambient push.
    pub weather: Option<Arc<dyn crate::weather::WeatherProvider>>,
    /// Whether weather temperatures are reported in imperial units (°F), from the
    /// household `weather_units` at boot. Mirrors `directions_imperial`.
    pub weather_imperial: bool,
}

impl LlmFactory {
    /// Default model for a backend label when the caller pins none.
    fn default_model(backend: &str) -> Option<&'static str> {
        match backend {
            "ollama" => Some("llama3.2"),
            "anthropic" | "claude" => Some("claude-opus-5"),
            "openai" | "gpt" => Some("gpt-4o-mini"),
            _ => None,
        }
    }

    /// Build a backend from a label (`ollama` / `anthropic` / `openai` / `mock`) and
    /// an optional model. Returns the trait object plus the canonical label and the
    /// resolved model so callers can report exactly what took effect.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &self,
        engine: LlmEngine,
        web_search: bool,
        search_provider: &str,
        search_api_key: Option<&str>,
        backend: &str,
        model: Option<&str>,
        anthropic_auth: AnthropicAuth,
    ) -> Result<(Arc<dyn LlmBackend>, String, Option<String>)> {
        // These only affect the ollama/anthropic arms under the `rig` feature;
        // silence unused warnings on native-only builds.
        let _ = (engine, web_search, search_provider, search_api_key);
        match backend.to_lowercase().as_str() {
            "mock" => Ok((Arc::new(MockLlm::default()), "mock".to_string(), None)),
            "anthropic" | "claude" => {
                let model = model
                    .or(Self::default_model("anthropic"))
                    .unwrap_or("claude-opus-5")
                    .to_string();
                let backend: Arc<dyn LlmBackend> = match anthropic_auth {
                    // Subscription (OAuth) uses the token provider; the rig engine has
                    // no subscription path, so subscription always uses native HTTP.
                    AnthropicAuth::Subscription => {
                        let token = self.anthropic_token.clone().context(
                            "the anthropic subscription backend needs a token — set \
                             ANTHROPIC_OAUTH_TOKEN (run `claude setup-token`) on the orchestrator",
                        )?;
                        Arc::new(AnthropicBackend::with_subscription(
                            &self.anthropic_base_url,
                            token,
                            &model,
                            self.anthropic_max_tokens,
                        ))
                    }
                    AnthropicAuth::ApiKey => {
                        let key = self.anthropic_api_key.clone().context(
                            "the anthropic backend requires ANTHROPIC_API_KEY on the orchestrator \
                             (or switch to subscription auth)",
                        )?;
                        match engine {
                            LlmEngine::Rig => Arc::new(crate::llm::rig::RigBackend::anthropic(
                                &self.anthropic_base_url,
                                &key,
                                &model,
                                self.anthropic_max_tokens,
                                crate::llm::rig::tools_from_config(
                                    web_search,
                                    search_provider,
                                    search_api_key,
                                    self.home_location.clone(),
                                    self.spotify.clone(),
                                    self.calendar.clone(),
                                    self.directions.clone(),
                                    self.cadora.clone(),
                                    self.weather.clone(),
                                    self.weather_imperial,
                                ),
                            )?),
                            _ => Arc::new(AnthropicBackend::new(
                                &self.anthropic_base_url,
                                key,
                                &model,
                                self.anthropic_max_tokens,
                            )),
                        }
                    }
                };
                Ok((backend, "anthropic".to_string(), Some(model)))
            }
            "openai" | "gpt" => {
                let key = self
                    .openai_api_key
                    .clone()
                    .context("the openai backend requires OPENAI_API_KEY on the orchestrator")?;
                let model = model
                    .or(Self::default_model("openai"))
                    .unwrap_or("gpt-4o-mini")
                    .to_string();
                // rig-openai is out of scope for v1: the OpenAI backend always uses
                // the native HTTP client, even under the rig engine.
                let backend: Arc<dyn LlmBackend> = Arc::new(OpenAiBackend::new(
                    &self.openai_base_url,
                    key,
                    &model,
                    self.openai_max_tokens,
                ));
                Ok((backend, "openai".to_string(), Some(model)))
            }
            "ollama" => {
                let model = model.unwrap_or("llama3.2").to_string();
                let backend: Arc<dyn LlmBackend> = match engine {
                    LlmEngine::Rig => Arc::new(crate::llm::rig::RigBackend::ollama(
                        &self.ollama_url,
                        &model,
                        crate::llm::rig::tools_from_config(
                            web_search,
                            search_provider,
                            search_api_key,
                            self.home_location.clone(),
                            self.spotify.clone(),
                            self.calendar.clone(),
                            self.directions.clone(),
                            self.cadora.clone(),
                            self.weather.clone(),
                            self.weather_imperial,
                        ),
                    )?),
                    _ => Arc::new(OllamaBackend::new(&self.ollama_url, &model)),
                };
                Ok((backend, "ollama".to_string(), Some(model)))
            }
            other => {
                anyhow::bail!(
                    "unknown LLM backend `{other}` (expected ollama/anthropic/openai/mock)"
                )
            }
        }
    }
}

/// The live, swappable settings the pipeline reads each turn.
#[derive(Clone)]
pub struct RuntimeSettings {
    /// The currently selected LLM backend.
    pub llm: Arc<dyn LlmBackend>,
    /// Which engine backs ollama/anthropic (native HTTP vs rig-core).
    pub engine: LlmEngine,
    /// Whether the rig web-search tool is enabled (rig engine only).
    pub web_search: bool,
    /// Web-search backend: `duckduckgo` (keyless) or `tavily` (needs a key).
    pub search_provider: String,
    /// API key for the search provider (Tavily). `None` = unset.
    pub search_api_key: Option<String>,
    /// Canonical label of the selected backend (`ollama` / `anthropic` / `mock`).
    pub llm_backend: String,
    /// The resolved model name, if the backend uses one.
    pub llm_model: Option<String>,
    /// Live Anthropic API key. Runtime-settable (config page) so the cloud backend
    /// can be selected without a restart; seeded from `ANTHROPIC_API_KEY` at boot.
    /// `None` = no key (selecting the api-key anthropic backend is rejected).
    pub anthropic_api_key: Option<String>,
    /// Live OpenAI API key. Runtime-settable; seeded from `OPENAI_API_KEY` at boot.
    pub openai_api_key: Option<String>,
    /// Live Anthropic subscription OAuth token. Runtime-settable (config page, when
    /// auth = subscription); seeded from `ANTHROPIC_OAUTH_TOKEN` at boot. `None` = no
    /// override (the provider falls back to env / the token command).
    pub anthropic_oauth_token: Option<String>,
    /// Live Mapbox token for the `directions_lookup` tool. Runtime-settable (config
    /// page Tools tab); seeded from `MAPBOX_TOKEN`/`MAPBOX_ACCESS_TOKEN` at boot.
    /// `None` = the tool isn't advertised.
    pub mapbox_token: Option<String>,
    /// How the Anthropic backend authenticates (API key vs subscription OAuth).
    pub anthropic_auth: AnthropicAuth,
    /// The Piper voice to synthesize with, or `None` for the server default.
    pub tts_voice: Option<String>,
    /// End-of-speech trailing silence (ms) the energy VAD waits for before
    /// finalizing the STT transcript. A/B-tunable from the device.
    pub end_silence_ms: u64,
    /// RMS (i16 units) above which a chunk counts as speech for the VAD.
    pub voice_rms_threshold: f64,
    /// Google Drive photo-slideshow credentials + linkage (orchestrator-owned;
    /// pulled by the device over Wyoming). Orthogonal to the LLM rebuild path.
    pub drive: DriveConfig,
    /// Canonical household + home information (location, units, people) injected
    /// into the per-turn prompt. Orthogonal to the LLM rebuild path.
    pub household: Household,
    /// Spotify voice-control credentials + linkage. Unlike Drive, a change here
    /// rebuilds the backend (via [`SpotifyConfig::controller`]) so the
    /// `spotify_control` tool is advertised/withdrawn live.
    pub spotify: SpotifyConfig,
    /// Cadora shopping-list linkage. Like Spotify, a change here rebuilds the backend
    /// (via [`CadoraConfig::controller`]) so the `shopping_list_add` tool is
    /// advertised/withdrawn live.
    pub cadora: CadoraConfig,
    /// The live System-1 fast-decision selection (plans/system1-fast-decisions.md), read
    /// from the per-turn snapshot so a config-page swap takes effect between turns.
    /// Bundled into one field so the widely-constructed `RuntimeSettings` only gains one.
    pub system1: System1Runtime,
}

/// Default System-1 HTTP base URL (local `laya-serve`).
pub const DEFAULT_SYSTEM1_BASE_URL: &str = "http://127.0.0.1:8000";
/// Default System-1 OpenRouter model id (the `jev` backend).
pub const DEFAULT_SYSTEM1_MODEL: &str = "typesafe/jev-1.13";
/// Default System-1 confidence floor.
pub const DEFAULT_SYSTEM1_MIN_CONFIDENCE: f64 = 0.85;

/// The live System-1 selection: the built engine plus the descriptor needed to rebuild
/// it on a config-page swap and report the current choice. `Default` = disabled
/// (`NoDecision`), so a `RuntimeSettings` literal can spell it `System1Runtime::default()`.
#[derive(Clone)]
pub struct System1Runtime {
    /// The live decision engine (`name() == "none"` ⇒ the turn logic skips the stage).
    pub engine: Arc<dyn crate::system1::DecisionEngine>,
    /// Selected backend label (`none`/`laya-serve`/`jev`/…).
    pub backend: String,
    /// HTTP base URL (`laya-serve` host, or the OpenRouter API root for `jev`).
    pub base_url: String,
    /// Model id for the `jev` backend (OpenRouter).
    pub model: String,
    /// Confidence floor below which a decision defers to System-2.
    pub min_confidence: f64,
    /// Allowed intents (empty = the built-in default set).
    pub intents: Vec<String>,
    /// Live OpenRouter API key for `jev`. Runtime-settable (config page); seeded from
    /// `OPENROUTER_API_KEY` at boot. `None` = unset.
    pub openrouter_api_key: Option<String>,
}

impl Default for System1Runtime {
    fn default() -> Self {
        Self {
            engine: crate::system1::none(),
            backend: "none".to_string(),
            base_url: DEFAULT_SYSTEM1_BASE_URL.to_string(),
            model: DEFAULT_SYSTEM1_MODEL.to_string(),
            min_confidence: DEFAULT_SYSTEM1_MIN_CONFIDENCE,
            intents: Vec::new(),
            openrouter_api_key: None,
        }
    }
}

/// A description of the live System-1 selection, for the config page. Never exposes the
/// OpenRouter key (only whether one is set).
#[derive(Debug, Clone, PartialEq)]
pub struct System1View {
    pub backend: String,
    pub base_url: String,
    pub model: String,
    pub min_confidence: f64,
    pub openrouter_key_set: bool,
    pub intents: Vec<String>,
}

/// A requested System-1 change (config page). Absent fields are left unchanged;
/// `openrouter_api_key` is tri-state (`None` = keep, `Some(None)` = clear,
/// `Some(Some(v))` = set).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct System1Update {
    pub backend: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub min_confidence: Option<f64>,
    pub openrouter_api_key: Option<Option<String>>,
    pub intents: Option<Vec<String>>,
}

/// A description of the settings currently in effect, for reporting back to the
/// device settings screen or the config page.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsView {
    pub llm_backend: String,
    pub llm_model: Option<String>,
    /// Whether an Anthropic API key is configured (env or runtime). The key itself
    /// is never exposed.
    pub anthropic_key_set: bool,
    /// Whether an OpenAI API key is configured. The key itself is never exposed.
    pub openai_key_set: bool,
    /// Whether a runtime Anthropic subscription OAuth token is set. Never exposed.
    pub anthropic_oauth_token_set: bool,
    /// Whether a Mapbox token (directions tool) is configured. Never exposed.
    pub mapbox_token_set: bool,
    /// Anthropic auth mode (API key vs subscription OAuth).
    pub anthropic_auth: AnthropicAuth,
    pub tts_voice: Option<String>,
    pub engine: LlmEngine,
    pub web_search: bool,
    pub search_provider: String,
    /// Whether a search API key is configured. The key itself is never exposed.
    pub search_key_set: bool,
    /// End-of-speech trailing silence (ms) the VAD waits for.
    pub end_silence_ms: u64,
    /// Speech-vs-noise RMS threshold for the VAD.
    pub voice_rms_threshold: f64,
}

/// A requested settings change. Absent fields are left unchanged; a `tts_voice` of
/// `Some(None)` clears the voice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SettingsUpdate {
    pub llm_backend: Option<String>,
    pub llm_model: Option<String>,
    /// Anthropic API key change: `None` = leave unchanged; `Some(None)` = clear;
    /// `Some(Some(v))` = set to `v`. Lets the cloud backend be enabled at runtime.
    pub anthropic_api_key: Option<Option<String>>,
    /// OpenAI API key change (same tri-state semantics as `anthropic_api_key`).
    pub openai_api_key: Option<Option<String>>,
    /// Anthropic subscription OAuth token change (tri-state). Applied without a
    /// backend rebuild — the shared token provider is updated in place.
    pub anthropic_oauth_token: Option<Option<String>>,
    /// New Anthropic auth mode, or `None` to leave it unchanged.
    pub anthropic_auth: Option<AnthropicAuth>,
    /// `None` = leave unchanged; `Some(None)` = clear; `Some(Some(v))` = set to `v`.
    pub tts_voice: Option<Option<String>>,
    /// Switch the LLM engine (native vs rig).
    pub engine: Option<LlmEngine>,
    /// Toggle the rig web-search tool.
    pub web_search: Option<bool>,
    /// Switch the search backend (`duckduckgo` / `tavily`).
    pub search_provider: Option<String>,
    /// `None` = leave unchanged; `Some(None)` = clear; `Some(Some(v))` = set.
    pub search_api_key: Option<Option<String>>,
    /// New end-of-speech trailing silence (ms) for the VAD, or `None` to leave it.
    pub end_silence_ms: Option<u64>,
    /// New speech-vs-noise RMS threshold for the VAD, or `None` to leave it.
    pub voice_rms_threshold: Option<f64>,
}

/// Thread-safe holder for the runtime settings plus the factory that rebuilds
/// backends on a swap. Shared (via `Arc`) by every connection.
pub struct SharedSettings {
    inner: RwLock<RuntimeSettings>,
    factory: LlmFactory,
    /// Where to persist changes, or `None` to keep settings in-memory only.
    persist_path: Option<PathBuf>,
}

impl SharedSettings {
    /// Create an in-memory-only shared holder (no persistence).
    pub fn new(factory: LlmFactory, initial: RuntimeSettings) -> Arc<Self> {
        Self::new_persistent(factory, initial, None)
    }

    /// Create a shared holder that persists every applied change to `persist_path`
    /// (when `Some`), reloaded at boot by the caller.
    pub fn new_persistent(
        factory: LlmFactory,
        initial: RuntimeSettings,
        persist_path: Option<PathBuf>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(initial),
            factory,
            persist_path,
        })
    }

    /// Snapshot the mutable settings for persistence.
    fn persisted_snapshot(s: &RuntimeSettings) -> PersistedSettings {
        PersistedSettings {
            engine: s.engine.as_str().to_string(),
            web_search: s.web_search,
            search_provider: s.search_provider.clone(),
            search_api_key: s.search_api_key.clone(),
            llm_backend: s.llm_backend.clone(),
            llm_model: s.llm_model.clone(),
            anthropic_api_key: s.anthropic_api_key.clone(),
            openai_api_key: s.openai_api_key.clone(),
            anthropic_oauth_token: s.anthropic_oauth_token.clone(),
            mapbox_token: s.mapbox_token.clone(),
            anthropic_auth: s.anthropic_auth.as_str().to_string(),
            tts_voice: s.tts_voice.clone(),
            end_silence_ms: s.end_silence_ms,
            voice_rms_threshold: s.voice_rms_threshold,
            drive: s.drive.clone(),
            household: s.household.clone(),
            spotify: s.spotify.clone(),
            cadora: s.cadora.clone(),
            system1_backend: s.system1.backend.clone(),
            system1_base_url: s.system1.base_url.clone(),
            system1_model: s.system1.model.clone(),
            system1_min_confidence: s.system1.min_confidence,
            system1_intents: s.system1.intents.clone(),
            openrouter_api_key: s.system1.openrouter_api_key.clone(),
        }
    }

    /// A fixed holder around an already-built backend whose LLM cannot be swapped
    /// (its factory has no credentials). Used by tests and any caller that only
    /// needs the immutable Phase-4 behavior. The TTS voice is still settable.
    pub fn fixed(
        llm: Arc<dyn LlmBackend>,
        llm_backend: impl Into<String>,
        tts_voice: Option<String>,
    ) -> Arc<Self> {
        let factory = LlmFactory {
            ollama_url: "http://127.0.0.1:11434".to_string(),
            anthropic_base_url: "https://api.anthropic.com".to_string(),
            anthropic_api_key: None,
            anthropic_max_tokens: 1024,
            openai_base_url: "https://api.openai.com".to_string(),
            openai_api_key: None,
            openai_max_tokens: 1024,
            anthropic_token: None,
            home_location: crate::directions::LiveHomeLocation::default(),
            spotify: None,
            cadora: None,
            calendar: None,
            directions: None,
            directions_provider: String::new(),
            directions_imperial: false,
            weather: None,
            weather_imperial: false,
        };
        Self::new(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".to_string(),
                search_api_key: None,
                llm_backend: llm_backend.into(),
                llm_model: None,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
        )
    }

    /// A snapshot of the live settings (cheap: `Arc`/`String` clones). Taken once
    /// per turn so a mid-turn swap never changes the backend under a running reply.
    pub fn snapshot(&self) -> RuntimeSettings {
        self.inner.read().unwrap().clone()
    }

    /// The current settings, for reporting to the device.
    pub fn view(&self) -> SettingsView {
        let s = self.inner.read().unwrap();
        SettingsView {
            llm_backend: s.llm_backend.clone(),
            llm_model: s.llm_model.clone(),
            anthropic_key_set: s
                .anthropic_api_key
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            openai_key_set: s.openai_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            anthropic_oauth_token_set: s
                .anthropic_oauth_token
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            mapbox_token_set: s.mapbox_token.as_deref().is_some_and(|k| !k.is_empty()),
            anthropic_auth: s.anthropic_auth,
            tts_voice: s.tts_voice.clone(),
            engine: s.engine,
            web_search: s.web_search,
            search_provider: s.search_provider.clone(),
            search_key_set: s.search_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            end_silence_ms: s.end_silence_ms,
            voice_rms_threshold: s.voice_rms_threshold,
        }
    }

    /// Apply a settings change, rebuilding the LLM backend if the backend or model
    /// changed. On success the swap is atomic and the new [`SettingsView`] is
    /// returned; on failure (e.g. anthropic requested with no key) the current
    /// settings are left untouched and the error is returned.
    pub fn apply(&self, update: &SettingsUpdate) -> Result<SettingsView> {
        // Any of these change which backend object we need, so rebuild the LLM.
        let needs_rebuild = update.llm_backend.is_some()
            || update.llm_model.is_some()
            || update.anthropic_auth.is_some()
            || update.engine.is_some()
            || update.web_search.is_some()
            || update.search_provider.is_some()
            || update.search_api_key.is_some()
            || update.anthropic_api_key.is_some()
            || update.openai_api_key.is_some();

        // Build the new LLM *before* taking the write lock so a failed build never
        // leaves the settings half-changed.
        let (rebuilt, targets) = if needs_rebuild {
            let current = self.inner.read().unwrap().clone();
            let target_backend = update
                .llm_backend
                .clone()
                .unwrap_or(current.llm_backend.clone());
            // A backend change with no explicit model resets to that backend's
            // default; a model-only change keeps the current backend.
            let target_model = match (&update.llm_backend, &update.llm_model) {
                (_, Some(m)) => Some(m.clone()),
                (Some(_), None) => None,
                (None, None) => current.llm_model.clone(),
            };
            let target_engine = update.engine.unwrap_or(current.engine);
            let target_auth = update.anthropic_auth.unwrap_or(current.anthropic_auth);
            let target_web_search = update.web_search.unwrap_or(current.web_search);
            let target_provider = update
                .search_provider
                .clone()
                .unwrap_or(current.search_provider.clone());
            let target_key = match &update.search_api_key {
                None => current.search_api_key.clone(),
                Some(k) => k.clone().filter(|s| !s.is_empty()),
            };
            // LLM provider keys: tri-state like the search key. These make the cloud
            // backend selectable at runtime, so build with a factory clone whose keys
            // reflect the target — never re-reading the environment.
            let target_anthropic_key = match &update.anthropic_api_key {
                None => current.anthropic_api_key.clone(),
                Some(k) => k.clone().filter(|s| !s.is_empty()),
            };
            let target_openai_key = match &update.openai_api_key {
                None => current.openai_api_key.clone(),
                Some(k) => k.clone().filter(|s| !s.is_empty()),
            };
            let mut factory = self.factory.clone();
            factory.anthropic_api_key = target_anthropic_key.clone();
            factory.openai_api_key = target_openai_key.clone();
            // Spotify isn't changed by a normal settings update, but the rebuilt tool
            // set must still carry the live controller — source it from current config.
            factory.spotify = current.spotify.controller();
            // Same for the Cadora shopping-list tool: keep the live controller so a
            // normal settings change never drops `shopping_list_add`.
            factory.cadora = current.cadora.controller();
            // Same for the directions tool: rebuild it from the live Mapbox token so a
            // normal settings change never drops `directions_lookup`.
            factory.directions = crate::directions::from_token(
                &factory.directions_provider,
                current.mapbox_token.as_deref(),
                factory.directions_imperial,
            );
            (
                Some(factory.build(
                    target_engine,
                    target_web_search,
                    &target_provider,
                    target_key.as_deref(),
                    &target_backend,
                    target_model.as_deref(),
                    target_auth,
                )?),
                Some((
                    target_engine,
                    target_auth,
                    target_web_search,
                    target_provider,
                    target_key,
                    target_anthropic_key,
                    target_openai_key,
                )),
            )
        } else {
            (None, None)
        };

        let mut w = self.inner.write().unwrap();
        if let Some((llm, label, model)) = rebuilt {
            let (engine, auth, web_search, provider, key, anthropic_key, openai_key) =
                targets.unwrap();
            w.llm = llm;
            w.llm_backend = label;
            w.llm_model = model;
            w.anthropic_auth = auth;
            w.engine = engine;
            w.web_search = web_search;
            w.search_provider = provider;
            w.search_api_key = key;
            w.anthropic_api_key = anthropic_key;
            w.openai_api_key = openai_key;
        }
        if let Some(voice) = &update.tts_voice {
            w.tts_voice = voice.clone().filter(|s| !s.is_empty());
        }
        // The Anthropic subscription OAuth token needs no backend rebuild: the shared
        // token provider is consulted per request, so updating its override cell (and
        // the stored value) takes effect immediately.
        if let Some(tok) = &update.anthropic_oauth_token {
            let tok = tok.clone().filter(|s| !s.is_empty());
            w.anthropic_oauth_token = tok.clone();
            if let Some(provider) = &self.factory.anthropic_token {
                provider.set_override(tok);
            }
        }
        // VAD tuning needs no backend rebuild — it's read from the per-turn snapshot
        // by `stream_to_transcript`. Clamp to sane ranges so a bad request can't wedge
        // end-of-speech detection.
        if let Some(ms) = update.end_silence_ms {
            w.end_silence_ms = ms.clamp(150, 5000);
        }
        if let Some(thr) = update.voice_rms_threshold {
            w.voice_rms_threshold = thr.clamp(0.0, 5000.0);
        }
        let view = SettingsView {
            llm_backend: w.llm_backend.clone(),
            llm_model: w.llm_model.clone(),
            anthropic_key_set: w
                .anthropic_api_key
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            openai_key_set: w.openai_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            anthropic_oauth_token_set: w
                .anthropic_oauth_token
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            mapbox_token_set: w.mapbox_token.as_deref().is_some_and(|k| !k.is_empty()),
            anthropic_auth: w.anthropic_auth,
            tts_voice: w.tts_voice.clone(),
            engine: w.engine,
            web_search: w.web_search,
            search_provider: w.search_provider.clone(),
            search_key_set: w.search_api_key.as_deref().is_some_and(|k| !k.is_empty()),
            end_silence_ms: w.end_silence_ms,
            voice_rms_threshold: w.voice_rms_threshold,
        };
        // Persist the new state (best-effort) after dropping the write lock so IO
        // never blocks a concurrent turn's snapshot.
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        Ok(view)
    }

    /// A snapshot of the live Google Drive config (client creds + refresh token +
    /// folder ids). Read by the `ambient-get-drive-token` control handler and the
    /// consent flow. Cheap clone.
    pub fn drive(&self) -> DriveConfig {
        self.inner.read().unwrap().drive.clone()
    }

    /// A snapshot of the live household + home information (location, units, people).
    /// Read per-turn when assembling the prompt and by the config dashboard. Cheap
    /// clone.
    pub fn household(&self) -> Household {
        self.inner.read().unwrap().household.clone()
    }

    /// The live home-location handle (the factory's shared cell, updated by
    /// [`apply_household`](Self::apply_household)). Handed to the ambient weather push
    /// so it tracks a config-page location edit without a restart.
    pub fn home_location(&self) -> crate::directions::LiveHomeLocation {
        self.factory.home_location.clone()
    }

    /// Replace the whole household record and persist it (best-effort, 0600). The
    /// dashboard household form is a full-record save (edit location/units + the
    /// roster together), so this sets rather than field-diffs; the value is
    /// [`Household::sanitized`] first (trim, drop blanks / nameless members). Never
    /// rebuilds the LLM — household is orthogonal to the backend. Returns the cleaned
    /// value that was stored.
    pub fn apply_household(&self, household: &Household) -> Household {
        let cleaned = household.sanitized();
        // Point the directions tool's live default origin at the new home location.
        // The factory's handle is the same cell every rebuilt backend's tool clones,
        // so this takes effect immediately without rebuilding the LLM.
        self.factory.home_location.set(cleaned.location.clone());
        let mut w = self.inner.write().unwrap();
        w.household = cleaned.clone();
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        cleaned
    }

    /// A snapshot of the live System-1 selection for the config page (never the key).
    pub fn system1_view(&self) -> System1View {
        let s = self.inner.read().unwrap();
        System1View {
            backend: s.system1.backend.clone(),
            base_url: s.system1.base_url.clone(),
            model: s.system1.model.clone(),
            min_confidence: s.system1.min_confidence,
            openrouter_key_set: s
                .system1
                .openrouter_api_key
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            intents: s.system1.intents.clone(),
        }
    }

    /// Directly install a System-1 engine object (no persist). Used at boot by
    /// `Pipeline::with_system1` and by tests to inject a scripted engine; the config
    /// page uses [`apply_system1`](Self::apply_system1) instead.
    pub fn set_system1(&self, engine: Arc<dyn crate::system1::DecisionEngine>) {
        let mut w = self.inner.write().unwrap();
        w.system1.backend = engine.name().to_string();
        w.system1.engine = engine;
    }

    /// Apply a System-1 config change: rebuild the engine (before taking the write lock,
    /// so a failed build leaves settings untouched), swap it in, and persist. Orthogonal
    /// to the LLM backend. Returns the resulting [`System1View`].
    pub fn apply_system1(&self, update: &System1Update) -> Result<System1View> {
        let (backend, base_url, model, min_conf, intents, api_key) = {
            let current = self.inner.read().unwrap();
            let s1 = &current.system1;
            let backend = update.backend.clone().unwrap_or_else(|| s1.backend.clone());
            let base_url = update.base_url.clone().unwrap_or_else(|| s1.base_url.clone());
            let model = update.model.clone().unwrap_or_else(|| s1.model.clone());
            let min_conf = update
                .min_confidence
                .unwrap_or(s1.min_confidence)
                .clamp(0.0, 1.0);
            let intents = update.intents.clone().unwrap_or_else(|| s1.intents.clone());
            let api_key = match &update.openrouter_api_key {
                None => s1.openrouter_api_key.clone(),
                Some(k) => k.clone().filter(|s| !s.is_empty()),
            };
            (backend, base_url, model, min_conf, intents, api_key)
        };

        // Build before the write lock so a bad backend can't half-apply.
        let engine = crate::system1::build(
            &backend,
            &base_url,
            &model,
            api_key.clone(),
            min_conf,
            intents.clone(),
        )?;

        let mut w = self.inner.write().unwrap();
        w.system1 = System1Runtime {
            engine,
            backend,
            base_url,
            model,
            min_confidence: min_conf,
            intents,
            openrouter_api_key: api_key,
        };
        let view = System1View {
            backend: w.system1.backend.clone(),
            base_url: w.system1.base_url.clone(),
            model: w.system1.model.clone(),
            min_confidence: w.system1.min_confidence,
            openrouter_key_set: w
                .system1
                .openrouter_api_key
                .as_deref()
                .is_some_and(|k| !k.is_empty()),
            intents: w.system1.intents.clone(),
        };
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        Ok(view)
    }

    /// Apply a Drive config change and persist it (best-effort, 0600). Drive is
    /// orthogonal to the LLM backend, so this never rebuilds anything. Empty-string
    /// sets are treated as clears. Returns the resulting [`DriveConfig`].
    pub fn apply_drive(&self, update: &DriveUpdate) -> DriveConfig {
        let mut w = self.inner.write().unwrap();
        if let Some(v) = &update.client_id {
            w.drive.client_id = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.client_secret {
            w.drive.client_secret = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.refresh_token {
            w.drive.refresh_token = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(ids) = &update.folder_ids {
            w.drive.folder_ids = ids
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = &update.scope {
            w.drive.scope = v.clone().filter(|s| !s.is_empty());
        }
        let result = w.drive.clone();
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        result
    }

    /// A snapshot of the live Spotify config (client creds + refresh token + device
    /// name). Read by the config-page status endpoint and the consent flow.
    pub fn spotify(&self) -> SpotifyConfig {
        self.inner.read().unwrap().spotify.clone()
    }

    /// Apply a Spotify config change, rebuild the LLM so the `spotify_control` tool
    /// is advertised/withdrawn to match, and persist (best-effort, 0600). Empty-string
    /// sets are treated as clears. Returns the resulting [`SpotifyConfig`].
    ///
    /// The rebuild is best-effort: if it fails (e.g. the current cloud backend can no
    /// longer build), the new Spotify config is still stored and persisted — the tool
    /// then activates on the next successful rebuild or restart — so consent is never
    /// lost to a transient backend error.
    pub fn apply_spotify(&self, update: &SpotifyUpdate) -> SpotifyConfig {
        // Compute the target config from a snapshot of the current settings.
        let current = self.inner.read().unwrap().clone();
        let mut target = current.spotify.clone();
        if let Some(v) = &update.client_id {
            target.client_id = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.client_secret {
            target.client_secret = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.refresh_token {
            target.refresh_token = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.device_name {
            target.device_name = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.scope {
            target.scope = v.clone().filter(|s| !s.is_empty());
        }

        // Rebuild the backend so the tool set reflects the new linkage. Build before
        // taking the write lock; on failure, fall through and still store the config.
        let mut factory = self.factory.clone();
        factory.anthropic_api_key = current.anthropic_api_key.clone();
        factory.openai_api_key = current.openai_api_key.clone();
        factory.spotify = target.controller();
        // Keep the Cadora shopping-list tool live across this rebuild.
        factory.cadora = current.cadora.controller();
        let rebuilt = factory
            .build(
                current.engine,
                current.web_search,
                &current.search_provider,
                current.search_api_key.as_deref(),
                &current.llm_backend,
                current.llm_model.as_deref(),
                current.anthropic_auth,
            )
            .map_err(|e| log::warn!("spotify: applied config but LLM rebuild failed: {e:#}"))
            .ok();

        let mut w = self.inner.write().unwrap();
        if let Some((llm, label, model)) = rebuilt {
            w.llm = llm;
            w.llm_backend = label;
            w.llm_model = model;
        }
        w.spotify = target.clone();
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        target
    }

    /// A snapshot of the live Cadora shopping-list config (base URL + link presence).
    /// Read by the config-page status endpoint and the pairing flow.
    pub fn cadora(&self) -> CadoraConfig {
        self.inner.read().unwrap().cadora.clone()
    }

    /// Apply a Cadora shopping-list config change, rebuild the LLM so the
    /// `shopping_list_add` tool is advertised/withdrawn to match, and persist
    /// (best-effort, 0600). Empty-string sets are treated as clears. Returns the
    /// resulting [`CadoraConfig`].
    ///
    /// Like [`Self::apply_spotify`], the rebuild is best-effort: if it fails the new
    /// config is still stored and persisted, so the tool activates on the next
    /// successful rebuild or restart — linkage is never lost to a transient error.
    pub fn apply_cadora(&self, update: &CadoraUpdate) -> CadoraConfig {
        let current = self.inner.read().unwrap().clone();
        let mut target = current.cadora.clone();
        if let Some(v) = &update.base_url {
            target.base_url = v.clone().filter(|s| !s.is_empty());
        }
        if let Some(v) = &update.link_token {
            target.link_token = v.clone().filter(|s| !s.is_empty());
        }

        // Rebuild the backend so the tool set reflects the new linkage. Build before
        // taking the write lock; on failure, fall through and still store the config.
        let mut factory = self.factory.clone();
        factory.anthropic_api_key = current.anthropic_api_key.clone();
        factory.openai_api_key = current.openai_api_key.clone();
        factory.spotify = current.spotify.controller();
        factory.cadora = target.controller();
        let rebuilt = factory
            .build(
                current.engine,
                current.web_search,
                &current.search_provider,
                current.search_api_key.as_deref(),
                &current.llm_backend,
                current.llm_model.as_deref(),
                current.anthropic_auth,
            )
            .map_err(|e| log::warn!("cadora: applied config but LLM rebuild failed: {e:#}"))
            .ok();

        let mut w = self.inner.write().unwrap();
        if let Some((llm, label, model)) = rebuilt {
            w.llm = llm;
            w.llm_backend = label;
            w.llm_model = model;
        }
        w.cadora = target.clone();
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        target
    }

    /// A snapshot of the live Mapbox token presence (never the value). Read by the
    /// config-page Tools tab status endpoint.
    pub fn mapbox_token_set(&self) -> bool {
        self.inner
            .read()
            .unwrap()
            .mapbox_token
            .as_deref()
            .is_some_and(|s| !s.is_empty())
    }

    /// Apply a directions-tool config change (the Mapbox token), rebuild the LLM so the
    /// `directions_lookup` tool is advertised/withdrawn to match, and persist
    /// (best-effort, 0600). An empty token is treated as a clear. Like
    /// [`Self::apply_spotify`], the rebuild is best-effort: on failure the token is
    /// still stored, so it activates on the next successful rebuild or restart. Returns
    /// whether a token is now set.
    pub fn apply_directions(&self, update: &DirectionsUpdate) -> bool {
        let current = self.inner.read().unwrap().clone();
        let target_token = match &update.mapbox_token {
            None => current.mapbox_token.clone(),
            Some(t) => t.clone().filter(|s| !s.is_empty()),
        };

        // Rebuild the backend so the tool set reflects the new token. Build before
        // taking the write lock; on failure, fall through and still store the token.
        let mut factory = self.factory.clone();
        factory.anthropic_api_key = current.anthropic_api_key.clone();
        factory.openai_api_key = current.openai_api_key.clone();
        factory.spotify = current.spotify.controller();
        factory.cadora = current.cadora.controller();
        factory.directions = crate::directions::from_token(
            &factory.directions_provider,
            target_token.as_deref(),
            factory.directions_imperial,
        );
        let rebuilt = factory
            .build(
                current.engine,
                current.web_search,
                &current.search_provider,
                current.search_api_key.as_deref(),
                &current.llm_backend,
                current.llm_model.as_deref(),
                current.anthropic_auth,
            )
            .map_err(|e| log::warn!("directions: applied token but LLM rebuild failed: {e:#}"))
            .ok();

        let mut w = self.inner.write().unwrap();
        if let Some((llm, label, model)) = rebuilt {
            w.llm = llm;
            w.llm_backend = label;
            w.llm_model = model;
        }
        w.mapbox_token = target_token.clone();
        let set = target_token.as_deref().is_some_and(|s| !s.is_empty());
        let snapshot = self
            .persist_path
            .is_some()
            .then(|| Self::persisted_snapshot(&w));
        drop(w);
        if let (Some(path), Some(snap)) = (&self.persist_path, snapshot) {
            persist(path, &snap);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factory_with_key(key: Option<&str>) -> LlmFactory {
        LlmFactory {
            ollama_url: "http://127.0.0.1:11434".to_string(),
            anthropic_base_url: "https://api.anthropic.com".to_string(),
            anthropic_api_key: key.map(String::from),
            anthropic_max_tokens: 256,
            openai_base_url: "https://api.openai.com".to_string(),
            openai_api_key: None,
            openai_max_tokens: 256,
            anthropic_token: None,
            home_location: crate::directions::LiveHomeLocation::default(),
            spotify: None,
            cadora: None,
            calendar: None,
            directions: None,
            directions_provider: String::new(),
            directions_imperial: false,
            weather: None,
            weather_imperial: false,
        }
    }

    fn shared(factory: LlmFactory) -> Arc<SharedSettings> {
        // Seed the live keys from the factory's env keys, exactly as `Config::
        // shared_settings` does at boot, so a later swap to that backend uses them.
        let anthropic_api_key = factory.anthropic_api_key.clone();
        let openai_api_key = factory.openai_api_key.clone();
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        SharedSettings::new(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".to_string(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key,
                openai_api_key,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
        )
    }

    #[test]
    fn apply_system1_swaps_and_reports_the_backend() {
        let s = shared(factory_with_key(None));
        // Starts disabled.
        assert_eq!(s.system1_view().backend, "none");
        assert_eq!(s.snapshot().system1.engine.name(), "none");

        // Swap to the local laya-serve backend (no network needed to construct it).
        let view = s
            .apply_system1(&System1Update {
                backend: Some("laya-serve".to_string()),
                base_url: Some("http://127.0.0.1:9999".to_string()),
                min_confidence: Some(0.9),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.backend, "laya-serve");
        assert!((view.min_confidence - 0.9).abs() < 1e-9);
        assert_eq!(s.snapshot().system1.engine.name(), "laya-serve");

        // An unknown backend is rejected and leaves the current engine untouched.
        let err = s.apply_system1(&System1Update {
            backend: Some("bogus".to_string()),
            ..Default::default()
        });
        assert!(err.is_err());
        assert_eq!(s.system1_view().backend, "laya-serve");

        // Back to disabled.
        s.apply_system1(&System1Update {
            backend: Some("none".to_string()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(s.snapshot().system1.engine.name(), "none");
    }

    #[test]
    fn subscription_backend_builds_with_a_token_provider() {
        use crate::llm::anthropic_auth::AnthropicTokenProvider;
        let mut factory = factory_with_key(None); // no API key…
        factory.anthropic_token = Some(Arc::new(AnthropicTokenProvider::with_fetcher(Arc::new(
            || Ok("tok".into()),
        ))));
        // …but subscription auth builds anyway, because a token provider is wired.
        let (_, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "anthropic",
                None,
                AnthropicAuth::Subscription,
            )
            .unwrap();
        assert_eq!(label, "anthropic");
        assert_eq!(model.as_deref(), Some("claude-opus-5"));
    }

    #[test]
    fn subscription_without_token_provider_is_rejected() {
        let factory = factory_with_key(None); // anthropic_token: None
                                              // `Arc<dyn LlmBackend>` isn't Debug, so take the error via `.err()`.
        let err = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "anthropic",
                None,
                AnthropicAuth::Subscription,
            )
            .err()
            .expect("subscription without a token provider must be rejected");
        assert!(format!("{err:#}").contains("ANTHROPIC_OAUTH_TOKEN"));
    }

    #[test]
    fn apply_persists_settings_and_they_reload() {
        let path = std::env::temp_dir().join(format!(
            "ambient_settings_test_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        let factory = factory_with_key(None);
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        s.apply(&SettingsUpdate {
            web_search: Some(true),
            search_provider: Some("tavily".into()),
            search_api_key: Some(Some("tvly-secret".into())),
            ..Default::default()
        })
        .unwrap();

        let p = load_persisted(&path).expect("settings file should exist");
        assert!(p.web_search);
        assert_eq!(p.search_provider, "tavily");
        assert_eq!(p.search_api_key.as_deref(), Some("tvly-secret"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn model_only_change_keeps_backend() {
        let s = shared(factory_with_key(None));
        let view = s
            .apply(&SettingsUpdate {
                llm_model: Some("qwen2.5".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.llm_backend, "ollama");
        assert_eq!(view.llm_model.as_deref(), Some("qwen2.5"));
    }

    #[test]
    fn backend_change_without_model_uses_default_model() {
        let s = shared(factory_with_key(Some("sk-test")));
        let view = s
            .apply(&SettingsUpdate {
                llm_backend: Some("anthropic".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.llm_backend, "anthropic");
        assert_eq!(view.llm_model.as_deref(), Some("claude-opus-5"));
    }

    #[test]
    fn anthropic_without_key_is_rejected_and_leaves_settings_unchanged() {
        let s = shared(factory_with_key(None));
        let before = s.view();
        let err = s
            .apply(&SettingsUpdate {
                llm_backend: Some("anthropic".to_string()),
                ..Default::default()
            })
            .unwrap_err();
        assert!(err.to_string().contains("ANTHROPIC_API_KEY"));
        assert_eq!(s.view(), before, "a failed swap is atomic");
    }

    #[test]
    fn runtime_anthropic_key_enables_the_backend_without_a_restart() {
        // No env/boot key, so anthropic is rejected up front.
        let s = shared(factory_with_key(None));
        assert!(!s.view().anthropic_key_set);
        assert!(s
            .apply(&SettingsUpdate {
                llm_backend: Some("anthropic".to_string()),
                ..Default::default()
            })
            .is_err());

        // Supply a key at runtime and select anthropic in the same change — it now
        // builds without a restart, and the view reports the key as set (never the
        // value: SettingsView carries only the boolean).
        let view = s
            .apply(&SettingsUpdate {
                llm_backend: Some("anthropic".to_string()),
                anthropic_api_key: Some(Some("sk-runtime".to_string())),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.llm_backend, "anthropic");
        assert_eq!(view.llm_model.as_deref(), Some("claude-opus-5"));
        assert!(view.anthropic_key_set);

        // A later unrelated change keeps the runtime key (unchanged tri-state).
        let view = s
            .apply(&SettingsUpdate {
                llm_model: Some("claude-haiku-4-5".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert!(view.anthropic_key_set);
        assert_eq!(view.llm_model.as_deref(), Some("claude-haiku-4-5"));

        // Clearing the key (Some(None)) while on anthropic is rejected atomically,
        // so the working backend is never torn down by a bad clear.
        let err = s
            .apply(&SettingsUpdate {
                anthropic_api_key: Some(None),
                ..Default::default()
            })
            .unwrap_err();
        assert!(err.to_string().contains("ANTHROPIC_API_KEY"));
        assert!(s.view().anthropic_key_set, "rejected clear is atomic");
    }

    #[test]
    fn runtime_key_is_persisted_so_a_restart_is_not_needed_either() {
        let path = std::env::temp_dir().join(format!(
            "ambient_key_persist_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        let factory = factory_with_key(None);
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        s.apply(&SettingsUpdate {
            llm_backend: Some("anthropic".into()),
            anthropic_api_key: Some(Some("sk-persist".into())),
            ..Default::default()
        })
        .unwrap();

        // The key lands in the persisted file, so the next boot loads it and the
        // cloud backend is selectable with no environment key and no restart.
        let p = load_persisted(&path).expect("settings file should exist");
        assert_eq!(p.llm_backend, "anthropic");
        assert_eq!(p.anthropic_api_key.as_deref(), Some("sk-persist"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tts_voice_can_be_set_and_cleared() {
        let s = shared(factory_with_key(None));
        let set = s
            .apply(&SettingsUpdate {
                tts_voice: Some(Some("en_US-amy-medium".to_string())),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(set.tts_voice.as_deref(), Some("en_US-amy-medium"));

        let cleared = s
            .apply(&SettingsUpdate {
                tts_voice: Some(None),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(cleared.tts_voice, None);
    }

    #[test]
    fn snapshot_is_stable_across_a_later_swap() {
        let s = shared(factory_with_key(None));
        let snap = s.snapshot();
        s.apply(&SettingsUpdate {
            llm_model: Some("other".to_string()),
            ..Default::default()
        })
        .unwrap();
        // The earlier snapshot still points at the original backend name.
        assert_eq!(snap.llm_model.as_deref(), Some("llama3.2"));
        assert_eq!(s.view().llm_model.as_deref(), Some("other"));
    }

    #[test]
    fn household_sanitize_trims_and_drops_blanks_and_nameless() {
        let messy = Household {
            location: Some("  Austin, Texas  ".into()),
            weather_units: Some("   ".into()), // blank → cleared
            members: vec![
                HouseholdMember {
                    name: "  Alice  ".into(),
                    emails: vec![" alice@example.com ".into(), "".into()],
                    phones: vec!["+1 555 0001".into()],
                    relationship: Some(" parent ".into()),
                },
                // No name → dropped entirely.
                HouseholdMember {
                    name: "   ".into(),
                    emails: vec!["ghost@example.com".into()],
                    ..Default::default()
                },
            ],
        };
        let clean = messy.sanitized();
        assert_eq!(clean.location.as_deref(), Some("Austin, Texas"));
        assert_eq!(clean.weather_units, None);
        assert_eq!(clean.members.len(), 1);
        assert_eq!(clean.members[0].name, "Alice");
        assert_eq!(clean.members[0].emails, vec!["alice@example.com"]);
        assert_eq!(clean.members[0].relationship.as_deref(), Some("parent"));
        assert!(!clean.is_empty());
        assert!(Household::default().is_empty());
    }

    #[test]
    fn spotify_config_controller_only_builds_when_linked() {
        let mut c = SpotifyConfig::default();
        assert!(!c.configured() && !c.linked() && c.controller().is_none());
        assert_eq!(c.device_label(), "Ambient"); // default device name
        c.client_id = Some("cid".into());
        c.client_secret = Some("sec".into());
        assert!(c.configured() && !c.linked() && c.controller().is_none());
        c.refresh_token = Some("rt".into());
        assert!(c.linked() && c.controller().is_some());
    }

    #[test]
    fn apply_household_persists_and_reloads() {
        let path = std::env::temp_dir().join(format!(
            "ambient_household_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        let factory = factory_with_key(None);
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        let stored = s.apply_household(&Household {
            location: Some("Boston, MA".into()),
            weather_units: Some("imperial".into()),
            members: vec![HouseholdMember {
                name: "Bob".into(),
                emails: vec!["bob@example.com".into()],
                phones: vec!["+1 555 0002".into()],
                relationship: None,
            }],
        });
        assert_eq!(stored.members.len(), 1);
        assert_eq!(s.household().location.as_deref(), Some("Boston, MA"));

        // It round-trips through the 0600 file so a restart keeps it.
        let p = load_persisted(&path).expect("settings file should exist");
        assert_eq!(p.household.location.as_deref(), Some("Boston, MA"));
        assert_eq!(p.household.weather_units.as_deref(), Some("imperial"));
        assert_eq!(p.household.members[0].name, "Bob");
        assert_eq!(p.household.members[0].emails, vec!["bob@example.com"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_spotify_links_and_persists() {
        let path = std::env::temp_dir().join(format!(
            "ambient_spotify_test_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        let factory = factory_with_key(None);
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        // Save creds (not linked yet), then link with a refresh token.
        s.apply_spotify(&SpotifyUpdate {
            client_id: Some(Some("cid".into())),
            client_secret: Some(Some("sec".into())),
            device_name: Some(Some("Kitchen".into())),
            ..Default::default()
        });
        assert!(s.spotify().configured() && !s.spotify().linked());
        s.apply_spotify(&SpotifyUpdate {
            refresh_token: Some(Some("rt".into())),
            ..Default::default()
        });
        assert!(s.spotify().linked());
        assert_eq!(s.spotify().device_label(), "Kitchen");

        // The persisted file carries the linkage so a restart needs no re-consent.
        let reloaded = load_persisted(&path).expect("settings file written");
        assert_eq!(reloaded.spotify.refresh_token.as_deref(), Some("rt"));
        assert_eq!(reloaded.spotify.device_name.as_deref(), Some("Kitchen"));

        // A blank secret on a later save must not wipe the stored one.
        s.apply_spotify(&SpotifyUpdate {
            client_secret: Some(None),
            ..Default::default()
        });
        assert!(!s.spotify().configured(), "secret cleared explicitly");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_directions_sets_mapbox_token_and_persists() {
        let path = std::env::temp_dir().join(format!(
            "ambient_directions_test_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        // A factory whose directions provider is mapbox (empty label ⇒ mapbox).
        let factory = factory_with_key(None);
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        assert!(!s.view().mapbox_token_set);
        assert!(!s.mapbox_token_set());
        let set = s.apply_directions(&DirectionsUpdate {
            mapbox_token: Some(Some("pk.test-token".into())),
        });
        assert!(set);
        assert!(s.view().mapbox_token_set);

        // Persisted so a restart keeps it; a blank re-save leaves it intact.
        let p = load_persisted(&path).expect("settings file written");
        assert_eq!(p.mapbox_token.as_deref(), Some("pk.test-token"));
        s.apply_directions(&DirectionsUpdate { mapbox_token: None });
        assert!(s.mapbox_token_set(), "blank re-save keeps the token");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_anthropic_oauth_token_overrides_provider_and_persists() {
        use crate::llm::anthropic_auth::AnthropicTokenProvider;
        let path = std::env::temp_dir().join(format!(
            "ambient_oauth_test_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        // Keep an Arc to the provider so we can assert the override was set (the factory
        // clone stored in SharedSettings shares the same instance).
        let provider = Arc::new(AnthropicTokenProvider::new(None));
        let mut factory = factory_with_key(None);
        factory.anthropic_token = Some(provider.clone());
        let (llm, label, model) = factory
            .build(
                LlmEngine::Native,
                false,
                "duckduckgo",
                None,
                "ollama",
                Some("llama3.2"),
                AnthropicAuth::ApiKey,
            )
            .unwrap();
        let s = SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine: LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: label,
                llm_model: model,
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: AnthropicAuth::ApiKey,
                tts_voice: None,
                end_silence_ms: DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: DEFAULT_VOICE_RMS_THRESHOLD,
                drive: DriveConfig::default(),
                household: Household::default(),
                spotify: SpotifyConfig::default(),
                cadora: CadoraConfig::default(),
                system1: System1Runtime::default(),
            },
            Some(path.clone()),
        );

        assert!(!s.view().anthropic_oauth_token_set);
        assert!(!provider.has_override());
        let view = s
            .apply(&SettingsUpdate {
                anthropic_oauth_token: Some(Some("oauth-xyz".into())),
                ..Default::default()
            })
            .unwrap();
        assert!(view.anthropic_oauth_token_set);
        assert!(
            provider.has_override(),
            "the shared provider got the override"
        );

        // Persisted (plaintext, 0600) so a restart re-applies it; never leaked in view.
        let p = load_persisted(&path).expect("settings file written");
        assert_eq!(p.anthropic_oauth_token.as_deref(), Some("oauth-xyz"));
        let _ = std::fs::remove_file(&path);
    }
}
