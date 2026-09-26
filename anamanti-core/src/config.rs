//! Runtime configuration for the orchestrator, loaded from a **per-instance JSON
//! file** (`anamanti.json` in the working directory by default, overridable with
//! `--config <path>`). This is where the **pluggable** LLM decision is bound:
//! `llm.backend` picks local (Ollama) vs cloud (Claude) vs the offline mock, and the
//! rest of the pipeline is handed a `dyn LlmBackend` — it never knows which was chosen.
//!
//! Only **secrets** remain environment variables: the provider API keys/tokens
//! (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MAPBOX_TOKEN`/
//! `MAPBOX_ACCESS_TOKEN`, `ANTHROPIC_OAUTH_TOKEN`) and the standard `RUST_LOG`.
//! Everything else — addresses, paths, identity, feature toggles, and the OAuth
//! *client* credentials for Spotify/Drive — lives in the JSON file.

use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::calendar::{CalendarSpec, IcalSubscription};
use crate::llm::anthropic_auth::{AnthropicAuth, AnthropicTokenProvider};
use crate::llm::catalog::ModelCatalog;
use crate::llm::{
    anthropic::AnthropicBackend, mock::MockLlm, ollama::OllamaBackend, openai::OpenAiBackend,
    LlmBackend,
};
use crate::music::{
    GroupSelector, ManagedProc, MpvControl, MusicDucker, MusicHub, MusicSupervisor, ProcSpec,
    SnapcastClient,
};
use crate::settings::{
    load_persisted, CadoraConfig, DriveConfig, Household, LlmEngine, LlmFactory, RuntimeSettings,
    SharedSettings, SpotifyConfig,
};

/// Default Google Drive OAuth scope for the photo slideshow (read-only).
pub const DEFAULT_DRIVE_SCOPE: &str = "https://www.googleapis.com/auth/drive.readonly";

/// The convention config-file path, looked up in the working directory when no
/// `--config <path>` is given. Absent → built-in defaults.
pub const DEFAULT_CONFIG_FILE: &str = "anamanti.json";

/// The default command that prints a fresh Anthropic subscription access token, used
/// when `llm.anthropic_token_cmd` is unset.
pub const DEFAULT_ANTHROPIC_TOKEN_CMD: &str = "ant auth print-credentials --access-token";

/// Interpret a config-file string that may carry the disable sentinel `off`/`none`/
/// (empty) as "feature disabled" (`None`), else `Some(trimmed)`.
fn disable_sentinel(v: &str) -> Option<&str> {
    match v.trim() {
        s if matches!(s.to_lowercase().as_str(), "off" | "none" | "") => None,
        s => Some(s),
    }
}

/// Default persona/system prompt: concise, speakable replies for an ambient
/// display. Kept short because the reply is spoken aloud via Piper.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a friendly, concise voice assistant for a home \
ambient display. Answer in one or two short spoken sentences. Do not use markdown, lists, or \
emoji. If you don't know, say so briefly.";

/// Which LLM backend the pipeline should use.
#[derive(Debug, Clone)]
pub enum LlmChoice {
    /// Deterministic offline echo backend (no server, no network).
    Mock,
    /// Local Ollama / llama.cpp endpoint.
    Ollama { url: String, model: String },
    /// Cloud Claude Messages API.
    Anthropic { model: String, max_tokens: u32 },
    /// Cloud OpenAI Chat Completions API.
    OpenAI { model: String, max_tokens: u32 },
}

/// Fully-resolved orchestrator configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address the device-facing Wyoming server binds to (advertised via mDNS).
    pub bind_addr: SocketAddr,
    /// Address the local HTTP **config page** binds to, or `None` to disable it.
    /// Defaults to loopback (`127.0.0.1:8730`) since the page has no auth.
    pub config_addr: Option<SocketAddr>,
    /// Human-readable mDNS instance name (the friendly label shown in the
    /// device's orchestrator dropdown).
    pub service_name: String,
    /// Stable mDNS selection key advertised in the `instance_id` TXT record. The
    /// device persists this to pin a specific orchestrator across restarts / IP
    /// changes, so it MUST be stable across restarts (never random per boot).
    /// Resolution order: the config file's `instance_id` → the **git branch code** of
    /// the working directory (so a copy running from a test branch/worktree identifies
    /// itself by its branch without extra config) → the sanitized `service_name`.
    /// The "local production" install lives outside a git checkout and sets
    /// `instance_id` in its `anamanti.json`, so it never falls through to the branch.
    pub instance_id: String,
    /// Downstream Wyoming STT (Whisper) address.
    pub stt_addr: SocketAddr,
    /// Downstream Wyoming TTS (Piper) address.
    pub tts_addr: SocketAddr,
    /// Optional Piper voice name.
    pub tts_voice: Option<String>,
    /// Directory holding Piper voice models (`<name>.onnx`). When set (Piper is
    /// co-located with the orchestrator), the settings voice dropdown lists only the
    /// voices actually present here; when `None`, it lists Piper's full advertised
    /// catalog. Set via the config file's `tts_voices_dir`.
    pub tts_voices_dir: Option<PathBuf>,
    /// Selected LLM backend.
    pub llm: LlmChoice,
    /// SQLite memory database path.
    pub db_path: PathBuf,
    /// Base system prompt/persona.
    pub system_prompt: String,
    /// The device's physical home location (e.g. "Austin, Texas"), injected into the
    /// prompt so location-relative questions (weather, sunset, nearby places) resolve
    /// an unqualified "here". `None` omits the location grounding. Set via the config
    /// file's `home_location`.
    pub home_location: Option<String>,
    /// Preferred measurement units for answers (e.g. "imperial" / "metric"), paired
    /// with `home_location`. Set via the config file's `weather_units`.
    pub weather_units: Option<String>,
    /// Idle timeout for a stalled turn.
    pub turn_timeout: Duration,
    /// Memory retrieval backend: `helix` (GraphRAG, default) or `sqlite` (FTS).
    pub memory_backend: MemoryBackendChoice,
    /// Append-only JSONL chat log path (always written; the ingester's queue).
    pub chatlog_path: PathBuf,
    /// Append-only JSONL prompt log path (debug/audit of the exact LLM prompt).
    pub promptlog_path: PathBuf,
    /// Embedded HelixDB on-disk store root (used when `memory_backend = helix`).
    pub helix_path: PathBuf,
    /// GraphRAG embedding + extraction settings (used when `memory_backend = helix`).
    pub graphrag: GraphRagConfig,
    /// Per-person speaker identification settings (speaker_id_plan.md).
    pub speaker: SpeakerConfig,
    /// STT engine selection: the downstream Wyoming Whisper server (`stt_addr`) or
    /// the in-process whisper.cpp engine (`plans/python-to-rust-whisper.md`).
    pub stt: SttConfig,
    /// Auto follow-up listening: reopen the mic (no wake word) when a reply is a
    /// question, and feed recent history into that turn's prompt.
    pub follow_up: FollowUpConfig,
    /// House-wide music routing (Snapcast) control-plane settings.
    pub music: MusicConfig,
    /// Weather feature settings (the `weather_lookup` tool + the ambient push).
    pub weather: WeatherSettings,
    /// System-1 fast-decision engine selection (plans/system1-fast-decisions.md).
    pub system1: System1Config,
    /// Where the runtime-swappable settings overlay is persisted (`settings_path` in
    /// the config file), or `None` to keep runtime settings in memory only. Defaults
    /// to `anamanti_settings.json`. This is a **separate** file from the boot config:
    /// the config page writes it and it overlays these seed values at the next boot.
    pub settings_path: Option<PathBuf>,
    /// Debug-only per-turn audio capture directory (AEC corpus). `None` disables it.
    pub audio_dump_dir: Option<PathBuf>,
    /// Initial Anthropic auth mode (`apikey`/`subscription`). Runtime-swappable.
    pub anthropic_auth: AnthropicAuth,
    /// Initial LLM engine (rig vs native). Runtime-swappable.
    pub engine: LlmEngine,
    /// Initial rig web-search enable. Runtime-swappable.
    pub web_search: bool,
    /// Initial web-search provider (`duckduckgo`/`tavily`). Runtime-swappable.
    pub search_provider: String,
    /// Command that prints a fresh Anthropic subscription token (`None` → the built-in
    /// `ant` default). Non-secret; the token itself stays in `ANTHROPIC_OAUTH_TOKEN`.
    pub anthropic_token_cmd: Option<String>,
    /// Ollama base URL used to (re)build the ollama backend at runtime, independent of
    /// which backend is currently selected.
    pub ollama_url: String,
    /// `max_tokens` cap for the Anthropic backend (independent of the current backend).
    pub anthropic_max_tokens: u32,
    /// `max_tokens` cap for the OpenAI backend (independent of the current backend).
    pub openai_max_tokens: u32,
    /// Web-calendar subscriptions for the `calendar_lookup` tool (empty → no tool).
    pub calendar_specs: Vec<CalendarSpec>,
    /// Cache TTL for fetched `.ics` feeds.
    pub calendar_cache_ttl: Duration,
    /// Routing provider label for the `directions_lookup` tool (empty → `mapbox`).
    pub directions_provider: String,
    /// Google Drive photo-slideshow seed (client creds + folders; the refresh token is
    /// minted by consent, never seeded). Overlaid by the persisted settings file.
    pub drive: DriveConfig,
    /// Spotify voice-control seed (client creds + refresh token + device name).
    /// Overlaid by the persisted settings file.
    pub spotify: SpotifyConfig,
    /// Cadora shopping-list seed (base URL + voice-link token). The token is normally
    /// minted by the config-page pairing flow. Overlaid by the persisted settings file.
    pub cadora: CadoraConfig,
}

/// Speaker-identification configuration. Off by default (`speaker.enabled`); a
/// missing model path degrades gracefully to the deterministic mock embedder.
#[derive(Debug, Clone)]
pub struct SpeakerConfig {
    /// Whether to identify speakers per turn (`speaker.enabled`).
    pub enabled: bool,
    /// Path to the speaker-embedding ONNX model (Phase E). Absent ⇒ household.
    pub model_path: Option<PathBuf>,
    /// Cosine ≥ this ⇒ confident match (centroid updated).
    pub match_threshold: f32,
    /// Cosine below this ⇒ mint a new anonymous cluster.
    pub new_threshold: f32,
    /// Minimum voiced audio (ms) before attempting identification.
    pub min_speech_ms: u32,
    /// Embedding dimensionality the ONNX model produces (ECAPA-TDNN → 192).
    pub embed_dims: usize,
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            match_threshold: 0.55,
            new_threshold: 0.40,
            min_speech_ms: 1200,
            embed_dims: 192,
        }
    }
}

/// Which STT engine transcribes a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttEngineKind {
    /// Dial the downstream Wyoming Whisper server at `stt_addr` (historical default).
    Wyoming,
    /// In-process whisper.cpp via `whisper-rs` (requires the `stt-whisper-local`
    /// build feature).
    WhisperLocal,
}

impl SttEngineKind {
    /// Parse the config label. `wyoming` (or `faster-whisper`) → the downstream
    /// server; `whisper-rs` / `whisper-local` / `local` → in-process.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wyoming" | "faster-whisper" | "whisper-wyoming" => Some(Self::Wyoming),
            "whisper-rs" | "whisper-local" | "whisper_local" | "local" => Some(Self::WhisperLocal),
            _ => None,
        }
    }
}

/// STT engine selection + in-process model settings (`stt` block). See
/// `plans/python-to-rust-whisper.md`. Note the downstream address lives in the
/// top-level `stt_addr` (used when `engine = wyoming`).
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Which engine transcribes (`wyoming` default, or `whisper-rs`).
    pub engine: SttEngineKind,
    /// Named model size for the in-process engine: `base` (default) or `small`,
    /// resolved to `<model_dir>/ggml-<model>.en.bin` unless `model_path` overrides.
    pub model: String,
    /// Directory holding the ggml model files (`ggml-base.en.bin`, …).
    pub model_dir: PathBuf,
    /// Explicit ggml model file; overrides `model`/`model_dir` when set.
    pub model_path: Option<PathBuf>,
    /// Decode language (`Some("en")`); `None` ⇒ auto-detect.
    pub language: Option<String>,
    /// Decode thread cap; `0` ⇒ a sensible default from host parallelism.
    pub num_threads: u32,
}

impl SttConfig {
    /// The ggml model file for the in-process engine: `model_path` if set, else
    /// `<model_dir>/ggml-<model>.en.bin`.
    pub fn resolved_model_path(&self) -> PathBuf {
        self.model_path
            .clone()
            .unwrap_or_else(|| self.model_dir.join(format!("ggml-{}.en.bin", self.model)))
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            engine: SttEngineKind::Wyoming,
            model: "base".to_string(),
            model_dir: PathBuf::from("models"),
            model_path: None,
            language: Some("en".to_string()),
            num_threads: 0,
        }
    }
}

/// Auto follow-up listening (`follow_up` block). After **every** spoken reply the
/// device reopens the mic and listens for more input with no wake word; that follow-up
/// turn's prompt is seeded with recent conversation history. The listen window differs
/// by reply: a **question** (ends with `?`) waits `question_wait_secs`, any other reply
/// waits `reply_wait_secs`, then the assistant sleeps if nothing was said. The window is
/// passed to the device in the `ambient-listen` frame and echoed back so the
/// orchestrator sizes the follow-up turn's no-speech VAD window to match. See
/// `plans/Plan.MD` (Follow-up listening) and `architecture.md` §4.
#[derive(Debug, Clone)]
pub struct FollowUpConfig {
    /// Master switch (`follow_up.enabled`, default on). Off ⇒ the orchestrator never
    /// sends an `ambient-listen` frame and every turn stays single-shot.
    pub enabled: bool,
    /// Optional safety ceiling on consecutive auto follow-ups (`follow_up.max_chain`).
    /// **`0` = unlimited** (the default): the loop is instead terminated by silence (an
    /// empty follow-up transcript sends no new `ambient-listen`). A non-zero value caps
    /// the chain, requiring a wake word again after that many auto-listens.
    pub max_chain: u32,
    /// Listen window (seconds) after a **question** reply — how long the mic stays open
    /// for the answer before sleeping (`follow_up.question_wait_secs`, default 10).
    pub question_wait_secs: u32,
    /// Listen window (seconds) after any **non-question** reply, before sleeping
    /// (`follow_up.reply_wait_secs`, default 5).
    pub reply_wait_secs: u32,
    /// How many recent turns to feed into a follow-up turn's prompt as history
    /// (`follow_up.history_turns`, default 5).
    pub history_turns: usize,
    /// Only include history turns completed within this many seconds of the follow-up
    /// (`follow_up.history_window_secs`, default 600 = 10 minutes).
    pub history_window_secs: i64,
}

impl Default for FollowUpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chain: 0,
            question_wait_secs: 10,
            reply_wait_secs: 5,
            history_turns: 5,
            history_window_secs: 600,
        }
    }
}

/// House-wide music routing (Snapcast) control-plane settings, from the config file's
/// `music` block. **Inert unless `enabled`** (`music.enabled`): the orchestrator only
/// ducks the music group while it speaks and (later) selects the active stream. It
/// never handles PCM — see `crate::music` and `plans/snapcast_routing_plan.md`.
#[derive(Debug, Clone)]
pub struct MusicConfig {
    /// Master switch (`music.enabled`). Off ⇒ the whole feature is dormant.
    pub enabled: bool,
    /// snapserver JSON-RPC control endpoint (`music.snapserver_addr`).
    pub snapserver_addr: SocketAddr,
    /// Duck the music group's volume while the assistant speaks
    /// (`music.duck_on_speech`, default on).
    pub duck_on_speech: bool,
    /// Volume percent to duck to during speech (`music.duck_percent`).
    pub duck_percent: u8,
    /// Which group to duck: `auto`, a group id (`music.group`), or the group playing a
    /// stream id (`music.stream`, most specific).
    pub group: GroupSelector,
    /// mpv JSON IPC socket for the web-URL player (`music.web_ipc`).
    pub mpv_ipc: Option<PathBuf>,
    /// Directory holding the `snapserver`/`librespot`/`mpv` binaries
    /// (`music.bin_dir`, default the Homebrew prefix bin).
    pub bin_dir: PathBuf,
    /// Directory holding the snapfifos (`music.run_dir`).
    pub run_dir: PathBuf,
    /// Directory for the supervised processes' log files (`music.log_dir`).
    pub log_dir: PathBuf,
    /// snapserver config path (`music.conf`).
    pub snapserver_conf: PathBuf,
    /// librespot Spotify Connect device name (`music.spotify_device_name`,
    /// default `Ambient`; the shared instance MusicPlan.md's tool targets).
    pub spotify_device_name: String,
    /// Auto-start the managed processes (snapserver/librespot/mpv) when the
    /// orchestrator boots, and stop them on shutdown (`music.autostart`,
    /// default on when music is enabled). The Music tab's manual buttons still work.
    pub autostart: bool,
}

impl Default for MusicConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapserver_addr: "127.0.0.1:1705".parse().unwrap(),
            duck_on_speech: true,
            duck_percent: 30,
            group: GroupSelector::Auto,
            mpv_ipc: Some(PathBuf::from("/tmp/ambient-mpv.sock")),
            bin_dir: PathBuf::from("/opt/homebrew/bin"),
            run_dir: PathBuf::from("/opt/homebrew/var/run/ambient"),
            log_dir: PathBuf::from("/opt/homebrew/var/log"),
            snapserver_conf: PathBuf::from("/opt/homebrew/etc/snapserver.conf"),
            spotify_device_name: "Ambient".to_string(),
            autostart: true,
        }
    }
}

/// Weather feature settings. Weather uses the keyless Open-Meteo API, so there is no
/// key to configure — just the master switch and how often the ambient indicator (the
/// icon + temperature beside the idle clock) is refreshed by the background push.
#[derive(Debug, Clone)]
pub struct WeatherSettings {
    /// Master switch (`weather.enabled`, default on). Off ⇒ the `weather_lookup` tool
    /// is not advertised and the ambient push does not run.
    pub enabled: bool,
    /// How often (seconds) the ambient current-conditions push refreshes
    /// (`weather.refresh_interval_secs`, default 1800 = 30 minutes). Clamped to a sane
    /// floor so a misconfiguration can't hammer the API.
    pub refresh_interval_secs: u64,
}

impl Default for WeatherSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            refresh_interval_secs: 1800,
        }
    }
}

impl WeatherSettings {
    /// The refresh interval as a `Duration`, clamped to at least 5 minutes.
    pub fn refresh_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.refresh_interval_secs.max(300))
    }
}

/// System-1 fast-decision settings (plans/system1-fast-decisions.md). Selects and
/// configures the pluggable [`crate::system1::DecisionEngine`] that runs before memory
/// recall + the LLM. Default `backend = "none"` reproduces today's behavior.
#[derive(Debug, Clone)]
pub struct System1Config {
    /// Which engine: `none` (default, disabled) | `mock` (tests) | `laya-serve` | `jev`
    /// | `laya-embedded`. The HTTP/embedded backends are wired in M1.
    pub backend: String,
    /// Base URL for the HTTP backends (`laya-serve` sidecar or OpenRouter for `jev`).
    pub base_url: String,
    /// OpenRouter model id for the `jev` backend.
    pub openrouter_model: String,
    /// Compute device for the in-process `laya-embedded` backend (`cpu` | `metal`).
    pub device: String,
    /// Checkpoint / Hugging Face repo id for the `laya-embedded` backend.
    pub model: String,
    /// Strict confidence floor below which a decision defers to System-2.
    pub min_confidence: f64,
    /// The intent labels the router is allowed to resolve (empty = the built-in set,
    /// filled in as M1+ handlers land).
    pub intents: Vec<String>,
}

impl Default for System1Config {
    fn default() -> Self {
        Self {
            backend: "none".to_string(),
            base_url: "http://127.0.0.1:8000".to_string(),
            openrouter_model: "typesafe/jev-1.13".to_string(),
            device: "metal".to_string(),
            model: "convaiinnovations/laya".to_string(),
            min_confidence: 0.85,
            intents: Vec::new(),
        }
    }
}

/// Which memory retrieval backend the pipeline uses for prompt context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryBackendChoice {
    /// SQLite FTS over the explicit/inferred memory store (recall fallback when
    /// Helix init fails, e.g. no `OPENAI_API_KEY`).
    Sqlite,
    /// Embedded HelixDB GraphRAG (vector KNN + graph expansion). The default.
    Helix,
}

/// Settings for the GraphRAG memory (embeddings + background entity extraction).
/// API keys are read from the environment at wiring time, not stored here.
#[derive(Debug, Clone)]
pub struct GraphRagConfig {
    /// OpenAI API base (overridable for testing).
    pub openai_base_url: String,
    /// Embedding model (default `text-embedding-3-small`).
    pub embed_model: String,
    /// Embedding dimensionality (native 1536; reducible via OpenAI's `dimensions`).
    pub embed_dims: usize,
    /// Anthropic API base for entity extraction.
    pub anthropic_base_url: String,
    /// Entity-extraction chat model (default Claude Haiku 4.5).
    pub extract_model: String,
    /// How often the background ingester drains the chat log.
    pub ingest_interval: Duration,
    /// KNN fan-out per vector search at recall time.
    pub recall_k: usize,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            openai_base_url: "https://api.openai.com".to_string(),
            embed_model: "text-embedding-3-small".to_string(),
            embed_dims: 1536,
            anthropic_base_url: "https://api.anthropic.com".to_string(),
            extract_model: "claude-haiku-4-5".to_string(),
            ingest_interval: Duration::from_secs(30),
            recall_k: 6,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Port 10700 is the conventional Wyoming satellite/host port.
            bind_addr: "0.0.0.0:10700".parse().unwrap(),
            // Config page on loopback only by default (no auth); override or
            // disable with config_addr in the config file.
            config_addr: Some("127.0.0.1:8730".parse().unwrap()),
            service_name: "Anamanti Core".to_string(),
            instance_id: "anamanti-core".to_string(),
            stt_addr: "127.0.0.1:10300".parse().unwrap(), // wyoming-faster-whisper default
            tts_addr: "127.0.0.1:10200".parse().unwrap(), // wyoming-piper default
            tts_voice: None,
            tts_voices_dir: None,
            llm: LlmChoice::Ollama {
                url: "http://127.0.0.1:11434".to_string(),
                model: "llama3.2".to_string(),
            },
            db_path: PathBuf::from("anamanti_memory.sqlite"),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            home_location: None,
            weather_units: None,
            turn_timeout: Duration::from_secs(30),
            memory_backend: MemoryBackendChoice::Helix,
            chatlog_path: PathBuf::from("anamanti_chatlog.jsonl"),
            promptlog_path: PathBuf::from("anamanti_promptlog.jsonl"),
            helix_path: PathBuf::from("anamanti_helix"),
            graphrag: GraphRagConfig::default(),
            speaker: SpeakerConfig::default(),
            stt: SttConfig::default(),
            follow_up: FollowUpConfig::default(),
            music: MusicConfig::default(),
            weather: WeatherSettings::default(),
            system1: System1Config::default(),
            settings_path: Some(PathBuf::from("anamanti_settings.json")),
            audio_dump_dir: None,
            anthropic_auth: AnthropicAuth::ApiKey,
            engine: LlmEngine::Rig,
            web_search: true,
            search_provider: "tavily".to_string(),
            anthropic_token_cmd: None,
            ollama_url: "http://127.0.0.1:11434".to_string(),
            anthropic_max_tokens: 1024,
            openai_max_tokens: 1024,
            calendar_specs: Vec::new(),
            calendar_cache_ttl: IcalSubscription::DEFAULT_TTL,
            directions_provider: String::new(),
            drive: DriveConfig {
                scope: Some(DEFAULT_DRIVE_SCOPE.to_string()),
                ..DriveConfig::default()
            },
            spotify: SpotifyConfig::default(),
            cadora: CadoraConfig::default(),
        }
    }
}

/// The git branch code of the current working directory, or `None` when this
/// process is not running inside a checkout (e.g. the "local production" install,
/// which runs a copied binary from outside a repo and sets `instance_id` in its
/// `anamanti.json`). Used as the default `instance_id` so a copy launched from a test
/// branch/worktree advertises itself by its branch with no extra configuration.
/// Detached HEAD (`branch == "HEAD"`) is treated as "no branch".
fn git_branch_code() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let branch = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if branch.is_empty() || branch == "HEAD" {
        None
    } else {
        Some(branch)
    }
}

/// Reduce a service name to a stable selection key: lowercase, alphanumerics and
/// hyphens only, collapsed/trimmed. Kept in sync with the device's expectation
/// that the `instance_id` TXT record is a stable, human-ish identifier.
fn sanitize_id(name: &str) -> String {
    let mapped: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed = mapped.trim_matches('-');
    if trimmed.is_empty() {
        "anamanti-core".to_string()
    } else {
        trimmed.to_lowercase()
    }
}

// ===========================================================================
// JSON file configuration
// ===========================================================================

/// The deserialized JSON config file. Every field is optional; an absent field
/// falls back to the built-in default (a 1:1 mirror of the historical env behavior),
/// so a partial file works. Unknown keys are rejected (`deny_unknown_fields`) so a
/// typo is a hard error rather than a silent revert-to-default.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub bind_addr: Option<SocketAddr>,
    /// Loopback config page `host:port`, or the disable sentinel `off`/`none`/(empty).
    pub config_addr: Option<String>,
    pub service_name: Option<String>,
    pub instance_id: Option<String>,
    pub stt_addr: Option<SocketAddr>,
    pub tts_addr: Option<SocketAddr>,
    pub tts_voice: Option<String>,
    pub tts_voices_dir: Option<PathBuf>,
    pub db_path: Option<PathBuf>,
    pub chatlog_path: Option<PathBuf>,
    pub promptlog_path: Option<PathBuf>,
    pub helix_path: Option<PathBuf>,
    /// Runtime settings overlay path, or the disable sentinel `off`/`none`/(empty).
    pub settings_path: Option<String>,
    pub audio_dump_dir: Option<PathBuf>,
    pub system_prompt: Option<String>,
    pub home_location: Option<String>,
    pub weather_units: Option<String>,
    pub turn_timeout_secs: Option<u64>,
    pub memory_backend: Option<String>,
    #[serde(default)]
    pub llm: FileLlm,
    #[serde(default)]
    pub graphrag: FileGraphRag,
    #[serde(default)]
    pub speaker: FileSpeaker,
    #[serde(default)]
    pub stt: FileStt,
    #[serde(default)]
    pub follow_up: FileFollowUp,
    #[serde(default)]
    pub music: FileMusic,
    #[serde(default)]
    pub weather: FileWeather,
    #[serde(default)]
    pub system1: FileSystem1,
    #[serde(default)]
    pub calendar: FileCalendar,
    #[serde(default)]
    pub directions: FileDirections,
    #[serde(default)]
    pub drive: FileDrive,
    #[serde(default)]
    pub spotify: FileSpotify,
    #[serde(default)]
    pub cadora: FileCadora,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileLlm {
    pub backend: Option<String>,
    pub engine: Option<String>,
    pub anthropic_auth: Option<String>,
    /// Command that prints a fresh subscription token (non-secret). The token itself
    /// stays in the `ANTHROPIC_OAUTH_TOKEN` env var.
    pub anthropic_token_cmd: Option<String>,
    pub web_search: Option<bool>,
    pub search_provider: Option<String>,
    #[serde(default)]
    pub ollama: FileOllama,
    #[serde(default)]
    pub anthropic: FileAnthropic,
    #[serde(default)]
    pub openai: FileOpenAi,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileOllama {
    pub url: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileAnthropic {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileOpenAi {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileGraphRag {
    pub openai_base_url: Option<String>,
    pub embed_model: Option<String>,
    pub embed_dims: Option<usize>,
    pub anthropic_base_url: Option<String>,
    pub extract_model: Option<String>,
    pub ingest_interval_secs: Option<u64>,
    pub recall_k: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileSpeaker {
    pub enabled: Option<bool>,
    pub model_path: Option<PathBuf>,
    pub match_threshold: Option<f32>,
    pub new_threshold: Option<f32>,
    pub min_speech_ms: Option<u32>,
    pub embed_dims: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileStt {
    pub engine: Option<String>,
    pub model: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub language: Option<String>,
    pub num_threads: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileFollowUp {
    pub enabled: Option<bool>,
    pub max_chain: Option<u32>,
    pub question_wait_secs: Option<u32>,
    pub reply_wait_secs: Option<u32>,
    pub history_turns: Option<usize>,
    pub history_window_secs: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileMusic {
    pub enabled: Option<bool>,
    pub snapserver_addr: Option<SocketAddr>,
    pub duck_on_speech: Option<bool>,
    pub duck_percent: Option<u8>,
    pub group: Option<String>,
    pub stream: Option<String>,
    /// mpv IPC socket path, or the disable sentinel `off`/`none`/(empty).
    pub web_ipc: Option<String>,
    pub bin_dir: Option<PathBuf>,
    pub run_dir: Option<PathBuf>,
    pub log_dir: Option<PathBuf>,
    pub conf: Option<PathBuf>,
    pub spotify_device_name: Option<String>,
    pub autostart: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileWeather {
    pub enabled: Option<bool>,
    pub refresh_interval_secs: Option<u64>,
}

/// The `system1` block of the config file (all fields optional; absent → defaults, and
/// an absent block leaves the engine disabled). See [`System1Config`].
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileSystem1 {
    pub backend: Option<String>,
    pub base_url: Option<String>,
    pub openrouter_model: Option<String>,
    pub device: Option<String>,
    pub model: Option<String>,
    pub min_confidence: Option<f64>,
    #[serde(default)]
    pub intents: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileCalendar {
    #[serde(default)]
    pub subscriptions: Vec<FileCalendarSub>,
    pub cache_ttl_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileCalendarSub {
    pub name: Option<String>,
    pub url: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDirections {
    pub provider: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDrive {
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    #[serde(default)]
    pub folder_ids: Vec<String>,
    pub scope: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileSpotify {
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub refresh_token: Option<String>,
    pub device_name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileCadora {
    /// Cadora app-server base URL (default `https://cadora-server.fly.dev`).
    pub base_url: Option<String>,
    /// A `vl_…` voice-link token, if seeded directly (normally minted via the
    /// config-page pairing flow instead).
    pub link_token: Option<String>,
}

/// Trim a config string and drop it when empty.
fn nonempty(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Convert the file's calendar subscriptions into [`CalendarSpec`]s, trimming and
/// dropping entries with no URL and auto-naming unnamed ones (`Calendar N`).
fn calendar_specs_from_file(subs: Vec<FileCalendarSub>) -> Vec<CalendarSpec> {
    subs.into_iter()
        .enumerate()
        .filter_map(|(i, sub)| {
            let url = sub.url.trim().to_string();
            if url.is_empty() {
                return None;
            }
            let name = nonempty(sub.name).unwrap_or_else(|| format!("Calendar {}", i + 1));
            Some(CalendarSpec { name, url })
        })
        .collect()
}

impl Config {
    /// Load configuration from a JSON file. With `Some(path)` the file is **required**
    /// (missing / unreadable / malformed → error). With `None` the convention path
    /// `anamanti.json` in the working directory is used when present; when absent, the
    /// built-in defaults are used (matching the historical "nothing configured" boot).
    pub fn load(cli_config: Option<&Path>) -> Result<Self> {
        match cli_config {
            Some(path) => {
                let data = std::fs::read_to_string(path)
                    .with_context(|| format!("reading config file {}", path.display()))?;
                let fc: FileConfig = serde_json::from_str(&data)
                    .with_context(|| format!("parsing config file {}", path.display()))?;
                Self::from_file(fc)
            }
            None => {
                let path = Path::new(DEFAULT_CONFIG_FILE);
                match std::fs::read_to_string(path) {
                    Ok(data) => {
                        let fc: FileConfig = serde_json::from_str(&data).with_context(|| {
                            format!("parsing config file {DEFAULT_CONFIG_FILE}")
                        })?;
                        Self::from_file(fc)
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        log::info!(
                            "no {DEFAULT_CONFIG_FILE} in the working directory; using built-in defaults"
                        );
                        Self::from_file(FileConfig::default())
                    }
                    Err(e) => Err(e).with_context(|| format!("reading {DEFAULT_CONFIG_FILE}")),
                }
            }
        }
    }

    /// Resolve a fully-built [`Config`] from a parsed [`FileConfig`], overlaying each
    /// present field onto the built-in defaults. Pure (no I/O), so it's unit-testable.
    pub fn from_file(fc: FileConfig) -> Result<Self> {
        let d = Config::default();

        let llm = match fc
            .llm
            .backend
            .as_deref()
            .unwrap_or("ollama")
            .to_lowercase()
            .as_str()
        {
            "mock" => LlmChoice::Mock,
            "anthropic" | "claude" => LlmChoice::Anthropic {
                model: fc
                    .llm
                    .anthropic
                    .model
                    .clone()
                    .unwrap_or_else(|| "claude-opus-5".to_string()),
                max_tokens: fc
                    .llm
                    .anthropic
                    .max_tokens
                    .unwrap_or(d.anthropic_max_tokens),
            },
            "openai" | "gpt" => LlmChoice::OpenAI {
                model: fc
                    .llm
                    .openai
                    .model
                    .clone()
                    .unwrap_or_else(|| "gpt-4o-mini".to_string()),
                max_tokens: fc.llm.openai.max_tokens.unwrap_or(d.openai_max_tokens),
            },
            _ => LlmChoice::Ollama {
                url: fc
                    .llm
                    .ollama
                    .url
                    .clone()
                    .unwrap_or_else(|| d.ollama_url.clone()),
                model: fc
                    .llm
                    .ollama
                    .model
                    .clone()
                    .unwrap_or_else(|| "llama3.2".to_string()),
            },
        };

        let memory_backend = match fc
            .memory_backend
            .as_deref()
            .unwrap_or("helix")
            .to_lowercase()
            .as_str()
        {
            "sqlite" | "fts" => MemoryBackendChoice::Sqlite,
            _ => MemoryBackendChoice::Helix,
        };

        let mut graphrag = GraphRagConfig::default();
        if let Some(v) = nonempty(fc.graphrag.embed_model) {
            graphrag.embed_model = v;
        }
        if let Some(v) = fc.graphrag.embed_dims {
            graphrag.embed_dims = v;
        }
        if let Some(v) = nonempty(fc.graphrag.extract_model) {
            graphrag.extract_model = v;
        }
        if let Some(v) = nonempty(fc.graphrag.openai_base_url) {
            graphrag.openai_base_url = v;
        }
        if let Some(v) = nonempty(fc.graphrag.anthropic_base_url) {
            graphrag.anthropic_base_url = v;
        }
        if let Some(secs) = fc.graphrag.ingest_interval_secs {
            graphrag.ingest_interval = Duration::from_secs(secs);
        }
        if let Some(v) = fc.graphrag.recall_k {
            graphrag.recall_k = v;
        }

        let sd = SpeakerConfig::default();
        let speaker = SpeakerConfig {
            enabled: fc.speaker.enabled.unwrap_or(sd.enabled),
            model_path: fc.speaker.model_path,
            match_threshold: fc.speaker.match_threshold.unwrap_or(sd.match_threshold),
            new_threshold: fc.speaker.new_threshold.unwrap_or(sd.new_threshold),
            min_speech_ms: fc.speaker.min_speech_ms.unwrap_or(sd.min_speech_ms),
            embed_dims: fc.speaker.embed_dims.unwrap_or(sd.embed_dims),
        };

        let sttd = SttConfig::default();
        let stt = SttConfig {
            engine: match fc.stt.engine.as_deref() {
                None => sttd.engine,
                Some(label) => SttEngineKind::from_label(label).with_context(|| {
                    format!("unknown stt.engine {label:?} (expected `wyoming` or `whisper-rs`)")
                })?,
            },
            model: nonempty(fc.stt.model).unwrap_or(sttd.model),
            model_dir: fc.stt.model_dir.unwrap_or(sttd.model_dir),
            model_path: fc.stt.model_path,
            // An explicit empty `language` means auto-detect (`None`); absent keeps
            // the default (`Some("en")`).
            language: match fc.stt.language {
                None => sttd.language,
                Some(s) if s.trim().is_empty() => None,
                Some(s) => Some(s),
            },
            num_threads: fc.stt.num_threads.unwrap_or(sttd.num_threads),
        };

        let fud = FollowUpConfig::default();
        let follow_up = FollowUpConfig {
            enabled: fc.follow_up.enabled.unwrap_or(fud.enabled),
            max_chain: fc.follow_up.max_chain.unwrap_or(fud.max_chain),
            question_wait_secs: fc
                .follow_up
                .question_wait_secs
                .unwrap_or(fud.question_wait_secs),
            reply_wait_secs: fc.follow_up.reply_wait_secs.unwrap_or(fud.reply_wait_secs),
            history_turns: fc.follow_up.history_turns.unwrap_or(fud.history_turns),
            history_window_secs: fc
                .follow_up
                .history_window_secs
                .unwrap_or(fud.history_window_secs),
        };

        let md = MusicConfig::default();
        let music = MusicConfig {
            enabled: fc.music.enabled.unwrap_or(md.enabled),
            snapserver_addr: fc.music.snapserver_addr.unwrap_or(md.snapserver_addr),
            duck_on_speech: fc.music.duck_on_speech.unwrap_or(md.duck_on_speech),
            duck_percent: fc
                .music
                .duck_percent
                .map(|p| p.min(100))
                .unwrap_or(md.duck_percent),
            group: GroupSelector::from_parts(fc.music.group.as_deref(), fc.music.stream.as_deref()),
            mpv_ipc: match fc.music.web_ipc {
                None => md.mpv_ipc.clone(),
                Some(v) => disable_sentinel(&v).map(PathBuf::from),
            },
            bin_dir: fc.music.bin_dir.unwrap_or_else(|| md.bin_dir.clone()),
            run_dir: fc.music.run_dir.unwrap_or_else(|| md.run_dir.clone()),
            log_dir: fc.music.log_dir.unwrap_or_else(|| md.log_dir.clone()),
            snapserver_conf: fc.music.conf.unwrap_or_else(|| md.snapserver_conf.clone()),
            spotify_device_name: nonempty(fc.music.spotify_device_name)
                .unwrap_or_else(|| md.spotify_device_name.clone()),
            autostart: fc.music.autostart.unwrap_or(md.autostart),
        };

        let wd = WeatherSettings::default();
        let weather = WeatherSettings {
            enabled: fc.weather.enabled.unwrap_or(wd.enabled),
            refresh_interval_secs: fc
                .weather
                .refresh_interval_secs
                .unwrap_or(wd.refresh_interval_secs),
        };

        let s1d = System1Config::default();
        let system1 = System1Config {
            backend: nonempty(fc.system1.backend).unwrap_or(s1d.backend),
            base_url: nonempty(fc.system1.base_url).unwrap_or(s1d.base_url),
            openrouter_model: nonempty(fc.system1.openrouter_model)
                .unwrap_or(s1d.openrouter_model),
            device: nonempty(fc.system1.device).unwrap_or(s1d.device),
            model: nonempty(fc.system1.model).unwrap_or(s1d.model),
            min_confidence: fc.system1.min_confidence.unwrap_or(s1d.min_confidence),
            intents: if fc.system1.intents.is_empty() {
                s1d.intents
            } else {
                fc.system1.intents
            },
        };

        // The config page: `off`/`none`/empty disables it, otherwise a host:port.
        let config_addr = match fc.config_addr {
            None => d.config_addr,
            Some(v) => match disable_sentinel(&v) {
                None => None,
                Some(s) => Some(
                    s.parse()
                        .with_context(|| format!("parsing config_addr=`{s}` as host:port"))?,
                ),
            },
        };

        let settings_path = match fc.settings_path {
            None => d.settings_path.clone(),
            Some(v) => disable_sentinel(&v).map(PathBuf::from),
        };

        let service_name = nonempty(fc.service_name).unwrap_or_else(|| d.service_name.clone());
        // Stable selection key: explicit config value, else the git branch code of the
        // working directory (a copy running from a test branch/worktree names itself by
        // its branch), else the sanitized service name. All must be stable across
        // restarts — persisted device selections depend on this not changing.
        let instance_id = nonempty(fc.instance_id)
            .or_else(git_branch_code)
            .unwrap_or_else(|| sanitize_id(&service_name));

        let calendar_specs = calendar_specs_from_file(fc.calendar.subscriptions);
        let calendar_cache_ttl = fc
            .calendar
            .cache_ttl_secs
            .map(Duration::from_secs)
            .unwrap_or(d.calendar_cache_ttl);

        let drive = DriveConfig {
            client_id: nonempty(fc.drive.client_id),
            client_secret: nonempty(fc.drive.client_secret),
            // The refresh token is minted by the consent flow, never seeded from config.
            refresh_token: None,
            folder_ids: fc
                .drive
                .folder_ids
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            scope: Some(
                nonempty(fc.drive.scope).unwrap_or_else(|| DEFAULT_DRIVE_SCOPE.to_string()),
            ),
        };

        let spotify = SpotifyConfig {
            client_id: nonempty(fc.spotify.client_id),
            client_secret: nonempty(fc.spotify.client_secret),
            refresh_token: nonempty(fc.spotify.refresh_token),
            device_name: nonempty(fc.spotify.device_name),
            scope: None,
        };

        let cadora = CadoraConfig {
            base_url: nonempty(fc.cadora.base_url),
            link_token: nonempty(fc.cadora.link_token),
        };

        let ollama_url = match &llm {
            LlmChoice::Ollama { url, .. } => url.clone(),
            _ => fc
                .llm
                .ollama
                .url
                .clone()
                .unwrap_or_else(|| d.ollama_url.clone()),
        };
        let anthropic_max_tokens = match &llm {
            LlmChoice::Anthropic { max_tokens, .. } => *max_tokens,
            _ => fc
                .llm
                .anthropic
                .max_tokens
                .unwrap_or(d.anthropic_max_tokens),
        };
        let openai_max_tokens = match &llm {
            LlmChoice::OpenAI { max_tokens, .. } => *max_tokens,
            _ => fc.llm.openai.max_tokens.unwrap_or(d.openai_max_tokens),
        };

        Ok(Self {
            bind_addr: fc.bind_addr.unwrap_or(d.bind_addr),
            config_addr,
            service_name,
            instance_id,
            stt_addr: fc.stt_addr.unwrap_or(d.stt_addr),
            tts_addr: fc.tts_addr.unwrap_or(d.tts_addr),
            tts_voice: nonempty(fc.tts_voice),
            tts_voices_dir: fc.tts_voices_dir.or(d.tts_voices_dir),
            llm,
            db_path: fc.db_path.unwrap_or(d.db_path),
            system_prompt: nonempty(fc.system_prompt).unwrap_or(d.system_prompt),
            home_location: nonempty(fc.home_location),
            weather_units: nonempty(fc.weather_units),
            turn_timeout: fc
                .turn_timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(d.turn_timeout),
            memory_backend,
            chatlog_path: fc.chatlog_path.unwrap_or(d.chatlog_path),
            promptlog_path: fc.promptlog_path.unwrap_or(d.promptlog_path),
            helix_path: fc.helix_path.unwrap_or(d.helix_path),
            graphrag,
            speaker,
            stt,
            follow_up,
            music,
            weather,
            system1,
            settings_path,
            audio_dump_dir: fc.audio_dump_dir,
            anthropic_auth: fc
                .llm
                .anthropic_auth
                .as_deref()
                .map(AnthropicAuth::from_label)
                .unwrap_or(d.anthropic_auth),
            engine: fc
                .llm
                .engine
                .as_deref()
                .map(LlmEngine::from_label)
                .unwrap_or(d.engine),
            web_search: fc.llm.web_search.unwrap_or(d.web_search),
            search_provider: nonempty(fc.llm.search_provider).unwrap_or(d.search_provider),
            anthropic_token_cmd: nonempty(fc.llm.anthropic_token_cmd),
            ollama_url,
            anthropic_max_tokens,
            openai_max_tokens,
            calendar_specs,
            calendar_cache_ttl,
            directions_provider: fc
                .directions
                .provider
                .map(|s| s.trim().to_string())
                .unwrap_or(d.directions_provider),
            drive,
            spotify,
            cadora,
        })
    }

    /// Build the selected System-1 decision engine (plans/system1-fast-decisions.md),
    /// mirroring [`Config::build_llm`]. Default `none` reproduces today's behavior.
    ///
    /// Build the System-1 engine straight from the config file's `system1` block (the
    /// OpenRouter key comes from the `OPENROUTER_API_KEY` env secret). The live boot path
    /// is [`Self::shared_settings`], which also applies the persisted overlay and makes
    /// the engine runtime-swappable; this helper is kept for direct/one-shot use.
    pub fn build_system1(&self) -> Result<Arc<dyn crate::system1::DecisionEngine>> {
        let key = env::var("OPENROUTER_API_KEY").ok().filter(|s| !s.is_empty());
        if self.system1.backend.eq_ignore_ascii_case("jev") && key.is_none() {
            log::warn!(
                "system1.backend=jev but OPENROUTER_API_KEY is unset; Jev requests will be \
                 rejected until it is provided (env or config page)."
            );
        }
        crate::system1::build(
            &self.system1.backend,
            &self.system1.base_url,
            &self.system1.openrouter_model,
            key,
            self.system1.min_confidence,
            self.system1.intents.clone(),
        )
    }

    /// Build the music ducker when music routing **and** duck-on-speech are both
    /// enabled; otherwise `None` (the whole path stays dormant). Best-effort at
    /// runtime — a missing/unreachable snapserver never breaks a turn.
    pub fn build_ducker(&self) -> Option<Arc<MusicDucker>> {
        if self.music.enabled && self.music.duck_on_speech {
            Some(Arc::new(MusicDucker::new(
                self.music.snapserver_addr,
                self.music.group.clone(),
                self.music.duck_percent,
            )))
        } else {
            None
        }
    }

    /// Build the mpv IPC control for the web-URL player, when music is enabled and
    /// an IPC socket is configured (`music.web_ipc`).
    pub fn mpv_control(&self) -> Option<MpvControl> {
        if !self.music.enabled {
            return None;
        }
        self.music.mpv_ipc.as_ref().map(MpvControl::new)
    }

    /// Build the process supervisor for the music sibling processes
    /// (snapserver / librespot / mpv), or `None` when music is disabled. The
    /// launch commands mirror `anamanti-core/deploy/snapcast/` (the runbook +
    /// launchd agents).
    pub fn build_supervisor(&self) -> Option<Arc<MusicSupervisor>> {
        if !self.music.enabled {
            return None;
        }
        let m = &self.music;
        let bin = |name: &str| m.bin_dir.join(name);
        let fifo = |name: &str| m.run_dir.join(name).display().to_string();
        let log = |name: &str| m.log_dir.join(name);
        let mpv_ipc = m
            .mpv_ipc
            .clone()
            .unwrap_or_else(|| PathBuf::from("/tmp/ambient-mpv.sock"));

        let mut specs = HashMap::new();
        specs.insert(
            ManagedProc::Snapserver,
            ProcSpec {
                program: bin("snapserver"),
                args: vec!["-c".into(), m.snapserver_conf.display().to_string()],
                log_path: log("ambient-snapserver.log"),
            },
        );
        specs.insert(
            ManagedProc::Librespot,
            ProcSpec {
                program: bin("librespot"),
                args: vec![
                    "--name".into(),
                    m.spotify_device_name.clone(),
                    "--backend".into(),
                    "pipe".into(),
                    "--device".into(),
                    fifo("snap-spotify"),
                    "--bitrate".into(),
                    "320".into(),
                    "--initial-volume".into(),
                    "100".into(),
                ],
                log_path: log("ambient-librespot.log"),
            },
        );
        specs.insert(
            ManagedProc::MpvWeb,
            ProcSpec {
                program: bin("mpv"),
                args: vec![
                    "--idle=yes".into(),
                    "--no-video".into(),
                    format!("--input-ipc-server={}", mpv_ipc.display()),
                    "--ao=pcm".into(),
                    "--ao-pcm-waveheader=no".into(),
                    format!("--ao-pcm-file={}", fifo("snap-web")),
                    "--audio-samplerate=48000".into(),
                    "--audio-channels=stereo".into(),
                    "--audio-format=s16".into(),
                ],
                log_path: log("ambient-mpv-web.log"),
            },
        );
        Some(Arc::new(MusicSupervisor::new(specs)))
    }

    /// Build the [`MusicHub`] (supervisor + snapserver control + mpv control) the
    /// config-page Music tab drives, or `None` when music is disabled.
    pub fn build_hub(&self) -> Option<MusicHub> {
        let supervisor = self.build_supervisor()?;
        Some(MusicHub {
            supervisor,
            snapcast: SnapcastClient::new(self.music.snapserver_addr),
            snapserver_addr: self.music.snapserver_addr.to_string(),
            mpv: self.mpv_control(),
        })
    }

    /// Instantiate the selected LLM backend behind the trait object the pipeline
    /// consumes. The Anthropic backend requires `ANTHROPIC_API_KEY`.
    pub fn build_llm(&self) -> Result<Arc<dyn LlmBackend>> {
        Ok(match &self.llm {
            LlmChoice::Mock => Arc::new(MockLlm::default()),
            LlmChoice::Ollama { url, model } => Arc::new(OllamaBackend::new(url, model)),
            LlmChoice::Anthropic { model, max_tokens } => {
                let key = env::var("ANTHROPIC_API_KEY")
                    .context("the anthropic backend (llm.backend) requires ANTHROPIC_API_KEY")?;
                Arc::new(AnthropicBackend::new(
                    "https://api.anthropic.com",
                    key,
                    model,
                    *max_tokens,
                ))
            }
            LlmChoice::OpenAI { model, max_tokens } => {
                let key = env::var("OPENAI_API_KEY")
                    .context("the openai backend (llm.backend) requires OPENAI_API_KEY")?;
                Arc::new(OpenAiBackend::new(
                    self.graphrag.openai_base_url.clone(),
                    key,
                    model,
                    *max_tokens,
                ))
            }
        })
    }

    /// The inputs a runtime backend swap (Phase 6) needs, captured from the config
    /// once so a later swap never re-reads anything. The provider API keys read here
    /// (still from the environment, since they're secrets) are only the **boot seed**:
    /// they flow into the live [`RuntimeSettings`], where the config page can override
    /// them at runtime, so a cloud backend can be enabled without a restart even if its
    /// key was unset at boot. If no key is present at boot *or* runtime, selecting that
    /// backend is rejected in-band.
    pub fn llm_factory(&self) -> LlmFactory {
        let imperial = crate::directions::units_are_imperial(self.weather_units.as_deref());
        LlmFactory {
            ollama_url: self.ollama_url.clone(),
            anthropic_base_url: "https://api.anthropic.com".to_string(),
            anthropic_api_key: env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.is_empty()),
            anthropic_max_tokens: self.anthropic_max_tokens,
            openai_base_url: self.graphrag.openai_base_url.clone(),
            openai_api_key: env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty()),
            openai_max_tokens: self.openai_max_tokens,
            anthropic_token: Some(Arc::new(AnthropicTokenProvider::new(
                self.anthropic_token_cmd.clone(),
            ))),
            // Seeded from the resolved Household in `shared_settings` (config → household
            // → this handle); starts empty here.
            home_location: crate::directions::LiveHomeLocation::default(),
            // Seeded per-build from the resolved Spotify config in `shared_settings`
            // (and refreshed by `apply`/`apply_spotify`).
            spotify: None,
            // Seeded per-build from the resolved Cadora config in `shared_settings`
            // (and refreshed by `apply`/`apply_cadora`).
            cadora: None,
            // Prebuilt from the config: the calendar source (web .ics) and the
            // directions provider (Mapbox). The Mapbox token is seeded from the env
            // secret here but is runtime-settable (Tools tab) — `shared_settings`
            // rebuilds `directions` from the resolved token, and `apply_directions`
            // rebuilds it on a config-page change.
            calendar: crate::calendar::build(self.calendar_specs.clone(), self.calendar_cache_ttl),
            directions: crate::directions::from_token(
                &self.directions_provider,
                self.initial_mapbox_token().as_deref(),
                imperial,
            ),
            directions_provider: self.directions_provider.clone(),
            directions_imperial: imperial,
            // Weather is keyless (Open-Meteo), so it's present whenever enabled; the
            // tool/push still no-op gracefully until a home location is set.
            weather: crate::weather::from_config(self.weather.enabled),
            weather_imperial: imperial,
        }
    }

    /// The selectable-model catalog for the settings dropdown, wired with the same
    /// provider endpoints/keys the factory uses (keys read from the environment) and
    /// the same Anthropic auth mode so subscription hosts list live models.
    pub fn model_catalog(&self) -> ModelCatalog {
        ModelCatalog::new(
            "https://api.anthropic.com",
            env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.is_empty()),
            self.graphrag.openai_base_url.clone(),
            env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty()),
        )
        .with_anthropic_auth(
            self.anthropic_auth,
            Some(Arc::new(AnthropicTokenProvider::new(
                self.anthropic_token_cmd.clone(),
            ))),
        )
    }

    /// The initial engine + web-search selection from the config. These seed the live
    /// [`RuntimeSettings`] and can be changed at runtime (config page / control frame).
    pub fn initial_engine(&self) -> LlmEngine {
        self.engine
    }
    pub fn initial_web_search(&self) -> bool {
        self.web_search
    }
    /// Initial search provider (`llm.search_provider`, default `tavily`).
    pub fn initial_search_provider(&self) -> String {
        self.search_provider.clone()
    }
    /// Initial search API key — a secret, so it's still read from `TAVILY_API_KEY`.
    pub fn initial_search_api_key(&self) -> Option<String> {
        env::var("TAVILY_API_KEY").ok().filter(|s| !s.is_empty())
    }

    /// Initial Mapbox token for the directions tool — a secret, seeded from
    /// `MAPBOX_TOKEN` / `MAPBOX_ACCESS_TOKEN`. Runtime-settable from the Tools tab.
    pub fn initial_mapbox_token(&self) -> Option<String> {
        env::var("MAPBOX_TOKEN")
            .ok()
            .or_else(|| env::var("MAPBOX_ACCESS_TOKEN").ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Initial Anthropic subscription OAuth token — a secret, seeded from
    /// `ANTHROPIC_OAUTH_TOKEN`. Runtime-settable from the Config tab (auth =
    /// subscription); overrides the token command when set.
    pub fn initial_anthropic_oauth_token(&self) -> Option<String> {
        env::var("ANTHROPIC_OAUTH_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Initial Google Drive photo-slideshow config seeded from the config file's
    /// `drive` block (client id/secret + folder ids). The refresh token is never
    /// seeded — it's minted by the consent flow. A persisted file overlays these at
    /// boot (see [`Self::shared_settings`]).
    pub fn initial_drive(&self) -> DriveConfig {
        self.drive.clone()
    }

    /// The initial household + home information seeded from the config file
    /// (`home_location` / `weather_units`). The member roster has no config form — it
    /// is dashboard-only — so it starts empty. A persisted file overlays these at boot
    /// (see [`Self::shared_settings`]), so a location edited on the config page wins
    /// over the config-file seed.
    pub fn initial_household(&self) -> Household {
        Household {
            location: self.home_location.clone(),
            weather_units: self.weather_units.clone(),
            members: Vec::new(),
        }
    }

    /// Initial Spotify voice-control config seeded from the config file's `spotify`
    /// block. Any of these may instead be set (or overridden) at runtime by the
    /// config-page consent flow; a persisted file overlays these at boot (see
    /// [`Self::shared_settings`]). Requires Spotify Premium to actually play.
    pub fn initial_spotify(&self) -> SpotifyConfig {
        self.spotify.clone()
    }

    /// Initial Cadora shopping-list config seeded from the config file's `cadora`
    /// block. The voice-link token is normally minted at runtime by the config-page
    /// pairing flow; a persisted file overlays these at boot (see
    /// [`Self::shared_settings`]).
    pub fn initial_cadora(&self) -> CadoraConfig {
        self.cadora.clone()
    }

    /// Build the shared, runtime-swappable settings (Phase 6): the initial backend
    /// selected by config plus the factory that rebuilds backends when the device
    /// changes them. The initial backend must build successfully (anthropic still
    /// needs its key at startup, matching [`Self::build_llm`]).
    pub fn shared_settings(&self) -> Result<Arc<SharedSettings>> {
        let factory = self.llm_factory();
        let persist_path = self.settings_path.clone();

        // Environment/config defaults, then overlay a persisted file when present
        // (page/device changes from a previous run win across restarts).
        let (backend_default, model_default) = match &self.llm {
            LlmChoice::Mock => ("mock".to_string(), None),
            LlmChoice::Ollama { model, .. } => ("ollama".to_string(), Some(model.clone())),
            LlmChoice::Anthropic { model, .. } => ("anthropic".to_string(), Some(model.clone())),
            LlmChoice::OpenAI { model, .. } => ("openai".to_string(), Some(model.clone())),
        };
        let mut engine = self.initial_engine();
        let mut web_search = self.initial_web_search();
        let mut search_provider = self.initial_search_provider();
        let mut search_api_key = self.initial_search_api_key();
        let mut backend = backend_default;
        let mut model = model_default;
        // Live provider keys start from the environment (the factory's boot seed) and
        // become runtime-settable; a persisted key overlays them so a key entered on
        // the config page enables the cloud backend without a restart.
        let mut anthropic_api_key = factory.anthropic_api_key.clone();
        let mut openai_api_key = factory.openai_api_key.clone();
        // Subscription OAuth token + Mapbox token: env seed (secrets), overlaid by any
        // persisted values below (a value entered on the config page wins).
        let mut anthropic_oauth_token = self.initial_anthropic_oauth_token();
        let mut mapbox_token = self.initial_mapbox_token();
        let mut anthropic_auth = self.anthropic_auth;
        let mut tts_voice = self.tts_voice.clone();

        let mut end_silence_ms = crate::settings::DEFAULT_END_SILENCE_MS;
        let mut voice_rms_threshold = crate::settings::DEFAULT_VOICE_RMS_THRESHOLD;
        // Google Drive photo config: config-file seed (client creds/folders), overlaid
        // by any persisted values below (the refresh token + page-set fields win).
        let mut drive = self.initial_drive();
        // Household + home info: config-file seed (location/units from home_location /
        // weather_units), overlaid by any persisted values below so a location fixed on
        // the config page — and the dashboard-only member roster — win.
        let mut household = self.initial_household();
        // Spotify voice-control config: same seed-then-persist-overlay pattern.
        let mut spotify = self.initial_spotify();
        // Cadora shopping-list config: same seed-then-persist-overlay pattern.
        let mut cadora = self.initial_cadora();
        // System-1 fast-decision selection: config-file seed, overlaid by persisted
        // values below. The OpenRouter key is an env secret seed (like the other keys).
        let mut system1_backend = self.system1.backend.clone();
        let mut system1_base_url = self.system1.base_url.clone();
        let mut system1_model = self.system1.openrouter_model.clone();
        let mut system1_min_confidence = self.system1.min_confidence;
        let mut system1_intents = self.system1.intents.clone();
        let mut openrouter_api_key = env::var("OPENROUTER_API_KEY").ok().filter(|s| !s.is_empty());

        if let Some(p) = persist_path.as_deref().and_then(load_persisted) {
            log::info!("loaded persisted settings");
            engine = LlmEngine::from_label(&p.engine);
            web_search = p.web_search;
            search_provider = p.search_provider;
            search_api_key = p.search_api_key;
            backend = p.llm_backend;
            model = p.llm_model;
            // Only override the env key when the persisted file actually carries one,
            // so a settings file without a key (serde default `None`) can't wipe a
            // working `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` from the environment.
            if p.anthropic_api_key.is_some() {
                anthropic_api_key = p.anthropic_api_key;
            }
            if p.openai_api_key.is_some() {
                openai_api_key = p.openai_api_key;
            }
            // Same guard for the OAuth / Mapbox tokens: only override the env seed when
            // the persisted file actually carries one, so an older file (field absent →
            // serde default `None`) can't wipe a working env token.
            if p.anthropic_oauth_token.is_some() {
                anthropic_oauth_token = p.anthropic_oauth_token;
            }
            if p.mapbox_token.is_some() {
                mapbox_token = p.mapbox_token;
            }
            anthropic_auth = AnthropicAuth::from_label(&p.anthropic_auth);
            tts_voice = p.tts_voice;
            end_silence_ms = p.end_silence_ms;
            voice_rms_threshold = p.voice_rms_threshold;
            // Overlay persisted Drive fields onto the config-file seed: a persisted
            // value wins (refresh token, page-set creds/folders), but keep the seed for
            // any field the persisted file leaves empty so setting a client id in
            // anamanti.json still takes effect after an older file is loaded.
            if p.drive.client_id.is_some() {
                drive.client_id = p.drive.client_id;
            }
            if p.drive.client_secret.is_some() {
                drive.client_secret = p.drive.client_secret;
            }
            if p.drive.refresh_token.is_some() {
                drive.refresh_token = p.drive.refresh_token;
            }
            if !p.drive.folder_ids.is_empty() {
                drive.folder_ids = p.drive.folder_ids;
            }
            if p.drive.scope.is_some() {
                drive.scope = p.drive.scope;
            }
            // Overlay persisted household onto the config-file seed: a persisted
            // location / units wins (fixed on the config page), but keep the seed for
            // any field the persisted file leaves empty so home_location still applies
            // after an older file (no household) is loaded. The member roster is
            // dashboard-only, so a persisted list always replaces the empty seed.
            if p.household.location.is_some() {
                household.location = p.household.location;
            }
            if p.household.weather_units.is_some() {
                household.weather_units = p.household.weather_units;
            }
            if !p.household.members.is_empty() {
                household.members = p.household.members;
            }
            // Overlay persisted Spotify fields onto the config-file seed (same rule as
            // Drive: a persisted value wins; keep the seed for anything left empty).
            if p.spotify.client_id.is_some() {
                spotify.client_id = p.spotify.client_id;
            }
            if p.spotify.client_secret.is_some() {
                spotify.client_secret = p.spotify.client_secret;
            }
            if p.spotify.refresh_token.is_some() {
                spotify.refresh_token = p.spotify.refresh_token;
            }
            if p.spotify.device_name.is_some() {
                spotify.device_name = p.spotify.device_name;
            }
            if p.spotify.scope.is_some() {
                spotify.scope = p.spotify.scope;
            }
            // Overlay persisted Cadora fields onto the config-file seed (same rule).
            if p.cadora.base_url.is_some() {
                cadora.base_url = p.cadora.base_url;
            }
            if p.cadora.link_token.is_some() {
                cadora.link_token = p.cadora.link_token;
            }
            // Overlay persisted System-1 selection onto the config-file seed. Only when
            // the persisted backend is non-empty (a real save), so an older settings
            // file — which lacks these fields (serde default "") — can't disable a
            // `system1.backend` set in anamanti.json.
            if !p.system1_backend.is_empty() {
                system1_backend = p.system1_backend;
                system1_base_url = p.system1_base_url;
                system1_model = p.system1_model;
                system1_min_confidence = p.system1_min_confidence;
                system1_intents = p.system1_intents;
            }
            if p.openrouter_api_key.is_some() {
                openrouter_api_key = p.openrouter_api_key;
            }
        }

        // Seed the directions tool's live default origin from the resolved household
        // location (config → household → this shared handle). The factory clone below
        // shares the same cell, so the initial backend's tool — and every later
        // rebuild — reads it; `apply_household` updates it on a config-page edit.
        factory.home_location.set(household.location.clone());
        // Seed the shared token provider's override from the resolved subscription
        // token (env overlaid by persisted). The Arc is shared with every rebuilt
        // backend, so a token entered on the config page later takes effect in place.
        if let Some(provider) = &factory.anthropic_token {
            provider.set_override(anthropic_oauth_token.clone());
        }

        // Build the initial backend with the resolved live keys (env overlaid by any
        // persisted key), never re-reading the environment during a later swap.
        let mut build_factory = factory.clone();
        build_factory.anthropic_api_key = anthropic_api_key.clone();
        build_factory.openai_api_key = openai_api_key.clone();
        // Seed the initial Spotify controller so the `spotify_control` tool is
        // advertised at boot when the account is already linked (env or persisted).
        build_factory.spotify = spotify.controller();
        // Seed the initial Cadora controller so the `shopping_list_add` tool is
        // advertised at boot when the shopping list is already linked.
        build_factory.cadora = cadora.controller();
        // Seed the initial directions provider from the resolved Mapbox token so the
        // `directions_lookup` tool is advertised at boot when a token is present.
        build_factory.directions = crate::directions::from_token(
            &build_factory.directions_provider,
            mapbox_token.as_deref(),
            build_factory.directions_imperial,
        );
        let (llm, llm_backend, llm_model) = build_factory
            .build(
                engine,
                web_search,
                &search_provider,
                search_api_key.as_deref(),
                &backend,
                model.as_deref(),
                anthropic_auth,
            )
            .context("building the initial LLM backend")?;
        // Build the initial System-1 engine from the resolved (seed → persisted) config.
        if system1_backend.eq_ignore_ascii_case("jev") && openrouter_api_key.is_none() {
            log::warn!(
                "system1.backend=jev but OPENROUTER_API_KEY is unset; Jev requests will be \
                 rejected until it is provided (env or config page)."
            );
        }
        let system1 = crate::system1::build(
            &system1_backend,
            &system1_base_url,
            &system1_model,
            openrouter_api_key.clone(),
            system1_min_confidence,
            system1_intents.clone(),
        )
        .context("building the initial System-1 engine")?;
        Ok(SharedSettings::new_persistent(
            factory,
            RuntimeSettings {
                llm,
                engine,
                web_search,
                search_provider,
                search_api_key,
                llm_backend,
                llm_model,
                anthropic_api_key,
                openai_api_key,
                anthropic_oauth_token,
                mapbox_token,
                anthropic_auth,
                tts_voice,
                end_silence_ms,
                voice_rms_threshold,
                drive,
                household,
                spotify,
                cadora,
                system1: crate::settings::System1Runtime {
                    engine: system1,
                    backend: system1_backend,
                    base_url: system1_base_url,
                    model: system1_model,
                    min_confidence: system1_min_confidence,
                    intents: system1_intents,
                    openrouter_api_key,
                },
            },
            persist_path,
        ))
    }

    /// Build the per-person [`SpeakerService`], or `None` when speaker ID is
    /// disabled (`speaker.enabled = false`). The registry lives in the same SQLite
    /// file as memory. When `speaker.model_path` is set the real ECAPA-TDNN ONNX
    /// embedder is loaded; otherwise it falls back to the deterministic mock
    /// embedder (with a loud warning, as the mock is not accurate for real voices).
    pub fn build_speaker_service(&self) -> Result<Option<Arc<crate::speaker::SpeakerService>>> {
        use crate::speaker::{
            MockSpeakerEmbedder, SpeakerEmbedder, SpeakerRegistry, SpeakerService,
            SpeakerThresholds,
        };
        if !self.speaker.enabled {
            return Ok(None);
        }
        let registry =
            SpeakerRegistry::open(&self.db_path).context("opening speaker registry database")?;
        let thresholds = SpeakerThresholds::from_ms(
            self.speaker.match_threshold,
            self.speaker.new_threshold,
            self.speaker.min_speech_ms,
        );
        let embedder: Arc<dyn SpeakerEmbedder> = match &self.speaker.model_path {
            Some(path) => {
                use crate::speaker::embed::OnnxSpeakerEmbedder;
                use crate::speaker::features::FbankConfig;
                match OnnxSpeakerEmbedder::open(
                    path,
                    self.speaker.embed_dims,
                    FbankConfig::default(),
                ) {
                    Ok(e) => {
                        log::info!("speaker embedder: ONNX model {}", path.display());
                        Arc::new(e)
                    }
                    Err(err) => {
                        log::error!(
                            "failed to load speaker model {} ({err:#}); using the mock embedder",
                            path.display()
                        );
                        Arc::new(MockSpeakerEmbedder::default())
                    }
                }
            }
            None => {
                log::warn!(
                    "speaker ID enabled with no model path; using the deterministic mock embedder \
                     (dev/testing only — not accurate for real voices). Set speaker.model_path \
                     to use the ONNX model."
                );
                Arc::new(MockSpeakerEmbedder::default())
            }
        };
        Ok(Some(Arc::new(SpeakerService::new(
            embedder, registry, thresholds,
        ))))
    }

    /// A short label for the selected backend (logging/settings).
    pub fn llm_label(&self) -> &'static str {
        match self.llm {
            LlmChoice::Mock => "mock",
            LlmChoice::Ollama { .. } => "ollama",
            LlmChoice::Anthropic { .. } => "anthropic",
            LlmChoice::OpenAI { .. } => "openai",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_id_defaults_match_service_name() {
        assert_eq!(Config::default().instance_id, "anamanti-core");
    }

    #[test]
    fn sanitize_id_maps_service_names_to_stable_keys() {
        assert_eq!(sanitize_id("Anamanti Core"), "anamanti-core");
        assert_eq!(sanitize_id("Test Mac"), "test-mac");
        assert_eq!(sanitize_id("Mac Mini (prod)"), "mac-mini--prod");
        assert_eq!(sanitize_id("***"), "anamanti-core");
    }

    fn parse(json: &str) -> FileConfig {
        serde_json::from_str(json).expect("valid FileConfig JSON")
    }

    #[test]
    fn empty_file_reproduces_defaults() {
        // `{}` overlays nothing, so every resolved value matches `Config::default()`
        // (except `instance_id`, which resolves via git branch / service name).
        let c = Config::from_file(parse("{}")).unwrap();
        let d = Config::default();
        assert_eq!(c.bind_addr, d.bind_addr);
        assert_eq!(c.config_addr, d.config_addr);
        assert_eq!(c.stt_addr, d.stt_addr);
        assert_eq!(c.tts_addr, d.tts_addr);
        assert_eq!(c.db_path, d.db_path);
        assert_eq!(c.helix_path, d.helix_path);
        assert_eq!(c.system_prompt, d.system_prompt);
        assert_eq!(c.memory_backend, d.memory_backend);
        assert_eq!(c.llm_label(), "ollama");
        assert_eq!(c.ollama_url, d.ollama_url);
        assert_eq!(c.engine, d.engine);
        assert!(c.web_search);
        assert_eq!(c.search_provider, "tavily");
        assert_eq!(c.anthropic_auth, AnthropicAuth::ApiKey);
        assert_eq!(c.settings_path, d.settings_path);
        assert!(c.audio_dump_dir.is_none());
        assert!(c.calendar_specs.is_empty());
        assert_eq!(c.calendar_cache_ttl, d.calendar_cache_ttl);
        assert_eq!(c.drive.scope.as_deref(), Some(DEFAULT_DRIVE_SCOPE));
        assert!(!c.speaker.enabled);
        assert!(c.music.enabled);
        // STT defaults to the downstream Wyoming engine.
        assert_eq!(c.stt.engine, SttEngineKind::Wyoming);
        assert_eq!(c.stt.model, "base");
        assert_eq!(c.stt.language.as_deref(), Some("en"));
    }

    #[test]
    fn stt_block_selects_in_process_engine() {
        let c = Config::from_file(parse(
            r#"{ "stt": { "engine": "whisper-rs", "model": "small",
                          "model_dir": "/models", "num_threads": 4 } }"#,
        ))
        .unwrap();
        assert_eq!(c.stt.engine, SttEngineKind::WhisperLocal);
        assert_eq!(c.stt.model, "small");
        assert_eq!(c.stt.num_threads, 4);
        assert_eq!(
            c.stt.resolved_model_path(),
            std::path::PathBuf::from("/models/ggml-small.en.bin")
        );
    }

    #[test]
    fn stt_model_path_overrides_dir_and_name() {
        let c = Config::from_file(parse(
            r#"{ "stt": { "model_path": "/opt/x.bin", "language": "" } }"#,
        ))
        .unwrap();
        assert_eq!(
            c.stt.resolved_model_path(),
            std::path::PathBuf::from("/opt/x.bin")
        );
        // An explicit empty language means auto-detect.
        assert_eq!(c.stt.language, None);
    }

    #[test]
    fn stt_unknown_engine_is_rejected() {
        let err = Config::from_file(parse(r#"{ "stt": { "engine": "vosk" } }"#)).unwrap_err();
        assert!(err.to_string().contains("stt.engine"), "{err}");
    }

    #[test]
    fn full_file_overlays_every_layer() {
        let c = Config::from_file(parse(
            r#"{
                "bind_addr": "0.0.0.0:10701",
                "config_addr": "127.0.0.1:8731",
                "service_name": "Test Mac",
                "instance_id": "test-key",
                "system_prompt": "hi",
                "home_location": "  Austin, Texas  ",
                "weather_units": "imperial",
                "turn_timeout_secs": 45,
                "memory_backend": "sqlite",
                "audio_dump_dir": "/tmp/dump",
                "settings_path": "custom_settings.json",
                "llm": {
                    "backend": "anthropic",
                    "engine": "native",
                    "anthropic_auth": "subscription",
                    "anthropic_token_cmd": "my-token-cmd",
                    "web_search": false,
                    "search_provider": "tavily",
                    "ollama": { "url": "http://ollama:11434", "model": "qwen2.5" },
                    "anthropic": { "model": "claude-x", "max_tokens": 2048 },
                    "openai": { "model": "gpt-x", "max_tokens": 512 }
                },
                "graphrag": { "embed_dims": 256, "ingest_interval_secs": 10, "recall_k": 9 },
                "speaker": { "enabled": true, "match_threshold": 0.7 },
                "music": { "enabled": true, "duck_percent": 250, "group": "g1" },
                "calendar": {
                    "subscriptions": [ { "url": "https://ex.com/a.ics" }, { "name": "Work", "url": "https://ex.com/w.ics" } ],
                    "cache_ttl_secs": 600
                },
                "directions": { "provider": "mapbox" },
                "drive": { "client_id": "cid", "folder_ids": ["f1", " ", "f2"] },
                "spotify": { "client_id": "sid", "refresh_token": "rt" }
            }"#,
        ))
        .unwrap();
        assert_eq!(c.bind_addr.to_string(), "0.0.0.0:10701");
        assert_eq!(
            c.config_addr.map(|a| a.to_string()).as_deref(),
            Some("127.0.0.1:8731")
        );
        assert_eq!(c.service_name, "Test Mac");
        assert_eq!(c.instance_id, "test-key");
        assert_eq!(c.home_location.as_deref(), Some("Austin, Texas")); // trimmed
        assert_eq!(c.weather_units.as_deref(), Some("imperial"));
        assert_eq!(c.turn_timeout, Duration::from_secs(45));
        assert_eq!(c.memory_backend, MemoryBackendChoice::Sqlite);
        assert_eq!(
            c.audio_dump_dir.as_deref(),
            Some(std::path::Path::new("/tmp/dump"))
        );
        assert_eq!(
            c.settings_path.as_deref(),
            Some(std::path::Path::new("custom_settings.json"))
        );
        assert_eq!(c.llm_label(), "anthropic");
        assert_eq!(c.engine, LlmEngine::Native);
        assert_eq!(c.anthropic_auth, AnthropicAuth::Subscription);
        assert_eq!(c.anthropic_token_cmd.as_deref(), Some("my-token-cmd"));
        assert!(!c.web_search);
        assert_eq!(c.search_provider, "tavily");
        // The non-selected backends' params still resolve from their sub-blocks.
        assert_eq!(c.ollama_url, "http://ollama:11434");
        assert_eq!(c.anthropic_max_tokens, 2048);
        assert_eq!(c.openai_max_tokens, 512);
        assert_eq!(c.graphrag.embed_dims, 256);
        assert_eq!(c.graphrag.ingest_interval, Duration::from_secs(10));
        assert_eq!(c.graphrag.recall_k, 9);
        assert!(c.speaker.enabled);
        assert_eq!(c.speaker.match_threshold, 0.7);
        assert!(c.music.enabled);
        assert_eq!(c.music.duck_percent, 100); // capped at 100
                                               // Calendar: bare URL auto-named, explicit name kept.
        assert_eq!(c.calendar_specs.len(), 2);
        assert_eq!(c.calendar_specs[0].name, "Calendar 1");
        assert_eq!(c.calendar_specs[1].name, "Work");
        assert_eq!(c.calendar_cache_ttl, Duration::from_secs(600));
        // Drive: blank folder id dropped; refresh token never seeded.
        assert_eq!(c.drive.client_id.as_deref(), Some("cid"));
        assert_eq!(c.drive.folder_ids, vec!["f1", "f2"]);
        assert!(c.drive.refresh_token.is_none());
        assert_eq!(c.spotify.client_id.as_deref(), Some("sid"));
        assert_eq!(c.spotify.refresh_token.as_deref(), Some("rt"));
    }

    #[test]
    fn disable_sentinels_turn_features_off() {
        let c = Config::from_file(parse(
            r#"{ "config_addr": "off", "settings_path": "none", "music": { "web_ipc": "" } }"#,
        ))
        .unwrap();
        assert!(c.config_addr.is_none());
        assert!(c.settings_path.is_none());
        assert!(c.music.mpv_ipc.is_none());
    }

    #[test]
    fn unknown_key_is_rejected() {
        // `deny_unknown_fields` turns a typo into a hard error, not a silent default.
        let err = serde_json::from_str::<FileConfig>(r#"{ "bnid_addr": "0.0.0.0:1" }"#)
            .expect_err("a misspelled key must be rejected");
        assert!(err.to_string().contains("bnid_addr"), "{err}");
    }

    #[test]
    fn instance_id_falls_back_to_sanitized_service_name_when_no_git() {
        // With an explicit service name and no instance_id, the key is the sanitized
        // service name — unless a git branch is detected. Assert it's one of the two
        // stable forms (never empty / random).
        let c = Config::from_file(parse(r#"{ "service_name": "Kitchen Mac" }"#)).unwrap();
        assert!(!c.instance_id.is_empty());
        assert_eq!(c.instance_id, c.instance_id.to_lowercase());
    }

    #[test]
    fn load_missing_explicit_config_is_an_error() {
        let missing = std::env::temp_dir().join(format!(
            "ambient_no_such_config_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&missing);
        assert!(Config::load(Some(&missing)).is_err());
    }

    #[test]
    fn shipped_example_config_is_valid() {
        // The committed `anamanti.example.json` must always parse under
        // `deny_unknown_fields` and resolve, so it can't drift out of sync with the
        // schema.
        let example = concat!(env!("CARGO_MANIFEST_DIR"), "/anamanti.example.json");
        let c = Config::load(Some(std::path::Path::new(example)))
            .expect("anamanti.example.json must be a valid config");
        assert_eq!(c.service_name, "Anamanti Core");
        assert_eq!(c.llm_label(), "ollama");
    }

    #[test]
    fn load_explicit_config_reads_the_file() {
        let path = std::env::temp_dir().join(format!(
            "ambient_load_config_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, r#"{ "service_name": "Loaded Mac" }"#).unwrap();
        let c = Config::load(Some(&path)).unwrap();
        assert_eq!(c.service_name, "Loaded Mac");
        let _ = std::fs::remove_file(&path);
    }
}
