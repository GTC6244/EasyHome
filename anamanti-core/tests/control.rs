//! Phase-6 control-protocol integration tests.
//!
//! These drive the *real* device-facing server accept loop over a loopback TCP
//! socket and exercise the project-local `ambient-*` control frames end-to-end:
//! describing + changing runtime settings and listing/deleting persistent memory.
//! No STT/TTS/LLM services are needed — control requests are answered directly
//! from the memory store and the shared settings.

use std::sync::Arc;
use std::time::Duration;

use anamanti_core::llm::anthropic_auth::AnthropicAuth;
use anamanti_core::memory::{MemoryKind, MemorySource, MemoryStore};
use anamanti_core::orchestrator::{Pipeline, ServiceConnector, TcpConnector};
use anamanti_core::settings::{
    CadoraConfig, DriveConfig, Household, LlmEngine, LlmFactory, RuntimeSettings, SharedSettings,
    SpotifyConfig, DEFAULT_END_SILENCE_MS, DEFAULT_VOICE_RMS_THRESHOLD,
};
use anamanti_core::wyoming::protocol::{read_event, types, write_event, WyomingEvent};
use serde_json::json;
use tokio::io::BufReader;
use tokio::net::{TcpListener, TcpStream};

/// Start the server on a loopback port and return its address + the shared memory
/// store (so the test can assert on persisted state) + settings.
async fn start_server() -> (std::net::SocketAddr, Arc<MemoryStore>, Arc<SharedSettings>) {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let factory = LlmFactory {
        ollama_url: "http://127.0.0.1:11434".into(),
        anthropic_base_url: "https://api.anthropic.com".into(),
        anthropic_api_key: None,
        anthropic_max_tokens: 128,
        openai_base_url: "https://api.openai.com".into(),
        openai_api_key: None,
        openai_max_tokens: 128,
        anthropic_token: None,
        home_location: anamanti_core::directions::LiveHomeLocation::default(),
        spotify: None,
        cadora: None,
        calendar: None,
        directions: None,
        directions_provider: String::new(),
        directions_imperial: false,
        weather: None,
        weather_imperial: false,
    };
    let (llm, backend, model) = factory
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
    let settings = SharedSettings::new(
        factory,
        RuntimeSettings {
            llm,
            engine: LlmEngine::Native,
            web_search: false,
            search_provider: "duckduckgo".to_string(),
            search_api_key: None,
            llm_backend: backend,
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
            system1: anamanti_core::settings::System1Runtime::default(),
        },
    );
    let pipeline = Pipeline::with_settings(
        settings.clone(),
        memory.clone(),
        "persona",
        Duration::from_secs(5),
    );
    // A connector is required by the API but control requests never touch it.
    let connector: Arc<dyn ServiceConnector> = Arc::new(TcpConnector {
        stt_addr: "127.0.0.1:1".parse().unwrap(),
        tts_addr: "127.0.0.1:1".parse().unwrap(),
    });

    // No provider keys → the catalog serves its static fallback (no network).
    let catalog = Arc::new(anamanti_core::llm::catalog::ModelCatalog::new(
        "http://unused",
        None,
        "http://unused",
        None,
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let notify = Arc::new(anamanti_core::notify::NotificationService::new());
    let weather = Arc::new(anamanti_core::weather::WeatherService::new());
    tokio::spawn(anamanti_core::server::serve(
        listener, pipeline, connector, catalog, None, None, notify, weather,
    ));
    (addr, memory, settings)
}

/// One control request/response round trip over a fresh connection.
async fn round_trip(addr: std::net::SocketAddr, request: WyomingEvent) -> WyomingEvent {
    let stream = TcpStream::connect(addr).await.unwrap();
    let (r, w) = stream.into_split();
    let mut reader = BufReader::new(r);
    let mut writer = w;
    write_event(&mut writer, &request).await.unwrap();
    read_event(&mut reader).await.unwrap().unwrap()
}

#[tokio::test]
async fn describe_then_set_settings_over_the_socket() {
    let (addr, _mem, settings) = start_server().await;

    let described = round_trip(addr, WyomingEvent::new(types::DESCRIBE_SETTINGS)).await;
    assert_eq!(described.event_type, types::SETTINGS);
    assert_eq!(described.data["llm_backend"], json!("ollama"));
    assert_eq!(described.data["tts_voice"], json!(null));

    // Change the model and pin a Piper voice.
    let set = round_trip(
        addr,
        WyomingEvent::with_data(
            types::SET_SETTINGS,
            json!({ "llm_model": "qwen2.5", "tts_voice": "en_US-amy-medium" }),
        ),
    )
    .await;
    assert_eq!(set.data["ok"], json!(true));
    assert_eq!(set.data["llm_model"], json!("qwen2.5"));
    assert_eq!(set.data["tts_voice"], json!("en_US-amy-medium"));

    // The swap is visible on the server's live settings.
    assert_eq!(settings.view().llm_model.as_deref(), Some("qwen2.5"));
    assert_eq!(
        settings.view().tts_voice.as_deref(),
        Some("en_US-amy-medium")
    );
}

#[tokio::test]
async fn list_models_over_the_socket() {
    let (addr, _mem, _settings) = start_server().await;

    let resp = round_trip(addr, WyomingEvent::new(types::LIST_MODELS)).await;
    assert_eq!(resp.event_type, types::MODELS);
    assert_eq!(resp.data["ok"], json!(true));
    let models = resp.data["models"].as_array().unwrap();
    // The static fallback (no keys in the test) includes both providers.
    assert!(models.iter().any(|m| m["id"] == json!("claude-opus-5")));
    assert!(models.iter().any(|m| m["id"] == json!("gpt-4o-mini")));
}

#[tokio::test]
async fn list_and_delete_memories_over_the_socket() {
    let (addr, mem, _settings) = start_server().await;
    let id = mem
        .add(
            MemoryKind::Fact,
            "the user likes tea",
            MemorySource::Explicit,
        )
        .unwrap();
    mem.add(MemoryKind::Preference, "likes jazz", MemorySource::Inferred)
        .unwrap();

    let listed = round_trip(addr, WyomingEvent::new(types::LIST_MEMORIES)).await;
    assert_eq!(listed.event_type, types::MEMORIES);
    let entries = listed.data["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);

    let deleted = round_trip(
        addr,
        WyomingEvent::with_data(types::DELETE_MEMORY, json!({ "id": id })),
    )
    .await;
    assert_eq!(deleted.event_type, types::MEMORY_RESULT);
    assert_eq!(deleted.data["ok"], json!(true));
    assert_eq!(deleted.data["count"], json!(1));
    assert_eq!(mem.count().unwrap(), 1);

    let cleared = round_trip(addr, WyomingEvent::new(types::CLEAR_MEMORIES)).await;
    assert_eq!(cleared.data["count"], json!(1));
    assert_eq!(mem.count().unwrap(), 0);
}

#[tokio::test]
async fn multiple_control_requests_reuse_one_connection() {
    let (addr, mem, _settings) = start_server().await;
    mem.add(MemoryKind::Fact, "one", MemorySource::Explicit)
        .unwrap();

    // Two requests back-to-back on the same socket (the accept loop handles frames
    // one at a time and keeps the connection open).
    let stream = TcpStream::connect(addr).await.unwrap();
    let (r, w) = stream.into_split();
    let mut reader = BufReader::new(r);
    let mut writer = w;

    write_event(&mut writer, &WyomingEvent::new(types::LIST_MEMORIES))
        .await
        .unwrap();
    let first = read_event(&mut reader).await.unwrap().unwrap();
    assert_eq!(first.data["entries"].as_array().unwrap().len(), 1);

    write_event(&mut writer, &WyomingEvent::new(types::DESCRIBE_SETTINGS))
        .await
        .unwrap();
    let second = read_event(&mut reader).await.unwrap().unwrap();
    assert_eq!(second.event_type, types::SETTINGS);
}
