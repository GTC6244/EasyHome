//! Project-local settings + memory control handler (Plan.MD Phase 6).
//!
//! The on-device settings screen manages the orchestrator's runtime LLM backend +
//! TTS voice and the persistent memory list. Those requests ride the same Wyoming
//! framing as a voice turn (device↔orchestrator hop only — see
//! `wyoming::protocol` `ambient-*` types) so the device reuses one discovered
//! endpoint and one codec. This module maps a control **request** event to a
//! **response** event; the server ([`crate::server`]) routes control frames here
//! and streams the response back.
//!
//! [`respond`] is a pure function of `(request, memory, settings)` so the whole
//! control surface is unit-testable without a socket.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::llm::anthropic_auth::AnthropicAuth;
use crate::llm::catalog::ModelCatalog;
use crate::memory::MemoryStore;
use crate::orchestrator::ServiceConnector;
use crate::settings::{LlmEngine, SettingsUpdate, SharedSettings};
use crate::speaker::SpeakerRegistry;
use crate::wyoming::protocol::{types, WyomingEvent};
use crate::wyoming::tts::{describe_voices, VoiceEntry};
use crate::wyoming::DynConnection;

/// Whether an incoming event is a Phase-6 control request the orchestrator should
/// answer (as opposed to the start of a voice turn).
pub fn is_control_request(event_type: &str) -> bool {
    matches!(
        event_type,
        types::DESCRIBE_SETTINGS
            | types::SET_SETTINGS
            | types::LIST_MEMORIES
            | types::DELETE_MEMORY
            | types::CLEAR_MEMORIES
            | types::LIST_SPEAKERS
            | types::NAME_SPEAKER
            | types::MERGE_SPEAKERS
            | types::DELETE_SPEAKER
            | types::LIST_MODELS
            | types::LIST_VOICES
            | types::GET_DRIVE_TOKEN
    )
}

/// Handle one control request over `device`: build the response and send it. Most
/// requests are answered synchronously by [`respond`]; `ambient-list-models` needs
/// an async catalog fetch and `ambient-list-voices` an async Piper `describe`, so
/// those are handled here.
#[allow(clippy::too_many_arguments)]
pub async fn handle_control(
    device: &mut DynConnection,
    request: &WyomingEvent,
    memory: &MemoryStore,
    settings: &SharedSettings,
    speaker: Option<&SpeakerRegistry>,
    catalog: &ModelCatalog,
    connector: &dyn ServiceConnector,
    voices_dir: Option<&Path>,
) -> Result<()> {
    let response = if request.event_type == types::LIST_MODELS {
        models_response(catalog).await
    } else if request.event_type == types::LIST_VOICES {
        voices_response(connector, voices_dir).await
    } else {
        respond(request, memory, settings, speaker)
    };
    device.send(&response).await
}

/// Build the `ambient-voices` response: Piper's advertised voice catalog, filtered
/// to the voices actually present on disk when a voices dir is configured. A
/// failure to reach Piper is reported in-band (`ok: false`) so it never drops the
/// device connection.
pub async fn voices_response(
    connector: &dyn ServiceConnector,
    voices_dir: Option<&Path>,
) -> WyomingEvent {
    match list_installed_voices(connector, voices_dir).await {
        Ok(voices) => {
            let voices: Vec<Value> = voices
                .into_iter()
                .map(|v| json!({ "name": v.name, "language": v.language, "label": v.label }))
                .collect();
            WyomingEvent::with_data(types::VOICES, json!({ "ok": true, "voices": voices }))
        }
        Err(e) => WyomingEvent::with_data(
            types::VOICES,
            json!({ "ok": false, "message": format!("{e:#}"), "voices": [] }),
        ),
    }
}

/// Fetch Piper's advertised voices and, when `voices_dir` is set, keep only those
/// whose `<name>.onnx` model is present there. Sorted by name, deduplicated.
async fn list_installed_voices(
    connector: &dyn ServiceConnector,
    voices_dir: Option<&Path>,
) -> Result<Vec<VoiceEntry>> {
    let conn = connector.connect_tts().await?;
    let mut catalog = describe_voices(conn).await?;
    if let Some(dir) = voices_dir {
        let installed = installed_voice_names(dir)?;
        catalog.retain(|v| installed.contains(&v.name));
    }
    catalog.sort_by(|a, b| a.name.cmp(&b.name));
    catalog.dedup_by(|a, b| a.name == b.name);
    Ok(catalog)
}

/// The set of voice names present in `dir`: the file stem of every `<name>.onnx`
/// (so `en_US-amy-medium.onnx` → `en_US-amy-medium`; the sibling `.onnx.json` is
/// ignored).
fn installed_voice_names(dir: &Path) -> Result<HashSet<String>> {
    let mut names = HashSet::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("scanning Piper voices dir {}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                names.insert(stem.to_string());
            }
        }
    }
    Ok(names)
}

/// Build the `ambient-models` response from the selectable-model catalog.
pub async fn models_response(catalog: &ModelCatalog) -> WyomingEvent {
    let models: Vec<Value> = catalog
        .models()
        .await
        .into_iter()
        .map(|m| json!({ "provider": m.provider, "id": m.id, "label": m.label }))
        .collect();
    WyomingEvent::with_data(types::MODELS, json!({ "ok": true, "models": models }))
}

/// Build the `ambient-drive-token` response: the Google Drive photo-slideshow
/// bundle the orchestrator owns (client creds + refresh token + folder ids). The
/// device pulls this and mints Drive access tokens on-device. The client secret and
/// refresh token ride the device↔orchestrator LAN hop only (never an off-the-shelf
/// Wyoming server); an unlinked orchestrator still answers `ok: true` with empty
/// fields so the device can degrade to local gradients.
pub fn drive_token_response(settings: &SharedSettings) -> WyomingEvent {
    let d = settings.drive();
    WyomingEvent::with_data(
        types::DRIVE_TOKEN,
        json!({
            "ok": true,
            "linked": d.linked(),
            "configured": d.configured(),
            "client_id": d.client_id.unwrap_or_default(),
            "client_secret": d.client_secret.unwrap_or_default(),
            "refresh_token": d.refresh_token.unwrap_or_default(),
            "folder_ids": d.folder_ids,
            "scope": d.scope.unwrap_or_default(),
        }),
    )
}

/// Build the response event for a control `request`. Never fails: a bad request or
/// a rejected settings change is reported in the response's `ok`/`message` fields
/// rather than raised, so a control error never drops the device connection.
pub fn respond(
    request: &WyomingEvent,
    memory: &MemoryStore,
    settings: &SharedSettings,
    speaker: Option<&SpeakerRegistry>,
) -> WyomingEvent {
    match request.event_type.as_str() {
        types::LIST_SPEAKERS => speakers_response(speaker),
        types::NAME_SPEAKER => name_speaker(request, speaker),
        types::MERGE_SPEAKERS => merge_speakers(request, memory, speaker),
        types::DELETE_SPEAKER => delete_speaker(request, speaker),
        types::GET_DRIVE_TOKEN => drive_token_response(settings),
        types::DESCRIBE_SETTINGS => settings_response(settings, true, "current settings"),
        types::SET_SETTINGS => match settings.apply(&parse_update(&request.data)) {
            Ok(_) => settings_response(settings, true, "settings applied"),
            Err(e) => settings_response(settings, false, &format!("{e:#}")),
        },
        types::LIST_MEMORIES => memories_response(memory),
        types::DELETE_MEMORY => {
            let id = request.data.get("id").and_then(Value::as_i64);
            match id {
                Some(id) => match memory.delete(id) {
                    Ok(removed) => memory_result(removed, usize::from(removed)),
                    Err(e) => memory_error(&format!("{e:#}")),
                },
                None => memory_error("delete-memory request missing integer `id`"),
            }
        }
        types::CLEAR_MEMORIES => match memory.clear() {
            Ok(n) => memory_result(true, n),
            Err(e) => memory_error(&format!("{e:#}")),
        },
        other => memory_error(&format!("unknown control request `{other}`")),
    }
}

/// Parse a `SettingsUpdate` from the `ambient-set-settings` data block. An absent
/// key leaves that setting unchanged; a present `tts_voice: null` clears the voice.
fn parse_update(data: &Value) -> SettingsUpdate {
    let string_field = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let tts_voice = match data.get("tts_voice") {
        None => None,                    // unchanged
        Some(Value::Null) => Some(None), // clear
        Some(v) => Some(v.as_str().filter(|s| !s.is_empty()).map(str::to_string)),
    };
    let engine =
        data.get("engine")
            .and_then(Value::as_str)
            .map(|e| match e.to_lowercase().as_str() {
                "rig" | "rig-core" | "rigcore" => LlmEngine::Rig,
                _ => LlmEngine::Native,
            });
    let web_search = data.get("web_search").and_then(Value::as_bool);
    let search_api_key = match data.get("search_api_key") {
        None => None,                    // unchanged
        Some(Value::Null) => Some(None), // clear
        Some(v) => Some(v.as_str().filter(|s| !s.is_empty()).map(str::to_string)),
    };
    let anthropic_auth = data
        .get("anthropic_auth")
        .and_then(Value::as_str)
        .map(AnthropicAuth::from_label);
    SettingsUpdate {
        llm_backend: string_field("llm_backend"),
        llm_model: string_field("llm_model"),
        // Provider API keys / tokens are intentionally NOT accepted from the device
        // control path — they are entered only on the orchestrator's loopback config
        // page, so cloud secrets never travel from or live on the shared Echo Show screen.
        anthropic_api_key: None,
        openai_api_key: None,
        anthropic_oauth_token: None,
        anthropic_auth,
        tts_voice,
        engine,
        web_search,
        search_provider: string_field("search_provider"),
        search_api_key,
        end_silence_ms: data.get("end_silence_ms").and_then(Value::as_u64),
        voice_rms_threshold: data.get("voice_rms_threshold").and_then(Value::as_f64),
    }
}

fn settings_response(settings: &SharedSettings, ok: bool, message: &str) -> WyomingEvent {
    let v = settings.view();
    WyomingEvent::with_data(
        types::SETTINGS,
        json!({
            "ok": ok,
            "message": message,
            "llm_backend": v.llm_backend,
            "llm_model": v.llm_model,
            // Read-only on the device: whether a provider key is configured, never
            // the key itself. Provider keys are entered only on the loopback config
            // page (see `webconfig`), not from the shared device screen.
            "anthropic_key_set": v.anthropic_key_set,
            "openai_key_set": v.openai_key_set,
            "anthropic_auth": v.anthropic_auth.as_str(),
            "tts_voice": v.tts_voice,
            "engine": engine_label(v.engine),
            "web_search": v.web_search,
            "search_provider": v.search_provider,
            "search_key_set": v.search_key_set,
            "end_silence_ms": v.end_silence_ms,
            "voice_rms_threshold": v.voice_rms_threshold,
        }),
    )
}

/// Canonical string label for an engine (for JSON responses).
fn engine_label(engine: LlmEngine) -> &'static str {
    match engine {
        LlmEngine::Native => "native",
        LlmEngine::Rig => "rig",
    }
}

fn memories_response(memory: &MemoryStore) -> WyomingEvent {
    match memory.list() {
        Ok(entries) => {
            let entries: Vec<Value> = entries
                .into_iter()
                .map(|m| {
                    json!({
                        "id": m.id,
                        "kind": m.kind.as_str(),
                        "content": m.content,
                        "source": m.source.as_str(),
                        "created_at": m.created_at,
                    })
                })
                .collect();
            WyomingEvent::with_data(types::MEMORIES, json!({ "ok": true, "entries": entries }))
        }
        Err(e) => WyomingEvent::with_data(
            types::MEMORIES,
            json!({ "ok": false, "message": format!("{e:#}"), "entries": [] }),
        ),
    }
}

fn memory_result(ok: bool, count: usize) -> WyomingEvent {
    WyomingEvent::with_data(
        types::MEMORY_RESULT,
        json!({ "ok": ok, "count": count as i64 }),
    )
}

fn memory_error(message: &str) -> WyomingEvent {
    WyomingEvent::with_data(
        types::MEMORY_RESULT,
        json!({ "ok": false, "count": 0, "message": message }),
    )
}

// ---- speaker identification control (speaker_id_plan.md Phase C) ----

/// The `ambient-speakers` response listing identified speakers.
fn speakers_response(speaker: Option<&SpeakerRegistry>) -> WyomingEvent {
    let Some(reg) = speaker else {
        return WyomingEvent::with_data(
            types::SPEAKERS,
            json!({ "ok": false, "message": SPEAKER_DISABLED, "speakers": [] }),
        );
    };
    match reg.list() {
        Ok(list) => {
            let speakers: Vec<Value> = list
                .into_iter()
                .map(|p| {
                    json!({
                        "id": p.id,
                        "name": p.name,
                        "labeled": p.labeled,
                        "samples": p.samples as i64,
                        "created_at": p.created_at,
                    })
                })
                .collect();
            WyomingEvent::with_data(types::SPEAKERS, json!({ "ok": true, "speakers": speakers }))
        }
        Err(e) => WyomingEvent::with_data(
            types::SPEAKERS,
            json!({ "ok": false, "message": format!("{e:#}"), "speakers": [] }),
        ),
    }
}

fn name_speaker(request: &WyomingEvent, speaker: Option<&SpeakerRegistry>) -> WyomingEvent {
    let Some(reg) = speaker else {
        return speaker_result(false, SPEAKER_DISABLED);
    };
    let id = request.data.get("id").and_then(Value::as_str);
    let name = request.data.get("name").and_then(Value::as_str);
    match (id, name) {
        (Some(id), Some(name)) if !name.trim().is_empty() => match reg.rename(id, name) {
            Ok(true) => speaker_result(true, &format!("named {id} \"{}\"", name.trim())),
            Ok(false) => speaker_result(false, "no such speaker"),
            Err(e) => speaker_result(false, &format!("{e:#}")),
        },
        _ => speaker_result(false, "name-speaker requires `id` and a non-empty `name`"),
    }
}

fn merge_speakers(
    request: &WyomingEvent,
    memory: &MemoryStore,
    speaker: Option<&SpeakerRegistry>,
) -> WyomingEvent {
    let Some(reg) = speaker else {
        return speaker_result(false, SPEAKER_DISABLED);
    };
    let keep = request.data.get("keep").and_then(Value::as_str);
    let drop = request.data.get("drop").and_then(Value::as_str);
    match (keep, drop) {
        (Some(keep), Some(drop)) => match reg.merge(keep, drop) {
            Ok(true) => {
                // Move the dropped speaker's memories onto the kept profile.
                let moved = memory.reassign_speaker(drop, keep).unwrap_or(0);
                speaker_result(
                    true,
                    &format!("merged {drop} into {keep} ({moved} memories moved)"),
                )
            }
            Ok(false) => speaker_result(false, "merge failed (unknown id, or same id)"),
            Err(e) => speaker_result(false, &format!("{e:#}")),
        },
        _ => speaker_result(false, "merge-speakers requires `keep` and `drop`"),
    }
}

fn delete_speaker(request: &WyomingEvent, speaker: Option<&SpeakerRegistry>) -> WyomingEvent {
    let Some(reg) = speaker else {
        return speaker_result(false, SPEAKER_DISABLED);
    };
    match request.data.get("id").and_then(Value::as_str) {
        Some(id) => match reg.delete(id) {
            Ok(true) => speaker_result(true, "deleted"),
            Ok(false) => speaker_result(false, "no such speaker"),
            Err(e) => speaker_result(false, &format!("{e:#}")),
        },
        None => speaker_result(false, "delete-speaker requires `id`"),
    }
}

fn speaker_result(ok: bool, message: &str) -> WyomingEvent {
    WyomingEvent::with_data(
        types::SPEAKER_RESULT,
        json!({ "ok": ok, "message": message }),
    )
}

const SPEAKER_DISABLED: &str = "speaker identification is disabled on the orchestrator";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryKind, MemorySource, MemoryStore};
    use crate::settings::{RuntimeSettings, SharedSettings};
    use std::sync::Arc;

    fn settings() -> Arc<SharedSettings> {
        SharedSettings::fixed(Arc::new(crate::llm::mock::MockLlm::default()), "mock", None)
    }

    fn req(event_type: &str, data: Value) -> WyomingEvent {
        WyomingEvent::with_data(event_type, data)
    }

    #[test]
    fn describe_reports_current_settings() {
        let s = SharedSettings::new(
            crate::settings::LlmFactory {
                ollama_url: "http://x".into(),
                anthropic_base_url: "http://y".into(),
                anthropic_api_key: None,
                anthropic_max_tokens: 10,
                openai_base_url: "http://z".into(),
                openai_api_key: None,
                openai_max_tokens: 10,
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
            },
            RuntimeSettings {
                llm: Arc::new(crate::llm::mock::MockLlm::default()),
                engine: crate::settings::LlmEngine::Native,
                web_search: false,
                search_provider: "duckduckgo".into(),
                search_api_key: None,
                llm_backend: "ollama".into(),
                llm_model: Some("llama3.2".into()),
                anthropic_api_key: None,
                openai_api_key: None,
                anthropic_oauth_token: None,
                mapbox_token: None,
                anthropic_auth: crate::llm::anthropic_auth::AnthropicAuth::ApiKey,
                tts_voice: Some("amy".into()),
                end_silence_ms: crate::settings::DEFAULT_END_SILENCE_MS,
                voice_rms_threshold: crate::settings::DEFAULT_VOICE_RMS_THRESHOLD,
                drive: crate::settings::DriveConfig::default(),
                household: crate::settings::Household::default(),
                spotify: crate::settings::SpotifyConfig::default(),
                cadora: crate::settings::CadoraConfig::default(),
                system1: crate::settings::System1Runtime::default(),
            },
        );
        let mem = MemoryStore::open_in_memory().unwrap();
        let resp = respond(&WyomingEvent::new(types::DESCRIBE_SETTINGS), &mem, &s, None);
        assert_eq!(resp.event_type, types::SETTINGS);
        assert_eq!(resp.data["ok"], json!(true));
        assert_eq!(resp.data["llm_backend"], json!("ollama"));
        assert_eq!(resp.data["llm_model"], json!("llama3.2"));
        assert_eq!(resp.data["tts_voice"], json!("amy"));
    }

    #[test]
    fn get_drive_token_is_a_control_request_and_reports_the_bundle() {
        use crate::settings::DriveUpdate;
        assert!(is_control_request(types::GET_DRIVE_TOKEN));
        let s = settings();
        // Unlinked orchestrator: ok with empty fields so the device degrades cleanly.
        let mem = MemoryStore::open_in_memory().unwrap();
        let resp = respond(&WyomingEvent::new(types::GET_DRIVE_TOKEN), &mem, &s, None);
        assert_eq!(resp.event_type, types::DRIVE_TOKEN);
        assert_eq!(resp.data["ok"], json!(true));
        assert_eq!(resp.data["linked"], json!(false));
        assert_eq!(resp.data["configured"], json!(false));
        assert_eq!(resp.data["refresh_token"], json!(""));

        // Fully linked: the bundle carries the creds + token for the device.
        s.apply_drive(&DriveUpdate {
            client_id: Some(Some("cid.apps".into())),
            client_secret: Some(Some("gocspx-secret".into())),
            refresh_token: Some(Some("1//refresh".into())),
            folder_ids: Some(vec!["1AbC".into()]),
            scope: Some(Some("scope-x".into())),
        });
        let resp = respond(&WyomingEvent::new(types::GET_DRIVE_TOKEN), &mem, &s, None);
        assert_eq!(resp.data["linked"], json!(true));
        assert_eq!(resp.data["configured"], json!(true));
        assert_eq!(resp.data["client_id"], json!("cid.apps"));
        assert_eq!(resp.data["client_secret"], json!("gocspx-secret"));
        assert_eq!(resp.data["refresh_token"], json!("1//refresh"));
        assert_eq!(resp.data["folder_ids"][0], json!("1AbC"));
        assert_eq!(resp.data["scope"], json!("scope-x"));
    }

    #[test]
    fn set_settings_applies_tts_voice() {
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let resp = respond(
            &req(
                types::SET_SETTINGS,
                json!({ "tts_voice": "en_US-amy-medium" }),
            ),
            &mem,
            &s,
            None,
        );
        assert_eq!(resp.data["ok"], json!(true));
        assert_eq!(resp.data["tts_voice"], json!("en_US-amy-medium"));
        assert_eq!(s.view().tts_voice.as_deref(), Some("en_US-amy-medium"));
    }

    #[test]
    fn set_settings_reports_rejected_backend_without_dropping() {
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        // `fixed` has no anthropic key, so selecting it must be rejected in-band.
        let resp = respond(
            &req(types::SET_SETTINGS, json!({ "llm_backend": "anthropic" })),
            &mem,
            &s,
            None,
        );
        assert_eq!(resp.event_type, types::SETTINGS);
        assert_eq!(resp.data["ok"], json!(false));
        assert!(resp.data["message"]
            .as_str()
            .unwrap()
            .contains("ANTHROPIC_API_KEY"));
        assert_eq!(
            s.view().llm_backend,
            "mock",
            "settings unchanged after reject"
        );
    }

    #[test]
    fn list_delete_and_clear_memories() {
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let id = mem
            .add(
                MemoryKind::Fact,
                "the user likes tea",
                MemorySource::Explicit,
            )
            .unwrap();
        mem.add(MemoryKind::Preference, "likes jazz", MemorySource::Inferred)
            .unwrap();

        let listed = respond(&WyomingEvent::new(types::LIST_MEMORIES), &mem, &s, None);
        assert_eq!(listed.event_type, types::MEMORIES);
        assert_eq!(listed.data["entries"].as_array().unwrap().len(), 2);
        assert_eq!(
            listed.data["entries"][1]["content"],
            json!("the user likes tea")
        );

        let deleted = respond(
            &req(types::DELETE_MEMORY, json!({ "id": id })),
            &mem,
            &s,
            None,
        );
        assert_eq!(deleted.event_type, types::MEMORY_RESULT);
        assert_eq!(deleted.data["ok"], json!(true));
        assert_eq!(deleted.data["count"], json!(1));

        let cleared = respond(&WyomingEvent::new(types::CLEAR_MEMORIES), &mem, &s, None);
        assert_eq!(cleared.data["ok"], json!(true));
        assert_eq!(cleared.data["count"], json!(1));
        assert_eq!(mem.count().unwrap(), 0);
    }

    #[test]
    fn delete_without_id_is_an_in_band_error() {
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let resp = respond(&WyomingEvent::new(types::DELETE_MEMORY), &mem, &s, None);
        assert_eq!(resp.event_type, types::MEMORY_RESULT);
        assert_eq!(resp.data["ok"], json!(false));
    }

    #[test]
    fn control_requests_are_recognized() {
        assert!(is_control_request(types::DESCRIBE_SETTINGS));
        assert!(is_control_request(types::SET_SETTINGS));
        assert!(is_control_request(types::LIST_MEMORIES));
        assert!(is_control_request(types::LIST_SPEAKERS));
        assert!(is_control_request(types::NAME_SPEAKER));
        assert!(is_control_request(types::MERGE_SPEAKERS));
        assert!(is_control_request(types::DELETE_SPEAKER));
        assert!(is_control_request(types::LIST_MODELS));
        assert!(is_control_request(types::LIST_VOICES));
        assert!(!is_control_request(types::AUDIO_START));
        assert!(!is_control_request(types::TRANSCRIPT));
    }

    #[test]
    fn speaker_requests_report_disabled_without_a_registry() {
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let resp = respond(&WyomingEvent::new(types::LIST_SPEAKERS), &mem, &s, None);
        assert_eq!(resp.event_type, types::SPEAKERS);
        assert_eq!(resp.data["ok"], json!(false));
        assert_eq!(resp.data["speakers"].as_array().unwrap().len(), 0);

        let resp = respond(
            &req(types::NAME_SPEAKER, json!({ "id": "spk-1", "name": "Sam" })),
            &mem,
            &s,
            None,
        );
        assert_eq!(resp.event_type, types::SPEAKER_RESULT);
        assert_eq!(resp.data["ok"], json!(false));
    }

    #[test]
    fn list_name_and_delete_speakers() {
        use crate::speaker::{MockSpeakerEmbedder, SpeakerEmbedder, SpeakerRegistry};
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let reg = SpeakerRegistry::open_in_memory().unwrap();
        let e = MockSpeakerEmbedder::default();
        let id = reg
            .create_cluster(&e.embed(&vec![1000i16; 20_000]).unwrap())
            .unwrap();

        // List shows the anonymous cluster.
        let listed = respond(
            &WyomingEvent::new(types::LIST_SPEAKERS),
            &mem,
            &s,
            Some(&reg),
        );
        assert_eq!(listed.event_type, types::SPEAKERS);
        assert_eq!(listed.data["ok"], json!(true));
        let arr = listed.data["speakers"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], json!(id));
        assert_eq!(arr[0]["labeled"], json!(false));

        // Name it.
        let named = respond(
            &req(types::NAME_SPEAKER, json!({ "id": id, "name": "Sam" })),
            &mem,
            &s,
            Some(&reg),
        );
        assert_eq!(named.event_type, types::SPEAKER_RESULT);
        assert_eq!(named.data["ok"], json!(true));
        assert_eq!(reg.get(&id).unwrap().unwrap().name.as_deref(), Some("Sam"));

        // Naming an unknown speaker fails in-band.
        let bad = respond(
            &req(types::NAME_SPEAKER, json!({ "id": "nope", "name": "X" })),
            &mem,
            &s,
            Some(&reg),
        );
        assert_eq!(bad.data["ok"], json!(false));

        // Delete it.
        let del = respond(
            &req(types::DELETE_SPEAKER, json!({ "id": id })),
            &mem,
            &s,
            Some(&reg),
        );
        assert_eq!(del.data["ok"], json!(true));
        assert_eq!(reg.count().unwrap(), 0);
    }

    #[test]
    fn merge_speakers_moves_memories() {
        use crate::memory::{MemoryKind, MemorySource};
        use crate::speaker::{MockSpeakerEmbedder, SpeakerEmbedder, SpeakerRegistry};
        let s = settings();
        let mem = MemoryStore::open_in_memory().unwrap();
        let reg = SpeakerRegistry::open_in_memory().unwrap();
        let e = MockSpeakerEmbedder::default();
        let keep = reg
            .create_cluster(&e.embed(&vec![1000i16; 20_000]).unwrap())
            .unwrap();
        let drop = reg
            .create_cluster(&e.embed(&vec![500i16; 20_000]).unwrap())
            .unwrap();
        mem.add_scoped(
            MemoryKind::Fact,
            "likes tea",
            MemorySource::Explicit,
            Some(&drop),
        )
        .unwrap();

        let resp = respond(
            &req(types::MERGE_SPEAKERS, json!({ "keep": keep, "drop": drop })),
            &mem,
            &s,
            Some(&reg),
        );
        assert_eq!(resp.data["ok"], json!(true));
        assert_eq!(reg.count().unwrap(), 1, "dropped profile removed");
        // The dropped speaker's memory now belongs to the kept speaker.
        let hits = mem.search_scoped("tea", Some(&keep), 10).unwrap();
        assert!(hits.iter().any(|m| m.content == "likes tea"));
    }

    #[tokio::test]
    async fn list_models_returns_the_static_fallback_without_keys() {
        // No provider keys → the catalog serves its curated fallback list.
        let catalog = ModelCatalog::new("http://unused", None, "http://unused", None);
        let resp = models_response(&catalog).await;
        assert_eq!(resp.event_type, types::MODELS);
        assert_eq!(resp.data["ok"], json!(true));
        let models = resp.data["models"].as_array().unwrap();
        assert!(models
            .iter()
            .any(|m| m["provider"] == json!("anthropic") && m["id"] == json!("claude-opus-5")));
        assert!(models
            .iter()
            .any(|m| m["provider"] == json!("openai") && m["id"] == json!("gpt-4o-mini")));
    }

    // ---- ambient-list-voices ----

    /// A connector whose `connect_tts` answers a `describe` with a fixed voice
    /// catalog (amy, lessac, and a not-installed German voice). `connect_stt` is
    /// never used by the voice path.
    struct VoiceConnector;

    #[async_trait::async_trait]
    impl crate::orchestrator::ServiceConnector for VoiceConnector {
        async fn connect_stt(&self) -> Result<crate::wyoming::DynConnection> {
            anyhow::bail!("stt not used in voice tests")
        }

        async fn connect_tts(&self) -> Result<crate::wyoming::DynConnection> {
            use crate::wyoming::protocol::write_event;
            let (client, server) = tokio::io::duplex(64 * 1024);
            tokio::spawn(async move {
                let (r, w) = tokio::io::split(server);
                let mut reader = tokio::io::BufReader::new(r);
                let mut writer = w;
                // Wait for the describe, then answer with an info catalog.
                let _ = crate::wyoming::protocol::read_event(&mut reader).await;
                let info = WyomingEvent::with_data(
                    types::INFO,
                    json!({ "tts": [{ "voices": [
                        { "name": "en_US-amy-medium", "languages": ["en_US"], "description": "amy (medium)" },
                        { "name": "en_US-lessac-medium", "languages": ["en_US"], "description": "lessac (medium)" },
                        { "name": "de_DE-thorsten-low", "languages": ["de_DE"], "description": "thorsten (low)" },
                    ]}]}),
                );
                let _ = write_event(&mut writer, &info).await;
            });
            let (r, w) = tokio::io::split(client);
            Ok(crate::wyoming::DynConnection::from_io(r, w))
        }
    }

    /// Create a unique temp dir seeded with `<name>.onnx` (+ a sibling `.onnx.json`
    /// that must be ignored) for each installed voice. Returned dir is caller-owned.
    fn temp_voices_dir(installed: &[&str]) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ambient-voices-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in installed {
            std::fs::write(dir.join(format!("{name}.onnx")), b"").unwrap();
            std::fs::write(dir.join(format!("{name}.onnx.json")), b"{}").unwrap();
        }
        dir
    }

    #[tokio::test]
    async fn voices_response_filters_to_installed_when_dir_is_set() {
        let dir = temp_voices_dir(&["en_US-amy-medium", "en_US-lessac-medium"]);
        let resp = voices_response(&VoiceConnector, Some(&dir)).await;
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(resp.event_type, types::VOICES);
        assert_eq!(resp.data["ok"], json!(true));
        let voices = resp.data["voices"].as_array().unwrap();
        let names: Vec<&str> = voices.iter().map(|v| v["name"].as_str().unwrap()).collect();
        // Only the two installed voices survive; the German voice is filtered out.
        assert_eq!(names, vec!["en_US-amy-medium", "en_US-lessac-medium"]);
        assert_eq!(voices[0]["language"], json!("en_US"));
        assert_eq!(voices[0]["label"], json!("amy (medium)"));
    }

    #[tokio::test]
    async fn voices_response_returns_full_catalog_without_a_dir() {
        // No voices dir configured (remote Piper): the full advertised list is returned.
        let resp = voices_response(&VoiceConnector, None).await;
        assert_eq!(resp.data["ok"], json!(true));
        let voices = resp.data["voices"].as_array().unwrap();
        assert_eq!(voices.len(), 3);
    }

    #[tokio::test]
    async fn voices_response_reports_scan_error_in_band() {
        // A missing voices dir surfaces as ok:false, never a dropped connection.
        let missing = std::env::temp_dir().join("ambient-voices-does-not-exist-xyz");
        let resp = voices_response(&VoiceConnector, Some(&missing)).await;
        assert_eq!(resp.event_type, types::VOICES);
        assert_eq!(resp.data["ok"], json!(false));
        assert_eq!(resp.data["voices"].as_array().unwrap().len(), 0);
    }
}
