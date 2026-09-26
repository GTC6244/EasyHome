// The embedded HelixDB engine has deeply nested generic types; computing the
// layout of the async runtime that awaits them needs a higher recursion limit than
// the default 128 (the `db` crate sets the same).
#![recursion_limit = "512"]
//! Ambient Smart Display — Mac Mini assistant orchestrator (Plan.MD Phase 4).
//!
//! Runs the "brain": a Wyoming server the Echo Show discovers over mDNS, wiring
//! downstream Whisper (STT) → a pluggable LLM + persistent SQLite memory → Piper
//! (TTS), and streaming the synthesized reply back to the device.
//!
//! Configuration is loaded from a per-instance JSON file (`anamanti.json` in the
//! working directory by default, overridable with `--config <path>`; see
//! `config.rs`). Only provider API keys/tokens remain environment variables. With
//! the defaults it advertises `_wyoming._tcp` on port 10700 and talks to a local
//! Whisper (10300), Piper (10200), and Ollama (11434).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::net::TcpListener;

use anamanti_core::config::{Config, MemoryBackendChoice, SttEngineKind};
use anamanti_core::discovery::MdnsAdvertiser;
use anamanti_core::memory::{ChatLog, GraphView, MemoryStore, PromptLog};
use anamanti_core::notify::NotificationService;
use anamanti_core::orchestrator::{self, Pipeline, TcpConnector};
use anamanti_core::server;
use anamanti_core::weather::WeatherService;
use anamanti_core::webconfig::{self, DebugSources};

/// Worker-thread stack size. The embedded HelixDB engine builds deep async state
/// machines whose stack usage exceeds tokio's 2 MiB default, especially in debug
/// builds; 16 MiB gives comfortable headroom.
const WORKER_STACK_SIZE: usize = 16 * 1024 * 1024;

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(WORKER_STACK_SIZE)
        .build()
        .context("building tokio runtime")?;
    // Run on a spawned task so all of `run` (including the embedded-HelixDB init)
    // executes on a large-stack worker thread rather than the main thread.
    runtime.block_on(async {
        tokio::spawn(run())
            .await
            .context("orchestrator task panicked")?
    })
}

async fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = parse_cli().context("parsing command-line arguments")?;
    let mut config = Config::load(cli.config.as_deref()).context("loading configuration")?;
    log::info!(
        "starting orchestrator: name={:?} instance_id={} bind={} stt={} tts={} llm={} db={}",
        config.service_name,
        config.instance_id,
        config.bind_addr,
        config.stt_addr,
        config.tts_addr,
        config.llm_label(),
        config.db_path.display(),
    );

    // Startup model check: when using Ollama, verify the configured model is
    // actually installed. A missing model otherwise fails silently as a per-turn
    // 404 (transcript shows, no reply). Surface it loudly at boot, and fall back
    // to an installed model so the assistant still responds.
    ensure_ollama_model(&mut config).await;

    let memory = Arc::new(MemoryStore::open(&config.db_path).context("opening memory store")?);
    log::info!("memory store holds {} entries", memory.count()?);

    // Runtime-swappable settings (Phase 6): the initial backend/voice come from
    // config; the on-device settings screen can change them between turns.
    let settings = config
        .shared_settings()
        .context("initializing LLM backend")?;

    // Selectable-model catalog for the settings dropdown (device + config page).
    // Live-fetched from the provider APIs (filtered to the last 12 months) with a
    // curated static fallback; shared by both front doors.
    let catalog = Arc::new(config.model_catalog());

    // Chat log (always on): every completed turn is recorded here, both as the
    // durable source of truth and as the ingestion queue for GraphRAG memory.
    let chatlog = Arc::new(
        ChatLog::open(&config.chatlog_path)
            .with_context(|| format!("opening chat log at {}", config.chatlog_path.display()))?,
    );
    log::info!("chat log at {}", config.chatlog_path.display());

    // Prompt log (always on): the exact assembled LLM prompt per turn, for the
    // orchestrator's debug GUI (`/prompts`). Separate from the chat log, which the
    // GraphRAG ingester consumes.
    let promptlog =
        Arc::new(PromptLog::open(&config.promptlog_path).with_context(|| {
            format!("opening prompt log at {}", config.promptlog_path.display())
        })?);
    log::info!("prompt log at {}", config.promptlog_path.display());

    {
        let v = settings.view();
        log::info!(
            "llm: engine={:?} backend={} web_search={} (change at runtime via the config page)",
            v.engine,
            v.llm_backend,
            v.web_search,
        );
        // Web search only runs on the rig engine; warn if it's on but the engine
        // isn't rig, so the tool would silently never fire.
        if v.web_search && v.engine != anamanti_core::settings::LlmEngine::Rig {
            log::warn!(
                "web_search is on but engine is not rig; the web-search tool only runs on \
                 the rig engine. Set llm.engine=\"rig\" in the config file (or switch it on \
                 the config page)."
            );
        }
    }

    // System-1 fast-decision engine (plans/system1-fast-decisions.md). Default `none`
    // (disabled) reproduces today's behavior; a bad/unimplemented backend fails loudly
    // at boot rather than silently.
    // System-1 fast-decision engine was built inside `shared_settings` (config seed +
    // persisted overlay) and lives in the runtime-swappable settings, so it can be
    // changed live from the config page. Log the selected backend.
    log::info!(
        "system1 decision engine: {}",
        settings.system1_view().backend
    );

    // Forecast provider (keyless Open-Meteo), shared by the System-1 weather fast path
    // and the ambient push. `None` when weather is disabled.
    let weather_provider = anamanti_core::weather::from_config(config.weather.enabled);

    let mut pipeline = Pipeline::with_settings(
        settings,
        memory,
        config.system_prompt.clone(),
        config.turn_timeout,
    )
    .with_chatlog(chatlog.clone())
    .with_promptlog(promptlog.clone())
    .with_follow_up(config.follow_up.clone())
    .with_audio_dump(config.audio_dump_dir.clone())
    .with_weather(weather_provider.clone());
    // Home location + household roster are grounded from the runtime settings
    // snapshot each turn (seeded from the config file's home_location at boot, then
    // editable from the config dashboard's Household tab), not fixed onto the pipeline.

    // Read-only handle onto the GraphRAG store for the debug GUI (`/helix`); stays
    // `None` on the SQLite backend or when the graph backend fails to initialize.
    let mut graph_view: Option<Arc<dyn GraphView>> = None;

    // Memory retrieval backend: SQLite FTS (default) or embedded HelixDB GraphRAG.
    if config.memory_backend == MemoryBackendChoice::Helix {
        match build_graphrag_recall(&config, &chatlog).await {
            Ok((recall, graph)) => {
                log::info!("memory backend: HelixDB GraphRAG (embedded, in-process)");
                pipeline = pipeline.with_recall(recall);
                graph_view = Some(graph);
            }
            Err(e) => {
                log::error!("GraphRAG init failed ({e:#}); falling back to SQLite FTS recall");
            }
        }
    } else {
        log::info!("memory backend: SQLite FTS");
    }

    // Per-person speaker identification (opt-in; speaker_id_plan.md). Failure or
    // absence degrades to the shared-household behavior.
    match config.build_speaker_service() {
        Ok(Some(speaker)) => {
            log::info!("speaker identification: enabled (per-person memory + context)");
            pipeline = pipeline.with_speaker(speaker);
        }
        Ok(None) => log::info!("speaker identification: disabled (shared household)"),
        Err(e) => {
            log::error!("speaker ID init failed ({e:#}); continuing with shared household");
        }
    }

    // STT engine selection (plans/python-to-rust-whisper.md). Default: the downstream
    // Wyoming Whisper server dialed via the connector below. `whisper-rs` loads
    // whisper.cpp in-process and needs the `stt-whisper-local` build feature.
    match config.stt.engine {
        SttEngineKind::Wyoming => {
            log::info!(
                "STT engine: wyoming (downstream Whisper at {})",
                config.stt_addr
            );
        }
        SttEngineKind::WhisperLocal => {
            #[cfg(feature = "stt-whisper-local")]
            {
                let model = config.stt.resolved_model_path();
                let model_str = model.to_string_lossy().into_owned();
                let engine = anamanti_core::stt::WhisperEngine::open(
                    &model_str,
                    config.stt.language.clone(),
                    config.stt.num_threads as i32,
                )
                .with_context(|| format!("loading in-process Whisper model {model_str}"))?;
                log::info!(
                    "STT engine: whisper-rs (in-process; model {model_str}, {} threads)",
                    config.stt.num_threads
                );
                pipeline = pipeline
                    .with_stt_engine(Arc::new(anamanti_core::stt::WhisperSttEngine::new(engine)));
            }
            #[cfg(not(feature = "stt-whisper-local"))]
            {
                anyhow::bail!(
                    "config selects stt.engine = whisper-rs, but this binary was built \
                     without the `stt-whisper-local` feature; rebuild with \
                     `--features stt-whisper-local`"
                );
            }
        }
    }

    let connector: Arc<dyn orchestrator::ServiceConnector> = Arc::new(TcpConnector {
        stt_addr: config.stt_addr,
        tts_addr: config.tts_addr,
    });

    // Optional local HTTP config page (no auth; loopback by default). Serves the
    // same runtime-swappable settings the device controls over Wyoming, so you can
    // change the LLM backend/model/voice live from a browser. Best-effort: a bind
    // failure disables the page but never stops the orchestrator.
    // The Music tab's control hub (process supervisor + snapserver/mpv control),
    // shared into the config page. `None` when music is disabled in the config file.
    let music_hub = config.build_hub();

    // Proactive-notification registry (Approach A): shared by the device-facing
    // server (which registers each device's persistent notify channel) and the
    // config page (whose "Notify" tab can push a test notification).
    let notify = Arc::new(NotificationService::new());

    // Ambient weather push (keyless Open-Meteo): the registry of persistent weather
    // channels the device dials, plus a periodic task that fetches current conditions
    // for the household location and fans them out so the icon + temperature beside the
    // idle clock stay fresh. Dormant (no task) when weather is disabled in the config.
    let weather_svc = Arc::new(WeatherService::new());
    if let Some(provider) = weather_provider.clone() {
        let imperial =
            anamanti_core::directions::units_are_imperial(config.weather_units.as_deref());
        anamanti_core::weather::service::spawn_periodic(
            weather_svc.clone(),
            provider,
            pipeline.settings().home_location(),
            imperial,
            config.weather.refresh_interval(),
        );
        log::info!(
            "weather push: every {}s (imperial={imperial})",
            config.weather.refresh_interval().as_secs()
        );
    }

    if let Some(config_addr) = config.config_addr {
        let settings = pipeline.settings().clone();
        let catalog = catalog.clone();
        let connector = connector.clone();
        let voices_dir = config.tts_voices_dir.clone();
        let music = music_hub.clone();
        let notify = notify.clone();
        let debug = DebugSources {
            memory: pipeline.memory().clone(),
            chatlog_path: config.chatlog_path.clone(),
            promptlog_path: config.promptlog_path.clone(),
            db_path: config.db_path.clone(),
            helix_path: config.helix_path.clone(),
            memory_backend: match config.memory_backend {
                MemoryBackendChoice::Helix => "helix",
                MemoryBackendChoice::Sqlite => "sqlite",
            }
            .to_string(),
            graph: graph_view.clone(),
        };
        match TcpListener::bind(config_addr).await {
            Ok(listener) => {
                let local = listener.local_addr().unwrap_or(config_addr);
                log::info!(
                    "config + debug pages on http://{local}/ \
                     (chat log, prompts, SQLite, HelixDB — no auth, keep it on a trusted network)"
                );
                tokio::spawn(async move {
                    if let Err(e) = webconfig::serve(
                        listener, settings, catalog, connector, voices_dir, debug, music, notify,
                    )
                    .await
                    {
                        log::error!("config page stopped: {e:#}");
                    }
                });
            }
            Err(e) => log::warn!("config page disabled: could not bind {config_addr}: {e:#}"),
        }
    }

    let listener = TcpListener::bind(config.bind_addr)
        .await
        .with_context(|| format!("binding device-facing server to {}", config.bind_addr))?;
    let local = listener.local_addr()?;

    // Advertise over mDNS so the Echo Show discovers us without a hardcoded IP.
    // Held for the process lifetime; unregisters on drop.
    let _mdns = MdnsAdvertiser::advertise(&config.service_name, &config.instance_id, local.port())
        .context("advertising Wyoming service over mDNS")?;

    // House-wide music routing (Snapcast) control plane. Dormant unless music is
    // enabled in the config file; when on, the ducker lowers the music group's volume
    // while the assistant speaks. Best-effort — a missing snapserver never breaks a turn.
    let ducker = config.build_ducker();
    if config.music.enabled {
        log::info!(
            "music routing on: snapserver={} duck_on_speech={} duck_to={}% group={:?}",
            config.music.snapserver_addr,
            config.music.duck_on_speech,
            config.music.duck_percent,
            config.music.group,
        );
        if let Some(p) = &config.music.mpv_ipc {
            log::info!("music web-URL player mpv IPC at {}", p.display());
        }
    }

    // Auto-start the managed music processes (snapserver/librespot/mpv) as part of
    // the orchestrator lifecycle, when enabled. The Music tab's buttons still work.
    if let Some(hub) = &music_hub {
        if config.music.autostart {
            log::info!("music: auto-starting managed processes (snapserver, librespot, mpv)");
            hub.start_all().await;
        }
    }

    log::info!("orchestrator ready on {local}; waiting for the device");

    let outcome = tokio::select! {
        res = server::serve(listener, pipeline, connector, catalog, config.tts_voices_dir.clone(), ducker, notify, weather_svc) => {
            res.context("device-facing server stopped")
        }
        _ = tokio::signal::ctrl_c() => {
            log::info!("shutdown signal received; stopping");
            Ok(())
        }
    };

    // On shutdown, stop the processes this orchestrator started (externally-started
    // ones are untracked and left alone).
    if let Some(hub) = &music_hub {
        log::info!("music: stopping managed processes");
        hub.stop_all().await;
    }

    outcome?;
    Ok(())
}

/// Parsed command-line arguments. The only flag is `--config <path>` (or
/// `--config=<path>`), which points at the per-instance JSON config file and
/// overrides the convention `anamanti.json`. Reading argv is not an env-var read, so
/// this is compatible with the JSON-only configuration model.
struct CliArgs {
    config: Option<PathBuf>,
}

/// Hand-rolled argv parser (no `clap` dependency for a single flag).
fn parse_cli() -> Result<CliArgs> {
    let mut config = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(val) = arg.strip_prefix("--config=") {
            config = Some(PathBuf::from(val));
        } else if arg == "--config" {
            let val = args
                .next()
                .context("--config requires a path argument (e.g. --config ./anamanti.json)")?;
            config = Some(PathBuf::from(val));
        } else if arg == "--help" || arg == "-h" {
            println!("Usage: anamanti-core [--config <path>]");
            std::process::exit(0);
        } else {
            anyhow::bail!("unknown argument `{arg}` (supported: --config <path>)");
        }
    }
    Ok(CliArgs { config })
}

/// Startup model check for the Ollama backend: verify the configured model is
/// installed; if not, fall back to an installed one (with a loud warning) so a
/// misconfigured/absent model fails visibly at boot instead of as a silent
/// per-turn 404. No-op for non-Ollama backends or if Ollama is unreachable.
async fn ensure_ollama_model(config: &mut Config) {
    use anamanti_core::config::LlmChoice;
    use anamanti_core::llm::ollama::{self, ModelCheck};

    let LlmChoice::Ollama { url, model } = &config.llm else {
        return;
    };
    let (url, model) = (url.clone(), model.clone());
    match ollama::available_models(&url).await {
        Ok(models) => match ollama::resolve_model(&model, &models) {
            ModelCheck::Available => log::info!("ollama model '{model}' is installed"),
            ModelCheck::FallBack(fallback) => {
                log::warn!(
                    "ollama model '{model}' is not installed at {url}; falling back to '{fallback}'. \
                     Installed: {models:?}. Set llm.ollama.model in the config file or run \
                     `ollama pull {model}`."
                );
                config.llm = LlmChoice::Ollama {
                    url,
                    model: fallback,
                };
            }
            ModelCheck::NonePulled => log::error!(
                "no models are installed in ollama at {url}; LLM turns will fail until you \
                 `ollama pull {model}` (or start Ollama)."
            ),
        },
        Err(e) => log::warn!(
            "could not query ollama models at {url} ({e:#}); proceeding with '{model}' \
             (turns will fail if it isn't installed)"
        ),
    }
}

/// Build the HelixDB GraphRAG recall backend and spawn the background ingester.
/// Requires `OPENAI_API_KEY` (embeddings); `ANTHROPIC_API_KEY` enables Claude
/// Haiku entity extraction (absent → pure-vector recall).
async fn build_graphrag_recall(
    config: &Config,
    chatlog: &Arc<ChatLog>,
) -> Result<(Arc<dyn anamanti_core::memory::Recall>, Arc<dyn GraphView>)> {
    use anamanti_core::memory::embed::{Embedder, OpenAiEmbedder};
    use anamanti_core::memory::entity::{
        AnthropicEntityExtractor, EntityExtractor, NoopEntityExtractor,
    };
    use anamanti_core::memory::helix::HelixMemory;
    use anamanti_core::memory::ingester::MemoryIngester;
    use anamanti_core::memory::HelixRecall;

    let g = &config.graphrag;
    let openai_key = std::env::var("OPENAI_API_KEY")
        .context("memory_backend=\"helix\" requires OPENAI_API_KEY for embeddings")?;
    let embedder: Arc<dyn Embedder> = Arc::new(OpenAiEmbedder::new(
        g.openai_base_url.clone(),
        openai_key,
        g.embed_model.clone(),
        g.embed_dims,
    ));

    let helix = Arc::new(
        HelixMemory::open_disk(config.helix_path.clone(), "ambient", embedder.dimensions())
            .await
            .context("opening embedded HelixDB store")?,
    );
    log::info!(
        "HelixDB store at {} ({} nodes)",
        config.helix_path.display(),
        helix.node_count().await.unwrap_or(0)
    );

    let extractor: Arc<dyn EntityExtractor> = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) if !key.is_empty() => Arc::new(AnthropicEntityExtractor::new(
            g.anthropic_base_url.clone(),
            key,
            g.extract_model.clone(),
        )),
        _ => {
            log::warn!(
                "ANTHROPIC_API_KEY absent; entity extraction disabled (recall is pure vector KNN)"
            );
            Arc::new(NoopEntityExtractor)
        }
    };

    let ingester = Arc::new(MemoryIngester::new(
        chatlog.path().to_path_buf(),
        helix.clone(),
        embedder.clone(),
        extractor,
    ));
    ingester.spawn(g.ingest_interval);
    log::info!(
        "background memory ingester running every {}s",
        g.ingest_interval.as_secs()
    );

    let recall: Arc<dyn anamanti_core::memory::Recall> =
        Arc::new(HelixRecall::new(helix.clone(), embedder, g.recall_k));
    let graph: Arc<dyn GraphView> = helix;
    Ok((recall, graph))
}
