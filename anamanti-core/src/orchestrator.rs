//! The assistant pipeline (Plan.MD Phase 4). Given a device-facing Wyoming
//! connection, one [`Pipeline::run_turn`] drives a full voice turn:
//!
//! 1. **STT** — forward the device's streamed PCM to the STT engine (downstream
//!    Wyoming Whisper or in-process whisper.cpp, behind the [`crate::stt::Transcriber`]
//!    seam), detect end-of-speech with the Core's energy VAD, finalize, then relay
//!    the `transcript` back to the device.
//! 2. **Memory + LLM** — apply explicit memory commands ("remember…"/"forget…")
//!    or auto-infer facts, build memory context, and stream a reply from the
//!    pluggable [`LlmBackend`].
//! 3. **TTS** — synthesize the reply with Piper and relay the audio frames back to
//!    the device over the same socket (architecture.md §4 SPEAKING).
//!
//! The pipeline acquires its downstream connections through a [`ServiceConnector`]
//! so production dials TCP while tests wire in-memory mock servers — the turn
//! logic is identical either way.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::time::{sleep_until, Instant};

use crate::audio_dump::TurnAudioDump;
use crate::config::FollowUpConfig;
use crate::llm::{DeviceAction, LlmBackend, LlmTurn, RecipeNav};
use crate::memory::chatlog::now_secs;
use crate::memory::promptlog::PromptLogRecord;
use crate::memory::{
    infer_memories, parse_command, ChatLog, ChatLogRecord, MemoryCommand, MemoryKind, MemorySource,
    MemoryStore, PromptLog, Recall, SqliteRecall,
};
use crate::settings::{Household, HouseholdMember, SharedSettings};
use crate::speaker::{SpeakerContext, SpeakerService};
use crate::stt::{SttEngine, SttEvent, Transcriber, WyomingTranscriber};
use crate::wyoming::protocol::{self, types, AudioFormat, WyomingEvent};
use crate::wyoming::tts::TtsSession;
use crate::wyoming::{DynConnection, DynRead, DynWrite};

/// Default no-speech window for an ordinary (wake-word) turn: if the user never
/// speaks, finalize (an empty transcript) after this long rather than hanging to the
/// turn timeout. A follow-up turn overrides this with its `follow_up.*_wait_secs`
/// listen window (see [`Pipeline::run_turn_after_start`]).
const DEFAULT_NO_SPEECH_FINALIZE: Duration = Duration::from_secs(6);

/// Minimum span of *consecutive* voiced audio (chunks above `voice_rms_threshold`)
/// required before we latch `speech_started` and switch a turn's finalize clock from
/// the `no_speech_finalize` window to the much shorter `end_silence` window. This
/// debounces the speech onset: a lone transient above the energy gate — the residual
/// TTS tail heard right after a follow-up mic reopens, room echo, or a brief noise —
/// would otherwise collapse a 10 s "wait for the user to answer" window into ~1 s and
/// cut the user off (observed in production as a follow-up turn that finalized with an
/// empty transcript). Real speech clears this in a single word; a blip does not.
const MIN_SPEECH_ONSET: Duration = Duration::from_millis(250);

/// Progress events surfaced as a turn runs (logging, tests, and — via the device
/// relay — the Phase-5 UI).
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    /// PCM is now streaming to the STT service.
    Streaming,
    /// The final transcript arrived from STT.
    Transcript(String),
    /// The turn's speaker was identified (or attributed to the shared household).
    Speaker(SpeakerContext),
    /// One reply-token fragment from the LLM (or a command confirmation).
    ReplyToken(String),
    /// The complete reply text.
    Reply(String),
    /// A memory entry was stored this turn.
    MemoryStored(String),
    /// TTS audio is now streaming back to the device.
    Speaking,
    /// The turn completed.
    Finished,
}

/// The result of driving one turn: either a turn ran to completion, or the device
/// had already disconnected (nothing to do). Lets the connection handler tell a
/// finished turn from a closed socket without busy-looping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    Completed,
    Disconnected,
}

/// Acquires downstream Wyoming connections. Abstracted so tests can substitute
/// in-process mock STT/TTS servers for real TCP dials.
#[async_trait]
pub trait ServiceConnector: Send + Sync {
    async fn connect_stt(&self) -> Result<DynConnection>;
    async fn connect_tts(&self) -> Result<DynConnection>;
}

/// Production connector: dials the configured Whisper and Piper TCP endpoints.
pub struct TcpConnector {
    pub stt_addr: std::net::SocketAddr,
    pub tts_addr: std::net::SocketAddr,
}

#[async_trait]
impl ServiceConnector for TcpConnector {
    async fn connect_stt(&self) -> Result<DynConnection> {
        DynConnection::connect_tcp(self.stt_addr)
            .await
            .context("connecting to STT (Whisper) service")
    }
    async fn connect_tts(&self) -> Result<DynConnection> {
        DynConnection::connect_tcp(self.tts_addr)
            .await
            .context("connecting to TTS (Piper) service")
    }
}

/// The shared, cheaply-cloneable pipeline state. One instance is reused across
/// every connection.
#[derive(Clone)]
pub struct Pipeline {
    settings: Arc<SharedSettings>,
    memory: Arc<MemoryStore>,
    /// Retrieval backend for prompt context (SQLite FTS by default; HelixDB
    /// GraphRAG when configured). Explicit/inferred writes still go to `memory`.
    recall: Arc<dyn Recall>,
    /// Optional append-only chat log; each completed turn is recorded for the
    /// background GraphRAG ingester. `None` disables logging.
    chatlog: Option<Arc<ChatLog>>,
    /// Optional append-only prompt log; the exact assembled LLM prompt per turn,
    /// for the debug GUI. `None` disables prompt logging.
    promptlog: Option<Arc<PromptLog>>,
    /// Optional per-person speaker identification (speaker_id_plan.md). `None`
    /// preserves the shared-household behavior — every turn attributes to
    /// [`crate::speaker::HOUSEHOLD_SPEAKER`].
    speaker: Option<Arc<SpeakerService>>,
    /// Debug-only per-turn audio capture directory (AEC corpus). `None` disables
    /// capture entirely; set from the config file's `audio_dump_dir`.
    audio_dump_dir: Option<PathBuf>,
    system_prompt: String,
    turn_timeout: Duration,
    /// Auto follow-up listening: when a reply is a question, tell the device to
    /// reopen the mic (no wake word) and feed recent history into that turn's prompt.
    /// Defaults to [`FollowUpConfig::default`] (enabled) until `with_follow_up` sets it.
    follow_up: FollowUpConfig,
    /// Forecast provider for the System-1 `weather` fast path (keyless Open-Meteo),
    /// shared with the ambient push. `None` when weather is disabled — the weather
    /// intent then defers to System-2. Set via `with_weather`.
    weather: Option<Arc<dyn crate::weather::WeatherProvider>>,
    /// STT engine used to transcribe each turn. `None` keeps the historical path:
    /// dial the downstream Wyoming Whisper server via the per-turn
    /// [`ServiceConnector`]. `Some` (e.g. the in-process whisper.cpp engine) supplies
    /// a session directly and the connector's `connect_stt` is never called.
    stt_engine: Option<Arc<dyn SttEngine>>,
}

impl Pipeline {
    /// Build a pipeline around the runtime-swappable [`SharedSettings`] (Phase 6):
    /// the LLM backend and TTS voice are read from a per-turn snapshot, so the
    /// device settings screen can change them between turns without a restart.
    pub fn with_settings(
        settings: Arc<SharedSettings>,
        memory: Arc<MemoryStore>,
        system_prompt: impl Into<String>,
        turn_timeout: Duration,
    ) -> Self {
        let recall: Arc<dyn Recall> = Arc::new(SqliteRecall::new(memory.clone()));
        Self {
            settings,
            memory,
            recall,
            chatlog: None,
            promptlog: None,
            speaker: None,
            audio_dump_dir: None,
            system_prompt: system_prompt.into(),
            turn_timeout,
            follow_up: FollowUpConfig::default(),
            weather: None,
            stt_engine: None,
        }
    }

    /// Set the debug-only per-turn audio capture directory (AEC corpus). `None`
    /// leaves capture disabled. Sourced from the config file's `audio_dump_dir`.
    pub fn with_audio_dump(mut self, dir: Option<PathBuf>) -> Self {
        self.audio_dump_dir = dir;
        self
    }

    /// Swap the retrieval backend used to build prompt context (e.g. the HelixDB
    /// GraphRAG backend). Defaults to SQLite FTS.
    pub fn with_recall(mut self, recall: Arc<dyn Recall>) -> Self {
        self.recall = recall;
        self
    }

    /// Attach an append-only chat log; each completed turn is recorded for the
    /// background GraphRAG ingester.
    pub fn with_chatlog(mut self, chatlog: Arc<ChatLog>) -> Self {
        self.chatlog = Some(chatlog);
        self
    }

    /// Attach an append-only prompt log; the exact assembled LLM prompt for each
    /// turn is recorded for the debug GUI (`webconfig.rs` → `/prompts`).
    pub fn with_promptlog(mut self, promptlog: Arc<PromptLog>) -> Self {
        self.promptlog = Some(promptlog);
        self
    }

    /// Enable per-person speaker identification. Without this the pipeline keeps the
    /// shared-household behavior (all turns → `household`).
    pub fn with_speaker(mut self, speaker: Arc<SpeakerService>) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Configure auto follow-up listening (question → reopen mic + history). Without
    /// this the pipeline uses [`FollowUpConfig::default`] (enabled).
    pub fn with_follow_up(mut self, follow_up: FollowUpConfig) -> Self {
        self.follow_up = follow_up;
        self
    }

    /// Install a System-1 fast-decision engine (plans/system1-fast-decisions.md) into the
    /// live settings, so the per-turn snapshot picks it up. The boot path builds the
    /// engine from config in [`SharedSettings`]; this is a convenience for tests and
    /// callers holding an already-built engine. The config page swaps it at runtime via
    /// [`SharedSettings::apply_system1`].
    pub fn with_system1(self, system1: Arc<dyn crate::system1::DecisionEngine>) -> Self {
        self.settings.set_system1(system1);
        self
    }

    /// Provide the forecast provider used by the System-1 `weather` fast path (shared
    /// with the ambient push). Without it, the weather intent defers to System-2.
    pub fn with_weather(mut self, weather: Option<Arc<dyn crate::weather::WeatherProvider>>) -> Self {
        self.weather = weather;
        self
    }

    /// Attach an in-process STT engine (e.g. whisper.cpp). Without this the pipeline
    /// dials the downstream Wyoming Whisper server via the per-turn
    /// [`ServiceConnector`] (the historical default).
    pub fn with_stt_engine(mut self, engine: Arc<dyn SttEngine>) -> Self {
        self.stt_engine = Some(engine);
        self
    }

    /// The speaker service, if enabled (the Phase-C control handler lists/renames
    /// its registry).
    pub fn speaker(&self) -> Option<&Arc<SpeakerService>> {
        self.speaker.as_ref()
    }

    /// Build a pipeline around a fixed LLM backend + voice (the Phase-4 behavior).
    /// The backend cannot be swapped at runtime (its settings holder has no
    /// credentials); the TTS voice can still be changed via a control frame.
    pub fn new(
        llm: Arc<dyn LlmBackend>,
        memory: Arc<MemoryStore>,
        system_prompt: impl Into<String>,
        tts_voice: Option<String>,
        turn_timeout: Duration,
    ) -> Self {
        let settings = SharedSettings::fixed(llm, "custom", tts_voice);
        Self::with_settings(settings, memory, system_prompt, turn_timeout)
    }

    /// The persistent memory store (the Phase-6 control handler lists/deletes it).
    pub fn memory(&self) -> &Arc<MemoryStore> {
        &self.memory
    }

    /// The runtime-swappable settings (the Phase-6 control handler reads/updates it).
    pub fn settings(&self) -> &Arc<SharedSettings> {
        &self.settings
    }

    /// Drive one voice turn over `device`. Returns `Ok(())` on a completed turn or
    /// a clean device disconnect; returns `Err` only on an unrecoverable pipeline
    /// failure (STT unreachable, LLM error, …).
    pub async fn run_turn(
        &self,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
    ) -> Result<TurnOutcome> {
        // 1. Wait for the device's `audio-start`; a clean close before that just
        //    ends the connection.
        let (format, followup_depth, followup_wait_secs, screen) = loop {
            match device.read().await? {
                Some(ev) if ev.event_type == types::AUDIO_START => {
                    break (
                        protocol::audio_format(&ev.data).unwrap_or(AudioFormat::PCM_16K_MONO),
                        protocol::followup_depth(&ev.data),
                        protocol::followup_wait_secs(&ev.data),
                        protocol::display_context(&ev.data),
                    );
                }
                Some(_) => continue, // ignore stray pre-turn frames
                None => return Ok(TurnOutcome::Disconnected),
            }
        };
        self.run_turn_after_start(
            device,
            connector,
            format,
            followup_depth,
            followup_wait_secs,
            screen,
            on_event,
        )
        .await
    }

    /// Drive a turn whose opening `audio-start` has already been read (the server
    /// consumes it to distinguish a turn from a Phase-6 control frame). Splitting
    /// this out lets one accept loop serve both turns and control on the same
    /// socket.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_turn_after_start(
        &self,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        format: AudioFormat,
        followup_depth: u32,
        followup_wait_secs: u32,
        screen: Option<protocol::DisplayContext>,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
    ) -> Result<TurnOutcome> {
        // Take one settings snapshot for the whole turn so a concurrent control
        // swap never changes the backend/voice mid-reply.
        let runtime = self.settings.snapshot();

        // Debug-only AEC corpus capture (no-op unless `audio_dump_dir` is configured):
        // records this turn's device mic (near-end) and Piper TTS (far-end reference).
        let dump = TurnAudioDump::for_turn(self.audio_dump_dir.as_deref());

        // 2. Open the STT stream and pump device PCM into it until the transcript.
        //    The concrete engine sits behind the `Transcriber` seam
        //    (`plans/python-to-rust-whisper.md`): an attached in-process engine (e.g.
        //    whisper.cpp) supplies a session directly; otherwise dial the downstream
        //    Wyoming Whisper server via the connector (the historical default).
        let mut stt: Box<dyn Transcriber> = match self.stt_engine.as_ref() {
            Some(engine) => engine.begin(format).await?,
            None => {
                let stt_conn = connector.connect_stt().await?;
                Box::new(WyomingTranscriber::begin(stt_conn, format).await?)
            }
        };
        on_event(TurnEvent::Streaming);

        let end_silence = Duration::from_millis(runtime.end_silence_ms);
        let voice_rms_threshold = runtime.voice_rms_threshold;
        // How long to wait for the user to *start* speaking before finalizing (and, on
        // silence, sleeping). A follow-up turn (the device auto-opened the mic, no wake
        // word) uses the window the reply that triggered it chose — 10 s after a
        // question, 5 s otherwise — echoed here on the `audio-start`. An ordinary turn
        // keeps the default.
        let no_speech_finalize = if followup_depth > 0 && followup_wait_secs > 0 {
            Duration::from_secs(followup_wait_secs as u64)
        } else {
            DEFAULT_NO_SPEECH_FINALIZE
        };
        let Some((transcript, voiced_pcm)) = self
            .stream_to_transcript(
                device,
                stt.as_mut(),
                end_silence,
                no_speech_finalize,
                voice_rms_threshold,
                format.rate,
                dump.as_ref(),
            )
            .await?
        else {
            // Device closed or timed out before a transcript — abandon the turn.
            let _ = stt.finish().await;
            return Ok(TurnOutcome::Completed);
        };
        let _ = stt.finish().await; // idempotent finalize (VAD already sent audio-stop)
        if let Some(d) = dump.as_ref() {
            d.set_transcript(&transcript);
        }
        on_event(TurnEvent::Transcript(transcript.clone()));

        // Relay the transcript to the device (renders on screen; ends its input).
        device
            .send(&WyomingEvent::transcript(&transcript))
            .await
            .ok();

        if transcript.trim().is_empty() {
            // No speech within the listen window: sleep. Send an `audio-stop` so the
            // device leaves SPEAKING and returns to idle immediately (it flips to
            // SPEAKING on the transcript above and would otherwise wait out its idle
            // watchdog for TTS that never comes). We do NOT send `ambient-listen`, so
            // the follow-up chain ends here — silence is the terminator.
            device.send(&WyomingEvent::audio_stop(0)).await.ok();
            on_event(TurnEvent::Finished);
            return Ok(TurnOutcome::Completed);
        }

        // 2b. Identify who is speaking (or attribute to the shared household), so
        //     memory writes/recall and the reply are per-person.
        let speaker = self.identify_speaker(&voiced_pcm);
        on_event(TurnEvent::Speaker(speaker.clone()));

        // 3 + 4. Memory + LLM → reply, relaying each token to the device for
        //    token-by-token rendering (Phase 5) AND synthesizing complete sentences
        //    with Piper as soon as they form, so playback begins before the full
        //    reply is generated (streaming TTS; architecture.md §4 SPEAKING).
        let (reply, memories_written) = self
            .respond_and_speak(
                &runtime,
                &transcript,
                &speaker,
                device,
                connector,
                followup_depth,
                screen.as_ref(),
                on_event,
                dump.as_ref(),
            )
            .await?;
        if let Some(d) = dump.as_ref() {
            d.set_reply(&reply);
        }
        on_event(TurnEvent::Reply(reply.clone()));

        // Record the completed turn for the background GraphRAG ingester. Never
        // let a logging failure break the turn.
        self.log_turn(&runtime, &speaker, &transcript, &reply, memories_written);

        on_event(TurnEvent::Finished);
        Ok(TurnOutcome::Completed)
    }

    /// Pump loop: forward device `audio-chunk`s to STT, detect end-of-speech, and
    /// return STT's `transcript`. `None` if the device hung up or the turn idled
    /// past `turn_timeout`.
    ///
    /// wyoming-faster-whisper does **not** do streaming VAD — it transcribes the
    /// buffered utterance only once it receives `audio-stop`. The device, meanwhile,
    /// streams continuously and waits for the transcript before it stops. So the
    /// orchestrator is the only party that can close the loop: it runs a simple
    /// energy VAD over the incoming PCM and, once speech has been followed by a
    /// short trailing silence, sends `audio-stop` to STT to finalize the transcript.
    /// Returns the transcript together with the utterance's **voiced** PCM (the
    /// chunks that passed the energy gate), so the caller can compute a speaker
    /// embedding without re-reading the socket. `None` on disconnect/timeout.
    #[allow(clippy::too_many_arguments)]
    async fn stream_to_transcript(
        &self,
        device: &mut DynConnection,
        stt: &mut dyn Transcriber,
        end_silence: std::time::Duration,
        no_speech_finalize: std::time::Duration,
        voice_rms_threshold: f64,
        mic_rate: u32,
        dump: Option<&TurnAudioDump>,
    ) -> Result<Option<(String, Vec<i16>)>> {
        // `voice_rms_threshold`: RMS (i16 units) above which a chunk counts as speech
        // rather than room noise. The Echo's far-field pickup is quiet (~50 idle,
        // several hundred+ while speaking). `end_silence`: trailing silence after
        // speech that marks end-of-utterance. Both come from the per-turn settings
        // snapshot so they are A/B-tunable from the device without a restart.
        //
        // If no speech is ever detected, still finalize after `no_speech_finalize` so a
        // silent or too-quiet utterance ends the turn instead of hanging to
        // `turn_timeout`. For a follow-up turn this is the caller's listen window
        // (`follow_up.*_wait_secs`), so "wait 10 s after a question / 5 s otherwise,
        // then sleep" is enforced here — the only place that can tell silence from a
        // user who is mid-sentence (the device runs no VAD).
        let turn_start = Instant::now();
        let mut last_voice = turn_start;
        let mut speech_started = false;
        // Consecutive voiced audio accumulated since the last non-voiced chunk, used to
        // debounce the speech onset (see `MIN_SPEECH_ONSET`). Reset by any non-voiced
        // chunk so only *sustained* voice latches `speech_started`.
        let mut voiced_run = Duration::ZERO;
        // True once we've sent `audio-stop` to STT and are just awaiting the result.
        let mut finalized = false;
        // Accumulated voiced PCM (samples from chunks above the energy gate), used
        // for the speaker embedding once the transcript arrives.
        let mut voiced_pcm: Vec<i16> = Vec::new();

        let mut deadline = Instant::now() + self.turn_timeout;
        loop {
            tokio::select! {
                biased;

                _ = sleep_until(deadline) => {
                    log::warn!("turn idle-timed-out waiting for STT transcript");
                    return Ok(None);
                }

                // Only read the device *before* we've finalized STT. `device.read()`
                // (read_line + read_exact) is NOT cancellation-safe, so if it were
                // still enabled once the STT transcript can arrive, the `sev` arm
                // winning the race would drop this future mid-PCM-payload and desync
                // the reader — the next `read_event` (barge-in watcher / next turn)
                // then reads binary as a header line ("stream did not contain valid
                // UTF-8"), spuriously aborting the reply. After `finalized` we ignore
                // device chunks anyway, so simply stop reading them: nothing is ever
                // mid-frame when the transcript lands. Leftover buffered chunks are
                // complete frames, harmlessly drained later.
                dev = device.read(), if !finalized => {
                    deadline = Instant::now() + self.turn_timeout;
                    match dev? {
                        Some(ev) if ev.event_type == types::AUDIO_CHUNK => {
                            if let Some(pcm) = ev.payload {
                                // Capture the near-end mic for the AEC corpus (the
                                // whole listening window, incl. pre/post-speech).
                                if let Some(d) = dump {
                                    d.push_mic(&pcm, mic_rate);
                                }
                                if !finalized {
                                    let now = Instant::now();
                                    let voiced = rms_i16_le(&pcm) > voice_rms_threshold;
                                    if voiced {
                                        last_voice = now;
                                        // Keep the voiced samples for speaker ID.
                                        append_pcm_i16_le(&mut voiced_pcm, &pcm);
                                    }
                                    // Debounce the onset: only latch `speech_started` after
                                    // MIN_SPEECH_ONSET of *consecutive* voiced audio, so a
                                    // lone transient (TTS tail, echo, noise) can't collapse
                                    // the no-speech window into the short end-silence one.
                                    if !speech_started {
                                        let prev_run = voiced_run;
                                        let latched;
                                        (voiced_run, latched) = voiced_onset_step(
                                            voiced,
                                            chunk_duration(pcm.len(), mic_rate),
                                            voiced_run,
                                        );
                                        if latched {
                                            speech_started = true;
                                            log::debug!("VAD: speech started");
                                        } else if !voiced && prev_run > Duration::ZERO {
                                            log::debug!(
                                                "VAD: discarded {:?} voiced transient below \
                                                 {:?} onset; no-speech window still open",
                                                prev_run, MIN_SPEECH_ONSET
                                            );
                                        }
                                    }
                                    stt.forward_pcm(pcm).await?;

                                    let ended = if speech_started {
                                        now.duration_since(last_voice) >= end_silence
                                    } else {
                                        now.duration_since(turn_start) >= no_speech_finalize
                                    };
                                    if ended {
                                        log::info!(
                                            "VAD: end-of-speech (speech_started={speech_started}); \
                                             finalizing STT"
                                        );
                                        stt.finish().await?;
                                        finalized = true;
                                    }
                                }
                                // After finalizing, drop further mic chunks: STT has
                                // its `audio-stop` and is transcribing.
                            }
                        }
                        // The device sends `audio-stop` only after it sees the
                        // transcript; before then, ignore other frames.
                        Some(_) => {}
                        None => return Ok(None), // device disconnected
                    }
                }

                sev = stt.read_event() => {
                    deadline = Instant::now() + self.turn_timeout;
                    match sev? {
                        Some(SttEvent::Transcript(text)) => {
                            // Guard against STT hallucinations on silence. If our energy
                            // VAD never latched `speech_started`, this turn finalized on
                            // the no-speech timeout: everything we forwarded to Whisper
                            // was silence/room noise. faster-whisper does NOT return an
                            // empty string on non-speech — it emits short training-set
                            // filler ("Thank you.", "I'm sorry.", "Thank you for
                            // watching.") — so any text here is not a real utterance,
                            // especially on a follow-up window (mic reopened with no wake
                            // word). Collapse it to an empty transcript, which the caller
                            // already treats as "no speech": sleep, send `audio-stop`,
                            // and end the follow-up chain (no `ambient-listen`). This
                            // stops the self-perpetuating phantom-reply loop in a quiet
                            // room. The trade-off is that a genuine utterance too quiet
                            // to clear `voice_rms_threshold` for `MIN_SPEECH_ONSET` is
                            // also dropped — consistent with the existing no-speech
                            // finalize, which already treats too-quiet audio as silence.
                            if !speech_started {
                                if !text.trim().is_empty() {
                                    log::info!(
                                        "VAD: discarding STT transcript {text:?} from a \
                                         no-speech finalize (no speech detected; likely \
                                         Whisper hallucination on silence)"
                                    );
                                }
                                return Ok(Some((String::new(), Vec::new())));
                            }
                            return Ok(Some((text, std::mem::take(&mut voiced_pcm))));
                        }
                        Some(SttEvent::Other) => {} // voice-started / voice-stopped etc.
                        None => anyhow::bail!("STT service closed before returning a transcript"),
                    }
                }
            }
        }
    }

    /// Apply memory policy, stream the LLM reply, and synthesize it with Piper as
    /// complete sentences arrive so playback begins before generation finishes.
    ///
    /// Each token is relayed to the device as a `reply-token` (for token-by-token
    /// on-screen rendering) and appended to a pending buffer; whenever that buffer
    /// holds a complete sentence it is flushed to TTS immediately. This removes the
    /// old "buffer the whole reply, then speak" barrier: time-to-first-audio drops
    /// from full-reply latency to first-sentence latency. Returns the full reply
    /// text and the memory entries written this turn.
    // These are all distinct per-turn inputs (runtime/transcript/speaker/device/
    // connector/event-sink/dump); bundling them into a struct would only relocate
    // the list, so the arg-count lint isn't worth appeasing here.
    #[allow(clippy::too_many_arguments)]
    async fn respond_and_speak(
        &self,
        runtime: &crate::settings::RuntimeSettings,
        transcript: &str,
        speaker: &SpeakerContext,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        followup_depth: u32,
        screen: Option<&protocol::DisplayContext>,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
        dump: Option<&TurnAudioDump>,
    ) -> Result<(String, Vec<String>)> {
        // Memory entries written this turn (for the chat log / ingester).
        let mut memories_written = Vec::new();
        // The scope memory writes/recall use: this person, or shared (household).
        let scope = speaker_scope(speaker);

        // Explicit command → apply, confirm, and speak the confirmation in one
        // chunk, skipping the LLM. (Instant, so it is not barge-in-interruptible.)
        if let Some(cmd) = parse_command(transcript) {
            match &cmd {
                MemoryCommand::Remember { content, .. } => memories_written.push(content.clone()),
                MemoryCommand::NameSpeaker(name) => {
                    memories_written.push(format!("The user's name is {name}"))
                }
                _ => {}
            }
            let reply = self.apply_command(cmd, scope, on_event)?;
            on_event(TurnEvent::ReplyToken(reply.clone()));
            let (_reader, writer) = device.split_mut();
            protocol::write_event(writer, &WyomingEvent::reply_token(&reply))
                .await
                .ok();
            on_event(TurnEvent::Speaking);
            let mut audio_started = false;
            self.speak_chunk(writer, runtime, connector, &reply, &mut audio_started, dump)
                .await?;
            if audio_started {
                protocol::write_event(writer, &WyomingEvent::audio_stop(0))
                    .await
                    .ok();
            }
            return Ok((reply, memories_written));
        }

        // System-1 fast-decision fork (plans/system1-fast-decisions.md; Plan.MD
        // 2026-09-25). Runs *before* memory recall + the LLM. Skipped entirely for the
        // default `none` engine, so ordinary builds are unaffected. On a confident
        // Resolve with a matching handler, the turn is answered here — skipping the
        // GraphRAG embedding round-trip (#3) and the rig+tools full completion (#1) —
        // and returns; otherwise it falls through to System-2 below.
        if runtime.system1.engine.name() != "none" {
            let req = crate::system1::DecisionRequest {
                transcript: transcript.to_string(),
                screen: None,        // M1: derive a label from the turn's `screen` context
                history: Vec::new(), // M1: recent turns for follow-up disambiguation
                // Home location grounds location-dependent intents (weather): the HTTP
                // engine retries an otherwise-deferred turn with this folded into the query.
                location: self.settings.home_location().get(),
            };
            match runtime.system1.engine.decide(&req).await {
                Ok(crate::system1::Decision::Resolve(r)) => {
                    log::info!(
                        "system1 ({}) resolved intent `{}` (conf {:.2})",
                        runtime.system1.engine.name(),
                        r.intent,
                        r.confidence,
                    );
                    if let Some(reply) = self
                        .handle_system1_intent(
                            &r,
                            runtime,
                            transcript,
                            device,
                            connector,
                            followup_depth,
                            on_event,
                            dump,
                        )
                        .await
                    {
                        // Fully handled on the fast path (widget + spoken reply +
                        // follow-up). The caller still logs the turn to the chat log.
                        return Ok((reply, memories_written));
                    }
                    log::debug!(
                        "system1 intent `{}` not handled on the fast path; deferring to System-2",
                        r.intent
                    );
                }
                Ok(crate::system1::Decision::Defer) => {}
                Err(e) => log::debug!(
                    "system1 ({}) decide failed ({e:#}); deferring to System-2",
                    runtime.system1.engine.name()
                ),
            }
        }

        // Inferred capture from an ordinary turn — attributed to this speaker.
        for (kind, content) in infer_memories(transcript) {
            self.memory
                .add_scoped(kind, &content, MemorySource::Inferred, scope)?;
            memories_written.push(content.clone());
            on_event(TurnEvent::MemoryStored(content));
        }

        // Build per-person memory context and stream the LLM reply. A recall
        // failure (e.g. a transient GraphRAG backend error) must not sink the turn —
        // proceed with no memory context rather than erroring.
        let context = self
            .build_context(transcript, scope)
            .await
            .unwrap_or_else(|e| {
                log::warn!("memory recall failed; answering without context: {e:#}");
                String::new()
            });
        // Ground "here" and who lives here from the canonical household record (live
        // per-turn snapshot, so config-page edits take effect without a restart).
        let household = &runtime.household;
        // Ground the model in the real wall-clock (it has no clock) and tell it who it
        // is speaking with, so time/date questions are answered from fact and it can
        // address the person by name / apply the right person's memory. When the
        // identified speaker matches a household member, the line notes who they are.
        let identity = speaker_identity_line(speaker, household);
        let mut system_prompt = format!(
            "{}\n\n{}\n\n{}",
            self.system_prompt,
            current_datetime_line(),
            identity
        );
        if let Some(line) = location_line(
            household.location.as_deref(),
            household.weather_units.as_deref(),
        ) {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&line);
        }
        if let Some(line) = household_line(&household.members) {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&line);
        }
        if !context.is_empty() {
            system_prompt.push_str("\n\nWhat you remember about this person:\n");
            system_prompt.push_str(&context);
        }
        // Tell the model what the display is currently showing (its "display context"),
        // so it can drive that screen by voice with the matching tool. Absent on an idle
        // display. Extensible per screen kind — see `display_context_line`.
        if let Some(screen) = screen {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&display_context_line(screen));
        }

        // Per-turn device-action channel: action tools (timers) push `DeviceAction`s
        // here and the drive loop below relays them to the device as `ambient-timer`
        // frames on the same socket. Unbounded so a tool's `invoke` never blocks.
        let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<DeviceAction>();
        // Record the exact prompt the model is about to see (debug GUI). Best-effort:
        // a logging failure must never break a turn.
        self.log_prompt(runtime, speaker, &system_prompt, transcript);

        // A follow-up turn (the device auto-opened the mic after a question reply, so
        // `followup_depth > 0`) is fed the recent conversation so the answer has
        // context; an ordinary single-shot turn carries no history (locked "single-turn
        // v1" decision). See plans/Plan.MD (Follow-up listening).
        let history = if followup_depth > 0 {
            self.recent_history()
        } else {
            Vec::new()
        };

        let mut stream = runtime
            .llm
            .respond(
                LlmTurn::new(system_prompt, transcript)
                    .with_actions(action_tx)
                    .with_history(history),
            )
            .await
            .with_context(|| format!("LLM backend `{}` failed", runtime.llm.name()))?;

        let mut reply = String::new();
        let mut pending = String::new();
        let mut speaking = false;
        // Whether the single, coalesced device-facing `audio-start` has been sent.
        // Piper emits an `audio-start`/`audio-stop` per sentence, but the device
        // ends its turn on the *first* `audio-stop` — so we forward exactly one
        // `audio-start` up front, relay only the chunks, and send one `audio-stop`
        // at the very end. The reply still streams sentence-by-sentence (low
        // time-to-first-audio) but reaches the device as one continuous stream.
        let mut audio_started = false;

        // Race the reply against a barge-in in an inner scope so both futures (and
        // their borrows of `reply`/`pending`/the split socket halves) are dropped
        // before we read `reply` back out below.
        let interrupted = {
            // Split the device socket so we can *concurrently* stream reply tokens +
            // TTS audio out on the writer while watching the reader for a barge-in.
            // The user talking over the assistant (a new wake word or on-device VAD)
            // sends an `ambient-interrupt` frame (and/or drops the socket); either
            // wins the `select!` below, cancels the driver, and aborts the LLM + TTS
            // at once instead of finishing a reply nobody is listening to.
            let (reader, writer) = device.split_mut();

            // The generation + speaking driver. Dropping this future (when a barge-in
            // wins the race) drops the LLM `stream` and any in-flight `TtsSession`,
            // cancelling the upstream Ollama/Anthropic request and Piper synthesis.
            let drive = async {
                while let Some(tok) = stream.next().await {
                    // Relay any device actions a tool emitted (e.g. a timer) to the
                    // device as they arrive, before rendering more of the reply.
                    drain_device_actions(&mut action_rx, writer).await;
                    let tok = tok?;
                    reply.push_str(&tok);
                    pending.push_str(&tok);
                    protocol::write_event(writer, &WyomingEvent::reply_token(&tok))
                        .await
                        .ok();
                    on_event(TurnEvent::ReplyToken(tok));

                    // Flush every complete sentence that has formed so far.
                    while let Some(sentence) = take_speakable(&mut pending, false) {
                        if !speaking {
                            on_event(TurnEvent::Speaking);
                            speaking = true;
                        }
                        if !self
                            .speak_chunk(
                                writer,
                                runtime,
                                connector,
                                &sentence,
                                &mut audio_started,
                                dump,
                            )
                            .await?
                        {
                            // Device closed mid-relay; stop generating.
                            return Ok::<(), anyhow::Error>(());
                        }
                    }
                }
                // Relay any device actions emitted late in the reply (e.g. a tool
                // call on the final round) before closing the turn.
                drain_device_actions(&mut action_rx, writer).await;
                // Speak any trailing clause left without terminal punctuation.
                if let Some(rest) = take_speakable(&mut pending, true) {
                    if !speaking {
                        on_event(TurnEvent::Speaking);
                    }
                    self.speak_chunk(writer, runtime, connector, &rest, &mut audio_started, dump)
                        .await?;
                }
                // Follow-up listen (after EVERY reply) + the final `audio-stop`. Shared
                // with the System-1 fast path via `emit_follow_up_and_stop`. Reaching
                // here means the reply completed (a barge-in would have dropped this
                // whole future), so we never reopen over an interruption.
                emit_follow_up_and_stop(
                    writer,
                    &self.follow_up,
                    audio_started,
                    &reply,
                    followup_depth,
                )
                .await;
                Ok(())
            };
            tokio::pin!(drive);

            let watch = watch_for_barge_in(reader);
            tokio::pin!(watch);

            let interrupted;
            tokio::select! {
                biased;
                // Barge-in (or device close) → cancel the reply immediately.
                _ = &mut watch => {
                    interrupted = true;
                }
                res = &mut drive => {
                    res?;
                    interrupted = false;
                }
            }
            interrupted
        };
        if interrupted {
            log::info!("barge-in: aborting in-flight LLM generation + TTS for this turn");
        }

        Ok((reply.trim().to_string(), memories_written))
    }

    /// Append the completed turn to the chat log, if one is attached. A failure is
    /// logged and swallowed — logging must never break a turn.
    fn log_turn(
        &self,
        runtime: &crate::settings::RuntimeSettings,
        speaker: &SpeakerContext,
        transcript: &str,
        reply: &str,
        memories_written: Vec<String>,
    ) {
        let Some(log) = &self.chatlog else {
            return;
        };
        let id = log.next_id();
        let record = ChatLogRecord {
            session_id: format!("s-{id}"),
            id,
            ts: now_secs(),
            transcript: transcript.to_string(),
            reply: reply.to_string(),
            memories_written,
            llm_backend: runtime.llm_backend.clone(),
            model: runtime.llm_model.clone(),
            speaker_id: speaker.speaker_id.clone(),
            speaker_name: speaker.name.clone(),
        };
        if let Err(e) = log.append(&record) {
            log::warn!("failed to append chat log record: {e:#}");
        }
    }

    /// Append the assembled LLM prompt to the prompt log, if one is attached. A
    /// failure is logged and swallowed — logging must never break a turn.
    fn log_prompt(
        &self,
        runtime: &crate::settings::RuntimeSettings,
        speaker: &SpeakerContext,
        system_prompt: &str,
        user_message: &str,
    ) {
        let Some(log) = &self.promptlog else {
            return;
        };
        let record = PromptLogRecord {
            id: log.next_id(),
            ts: now_secs(),
            llm_backend: runtime.llm_backend.clone(),
            model: runtime.llm_model.clone(),
            speaker_id: speaker.speaker_id.clone(),
            speaker_name: speaker.name.clone(),
            system_prompt: system_prompt.to_string(),
            user_message: user_message.to_string(),
        };
        if let Err(e) = log.append(&record) {
            log::warn!("failed to append prompt log record: {e:#}");
        }
    }

    /// Apply an explicit memory command; returns the spoken confirmation.
    fn apply_command(
        &self,
        cmd: MemoryCommand,
        speaker_id: Option<&str>,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
    ) -> Result<String> {
        Ok(match cmd {
            MemoryCommand::Remember { kind, content } => {
                self.memory
                    .add_scoped(kind, &content, MemorySource::Explicit, speaker_id)?;
                on_event(TurnEvent::MemoryStored(content));
                "Okay, I'll remember that.".to_string()
            }
            MemoryCommand::ForgetMatching(query) => {
                let n = self.memory.forget_matching(&query)?;
                if n > 0 {
                    "Okay, I've forgotten that.".to_string()
                } else {
                    "I didn't have anything about that.".to_string()
                }
            }
            MemoryCommand::ForgetLast => match self.memory.forget_last()? {
                Some(_) => "Okay, I've forgotten that.".to_string(),
                None => "There was nothing to forget.".to_string(),
            },
            MemoryCommand::NameSpeaker(name) => {
                // Record the name as a per-person fact (parity with inferred capture)…
                let fact = format!("The user's name is {name}");
                self.memory.add_scoped(
                    MemoryKind::Fact,
                    &fact,
                    MemorySource::Explicit,
                    speaker_id,
                )?;
                on_event(TurnEvent::MemoryStored(fact));
                // …and attach it to the voiceprint profile, when a person was identified.
                if let (Some(svc), Some(id)) = (&self.speaker, speaker_id) {
                    if let Err(e) = svc.registry().rename(id, &name) {
                        log::warn!("failed to name speaker {id}: {e:#}");
                    }
                }
                format!("Nice to meet you, {name}!")
            }
        })
    }

    /// Gather memory entries relevant to the transcript as prompt context for this
    /// speaker (their own entries plus shared), via the configured recall backend
    /// (SQLite FTS by default; HelixDB GraphRAG when set).
    async fn build_context(&self, transcript: &str, speaker_id: Option<&str>) -> Result<String> {
        let hits = self.recall.recall(transcript, speaker_id, 8).await?;
        Ok(hits
            .iter()
            .map(|c| format!("- {c}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// The recent conversation, as `(user, assistant)` pairs oldest-first, for a
    /// **follow-up** turn's prompt: the last `follow_up.history_turns` chat-log turns
    /// that completed within `follow_up.history_window_secs`. Empty when no chat log is
    /// attached, the feature is off, or nothing recent qualifies. Recency-scoped rather
    /// than device-scoped (the chat log has no device id), so back-to-back conversations
    /// on one display thread correctly; two displays talking at once could interleave
    /// (an accepted v1 limitation — see plans/Plan.MD). A read failure yields no history
    /// rather than breaking the turn.
    fn recent_history(&self) -> Vec<(String, String)> {
        let Some(log) = &self.chatlog else {
            return Vec::new();
        };
        if self.follow_up.history_turns == 0 {
            return Vec::new();
        }
        let cutoff = now_secs() - self.follow_up.history_window_secs;
        let recent = crate::memory::chatlog::read_tail(log.path(), self.follow_up.history_turns)
            .unwrap_or_else(|e| {
                log::warn!("follow-up: reading recent chat history failed: {e:#}");
                Vec::new()
            });
        // `read_tail` is newest-first: keep only in-window turns with real content, then
        // reverse to chronological (oldest-first) order for the replayed message list.
        recent
            .into_iter()
            .filter(|r| r.ts >= cutoff)
            .filter(|r| !r.transcript.trim().is_empty() && !r.reply.trim().is_empty())
            .map(|r| (r.transcript, r.reply))
            .rev()
            .collect()
    }

    /// Identify the turn's speaker via the [`SpeakerService`], degrading gracefully
    /// to the shared household on any failure (or when identification is disabled).
    fn identify_speaker(&self, voiced_pcm: &[i16]) -> SpeakerContext {
        let Some(svc) = &self.speaker else {
            return SpeakerContext::household();
        };
        match svc.identify_and_attribute(voiced_pcm) {
            Ok(ctx) => ctx,
            Err(e) => {
                log::warn!("speaker identification failed; attributing to household: {e:#}");
                SpeakerContext::household()
            }
        }
    }

    /// Synthesize one chunk of reply text with Piper and relay its audio frames to
    /// the device over the same socket. A fresh downstream Piper connection is used
    /// per chunk so this works with any Wyoming TTS server (whether or not it keeps
    /// a connection open across `synthesize` requests).
    ///
    /// Returns `Ok(false)` if the device closed mid-relay — treated as a graceful
    /// stop (e.g. the Phase-3 client that ends on transcript, or a barge-in that
    /// dropped the socket), not a turn failure — and `Ok(true)` otherwise. An
    /// empty/whitespace chunk is a no-op that returns `Ok(true)`.
    /// Dispatch a System-1 [`Resolution`](crate::system1::Resolution) to its handler.
    /// Returns `Some(reply)` when the turn was fully answered on the fast path (so the
    /// caller returns without touching recall/the LLM), or `None` to defer to System-2
    /// (unknown intent, or a handler precondition that failed *before* anything was sent
    /// to the device — so deferring can't double-speak).
    #[allow(clippy::too_many_arguments)]
    async fn handle_system1_intent(
        &self,
        r: &crate::system1::Resolution,
        runtime: &crate::settings::RuntimeSettings,
        _transcript: &str,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        followup_depth: u32,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
        dump: Option<&TurnAudioDump>,
    ) -> Option<String> {
        match r.intent.as_str() {
            "weather" => {
                self.handle_weather(runtime, device, connector, followup_depth, on_event, dump)
                    .await
            }
            "timer" => {
                self.handle_timer(
                    runtime,
                    _transcript,
                    device,
                    connector,
                    followup_depth,
                    on_event,
                    dump,
                )
                .await
            }
            _ => None, // no fast-path handler yet → defer
        }
    }

    /// The System-1 `timer` fast path: parse an unambiguous duration from the transcript,
    /// start the timer on the device, and speak a confirmation — skipping the LLM. Returns
    /// `None` (defer) when no clear duration is present (e.g. "cancel my timer", or a
    /// free-form request), so the robust System-2 timer tool still handles those.
    #[allow(clippy::too_many_arguments)]
    async fn handle_timer(
        &self,
        runtime: &crate::settings::RuntimeSettings,
        transcript: &str,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        followup_depth: u32,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
        dump: Option<&TurnAudioDump>,
    ) -> Option<String> {
        let secs = crate::system1::parse_duration_secs(transcript)?; // unclear → defer
        // Guard rails: ignore absurd durations (>24h) — defer to System-2.
        if secs == 0 || secs > 24 * 3600 {
            log::debug!("system1 timer: duration {secs}s out of range; deferring to System-2");
            return None;
        }

        // Committed: own the turn from here (never fall through → no double-speak).
        let reply = format!("Timer set for {}.", human_duration(secs));
        let (_reader, writer) = device.split_mut();
        protocol::write_event(writer, &WyomingEvent::timer_start(secs, None))
            .await
            .ok();
        on_event(TurnEvent::ReplyToken(reply.clone()));
        protocol::write_event(writer, &WyomingEvent::reply_token(&reply))
            .await
            .ok();
        on_event(TurnEvent::Speaking);
        let mut audio_started = false;
        if let Err(e) = self
            .speak_chunk(writer, runtime, connector, &reply, &mut audio_started, dump)
            .await
        {
            log::warn!("system1 timer: TTS failed ({e:#}); timer started, text still sent");
        }
        emit_follow_up_and_stop(writer, &self.follow_up, audio_started, &reply, followup_depth)
            .await;
        Some(reply)
    }

    /// The System-1 `weather` fast path: fetch the forecast (keyless Open-Meteo), open
    /// the full-screen weather widget *before* speaking, then speak a short templated
    /// summary and invite a follow-up. Skips memory recall + the LLM entirely. Returns
    /// `None` (defer) only on a precondition that fails before anything is emitted (no
    /// provider, no home location, or the fetch errors).
    async fn handle_weather(
        &self,
        runtime: &crate::settings::RuntimeSettings,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        followup_depth: u32,
        on_event: &mut (dyn FnMut(TurnEvent) + Send),
        dump: Option<&TurnAudioDump>,
    ) -> Option<String> {
        let provider = self.weather.as_ref()?; // weather disabled → defer
        let location = runtime
            .household
            .location
            .clone()
            .unwrap_or_default();
        if location.trim().is_empty() {
            log::debug!("system1 weather: no home location configured; deferring to System-2");
            return None;
        }
        let imperial =
            crate::directions::units_are_imperial(runtime.household.weather_units.as_deref());
        let report = match provider.fetch(&location, imperial).await {
            Ok(report) => report,
            Err(e) => {
                log::warn!("system1 weather fetch failed ({e:#}); deferring to System-2");
                return None; // nothing emitted yet — safe to defer
            }
        };

        // Committed: from here we own the turn and must not fall through (that would
        // double-speak). Best-effort writes mirror the normal reply path.
        let reply = weather_summary(&report);
        let (_reader, writer) = device.split_mut();
        // Widget first, so the screen changes the instant the intent resolves.
        protocol::write_event(
            writer,
            &WyomingEvent::weather_show(
                serde_json::to_value(&report).unwrap_or(serde_json::Value::Null),
            ),
        )
        .await
        .ok();
        on_event(TurnEvent::ReplyToken(reply.clone()));
        protocol::write_event(writer, &WyomingEvent::reply_token(&reply))
            .await
            .ok();
        on_event(TurnEvent::Speaking);
        let mut audio_started = false;
        if let Err(e) = self
            .speak_chunk(writer, runtime, connector, &reply, &mut audio_started, dump)
            .await
        {
            log::warn!("system1 weather: TTS failed ({e:#}); widget + text still sent");
        }
        emit_follow_up_and_stop(writer, &self.follow_up, audio_started, &reply, followup_depth)
            .await;
        Some(reply)
    }

    async fn speak_chunk(
        &self,
        writer: &mut DynWrite,
        runtime: &crate::settings::RuntimeSettings,
        connector: &dyn ServiceConnector,
        text: &str,
        audio_started: &mut bool,
        dump: Option<&TurnAudioDump>,
    ) -> Result<bool> {
        let text = sanitize_for_tts(text);
        if text.trim().is_empty() {
            return Ok(true);
        }
        let tts_conn = connector.connect_tts().await?;
        let mut tts = TtsSession::begin(tts_conn, &text, runtime.tts_voice.as_deref()).await?;

        // Piper announces its rate in each `audio-start`; track it so dumped TTS
        // (the far-end reference) is tagged with the right sample rate.
        let mut tts_rate = AudioFormat::default().rate;
        while let Some(ev) = tts.next_audio().await? {
            if ev.event_type == types::AUDIO_START {
                if let Some(fmt) = protocol::audio_format(&ev.data) {
                    tts_rate = fmt.rate;
                }
            }
            if ev.event_type == types::AUDIO_CHUNK {
                if let (Some(d), Some(pcm)) = (dump, ev.payload.as_deref()) {
                    d.push_tts(pcm, tts_rate);
                }
            }
            // Coalesce this sentence's Piper stream into the turn's single
            // device-facing stream: forward exactly one `audio-start`, relay every
            // `audio-chunk`, and swallow the per-sentence `audio-stop` (the turn
            // sends one final `audio-stop`). The device ends its turn on the first
            // `audio-stop` it sees, so a per-sentence stop would truncate the reply.
            let forward = match ev.event_type.as_str() {
                types::AUDIO_START => {
                    if *audio_started {
                        false
                    } else {
                        *audio_started = true;
                        true
                    }
                }
                types::AUDIO_STOP => false,
                _ => true,
            };
            if forward && protocol::write_event(writer, &ev).await.is_err() {
                log::info!("device closed before TTS playback finished; stopping relay");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Synthesize a standalone phrase with Piper and stream it to the device as one
    /// self-contained audio stream (`audio-start` → `audio-chunk`… → `audio-stop`).
    ///
    /// Unlike [`Pipeline::speak_chunk`] this is not part of a voice turn: it answers
    /// an out-of-band `ambient-speak` request (see [`crate::server`]) so an on-device
    /// timer can voice its "Time's up …" announcement in the real Piper voice when
    /// the Mac is reachable. Best-effort — a device that drops the socket mid-stream
    /// is a graceful stop, not an error.
    pub async fn announce(
        &self,
        device: &mut DynConnection,
        connector: &dyn ServiceConnector,
        text: &str,
    ) -> Result<()> {
        let runtime = self.settings.snapshot();
        let (_reader, writer) = device.split_mut();
        let mut audio_started = false;
        self.speak_chunk(writer, &runtime, connector, text, &mut audio_started, None)
            .await?;
        // `speak_chunk` swallows Piper's per-request `audio-stop`; send the single
        // terminating stop so the device ends playback (only if a stream was opened).
        if audio_started {
            let _ = protocol::write_event(writer, &WyomingEvent::audio_stop(0)).await;
        }
        Ok(())
    }
}

/// The memory scope for a speaker: `None` (shared/household) for the sentinel
/// household context, else the concrete `speaker_id`.
fn speaker_scope(speaker: &SpeakerContext) -> Option<&str> {
    if speaker.is_household() {
        None
    } else {
        Some(speaker.speaker_id.as_str())
    }
}

/// The system-prompt line telling the model who it is speaking with. When the
/// identified speaker's name matches a [`Household`] member (speaker_id_plan.md
/// reconciliation), the line notes that they live here — and their relationship when
/// one is recorded — so the model has the right context for that specific person.
fn speaker_identity_line(speaker: &SpeakerContext, household: &Household) -> String {
    match &speaker.name {
        Some(name) => match household.member_matching(name) {
            // Prefer the household's canonical name (correct spelling/capitalization)
            // over the raw voiceprint label — that's the point of the roster.
            Some(m) => {
                let canonical = m.name.trim();
                match m
                    .relationship
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    Some(rel) => {
                        format!("You are speaking with {canonical} ({rel}), who lives here.")
                    }
                    None => format!("You are speaking with {canonical}, who lives here."),
                }
            }
            None => format!("You are speaking with {name}."),
        },
        None if speaker.is_household() => {
            "You are speaking with a member of the household.".to_string()
        }
        None => "You are speaking with a household member you haven't been introduced to yet. \
                 If they tell you their name, greet them by it."
            .to_string(),
    }
}

/// Append little-endian PCM16 bytes to an `i16` sample buffer (a trailing odd byte,
/// never expected from a well-formed frame, is ignored).
fn append_pcm_i16_le(out: &mut Vec<i16>, pcm: &[u8]) {
    out.reserve(pcm.len() / 2);
    for c in pcm.chunks_exact(2) {
        out.push(i16::from_le_bytes([c[0], c[1]]));
    }
}

/// Watch the device→orchestrator half of the socket for a **barge-in** while the
/// assistant is replying. Resolves (ending the race in [`Pipeline::respond_and_speak`])
/// when the device sends an `ambient-interrupt` frame or closes the socket; any
/// other stray frame during playback (e.g. the device's own `audio-stop` from the
/// STREAMING→SPEAKING edge) is consumed and ignored so it can't be mistaken for a
/// barge-in.
///
/// This is a single long-lived future (never re-created inside a `select!` arm), so
/// it is cancellation-safe: whenever it is dropped it is parked on a fresh
/// `read_event` with no partially-consumed frame.
async fn watch_for_barge_in(reader: &mut DynRead) {
    loop {
        match protocol::read_event(reader).await {
            Ok(Some(ev)) if ev.is_interrupt() => return,
            Ok(Some(_)) => continue, // stray frame during playback — ignore
            Ok(None) => return,      // device closed the socket
            Err(e) => {
                log::debug!("barge-in watcher read error (treating as disconnect): {e:#}");
                return;
            }
        }
    }
}

/// The longest run of text (in characters) we let accumulate without terminal
/// punctuation before flushing it to TTS anyway. Bounds time-to-first-audio on a
/// long unpunctuated clause.
const TTS_MAX_CHUNK_CHARS: usize = 240;

/// Pull the next speakable chunk out of `pending`, draining it from the buffer.
///
/// A chunk is a complete sentence — text up to and including a `.`/`!`/`?` or a
/// newline — or, to bound latency on a long clause that never terminates, the
/// leading [`TTS_MAX_CHUNK_CHARS`] broken at the last whitespace. When `flush` is
/// set (end of the token stream) the entire remaining buffer is returned. Returns
/// `None` when there is nothing complete to speak yet.
fn take_speakable(pending: &mut String, flush: bool) -> Option<String> {
    if flush {
        let rest = pending.trim().to_string();
        pending.clear();
        return if rest.is_empty() { None } else { Some(rest) };
    }

    // End of the first sentence, if one has arrived.
    let boundary = pending
        .char_indices()
        .find(|(_, ch)| matches!(ch, '.' | '!' | '?' | '\n'))
        .map(|(i, ch)| i + ch.len_utf8());

    let cut = match boundary {
        Some(b) => b,
        None => {
            if pending.chars().count() < TTS_MAX_CHUNK_CHARS {
                return None;
            }
            // Overlong unterminated clause: break at the last space/newline within
            // the cap (both are single-byte, so `+ 1` stays on a char boundary).
            let cap = pending
                .char_indices()
                .nth(TTS_MAX_CHUNK_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(pending.len());
            pending[..cap]
                .rfind([' ', '\n'])
                .map(|w| w + 1)
                .unwrap_or(cap)
        }
    };

    let chunk: String = pending.drain(..cut).collect();
    let chunk = chunk.trim().to_string();
    if chunk.is_empty() {
        // The drained span was only whitespace/punctuation — try the next boundary.
        return take_speakable(pending, false);
    }
    Some(chunk)
}

/// Strip Markdown decoration characters an LLM may emit (`* _ ` # ~`) so Piper does
/// not read them aloud (e.g. "asterisk asterisk"). Kept deliberately light: it only
/// removes standalone decoration glyphs, leaving words and punctuation intact.
fn sanitize_for_tts(text: &str) -> String {
    text.chars()
        .filter(|c| !matches!(c, '*' | '`' | '#' | '_' | '~'))
        .collect()
}

/// A one-line statement of the current local date + time + timezone, injected into
/// the LLM system prompt each turn so the model answers time/date questions from
/// fact instead of hallucinating (it has no clock of its own). Example:
/// `Current date and time: Monday, 16 September 2026, 6:36 PM EDT (UTC-04:00).`
///
/// Phrased *non-leadingly*: the model must only use it when the user actually asks
/// about the time/date, and must not volunteer it otherwise. Without that guard the
/// prominently-stated clock became the most salient fact in context, so on a vague or
/// mis-transcribed request the model would default to reciting the time.
fn current_datetime_line() -> String {
    let now = chrono::Local::now();
    format!(
        "For reference, the current date and time is {} ({}). Use this only to answer \
         questions that are explicitly about the time, date, or day of week; do not \
         mention the time or date otherwise, and never bring it up on your own.",
        now.format("%A, %-d %B %Y, %-I:%M %p %Z"),
        now.format("UTC%:z"),
    )
}

/// A system-prompt line grounding the assistant in the device's physical location
/// (and preferred units), so location-relative questions — weather, sunset, nearby
/// places — resolve an unqualified "here" to the configured home rather than the
/// model guessing. Returns `None` when no home location is configured, so the prompt
/// omits the line entirely. Phrased non-leadingly (like the clock line) so it only
/// matters when the user actually asks something location-relative.
fn location_line(location: Option<&str>, units: Option<&str>) -> Option<String> {
    let location = location.map(str::trim).filter(|s| !s.is_empty())?;
    let mut line = format!(
        "This device is located in {location}. When the user asks about local conditions \
         (weather, sunset, nearby places) without naming a place, assume {location}. Do \
         not mention the location otherwise."
    );
    if let Some(units) = units.map(str::trim).filter(|s| !s.is_empty()) {
        line.push_str(&format!(
            " Prefer {units} units for temperatures and measurements."
        ));
    }
    Some(line)
}

/// A system-prompt block naming the people who live in the home (the canonical
/// household directory), so the model can address them correctly and has their
/// contact details on hand when a request needs one. Phrased non-leadingly (like the
/// clock/location lines): it must only be used when the user actually asks something
/// that needs it, never volunteered. Returns `None` when the roster is empty so the
/// prompt omits the block entirely.
fn household_line(members: &[HouseholdMember]) -> Option<String> {
    let named: Vec<&HouseholdMember> = members
        .iter()
        .filter(|m| !m.name.trim().is_empty())
        .collect();
    if named.is_empty() {
        return None;
    }
    let mut block = String::from(
        "The people who live in this home (use only when a request needs to know who \
         is here or how to reach them; do not recite this otherwise):",
    );
    for m in named {
        let mut detail = Vec::new();
        if let Some(rel) = m
            .relationship
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            detail.push(rel.to_string());
        }
        if !m.emails.is_empty() {
            detail.push(format!("email {}", m.emails.join(", ")));
        }
        if !m.phones.is_empty() {
            detail.push(format!("phone {}", m.phones.join(", ")));
        }
        if detail.is_empty() {
            block.push_str(&format!("\n- {}", m.name.trim()));
        } else {
            block.push_str(&format!("\n- {} ({})", m.name.trim(), detail.join("; ")));
        }
    }
    Some(block)
}

/// A prompt line describing what the display is currently showing (its "display
/// context"), so the model can drive that screen by voice. Dispatches per screen kind;
/// add an arm here (plus a device-side setter and a `DisplayContext` variant) when a new
/// screen — music, weather, photos — wants voice control.
fn display_context_line(ctx: &protocol::DisplayContext) -> String {
    match ctx {
        protocol::DisplayContext::Recipe(screen) => recipe_screen_line(screen),
        protocol::DisplayContext::Weather(screen) => weather_screen_line(screen),
    }
}

/// The prompt line for the weather screen: names what the forecast currently shows, so
/// the model can answer follow-ups in context ("what about tomorrow" → `weather_lookup`
/// for the same place) or `close_weather` when the user says to close it.
fn weather_screen_line(screen: &protocol::WeatherScreen) -> String {
    let unit = if screen.units == "imperial" { "F" } else { "C" };
    let place = screen.location.trim();
    let at = if place.is_empty() {
        String::new()
    } else {
        format!(" for {place}")
    };
    let desc = screen.description.trim();
    let conditions = if desc.is_empty() {
        String::new()
    } else {
        format!(", currently {desc} at {}°{unit}", screen.temp)
    };
    format!(
        "The weather screen is currently open on the display, showing the forecast{at}{conditions}. \
         Use the `weather_lookup` tool to update it (e.g. another day or place) and \
         `close_weather` to close it."
    )
}

/// The prompt line for the recipe screen: names the active tab and whether the pane is
/// scrolled to the top/bottom, so the model can decide whether a `recipe_control`
/// (switch tab / scroll) or `close_recipe` call is useful.
fn recipe_screen_line(screen: &protocol::RecipeScreen) -> String {
    let tab = match screen.tab.as_str() {
        "ingredients" => "Ingredients",
        "steps" => "Steps",
        _ => "Overview",
    };
    let title = screen.title.trim();
    let dish = if title.is_empty() {
        "a recipe".to_string()
    } else {
        format!("the recipe for \"{title}\"")
    };
    let position = if screen.at_top && screen.at_bottom {
        " The whole tab fits on screen."
    } else if screen.at_top {
        " It is scrolled to the top."
    } else if screen.at_bottom {
        " It is scrolled to the bottom."
    } else {
        " It is scrolled partway."
    };
    format!(
        "The recipe screen is currently open on the display, showing {dish} \
         ({ing} ingredients, {steps} steps) on the {tab} tab.{position} Use the \
         `recipe_control` tool to switch tabs or scroll it, or `close_recipe` to close it.",
        ing = screen.ingredient_count,
        steps = screen.step_count,
    )
}

/// Drain every currently-pending [`DeviceAction`] from the per-turn channel and
/// relay it to the device as an `ambient-timer` frame on `writer`. Non-blocking: it
/// only takes actions already queued (a tool's `invoke` runs synchronously during
/// the LLM turn, so its action is enqueued before we get here). Write failures are
/// swallowed — the device dropping mid-turn is handled by the surrounding turn logic.
/// Close out a spoken turn: send the follow-up `anamanti-listen` frame (when enabled,
/// audio was produced, and the chain cap allows it) **before** the final `audio-stop`
/// (the device ends its turn on the first stop). Shared by the normal reply path and the
/// System-1 fast path so both invite follow-ups identically.
async fn emit_follow_up_and_stop<W>(
    writer: &mut W,
    follow_up: &FollowUpConfig,
    audio_started: bool,
    reply: &str,
    followup_depth: u32,
) where
    W: tokio::io::AsyncWrite + Unpin,
{
    let within_cap = follow_up.max_chain == 0 || followup_depth < follow_up.max_chain;
    if follow_up.enabled && audio_started && within_cap {
        let next_depth = followup_depth + 1;
        let wait_secs = if reply_is_question(reply) {
            follow_up.question_wait_secs
        } else {
            follow_up.reply_wait_secs
        };
        log::info!("follow-up: asking device to listen for {wait_secs}s (depth {next_depth})");
        protocol::write_event(writer, &WyomingEvent::listen(next_depth, wait_secs))
            .await
            .ok();
    }
    // Close the single coalesced device-facing audio stream.
    if audio_started {
        protocol::write_event(writer, &WyomingEvent::audio_stop(0))
            .await
            .ok();
    }
}

/// A speakable duration, e.g. `600` → "10 minutes", `5400` → "1 hour 30 minutes",
/// `90` → "1 minute 30 seconds". Used in the System-1 timer confirmation.
fn human_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    let mut parts = Vec::new();
    let unit = |n: u64, singular: &str| {
        if n == 1 {
            format!("1 {singular}")
        } else {
            format!("{n} {singular}s")
        }
    };
    if h > 0 {
        parts.push(unit(h, "hour"));
    }
    if m > 0 {
        parts.push(unit(m, "minute"));
    }
    if s > 0 {
        parts.push(unit(s, "second"));
    }
    if parts.is_empty() {
        return "0 seconds".to_string();
    }
    parts.join(" ")
}

/// A short, speakable one-line summary of a forecast for the System-1 weather fast path
/// (the full detail is on the widget). Metric/imperial follows the report's units.
fn weather_summary(report: &crate::weather::WeatherReport) -> String {
    let deg = if report.units == "imperial" {
        "°F"
    } else {
        "°C"
    };
    let c = &report.current;
    let lead = {
        let d = c.description.trim();
        if d.is_empty() {
            String::new()
        } else {
            format!("{d}, ")
        }
    };
    format!(
        "{lead}it's {}{deg} in {}. Today's high is {}{deg}, low {}{deg}.",
        c.temp, report.location_label, c.high, c.low
    )
}

async fn drain_device_actions<W>(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<DeviceAction>,
    writer: &mut W,
) where
    W: tokio::io::AsyncWrite + Unpin,
{
    while let Ok(action) = rx.try_recv() {
        let event = match action {
            DeviceAction::StartTimer {
                label,
                duration_secs,
            } => WyomingEvent::timer_start(duration_secs, label.as_deref()),
            DeviceAction::CancelTimer { label } => WyomingEvent::timer_cancel(label.as_deref()),
            DeviceAction::ShowRecipe(recipe) => WyomingEvent::recipe(
                serde_json::to_value(&recipe).unwrap_or(serde_json::Value::Null),
            ),
            DeviceAction::DismissRecipe => WyomingEvent::recipe_dismiss(),
            DeviceAction::ShowWeather(report) => WyomingEvent::weather_show(
                serde_json::to_value(&report).unwrap_or(serde_json::Value::Null),
            ),
            DeviceAction::DismissWeather => WyomingEvent::weather_dismiss(),
            DeviceAction::RecipeControl(nav) => match nav {
                RecipeNav::TabOverview => WyomingEvent::recipe_navigate("overview"),
                RecipeNav::TabIngredients => WyomingEvent::recipe_navigate("ingredients"),
                RecipeNav::TabSteps => WyomingEvent::recipe_navigate("steps"),
                RecipeNav::ScrollUp => WyomingEvent::recipe_scroll("up"),
                RecipeNav::ScrollDown => WyomingEvent::recipe_scroll("down"),
                RecipeNav::ScrollTop => WyomingEvent::recipe_scroll("top"),
                RecipeNav::ScrollBottom => WyomingEvent::recipe_scroll("bottom"),
            },
        };
        if let Err(e) = protocol::write_event(writer, &event).await {
            log::warn!("failed to relay device action to the device: {e:#}");
            break;
        }
    }
}

/// Whether a fully-assembled reply should trigger a follow-up listen: its last
/// meaningful character is a question mark. Trailing whitespace and a closing quote
/// or bracket are ignored (so `... right?"` still counts). A deliberately simple v1
/// heuristic — see plans/Plan.MD (Follow-up listening) for the rationale and known
/// misfires (e.g. rhetorical questions).
fn reply_is_question(reply: &str) -> bool {
    let trimmed = reply.trim_end_matches(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '»')
    });
    trimmed.ends_with('?') || trimmed.ends_with('？')
}

/// Root-mean-square amplitude (in `i16` units) of a little-endian PCM16 buffer,
/// used by the turn's energy VAD to tell speech from room noise. A trailing odd
/// byte (never expected from a well-formed frame) is ignored.
fn rms_i16_le(pcm: &[u8]) -> f64 {
    let mut sum_sq = 0f64;
    let mut n = 0u64;
    for c in pcm.chunks_exact(2) {
        let s = i16::from_le_bytes([c[0], c[1]]) as f64;
        sum_sq += s * s;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// Wall-clock duration of one i16-LE mono PCM chunk of `byte_len` bytes at
/// `sample_rate` Hz. Zero when the rate is unknown so it never contributes to the
/// onset debounce (see [`voiced_onset_step`]).
fn chunk_duration(byte_len: usize, sample_rate: u32) -> Duration {
    if sample_rate == 0 {
        return Duration::ZERO;
    }
    // 2 bytes per i16 sample.
    Duration::from_secs_f64((byte_len / 2) as f64 / sample_rate as f64)
}

/// One step of the speech-onset debounce. Folds the current chunk into the running
/// span of *consecutive* voiced audio and reports whether that span has now reached
/// [`MIN_SPEECH_ONSET`] (i.e. `speech_started` should latch). A non-voiced chunk
/// resets the run to zero, so only sustained voice — not a lone transient like the
/// residual TTS tail after a follow-up mic reopens — flips the turn out of its
/// no-speech window. Pure (takes no clock) so the debounce is unit-testable.
///
/// Returns `(updated_run, latched_now)`. Callers only invoke this while
/// `speech_started` is still false; once latched it stays latched.
fn voiced_onset_step(voiced: bool, chunk_dur: Duration, voiced_run: Duration) -> (Duration, bool) {
    if !voiced {
        return (Duration::ZERO, false);
    }
    let run = voiced_run.saturating_add(chunk_dur);
    (run, run >= MIN_SPEECH_ONSET)
}

#[cfg(test)]
mod location_tests {
    use super::location_line;

    #[test]
    fn grounds_configured_home_and_units() {
        let line = location_line(Some("Austin, Texas"), Some("imperial")).unwrap();
        assert!(line.contains("Austin, Texas"));
        assert!(line.contains("imperial"));
    }

    #[test]
    fn omits_units_when_unset_but_keeps_location() {
        let line = location_line(Some("Paris"), None).unwrap();
        assert!(line.contains("Paris"));
        assert!(!line.to_lowercase().contains("units"));
    }

    #[test]
    fn no_location_yields_no_line() {
        assert!(location_line(None, Some("metric")).is_none());
        assert!(location_line(Some("   "), None).is_none());
    }
}

#[cfg(test)]
mod household_tests {
    use super::{household_line, speaker_identity_line};
    use crate::settings::{Household, HouseholdMember};
    use crate::speaker::SpeakerContext;

    fn member(name: &str, emails: &[&str], phones: &[&str], rel: Option<&str>) -> HouseholdMember {
        HouseholdMember {
            name: name.to_string(),
            emails: emails.iter().map(|s| s.to_string()).collect(),
            phones: phones.iter().map(|s| s.to_string()).collect(),
            relationship: rel.map(str::to_string),
        }
    }

    fn named_speaker(name: &str) -> SpeakerContext {
        SpeakerContext {
            speaker_id: "spk-1".to_string(),
            name: Some(name.to_string()),
            is_new: false,
            confidence: 0.9,
        }
    }

    #[test]
    fn empty_roster_yields_no_block() {
        assert!(household_line(&[]).is_none());
        // A single nameless entry is also nothing to say.
        assert!(household_line(&[member("  ", &["x@y.z"], &[], None)]).is_none());
    }

    #[test]
    fn lists_people_with_their_contact_details() {
        let line = household_line(&[
            member(
                "Alice",
                &["alice@example.com"],
                &["+1 555 0001"],
                Some("parent"),
            ),
            member("Bob", &[], &[], None),
        ])
        .unwrap();
        assert!(line.contains("Alice"));
        assert!(line.contains("alice@example.com"));
        assert!(line.contains("+1 555 0001"));
        assert!(line.contains("parent"));
        // Bob has no details, so he appears as a bare name (no empty parens).
        assert!(line.contains("- Bob"));
        assert!(!line.contains("Bob ()"));
    }

    #[test]
    fn identity_line_reconciles_a_named_speaker_with_the_roster() {
        let hh = Household {
            members: vec![member("Alice", &[], &[], Some("parent"))],
            ..Default::default()
        };
        // Case-insensitive match on the voiceprint's name links to the member.
        let line = speaker_identity_line(&named_speaker("alice"), &hh);
        assert!(line.contains("Alice"), "keeps the spoken-to name: {line}");
        assert!(line.contains("parent"), "notes the relationship: {line}");
        assert!(
            line.contains("lives here"),
            "notes they're a resident: {line}"
        );
    }

    #[test]
    fn identity_line_for_a_matched_member_without_a_relationship() {
        let hh = Household {
            members: vec![member("Bob", &[], &[], None)],
            ..Default::default()
        };
        let line = speaker_identity_line(&named_speaker("Bob"), &hh);
        assert!(line.contains("Bob") && line.contains("lives here"));
        assert!(
            !line.contains('('),
            "no empty parens without a relationship: {line}"
        );
    }

    #[test]
    fn identity_line_stays_plain_for_a_non_member() {
        let line = speaker_identity_line(&named_speaker("Zoe"), &Household::default());
        assert_eq!(line, "You are speaking with Zoe.");
    }
}

#[cfg(test)]
mod vad_tests {
    use super::{chunk_duration, rms_i16_le, voiced_onset_step, MIN_SPEECH_ONSET};
    use std::time::Duration;

    fn pcm(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(rms_i16_le(&pcm(&[0, 0, 0, 0])), 0.0);
        assert_eq!(rms_i16_le(&[]), 0.0);
    }

    #[test]
    fn rms_tracks_amplitude() {
        // A constant ±1000 signal has RMS 1000; loud speech reads far above the
        // 120-unit voice threshold while a quiet ±30 noise floor stays below it.
        assert!((rms_i16_le(&pcm(&[1000, -1000, 1000, -1000])) - 1000.0).abs() < 1e-6);
        assert!(rms_i16_le(&pcm(&[30, -30, 25, -20])) < 120.0);
        assert!(rms_i16_le(&pcm(&[800, -600, 700, -900])) > 120.0);
    }

    #[test]
    fn chunk_duration_is_bytes_over_rate() {
        // 640 bytes = 320 i16 samples; at 16 kHz that is 20 ms.
        assert_eq!(chunk_duration(640, 16_000), Duration::from_millis(20));
        // Unknown rate contributes nothing to the onset (avoids div-by-zero).
        assert_eq!(chunk_duration(640, 0), Duration::ZERO);
    }

    #[test]
    fn onset_needs_sustained_voice_to_latch() {
        // Feed 20 ms voiced chunks; the onset must not latch until the accumulated
        // run reaches MIN_SPEECH_ONSET.
        let chunk = Duration::from_millis(20);
        let mut run = Duration::ZERO;
        let mut latched = false;
        let mut chunks = 0;
        while !latched {
            (run, latched) = voiced_onset_step(true, chunk, run);
            chunks += 1;
            assert!(chunks < 1000, "onset never latched");
        }
        // 250 ms / 20 ms = 13 chunks (12 chunks = 240 ms is still below threshold).
        assert_eq!(chunks, (MIN_SPEECH_ONSET.as_millis() as u64).div_ceil(20));
        assert!(run >= MIN_SPEECH_ONSET);
    }

    #[test]
    fn lone_transient_does_not_latch_and_resets() {
        // The production failure: one short voiced blip (the residual TTS tail after a
        // follow-up mic reopens) followed by silence. It must never latch speech, and
        // the run must reset so the no-speech window stays open.
        let (run, latched) = voiced_onset_step(true, Duration::from_millis(20), Duration::ZERO);
        assert!(!latched, "a single 20 ms blip must not count as speech");
        assert!(run < MIN_SPEECH_ONSET);

        // The following non-voiced chunk clears the run entirely.
        let (run, latched) = voiced_onset_step(false, Duration::from_millis(20), run);
        assert!(!latched);
        assert_eq!(run, Duration::ZERO, "non-voiced chunk resets the onset run");
    }

    #[test]
    fn gap_before_threshold_resets_progress() {
        // Voiced runs that are broken by silence before reaching the threshold must
        // restart from zero, so intermittent transients can't accumulate into a latch.
        let chunk = Duration::from_millis(100);
        let (run, latched) = voiced_onset_step(true, chunk, Duration::ZERO); // 100 ms
        assert!(!latched);
        let (run, _) = voiced_onset_step(false, chunk, run); // reset
        assert_eq!(run, Duration::ZERO);
        let (run, latched) = voiced_onset_step(true, chunk, run); // 100 ms again, not 200
        assert!(!latched);
        assert_eq!(run, Duration::from_millis(100));
    }
}

#[cfg(test)]
mod segmenter_tests {
    use super::{sanitize_for_tts, take_speakable, TTS_MAX_CHUNK_CHARS};

    #[test]
    fn waits_for_a_complete_sentence() {
        let mut p = String::from("Hello there");
        assert_eq!(take_speakable(&mut p, false), None, "no terminator yet");
        assert_eq!(p, "Hello there", "buffer untouched");
    }

    #[test]
    fn flushes_each_complete_sentence_in_order() {
        let mut p = String::from("First one. Second one! Third?");
        assert_eq!(take_speakable(&mut p, false).as_deref(), Some("First one."));
        assert_eq!(
            take_speakable(&mut p, false).as_deref(),
            Some("Second one!")
        );
        assert_eq!(take_speakable(&mut p, false).as_deref(), Some("Third?"));
        assert_eq!(take_speakable(&mut p, false), None);
        assert!(p.is_empty());
    }

    #[test]
    fn splits_on_newlines_too() {
        let mut p = String::from("Line one\nleftover");
        assert_eq!(take_speakable(&mut p, false).as_deref(), Some("Line one"));
        assert_eq!(take_speakable(&mut p, false), None);
        assert_eq!(p, "leftover");
    }

    #[test]
    fn flush_returns_trailing_clause() {
        let mut p = String::from("no terminator here");
        assert_eq!(take_speakable(&mut p, false), None);
        assert_eq!(
            take_speakable(&mut p, true).as_deref(),
            Some("no terminator here")
        );
        assert_eq!(take_speakable(&mut p, true), None, "empty flush is None");
    }

    #[test]
    fn overlong_unterminated_clause_is_broken_at_whitespace() {
        // A long run with no sentence terminator flushes near the cap at a space
        // boundary, never mid-word, so time-to-first-audio stays bounded.
        let word = "word ";
        let mut p = word.repeat(TTS_MAX_CHUNK_CHARS); // far longer than the cap
        let chunk = take_speakable(&mut p, false).expect("caps long clause");
        assert!(chunk.chars().count() <= TTS_MAX_CHUNK_CHARS);
        assert!(chunk.starts_with("word"));
        assert!(!chunk.ends_with("wor"), "must not split mid-word");
    }

    #[test]
    fn sanitize_strips_markdown_decoration() {
        assert_eq!(
            sanitize_for_tts("**bold** and _em_ and `code`"),
            "bold and em and code"
        );
        assert_eq!(sanitize_for_tts("# Heading"), " Heading");
        assert_eq!(sanitize_for_tts("plain text, ok."), "plain text, ok.");
    }
}
