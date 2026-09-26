//! End-to-end pipeline integration tests (Plan.MD Phase 4).
//!
//! These wire the orchestrator against in-memory mock Whisper (STT) and Piper
//! (TTS) servers plus a mock Echo Show, and drive a full turn through the real
//! [`Pipeline`] — proving STT relay, memory policy, LLM reply, and TTS playback
//! interoperate over the exact Wyoming wire format the device speaks, without any
//! external services.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::{split, BufReader, DuplexStream};

use anamanti_core::config::FollowUpConfig;
use anamanti_core::llm::mock::MockLlm;
use anamanti_core::llm::{LlmBackend, LlmTurn, ReplyStream};
use anamanti_core::memory::chatlog::now_secs;
use anamanti_core::memory::{ChatLog, ChatLogRecord, MemoryKind, MemoryStore};
use anamanti_core::orchestrator::{Pipeline, ServiceConnector, TurnEvent};
use anamanti_core::speaker::{
    MockSpeakerEmbedder, SpeakerContext, SpeakerRegistry, SpeakerService, SpeakerThresholds,
};
use anamanti_core::wyoming::protocol::{read_event, types, write_event, AudioFormat, WyomingEvent};
use anamanti_core::wyoming::DynConnection;
use serde_json::json;

/// A connector backed by in-process mock STT/TTS servers over duplex pipes.
struct MockConnector {
    transcript: String,
    /// Captures, in order, each chunk of text the TTS server was asked to
    /// synthesize. With streaming TTS a turn produces one entry per spoken
    /// sentence; a single-chunk reply produces exactly one entry.
    synthesized: Arc<Mutex<Vec<String>>>,
}

impl MockConnector {
    fn new(transcript: &str) -> Self {
        Self {
            transcript: transcript.to_string(),
            synthesized: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ServiceConnector for MockConnector {
    async fn connect_stt(&self) -> Result<DynConnection> {
        let (client, server) = tokio::io::duplex(64 * 1024);
        let transcript = self.transcript.clone();
        tokio::spawn(mock_stt(server, transcript));
        let (r, w) = split(client);
        Ok(DynConnection::from_io(r, w))
    }

    async fn connect_tts(&self) -> Result<DynConnection> {
        let (client, server) = tokio::io::duplex(64 * 1024);
        // One connect per spoken chunk (streaming TTS); each pushes its text.
        let captured = self.synthesized.clone();
        tokio::spawn(mock_tts(server, captured));
        let (r, w) = split(client);
        Ok(DynConnection::from_io(r, w))
    }
}

/// Mock Whisper: reads `transcribe`/`audio-start`, and once the first forwarded
/// `audio-chunk` arrives (server-side VAD firing) emits the final `transcript`.
async fn mock_stt(server: DuplexStream, transcript: String) {
    let (r, w) = split(server);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    while let Ok(Some(ev)) = read_event(&mut reader).await {
        if ev.event_type == types::AUDIO_CHUNK {
            let _ = write_event(&mut writer, &WyomingEvent::transcript(&transcript)).await;
            break;
        }
    }
    // Stay alive (draining) until the orchestrator drops the connection.
    while let Ok(Some(_)) = read_event(&mut reader).await {}
}

/// Mock Piper: reads the `synthesize` request, records the text, and emits a short
/// audio stream (`audio-start` → one `audio-chunk` → `audio-stop`).
async fn mock_tts(server: DuplexStream, captured: Arc<Mutex<Vec<String>>>) {
    let (r, w) = split(server);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    if let Ok(Some(ev)) = read_event(&mut reader).await {
        if ev.event_type == types::SYNTHESIZE {
            let text = ev
                .data
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string();
            captured.lock().unwrap().push(text);
            let fmt = AudioFormat::PCM_16K_MONO;
            for e in [
                WyomingEvent::audio_start(fmt, 0),
                WyomingEvent::audio_chunk(fmt, 0, vec![0u8; 8]),
                WyomingEvent::audio_stop(20),
            ] {
                let _ = write_event(&mut writer, &e).await;
            }
        }
    }
}

/// Drive the device side of a turn: stream audio, then read the relayed transcript,
/// the streamed reply tokens, and the returned TTS audio frames. Returns
/// `(transcript_seen, reply_text, tts_event_kinds)`.
async fn drive_device(io: DuplexStream) -> (String, String, Vec<String>) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    write_event(&mut writer, &WyomingEvent::audio_start(fmt, 0))
        .await
        .unwrap();
    write_event(
        &mut writer,
        // Voiced audio (clears the energy gate for MIN_SPEECH_ONSET) so the
        // orchestrator latches `speech_started` and accepts the transcript. A
        // sub-threshold chunk would now be treated as a no-speech finalize and the
        // transcript discarded as a likely STT hallucination (see stream_to_transcript).
        &WyomingEvent::audio_chunk(fmt, 0, voiced_chunk_bytes(180.0, 1500)),
    )
    .await
    .unwrap();
    write_event(
        &mut writer,
        &WyomingEvent::audio_chunk(fmt, 20, vec![3, 0, 4, 0]),
    )
    .await
    .unwrap();

    // The relayed transcript arrives first.
    let ev = read_event(&mut reader).await.unwrap().unwrap();
    let transcript = ev.transcript_text().unwrap_or_default().to_string();
    // Real device ends its input stream once it has the transcript.
    write_event(&mut writer, &WyomingEvent::audio_stop(40))
        .await
        .unwrap();

    // Then the streamed reply tokens (Phase 5) arrive, followed by the synthesized
    // reply audio. Accumulate the reply text and collect the TTS frame kinds.
    let mut reply = String::new();
    let mut kinds = Vec::new();
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if let Some(tok) = ev.reply_token_text() {
            reply.push_str(tok);
            continue;
        }
        // Every reply now ends with a follow-up-listen frame (before the audio-stop);
        // it isn't part of the TTS stream this helper reports on, so skip it.
        if ev.event_type == types::LISTEN {
            continue;
        }
        let t = ev.event_type.clone();
        kinds.push(t.clone());
        if t == types::AUDIO_STOP {
            break;
        }
    }
    (transcript, reply, kinds)
}

/// Like [`drive_device`] but drains every frame until the pipeline closes the
/// socket. Returns `(transcript, reply_text, audio_start_count, audio_stop_count)`.
/// The device-facing audio is coalesced into a single stream, so a well-formed
/// turn yields exactly one `audio-start` and one `audio-stop` however many
/// sentences were synthesized.
async fn drive_device_until_close(io: DuplexStream) -> (String, String, usize, usize) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    write_event(&mut writer, &WyomingEvent::audio_start(fmt, 0))
        .await
        .unwrap();
    write_event(
        &mut writer,
        // Voiced audio (clears the energy gate for MIN_SPEECH_ONSET) so the
        // orchestrator latches `speech_started` and accepts the transcript. A
        // sub-threshold chunk would now be treated as a no-speech finalize and the
        // transcript discarded as a likely STT hallucination (see stream_to_transcript).
        &WyomingEvent::audio_chunk(fmt, 0, voiced_chunk_bytes(180.0, 1500)),
    )
    .await
    .unwrap();

    let ev = read_event(&mut reader).await.unwrap().unwrap();
    let transcript = ev.transcript_text().unwrap_or_default().to_string();
    write_event(&mut writer, &WyomingEvent::audio_stop(40))
        .await
        .unwrap();

    let mut reply = String::new();
    let mut audio_starts = 0;
    let mut audio_stops = 0;
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if let Some(tok) = ev.reply_token_text() {
            reply.push_str(tok);
        } else if ev.event_type == types::AUDIO_START {
            audio_starts += 1;
        } else if ev.event_type == types::AUDIO_STOP {
            audio_stops += 1;
        }
    }
    (transcript, reply, audio_starts, audio_stops)
}

/// Drive the device side of a *silent* turn: stream room-noise-level audio that
/// never clears the energy gate, then drain the relayed frames until the socket
/// closes. Returns `(relayed_transcript, reply_text, saw_listen)`.
async fn drive_device_silent(io: DuplexStream) -> (String, String, bool) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    write_event(&mut writer, &WyomingEvent::audio_start(fmt, 0))
        .await
        .unwrap();
    // Sub-threshold audio: below `voice_rms_threshold`, so the orchestrator's VAD
    // never latches speech and this finalizes as a no-speech turn — even though the
    // mock STT still "hears" (hallucinates) a transcript from it.
    write_event(
        &mut writer,
        &WyomingEvent::audio_chunk(fmt, 0, vec![1, 0, 2, 0]),
    )
    .await
    .unwrap();

    let mut transcript = String::new();
    let mut reply = String::new();
    let mut saw_listen = false;
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if ev.is_transcript() {
            transcript = ev.transcript_text().unwrap_or_default().to_string();
            continue;
        }
        if let Some(tok) = ev.reply_token_text() {
            reply.push_str(tok);
            continue;
        }
        if ev.event_type == types::LISTEN {
            saw_listen = true;
        }
        // The orchestrator ends a silent turn with an `audio-stop` (and no
        // `ambient-listen`); that's our cue that the turn is over.
        if ev.event_type == types::AUDIO_STOP {
            break;
        }
    }
    (transcript, reply, saw_listen)
}

fn build_pipeline(memory: Arc<MemoryStore>) -> Pipeline {
    Pipeline::new(
        Arc::new(MockLlm::new("You said: {msg}")),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
}

async fn run_one_turn(
    pipeline: &Pipeline,
    connector: &MockConnector,
) -> ((String, String, Vec<String>), Vec<TurnEvent>) {
    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);

    let device_task = tokio::spawn(drive_device(dev_test));

    let mut events = Vec::new();
    {
        let mut on_event = |e: TurnEvent| events.push(e);
        pipeline
            .run_turn(&mut device, connector, &mut on_event)
            .await
            .expect("turn runs");
    }

    let device_out = device_task.await.unwrap();
    (device_out, events)
}

/// An out-of-band `ambient-speak` announcement (a fired timer's "Time's up …")
/// synthesizes the phrase with Piper and streams one self-contained audio stream
/// (`audio-start` → `audio-chunk`… → `audio-stop`) back to the device, outside any
/// voice turn.
#[tokio::test]
async fn announce_synthesizes_and_streams_audio_to_device() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory);
    let connector = MockConnector::new(""); // transcript unused for announce

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);

    // Device side: collect the streamed announcement audio frame kinds until stop.
    let reader_task = tokio::spawn(async move {
        let (r, _w) = split(dev_test);
        let mut reader = BufReader::new(r);
        let mut kinds = Vec::new();
        while let Ok(Some(ev)) = read_event(&mut reader).await {
            let kind = ev.event_type.clone();
            kinds.push(kind.clone());
            if kind == types::AUDIO_STOP {
                break;
            }
        }
        kinds
    });

    pipeline
        .announce(&mut device, &connector, "Time's up for pasta")
        .await
        .expect("announce streams audio");
    drop(device); // close the writer so the reader task can finish

    let kinds = reader_task.await.unwrap();
    assert_eq!(kinds.first().map(String::as_str), Some(types::AUDIO_START));
    assert!(
        kinds.iter().any(|k| k == types::AUDIO_CHUNK),
        "got some audio: {kinds:?}"
    );
    assert_eq!(kinds.last().map(String::as_str), Some(types::AUDIO_STOP));

    // Exactly the requested phrase was synthesized by Piper.
    assert_eq!(
        *connector.synthesized.lock().unwrap(),
        vec!["Time's up for pasta".to_string()]
    );
}

#[tokio::test]
async fn full_turn_streams_transcript_reply_and_tts_audio() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory.clone());
    let connector = MockConnector::new("turn on the lights");

    let ((transcript, reply, tts_kinds), events) = run_one_turn(&pipeline, &connector).await;

    assert_eq!(transcript, "turn on the lights");
    // The reply was relayed to the device token-by-token (Phase 5) and reassembles
    // to the full LLM reply.
    assert_eq!(reply, "You said: turn on the lights");
    assert_eq!(
        tts_kinds,
        vec![
            types::AUDIO_START.to_string(),
            types::AUDIO_CHUNK.to_string(),
            types::AUDIO_STOP.to_string(),
        ],
        "device receives a complete TTS audio stream"
    );

    assert!(events.contains(&TurnEvent::Transcript("turn on the lights".to_string())));
    assert!(events.contains(&TurnEvent::Reply(
        "You said: turn on the lights".to_string()
    )));
    assert!(events.contains(&TurnEvent::Speaking));
    assert_eq!(events.last(), Some(&TurnEvent::Finished));

    // Piper was asked to synthesize exactly the LLM reply. This reply has no
    // sentence-terminal punctuation, so it flushes as a single chunk at end of
    // stream.
    assert_eq!(
        *connector.synthesized.lock().unwrap(),
        vec!["You said: turn on the lights".to_string()]
    );
}

/// A silent room must not produce a phantom turn. The STT server hallucinates
/// "Thank you." on the forwarded silence (as faster-whisper does on non-speech
/// audio), but because the orchestrator's energy VAD never detected speech, that
/// transcript is discarded: no reply, no TTS, and — crucially — no `ambient-listen`,
/// so the follow-up chain terminates instead of looping on its own hallucinations.
#[tokio::test]
async fn silent_turn_discards_stt_hallucination_and_ends_the_chain() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory);
    // The canonical Whisper-on-silence hallucination.
    let connector = MockConnector::new("Thank you.");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_silent(dev_test));

    let mut events = Vec::new();
    {
        let mut on_event = |e: TurnEvent| events.push(e);
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    let (transcript, reply, saw_listen) = device_task.await.unwrap();

    // The hallucinated text never reaches the device as a real transcript…
    assert_eq!(transcript, "", "hallucinated transcript must be discarded");
    // …no reply is generated or spoken…
    assert_eq!(reply, "", "no reply for a discarded no-speech turn");
    assert!(
        connector.synthesized.lock().unwrap().is_empty(),
        "TTS must not run for a discarded no-speech turn"
    );
    // …and no follow-up listen is issued, so a quiet room can't self-perpetuate.
    assert!(
        !saw_listen,
        "no ambient-listen after a discarded no-speech turn"
    );
    // The pipeline still completes cleanly (sleeps on silence).
    assert_eq!(events.last(), Some(&TurnEvent::Finished));
}

#[tokio::test]
async fn two_devices_share_one_pipeline_and_each_reply_goes_to_its_own_socket() {
    // Multiple displays attach to a single orchestrator. The shared `Pipeline`
    // is reused across connections, but each turn drives its own borrowed socket,
    // so a reply must land on the device that asked — never cross-routed to the
    // other display. Run two turns concurrently through one shared pipeline with
    // distinct transcripts and assert each device sees only its own transcript +
    // reply.
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory);

    let connector_a = MockConnector::new("device A question");
    let connector_b = MockConnector::new("device B question");

    let ((out_a, _ev_a), (out_b, _ev_b)) = tokio::join!(
        run_one_turn(&pipeline, &connector_a),
        run_one_turn(&pipeline, &connector_b),
    );

    let (transcript_a, reply_a, _) = out_a;
    let (transcript_b, reply_b, _) = out_b;

    assert_eq!(transcript_a, "device A question");
    assert_eq!(reply_a, "You said: device A question");
    assert_eq!(transcript_b, "device B question");
    assert_eq!(reply_b, "You said: device B question");

    // Cross-routing check: neither device saw the other's content.
    assert!(
        !reply_a.contains('B'),
        "device A reply leaked device B: {reply_a:?}"
    );
    assert!(
        !reply_b.contains('A'),
        "device B reply leaked device A: {reply_b:?}"
    );
}

/// An LLM that emits one complete sentence and then hangs forever, so a turn is
/// still generating when a barge-in arrives — letting us prove the orchestrator
/// aborts the in-flight reply instead of blocking on the stalled backend.
struct FirstThenHangLlm;

#[async_trait]
impl anamanti_core::llm::LlmBackend for FirstThenHangLlm {
    fn name(&self) -> &str {
        "first-then-hang"
    }
    async fn respond(
        &self,
        _turn: anamanti_core::llm::LlmTurn,
    ) -> Result<anamanti_core::llm::ReplyStream> {
        use futures_util::stream::{self, StreamExt};
        let s = stream::iter(vec![Ok("First sentence. ".to_string())]).chain(stream::pending());
        Ok(Box::pin(s))
    }
}

/// Drive the device side of a barge-in: stream audio, read the transcript, then —
/// after the first spoken sentence's audio arrives — send an `ambient-interrupt`
/// and keep draining until the socket closes.
async fn drive_device_and_barge_in(io: DuplexStream) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    write_event(&mut writer, &WyomingEvent::audio_start(fmt, 0))
        .await
        .unwrap();
    write_event(
        &mut writer,
        // Voiced audio (clears the energy gate for MIN_SPEECH_ONSET) so the
        // orchestrator latches `speech_started` and accepts the transcript. A
        // sub-threshold chunk would now be treated as a no-speech finalize and the
        // transcript discarded as a likely STT hallucination (see stream_to_transcript).
        &WyomingEvent::audio_chunk(fmt, 0, voiced_chunk_bytes(180.0, 1500)),
    )
    .await
    .unwrap();

    // Transcript, then end our mic input (STREAMING → SPEAKING).
    let _ = read_event(&mut reader).await.unwrap().unwrap();
    write_event(&mut writer, &WyomingEvent::audio_stop(40))
        .await
        .unwrap();

    // Once the first sentence's audio starts arriving, barge in. (Per-sentence
    // audio-stops are coalesced away now, so trigger on the first audio-chunk.)
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if ev.event_type == types::AUDIO_CHUNK {
            write_event(&mut writer, &WyomingEvent::interrupt())
                .await
                .unwrap();
            break;
        }
    }
    // Drain anything still in flight until the server closes.
    while (read_event(&mut reader).await).unwrap_or(None).is_some() {}
}

#[tokio::test]
async fn barge_in_aborts_an_in_flight_reply() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(FirstThenHangLlm),
        memory,
        "test persona",
        None,
        Duration::from_secs(30),
    );
    let connector = MockConnector::new("say something");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_and_barge_in(dev_test));

    // The turn must return promptly on barge-in even though the LLM never finishes.
    let mut on_event = |_e: TurnEvent| {};
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        pipeline.run_turn(&mut device, &connector, &mut on_event),
    )
    .await
    .expect("barge-in aborts the turn instead of hanging on the stalled LLM");
    outcome.expect("turn runs");
    drop(device);
    device_task.await.unwrap();

    // Only the first sentence was ever synthesized; the hung remainder was aborted.
    assert_eq!(
        *connector.synthesized.lock().unwrap(),
        vec!["First sentence.".to_string()],
    );
}

#[tokio::test]
async fn multi_sentence_reply_is_synthesized_sentence_by_sentence() {
    // A reply with two sentences should be flushed to Piper as two separate
    // `synthesize` chunks (streaming TTS), producing two audio bursts — not one
    // whole-reply synthesis. This is what lets playback of sentence 1 begin before
    // sentence 2 has finished generating.
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(MockLlm::new("First part. Second part.")),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    );
    let connector = MockConnector::new("say something");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_until_close(dev_test));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device); // close the socket so the device driver sees EOF

    let (transcript, reply, audio_starts, audio_stops) = device_task.await.unwrap();
    assert_eq!(transcript, "say something");
    assert_eq!(reply, "First part. Second part.");
    assert_eq!(
        *connector.synthesized.lock().unwrap(),
        vec!["First part.".to_string(), "Second part.".to_string()],
        "each sentence is synthesized as its own Piper chunk (streaming TTS)"
    );
    // ...but the device sees one coalesced audio stream, so its turn state machine
    // (which ends on the first audio-stop) plays the whole reply, not just sentence 1.
    assert_eq!(audio_starts, 1, "one coalesced audio-start for the turn");
    assert_eq!(audio_stops, 1, "one coalesced audio-stop for the turn");
}

#[tokio::test]
async fn inferred_facts_are_persisted_across_a_turn() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory.clone());
    let connector = MockConnector::new("my name is Sam");

    let (_, events) = run_one_turn(&pipeline, &connector).await;

    assert!(events
        .iter()
        .any(|e| matches!(e, TurnEvent::MemoryStored(c) if c == "The user's name is Sam")));
    let stored = memory.list().unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].content, "The user's name is Sam");
    assert_eq!(stored[0].kind, MemoryKind::Fact);
}

// ---- Phase B: per-person identification end to end -------------------------

/// Synthesize `ms` of a pure tone at `hz` as little-endian PCM16 bytes — a mock
/// "voice" loud enough to pass the orchestrator's energy gate and long enough to
/// clear the speaker-ID minimum-speech floor.
fn voiced_chunk_bytes(hz: f32, ms: usize) -> Vec<u8> {
    let n = 16_000usize * ms / 1000;
    let mut out = Vec::with_capacity(n * 2);
    for t in 0..n {
        let x = (std::f32::consts::TAU * hz * t as f32 / 16_000.0).sin();
        out.extend_from_slice(&((x * 8000.0) as i16).to_le_bytes());
    }
    out
}

/// Like [`drive_device`] but streams a single ~1.5 s voiced chunk at pitch `hz`
/// (so speaker ID runs), returning `(transcript_seen, reply_text)`.
async fn drive_device_voiced(io: DuplexStream, hz: f32) -> (String, String) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    write_event(&mut writer, &WyomingEvent::audio_start(fmt, 0))
        .await
        .unwrap();
    write_event(
        &mut writer,
        &WyomingEvent::audio_chunk(fmt, 0, voiced_chunk_bytes(hz, 1500)),
    )
    .await
    .unwrap();

    let ev = read_event(&mut reader).await.unwrap().unwrap();
    let transcript = ev.transcript_text().unwrap_or_default().to_string();
    write_event(&mut writer, &WyomingEvent::audio_stop(1520))
        .await
        .unwrap();

    let mut reply = String::new();
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if let Some(tok) = ev.reply_token_text() {
            reply.push_str(tok);
            continue;
        }
        if ev.event_type == types::AUDIO_STOP {
            break;
        }
    }
    (transcript, reply)
}

/// Drive one voiced turn at pitch `hz` through a speaker-enabled pipeline.
async fn run_voiced_turn(
    pipeline: &Pipeline,
    transcript: &str,
    hz: f32,
) -> (String, Vec<TurnEvent>) {
    let connector = MockConnector::new(transcript);
    let (dev_pipeline, dev_test) = tokio::io::duplex(256 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_voiced(dev_test, hz));

    let mut events = Vec::new();
    {
        let mut on_event = |e: TurnEvent| events.push(e);
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    let (_, reply) = device_task.await.unwrap();
    (reply, events)
}

/// Pull the identified [`SpeakerContext`] out of a turn's events.
fn speaker_of(events: &[TurnEvent]) -> SpeakerContext {
    events
        .iter()
        .find_map(|e| match e {
            TurnEvent::Speaker(s) => Some(s.clone()),
            _ => None,
        })
        .expect("a Speaker event was emitted")
}

#[tokio::test]
async fn per_person_identification_scopes_memory_and_context() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let speaker = Arc::new(SpeakerService::new(
        Arc::new(MockSpeakerEmbedder::default()),
        SpeakerRegistry::open_in_memory().unwrap(),
        SpeakerThresholds::default(),
    ));
    // The reply echoes the system prompt so we can assert the identity line.
    let pipeline = Pipeline::new(
        Arc::new(MockLlm::new("[{sys}]")),
        memory.clone(),
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_speaker(speaker.clone());

    // Turn 1 — Sam introduces himself (low pitch). A new cluster is minted and the
    // inferred name fact is attributed to it.
    let (_, ev1) = run_voiced_turn(&pipeline, "my name is Sam", 180.0).await;
    let sam = speaker_of(&ev1);
    assert!(sam.is_new, "first voice creates a cluster");
    assert!(!sam.is_household());

    // Name the cluster (Phase C ships the voice/settings UX; here we set it directly).
    speaker.registry().rename(&sam.speaker_id, "Sam").unwrap();

    // Turn 2 — same voice, a preference. Matches Sam (not a new cluster) and the
    // reply's system prompt now greets him by name.
    let (reply2, ev2) = run_voiced_turn(&pipeline, "I like jazz", 180.0).await;
    let sam2 = speaker_of(&ev2);
    assert_eq!(sam2.speaker_id, sam.speaker_id, "same voice re-identifies");
    assert!(!sam2.is_new);
    assert!(
        reply2.contains("You are speaking with Sam."),
        "identity line reached the LLM: {reply2:?}"
    );

    // Turn 3 — a different voice (high pitch): a distinct cluster.
    let (_, ev3) = run_voiced_turn(&pipeline, "I like tea", 600.0).await;
    let dana = speaker_of(&ev3);
    assert_ne!(
        dana.speaker_id, sam.speaker_id,
        "different voice, new cluster"
    );
    assert!(dana.is_new);

    // Memory is per-person: Sam's scope has his name + jazz, never Dana's tea.
    let sam_hits: Vec<String> = memory
        .search_scoped("jazz tea Sam", Some(&sam.speaker_id), 10)
        .unwrap()
        .into_iter()
        .map(|m| m.content)
        .collect();
    assert!(sam_hits.iter().any(|c| c.contains("jazz")));
    assert!(sam_hits.iter().any(|c| c.contains("name is Sam")));
    assert!(!sam_hits.iter().any(|c| c.contains("tea")));

    let dana_hits: Vec<String> = memory
        .search_scoped("jazz tea", Some(&dana.speaker_id), 10)
        .unwrap()
        .into_iter()
        .map(|m| m.content)
        .collect();
    assert!(dana_hits.iter().any(|c| c.contains("tea")));
    assert!(!dana_hits.iter().any(|c| c.contains("jazz")));

    assert_eq!(speaker.registry().count().unwrap(), 2, "exactly two people");
}

// ---- Follow-up listening (question → reopen mic + history) ------------------

/// Drive the device side of a turn whose opening `audio-start` carries the given
/// follow-up `depth` (0 = an ordinary wake-word turn). Returns `(frame types in order
/// with reply-tokens filtered out, the `wait_secs` of the first `ambient-listen` seen)`.
async fn drive_device_collect_kinds(
    io: DuplexStream,
    followup_depth: u32,
) -> (Vec<String>, Option<u32>) {
    let (r, w) = split(io);
    let mut reader = BufReader::new(r);
    let mut writer = w;
    let fmt = AudioFormat::PCM_16K_MONO;

    let mut start = WyomingEvent::audio_start(fmt, 0);
    if followup_depth > 0 {
        if let serde_json::Value::Object(map) = &mut start.data {
            map.insert("followup".into(), json!(true));
            map.insert("followup_depth".into(), json!(followup_depth));
        }
    }
    write_event(&mut writer, &start).await.unwrap();
    write_event(
        &mut writer,
        // Voiced audio (clears the energy gate for MIN_SPEECH_ONSET) so the
        // orchestrator latches `speech_started` and accepts the transcript. A
        // sub-threshold chunk would now be treated as a no-speech finalize and the
        // transcript discarded as a likely STT hallucination (see stream_to_transcript).
        &WyomingEvent::audio_chunk(fmt, 0, voiced_chunk_bytes(180.0, 1500)),
    )
    .await
    .unwrap();

    let ev = read_event(&mut reader).await.unwrap().unwrap();
    assert!(ev.is_transcript());
    write_event(&mut writer, &WyomingEvent::audio_stop(40))
        .await
        .unwrap();

    let mut kinds = Vec::new();
    let mut listen_wait: Option<u32> = None;
    while let Some(ev) = read_event(&mut reader).await.unwrap() {
        if ev.reply_token_text().is_some() {
            continue;
        }
        if ev.event_type == types::LISTEN && listen_wait.is_none() {
            listen_wait = Some(
                ev.data
                    .get("wait_secs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            );
        }
        kinds.push(ev.event_type.clone());
    }
    (kinds, listen_wait)
}

#[tokio::test]
async fn question_reply_reopens_the_mic_with_the_long_window() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    // A reply that ends in a question mark gets the longer (10 s) listen window.
    let pipeline = Pipeline::new(
        Arc::new(MockLlm::new("Which one?")),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    );
    let connector = MockConnector::new("play some music");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_collect_kinds(dev_test, 0));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device);
    let (kinds, listen_wait) = device_task.await.unwrap();

    // The device got an `ambient-listen` frame, before the final audio-stop (the
    // device ends its turn on the first audio-stop), with the 10 s question window.
    let listen = kinds
        .iter()
        .position(|k| k == types::LISTEN)
        .unwrap_or_else(|| panic!("expected an ambient-listen frame; saw {kinds:?}"));
    let stop = kinds
        .iter()
        .rposition(|k| k == types::AUDIO_STOP)
        .expect("audio-stop sent");
    assert!(
        listen < stop,
        "listen must precede audio-stop; saw {kinds:?}"
    );
    assert_eq!(listen_wait, Some(10), "question → 10 s window");
}

#[tokio::test]
async fn statement_reply_reopens_the_mic_with_the_short_window() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    // Every reply now reopens the mic; a non-question reply gets the shorter (5 s) window.
    let pipeline = build_pipeline(memory);
    let connector = MockConnector::new("turn on the lights");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_collect_kinds(dev_test, 0));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device);
    let (kinds, listen_wait) = device_task.await.unwrap();

    assert!(
        kinds.iter().any(|k| k == types::LISTEN),
        "every reply reopens the mic; saw {kinds:?}"
    );
    assert_eq!(listen_wait, Some(5), "non-question → 5 s window");
}

#[tokio::test]
async fn followup_chain_stops_at_max_chain_when_capped() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    // With an explicit non-zero cap, an incoming turn already at the cap must NOT
    // reopen the mic again (the default cap is 0 = unlimited; here we set 2).
    let pipeline = Pipeline::new(
        Arc::new(MockLlm::new("Which one?")),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_follow_up(FollowUpConfig {
        enabled: true,
        max_chain: 2,
        question_wait_secs: 10,
        reply_wait_secs: 5,
        history_turns: 5,
        history_window_secs: 600,
    });
    let connector = MockConnector::new("play some music");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    // Incoming depth already at the cap (2).
    let device_task = tokio::spawn(drive_device_collect_kinds(dev_test, 2));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device);
    let (kinds, _) = device_task.await.unwrap();

    assert!(
        !kinds.iter().any(|k| k == types::LISTEN),
        "the chain must stop at max_chain; saw {kinds:?}"
    );
}

/// An LLM backend that records the [`LlmTurn`] it was handed (so a test can assert
/// what history the pipeline fed it) and replies with a fixed, non-question phrase.
struct CapturingLlm {
    seen: Arc<Mutex<Option<LlmTurn>>>,
}

#[async_trait]
impl LlmBackend for CapturingLlm {
    fn name(&self) -> &str {
        "capturing"
    }
    async fn respond(&self, turn: LlmTurn) -> Result<ReplyStream> {
        *self.seen.lock().unwrap() = Some(turn);
        use futures_util::stream::{self};
        Ok(Box::pin(stream::once(async { Ok("Got it.".to_string()) })))
    }
}

#[tokio::test]
async fn follow_up_turn_includes_recent_history_in_the_prompt() {
    // Pre-populate a chat log with a recent turn, then drive a FOLLOW-UP turn and
    // assert the pipeline replayed that turn as conversation history to the LLM.
    let dir = std::env::temp_dir().join(format!("ambient_followup_hist_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("chatlog.jsonl");
    let _ = std::fs::remove_file(&path);
    let log = Arc::new(ChatLog::open(&path).unwrap());
    log.append(&ChatLogRecord {
        id: log.next_id(),
        ts: now_secs(),
        session_id: "s-prev".to_string(),
        transcript: "what's the weather".to_string(),
        reply: "Where are you?".to_string(),
        memories_written: vec![],
        llm_backend: "mock".to_string(),
        model: None,
        speaker_id: "household".to_string(),
        speaker_name: None,
    })
    .unwrap();

    let seen = Arc::new(Mutex::new(None));
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(CapturingLlm { seen: seen.clone() }),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_chatlog(log.clone());
    let connector = MockConnector::new("in Austin");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    // depth 1 → this is a follow-up turn, so history should be included.
    let device_task = tokio::spawn(drive_device_collect_kinds(dev_test, 1));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device);
    device_task.await.unwrap();

    let turn = seen.lock().unwrap().take().expect("LLM was called");
    assert_eq!(
        turn.history,
        vec![(
            "what's the weather".to_string(),
            "Where are you?".to_string()
        )],
        "the follow-up turn replays the recent turn as history"
    );
    assert_eq!(turn.user_message, "in Austin");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn ordinary_turn_carries_no_history() {
    // The same chat log, but an ORDINARY turn (depth 0) must stay single-turn.
    let dir = std::env::temp_dir().join(format!("ambient_nohist_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("chatlog.jsonl");
    let _ = std::fs::remove_file(&path);
    let log = Arc::new(ChatLog::open(&path).unwrap());
    log.append(&ChatLogRecord {
        id: log.next_id(),
        ts: now_secs(),
        session_id: "s-prev".to_string(),
        transcript: "hello".to_string(),
        reply: "Hi there.".to_string(),
        memories_written: vec![],
        llm_backend: "mock".to_string(),
        model: None,
        speaker_id: "household".to_string(),
        speaker_name: None,
    })
    .unwrap();

    let seen = Arc::new(Mutex::new(None));
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(CapturingLlm { seen: seen.clone() }),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_chatlog(log.clone());
    let connector = MockConnector::new("what's up");

    let (dev_pipeline, dev_test) = tokio::io::duplex(64 * 1024);
    let (pr, pw) = split(dev_pipeline);
    let mut device = DynConnection::from_io(pr, pw);
    let device_task = tokio::spawn(drive_device_collect_kinds(dev_test, 0));

    {
        let mut on_event = |_e: TurnEvent| {};
        pipeline
            .run_turn(&mut device, &connector, &mut on_event)
            .await
            .expect("turn runs");
    }
    drop(device);
    device_task.await.unwrap();

    let turn = seen.lock().unwrap().take().expect("LLM was called");
    assert!(
        turn.history.is_empty(),
        "an ordinary turn stays single-turn (no history)"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn explicit_remember_command_short_circuits_the_llm() {
    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = build_pipeline(memory.clone());
    let connector = MockConnector::new("remember I like my coffee black");

    let (_, events) = run_one_turn(&pipeline, &connector).await;

    // The reply is the confirmation, not the mock LLM's "You said: …" echo.
    assert!(events.contains(&TurnEvent::Reply("Okay, I'll remember that.".to_string())));
    // The confirmation is what gets synthesized (one chunk).
    assert_eq!(
        *connector.synthesized.lock().unwrap(),
        vec!["Okay, I'll remember that.".to_string()]
    );

    let stored = memory.list().unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].content, "I like my coffee black");
    assert_eq!(stored[0].kind, MemoryKind::Preference);
}

/// System-1 fast path: when the decision engine resolves the `weather` intent, the
/// pipeline fetches the forecast, shows the full-screen weather widget, and speaks a
/// templated summary — **without** ever calling the LLM (System-2). A panicking LLM
/// backend proves the LLM is skipped. (plans/system1-fast-decisions.md, M1)
#[tokio::test]
async fn system1_weather_resolve_shows_widget_and_skips_the_llm() {
    use anamanti_core::settings::Household;
    use anamanti_core::system1::mock::MockDecider;
    use anamanti_core::weather::{CurrentConditions, WeatherProvider, WeatherReport};

    // Reaching System-2 on a resolved weather turn is a bug — panic loudly if it does.
    struct PanicLlm;
    #[async_trait]
    impl LlmBackend for PanicLlm {
        fn name(&self) -> &str {
            "panic"
        }
        async fn respond(&self, _turn: LlmTurn) -> Result<ReplyStream> {
            panic!("System-2 LLM must not run when System-1 resolves the weather intent");
        }
    }

    struct FakeWeather;
    #[async_trait]
    impl WeatherProvider for FakeWeather {
        async fn fetch(&self, location: &str, imperial: bool) -> Result<WeatherReport> {
            Ok(WeatherReport {
                location_label: location.to_string(),
                units: if imperial { "imperial".into() } else { "metric".into() },
                current: CurrentConditions {
                    temp: 18,
                    feels_like: 17,
                    weather_code: 3,
                    is_day: true,
                    high: 21,
                    low: 12,
                    description: "Partly cloudy".into(),
                },
                daily: Vec::new(),
            })
        }
    }

    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(PanicLlm),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_system1(Arc::new(MockDecider::resolve("weather", 0.99)))
    .with_weather(Some(Arc::new(FakeWeather)));

    // The weather fast path needs a home location; set it on the live settings.
    let hh = Household {
        location: Some("Austin, Texas".to_string()),
        weather_units: Some("imperial".to_string()),
        members: Vec::new(),
    };
    pipeline.settings().apply_household(&hh);

    let connector = MockConnector::new("what's the weather");
    let ((transcript, reply, kinds), _events) = run_one_turn(&pipeline, &connector).await;

    assert_eq!(transcript, "what's the weather");
    // The device was shown the full-screen weather widget.
    assert!(
        kinds.iter().any(|k| k == types::WEATHER),
        "expected an anamanti-weather frame, got {kinds:?}"
    );
    // The spoken reply is the templated forecast (a System-2 reply would not name the
    // location/high, and PanicLlm would have panicked).
    assert!(reply.contains("Austin, Texas"), "reply was: {reply:?}");
    assert!(reply.contains("high"), "reply was: {reply:?}");
    // Exactly the summary was synthesized by Piper.
    let synths = connector.synthesized.lock().unwrap().clone();
    assert_eq!(synths.len(), 1, "one TTS chunk, got {synths:?}");
    assert!(synths[0].contains("Austin, Texas"));
    // A well-formed turn ends with a single coalesced audio-stop.
    assert_eq!(kinds.last().map(String::as_str), Some(types::AUDIO_STOP));
}

/// System-1 fast path: a clear "set a timer for N" resolves on-device (a timer frame +
/// spoken confirmation) without the LLM; an unclear timer request would defer. A
/// panicking LLM proves System-2 is skipped. (plans/system1-fast-decisions.md, M3)
#[tokio::test]
async fn system1_timer_resolve_starts_timer_and_skips_the_llm() {
    use anamanti_core::system1::mock::MockDecider;

    struct PanicLlm;
    #[async_trait]
    impl LlmBackend for PanicLlm {
        fn name(&self) -> &str {
            "panic"
        }
        async fn respond(&self, _turn: LlmTurn) -> Result<ReplyStream> {
            panic!("System-2 LLM must not run when System-1 resolves a clear timer");
        }
    }

    let memory = Arc::new(MemoryStore::open_in_memory().unwrap());
    let pipeline = Pipeline::new(
        Arc::new(PanicLlm),
        memory,
        "test persona",
        None,
        Duration::from_secs(5),
    )
    .with_system1(Arc::new(MockDecider::resolve("timer", 0.99)));

    let connector = MockConnector::new("set a timer for 10 minutes");
    let ((transcript, reply, kinds), _events) = run_one_turn(&pipeline, &connector).await;

    assert_eq!(transcript, "set a timer for 10 minutes");
    assert!(
        kinds.iter().any(|k| k == types::TIMER),
        "expected an anamanti-timer frame, got {kinds:?}"
    );
    assert_eq!(reply, "Timer set for 10 minutes.");
    assert_eq!(kinds.last().map(String::as_str), Some(types::AUDIO_STOP));
}
