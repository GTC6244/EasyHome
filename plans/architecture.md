# Architecture

Detailed technical design for **Anamanti** (the ambient smart-display voice
assistant). This document is the source of truth for how the system is
structured; `Plan.MD` tracks phased delivery and open questions.

The two components: **Anamanti Core** (the Mac-side brain — STT ↔ LLM + memory ↔
TTS; crate `anamanti_core`, binary `anamanti-core`) and **Anamanti Display** (the
Echo Show app; package `com.anamanti.anamanti_display`).

---

## 1. System overview

Two cooperating nodes on a trusted LAN:

- **Echo Show 8 (LineageOS)** — the *edge* device. Captures audio, detects the
  wake word offline, streams audio to the Mac, renders the conversation, and
  plays back the spoken reply. Constrained: ~1 GB RAM, mobile-class CPU.
- **M4 Mac Mini** — the *brain* (**Anamanti Core**). Runs the Wyoming STT server,
  a pluggable LLM, and the Wyoming TTS (Piper) server.

They communicate over a single Wyoming Protocol TCP connection. The Mac's
service is located via mDNS, so neither node hardcodes an IP.

```
┌─────────────────────────── Echo Show 8 ───────────────────────────┐
│                                                                    │
│  Flutter UI (Dart)                                                 │
│   ├─ Ambient/idle dashboard                                        │
│   ├─ Live transcript view                                          │
│   └─ Streaming reply view                                          │
│        ▲                                                           │
│        │  flutter_rust_bridge v2 (StreamSink events, callbacks)    │
│        ▼                                                           │
│  Rust System Engine                                                │
│   ├─ Audio Capture   (cpal → oboe, 16 kHz mono i16)                │
│   ├─ Ring Buffer     (pre-allocated, lock-light)                   │
│   ├─ Wake Word       (tract-onnx + openWakeWord)                   │
│   ├─ Wyoming Client  (tokio TCP state machine)                     │
│   ├─ mDNS Resolver   (_wyoming._tcp)                               │
│   └─ Audio Playback  (cpal → oboe, TTS frames)                     │
│                                                                    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ TCP · Wyoming Protocol · newline JSON + PCM
                               ▼
┌─────────────────────────── M4 Mac Mini ───────────────────────────┐
│  Wyoming STT (Whisper / CoreML)  · Anamanti Core energy VAD        │
│        │ final transcript                                          │
│        ▼                                                           │
│  LLM backend  (trait-based, pluggable)  ◄─► Persistent Memory      │
│   ├─ Local backend  (Ollama / llama.cpp)          (local DB:       │
│   └─ Cloud backend  (Claude / OpenAI)              facts + prefs)  │
│        │ streamed reply tokens                                     │
│        ▼                                                           │
│  Wyoming TTS (Piper)  → synthesized audio frames                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component responsibilities

### 2.1 Rust System Engine (Echo Show)

The single owner of all real-time, resource-sensitive work. Chosen for
predictable memory use and no GC pauses under the 1 GB limit.

- **Audio Capture** — `cpal` pulls raw mono 16-bit PCM blocks from the mic array
  at the device-native rate (downmixed to mono in the real-time callback), then a
  linear resampler converts to 16 kHz off the RT path. On Android, `cpal` 0.18
  drives the NDK's **AAudio** input backend (the earlier `oboe` assumption is
  superseded — see `Plan.MD` Phase 2); on the macOS host it uses coreaudio.
- **Ring Buffer** — a pre-allocated single-producer/single-consumer circular
  buffer (`ringbuf::HeapRb`, allocated once) decouples the capture callback from
  consumers (wake word + network) with lock-free atomic index updates and no
  per-frame heap churn.
- **Wake Word** — a low-overhead thread continuously scores buffer windows with
  `tract-onnx` running the openWakeWord `.onnx` model chain (melspectrogram →
  feature/embedding → classifier). Fully offline; no audio leaves the device
  before a trigger.
- **Wyoming Client** — a `tokio` TCP state machine (see §4) that frames outgoing
  audio and parses incoming events.
- **mDNS Resolver** — browses `_wyoming._tcp`, resolves host/port, caches the
  last-known endpoint for fast reconnect.
- **Audio Playback** — `cpal`/`oboe` output stream that plays returned TTS audio
  frames. Symmetric with capture; keeps all audio in one layer.
- **Camera proximity sensor** (Android) — the front camera doubles as a proximity
  sensor. A thin Kotlin `CameraBridge` shim (twin of the `AudioRecord` `MicBridge`)
  opens the camera via Camera2 at 176×144 / ~5 fps and pushes packed luma frames over
  JNI; the Rust `camera::presence::PresenceDetector` measures frame-to-frame motion
  (mean absolute luma delta) and emits a `Presence` event when someone approaches or
  the room goes quiet. No ML / no face recognition / no frames leave the device. The
  brightness *actuation* is presentation and lives in Flutter (§2.2), not here.

### 2.2 Flutter UI (Echo Show)

- Always-on landscape layout tuned for the 8-inch display.
- Consumes FRB-generated `StreamSink` events; no polling.
- Primary states: **idle/ambient**, **live transcript**, **thinking**,
  **streaming reply**, **speaking**.
- **Idle/ambient screen:** photo slideshow from a chosen Google Photos/Drive
  folder via **on-device OAuth** (keeps running when the assistant is
  disconnected).
- **Settings screen:** LLM backend, TTS voice, wake word, photo source (on-device
  Google auth + folder picker), and **memory management** (view/delete entries).
- Renders reply tokens smoothly as they arrive.
- **Screen brightness:** the camera proximity sensor's `Presence` events fold into
  `AssistantState.userPresent`; `ScreenBrightnessController` actuates the window
  backlight (bright on approach, dimmed when quiet) via a `MethodChannel` to
  `MainActivity`. Sensing is Rust (§2.1); only the actuation is here. The **dim
  delay** — how long the screen stays bright after the room goes quiet before it
  dims to the away-mode clock — is a device-local setting (`AppSettings.dimDelaySecs`
  → the proximity detector's release window; Settings → *Display*).

### 2.3 Mac Mini services

- **STT (Whisper)** — two interchangeable engines behind the
  `anamanti-core/src/stt/` `Transcriber` seam (`plans/python-to-rust-whisper.md`),
  selected by the `stt.engine` config key:
  - **`wyoming`** (default) — the external `wyoming-faster-whisper` Python server
    over the Wyoming Protocol (`stt_addr`, port 10300).
  - **`whisper-rs`** — **in-process** whisper.cpp (`stt-whisper-local` build
    feature; optional `metal`/`coreml` accel). No separate process, no Python: the
    Core loads a ggml model (`base`/`small`) and transcribes in a `spawn_blocking`
    task. This is the deploy-simplification path, not a speed change.
  Either way the transcript is the same Whisper-class text; end-of-speech is the
  Core's energy VAD (below), never the engine.
- **LLM backend** — a trait/interface with a streaming
  `respond(transcript) -> token stream`. Two interchangeable implementations:
  local (Ollama/llama.cpp) and cloud (Claude/OpenAI). Selection is config-driven.
  Reads/writes the persistent memory store to build context and record new facts.
- **Persistent Memory** — a **SQLite** DB on the Mac is the store of record for
  long-term facts and user preferences across sessions. Policy is **explicit +
  inferred**: entries are added on request ("remember…") and auto-extracted from
  conversation turns. Managed via a settings list (view/delete) and voice
  ("forget that"); kept until cleared. Private to the LAN; survives device
  reflashes. **Retrieval/recall runs behind a `memory::Recall` seam and defaults
  to the embedded HelixDB GraphRAG backend** (`anamanti-core/src/memory/helix.rs`;
  in-process, no server/Docker): each completed turn is appended to
  `anamanti_chatlog.jsonl` and a background ingester (`memory/ingester.rs`) embeds
  it (OpenAI `text-embedding-3-small`, `OPENAI_API_KEY`) into a graph of
  `User/Turn/Memory/Entity` nodes, so recall does vector KNN + a graph hop rather
  than keyword FTS. If `OPENAI_API_KEY` is absent or the graph fails to open,
  recall **falls back to SQLite FTS** (writes are unaffected); `ANAMANTI_MEMORY_BACKEND=sqlite`
  forces FTS. **Per-person:** an opt-in local voiceprint embedder
  (`anamanti-core/src/speaker/`) identifies who is speaking from the utterance PCM and scopes
  memory writes/recall and the prompt to that person (a shared "household" scope is
  the floor); see `speaker_id_plan.md`.
- **End-of-speech / VAD** — runs in the **Anamanti Core**, for **both** STT engines
  (neither does streaming VAD: `wyoming-faster-whisper` transcribes only on
  `audio-stop`, and whisper.cpp transcribes the buffered utterance on finalize). The
  Anamanti Core scores per-chunk RMS energy over the incoming PCM and, after speech
  followed by ~900 ms of trailing silence (or a 6 s no-speech fallback), finalizes
  the transcriber (`anamanti-core/src/orchestrator.rs`, `stream_to_transcript`). The
  Echo Show device still runs **no VAD of its own** — it streams continuously and
  waits for the transcript.
- **Wyoming TTS (Piper)** — synthesizes the reply into audio frames streamed back
  to the device.

> **Implementation (Phase 4, `/mac`):** the STT, LLM, memory, and TTS pieces above
> are wired by a standalone Rust crate `anamanti_core` (`/mac`). It is a
> Wyoming **server** to the Echo Show and a Wyoming **client** to the off-the-shelf
> Whisper (STT) and Piper (TTS) Wyoming servers, with the pluggable `LlmBackend`
> trait (Ollama / Claude / mock) and the SQLite+FTS5 memory store in the middle
> (`anamanti-core/src/{Anamanti Core,server,discovery,llm,memory,wyoming}.rs`). It advertises
> `_wyoming._tcp` over mDNS, symmetric with the device's Phase-3 browse. The
> Wyoming wire codec is re-implemented there (byte-identical to the device's) since
> the device crate is an Android `cdylib` and can't be shared as a Mac library.

---

## 3. Interop boundary (flutter_rust_bridge v2)

- FRB v2 generates the JNI bindings and the Dart API from Rust signatures.
- Data flows **Rust → Dart** over a **single** generated `StreamSink`:
  - `start_wake_word_engine` returns one `Stream<WakeWordEvent>` covering the whole
    turn lifecycle. The `WakeWordEventKind` tag spans: `started`, `status`, `level`,
    `detected` (Phase 2); `connecting`, `streaming`, `transcript`, `disconnected`
    (Phase 3 Wyoming turn); `replyToken`, `speaking`, `speakingDone` (Phase 5); and
    `stopped` / `error`. `speakingDone` is UI-only: it fires when the reply audio has
    finished playing out of the ring (drained or barge-in-flushed), so the UI can keep
    the reply text on screen for exactly as long as it is being read, then remove it.
    Modeled as a flat struct tagged by a unit-only `WakeWordEventKind` enum
    (payload fields carry neutral defaults when not relevant), so the boundary needs
    no `freezed` codegen and there is exactly one stream to manage.
  - The UI split (transcript vs. reply vs. phase) happens **Dart-side**, not on the
    boundary. `AssistantController` (`anamanti-display/lib/src/engine/assistant_controller.dart`)
    folds this single event stream into an observable `AssistantState` / `TurnPhase`
    (`idle → listening → connecting → thinking → speaking`, plus `error`) that
    widgets watch. There are no separate `transcript_stream` / `reply_token_stream`
    / `state_stream` sinks on the FRB boundary.
- Control flows **Dart → Rust** via generated function calls: `start_wake_word_engine`
  / `stop_wake_word_engine` (Phase 2–3) plus the Phase-6 settings functions
  (`fetch_orchestrator_settings`, `update_orchestrator_settings`, `list_memories`,
  `delete_memory`, `clear_memories` in `anamanti-display/rust/src/api/settings.rs`). The settings
  functions are async: each stands up a small current-thread `tokio` runtime and
  drives one Wyoming control round trip, so FRB returns a `Future` off the UI
  isolate.
- Zero-copy is preferred for audio-adjacent buffers; UI text uses ordinary
  generated types.

### Phase 6 — settings + memory control protocol

- The **wake word, thresholds, and photo source** are device-local: the Flutter
  settings screen persists them (`SettingsStore` → JSON) and applies them by
  rebuilding the `WakeWordConfig` and restarting the engine / refreshing the
  slideshow.
- The **LLM backend + model and TTS voice** live on the Mac and are read/changed
  over a **project-local control protocol** on the device↔Anamanti Core hop —
  `anamanti-*` Wyoming frames (`describe`/`set` settings; `list`/`delete`/`clear`
  memories; `list-models`; `list-voices`) that ride the existing framing (byte-identical `types`
  in both crates, no off-the-shelf server sees them). The Anamanti Core's `Pipeline`
  reads a per-turn snapshot of runtime-swappable `SharedSettings`, so a
  backend/voice change takes effect on the next turn with no restart; the accept
  loop routes control frames to `control::handle_control` and audio-start frames to
  a turn. Backends are `ollama` (local), `anthropic` and `openai` (cloud), and
  `mock`; selecting a cloud backend needs its API key (`ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY`), else the change is rejected in-band (never dropping the
  connection). The key can come from the environment at boot **or** be entered at
  runtime on the loopback config page: `SettingsUpdate` carries optional
  `anthropic_api_key` / `openai_api_key` fields (same tri-state as the search key —
  absent = keep, `null` = clear, value = set), so a cloud backend can be enabled
  without a restart. Runtime keys live in `SharedSettings` (env is only the boot
  seed) and persist to `anamanti_settings.json` (0600). Key **entry** is deliberately
  config-page-only — the Wyoming/device control path never accepts a provider key, so
  cloud secrets never live on the shared Echo Show screen; the device is told only
  whether a key is set (`anthropic_key_set` / `openai_key_set`), never its value.
- **Model selection** is a drop-down of **specific Anthropic / OpenAI models from
  the last 12 months**, produced by `llm::catalog::ModelCatalog`: it live-queries
  each provider's `GET /v1/models` (Anthropic `created_at`, OpenAI `created`),
  filters to the trailing 12 months (and OpenAI to chat models), caches the result,
  and falls back to a curated built-in list when a key is missing/offline. The list
  is exposed to the device via `anamanti-list-models` → `anamanti-models` and to the
  browser via `GET /models` on the config page. The chosen `llm_model` flows through
  the same `SharedSettings::apply` → `LlmFactory::build` path into the concrete
  backend's request, and persists to `anamanti_settings.json`.
- **TTS voice selection** is likewise a drop-down. On `anamanti-list-voices`, the
  Anamanti Core asks Piper for its advertised catalog (a downstream Wyoming
  `describe` → `info`) and, when `ANAMANTI_TTS_VOICES_DIR` points at Piper's model
  dir (Piper co-located on the Mac), intersects it with the `<name>.onnx` files
  actually on disk so only installed voices are offered; with no dir set it returns
  the full advertised list. Exposed to the device via `anamanti-list-voices` →
  `anamanti-voices` (FRB `list_voices` → `listVoices`) and to the browser via
  `GET /voices` on the config page. Both the settings screen and the config page
  render a dropdown (with a "Server default" entry) and fall back to a free-text
  field when the list is unavailable; a hand-set voice not in the list stays
  selectable.
- **Anthropic auth mode** (`llm::anthropic_auth`): a per-provider toggle selects
  **API key** (`x-api-key` from `ANTHROPIC_API_KEY`) or **subscription OAuth**
  (`Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20`). Subscription tokens
  come from `AnthropicTokenProvider` — `ANTHROPIC_OAUTH_TOKEN` (from
  `claude setup-token`) or a token-printing command (`ANAMANTI_ANTHROPIC_TOKEN_CMD`,
  default the `ant` CLI), cached with a short TTL and refreshed on a 401. Both the
  chat backend and the catalog share the provider so listing + turns authenticate
  identically. The mode rides the settings protocol (`anthropic_auth` field) and the
  config page toggle; OpenAI is API-key-only.
- **Debug/inspection pages** (`webconfig.rs`, same loopback HTTP server as the
  config page, strictly read-only): `GET /chatlog`, `/prompts`, `/sqlite`, `/helix`
  render the recent chat log, the exact assembled LLM prompt per turn (captured to
  a separate `anamanti_promptlog.jsonl` via `PromptLog`), the SQLite memory rows, and
  the HelixDB GraphRAG node stats + a sample of nodes (behind the read-only
  `memory::GraphView` seam; reports "disabled" on the SQLite backend). Each has a
  `*.json` data endpoint the page fetches. No auth — keep the config address on a
  trusted network.
- **Household / home context** (`settings::Household`, editable at `GET /household`
  on the config page): a persisted record of the home **location + units** and a
  **roster of people** (name, emails, phones, relationship). Location/units seed from
  `ANAMANTI_HOME_LOCATION`/`ANAMANTI_WEATHER_UNITS` at first boot, then are editable
  (with the people) from the Household tab and persisted to `anamanti_settings.json`
  (0600 — PII). It is read from the **per-turn settings snapshot** and injected into
  the system prompt (`orchestrator::location_line` grounds an unqualified "here";
  `household_line` lists who lives here so the model can address them and has their
  contact details), so page edits take effect with no restart. The **directions
  tool's default origin** reads the same location via a shared `LiveHomeLocation`
  handle (`directions`), so a location edit re-homes routing without rebuilding the
  backend. An **identified speaker** (speaker_id_plan.md) whose name matches a roster
  member is reconciled to it (`Household::member_matching`), so the identity line uses
  the canonical name + relationship.
- **Memory management** is dual: the settings list (this control protocol) plus
  voice ("remember…", "forget that") applied on the Mac during a turn (Phase 4).
- **On-device Google OAuth** for the photo folder is wired as a seam
  (`GoogleAuthenticator`); the default is an honest stub because a real client ID
  can't be provisioned in this environment.

---

## 4. Wyoming client state machine (Rust)

```
        ┌────────────────────────────────────────────────────────┐
        ▼                                                        │
     ┌──────┐  wake word fires   ┌───────────┐  socket open   ┌──────────┐
     │ IDLE │ ─────────────────► │ TRIGGERED │ ─────────────► │STREAMING │
     └──────┘                    └───────────┘  send audio-   └──────────┘
        ▲   socket dormant/closed              start header        │
        │                                                          │  send PCM frames;
        │                                                          │  read transcript events
        │                                                          ▼
        │                        ┌───────────┐   final txt   ┌──────────┐
        │   playback done /      │ SPEAKING  │ ◄──────────── │ THINKING │
        └──── reset ──────────── │ (play TTS)│  TTS frames   │ (LLM)    │
                                 └───────────┘               └──────────┘
```

- **IDLE** — wake word evaluation runs; TCP socket dormant/closed.
- **TRIGGERED** — wake word fires; open TCP, send Wyoming `audio-start` header.
- **STREAMING** — send raw PCM chunks in Wyoming frames; concurrently read
  `transcript` events on the same socket. The device streams continuously and runs
  no VAD; **the Anamanti Core detects end-of-speech** (energy VAD over the PCM) and
  sends `audio-stop` to the STT server, which then returns the final transcript.
- **STREAMING → (optional) System-1 fast decision:** once the transcript is final, an
  **optional pluggable System-1 decision engine** may run on the Anamanti Core *before* THINKING
  (before memory recall and the LLM). It scores a fixed routing question set in a single
  non-autoregressive forward pass and either **Resolves** a common intent or **Defers**:
  - **Resolve** (confident, closed intent): drive the matching **existing `DeviceAction`** and
    speak a short templated line via Piper, then finish the turn — **skipping memory recall and the
    LLM entirely**. Implemented intents: **weather** (Open-Meteo → `DeviceAction::ShowWeather`, so
    the widget opens *before* speech) and **timer** (a conservative duration parser →
    `DeviceAction::StartTimer`; cancels/free-form defer). Resolved replies are **not**
    wake-word-interruptible in v1 (sub-second line). Chat-log write, inferred memory, and the
    follow-up `listen` window still happen (the fast path shares `emit_follow_up_and_stop` with the
    normal reply path). Because the engine is a *typed classifier* (intent label, no free-form
    slots), only intents with defaulted/parseable arguments are eligible; the rest defer.
  - **Defer** (ambiguous, open-ended, low confidence, disabled, or error): fall through to THINKING
    unchanged.
  The engine is pluggable behind a trait (like the LLM) and shares the `/v1/systemone` wire
  contract, so a local **`laya-serve`** sidecar (default) or **Jev/OpenRouter** (cloud fallback)
  are swappable from config. Default is off (`system1.backend = none`), reproducing today's flow.
  Full design: [`system1-fast-decisions.md`](./system1-fast-decisions.md).
- **THINKING** — STT final transcript handed to the LLM backend (which
  consults persistent memory); reply tokens stream back and render.
- **THINKING → SPEAKING (streaming TTS):** the Anamanti Core does **not** buffer the
  whole reply before speaking. It segments the LLM token stream into sentences and
  synthesizes each with Piper as soon as it forms, so the first audio reaches the
  device ~first-sentence latency (~2 s) instead of full-reply latency (~8 s). The
  per-sentence Piper bursts are **coalesced into one device-facing audio stream** (one
  `audio-start`, all chunks, one final `audio-stop`) because the device ends its turn
  on the first `audio-stop`.
- **SPEAKING** — the coalesced Piper audio stream is played via `cpal`/`oboe`. The
  turn returns to IDLE on the first `audio-stop` (above), but the audio keeps
  draining from the ring for seconds after; the device tracks ring occupancy
  (`PlaybackSink::pending()`) and emits a UI-only `speakingDone` once it empties, so
  the on-screen reply text stays up for the whole utterance and is removed only when
  the audio actually stops (a barge-in flush zeroes the ring and triggers it too).
- **Barge-in:** wake-word scoring keeps running during THINKING and SPEAKING. A wake
  word mid-reply **always flushes the playback ring immediately** (silencing the reply
  even after the turn has technically ended — the Anamanti Core relays audio faster than
  real-time, so the turn reaches IDLE while audio is still draining from the buffer),
  sends an `anamanti-interrupt` frame so the Anamanti Core **aborts the in-flight LLM +
  TTS**, and starts a fresh turn.
- **Follow-up listen (reopen the mic after every reply):** after each reply the
  Anamanti Core sends an `anamanti-listen` frame (above) before the final `audio-stop`,
  with a `wait_secs` window (10 s after a question, 5 s otherwise). The device records
  it during SPEAKING, ends the turn normally, and — once the reply audio has drained
  from the playback ring (the same `pending()==0` signal that emits `speakingDone`, so
  the mic never records the tail of the TTS) — **spawns a fresh turn with no wake word**,
  reusing the barge-in restart machinery. The follow-up turn keeps the raised in-turn
  wake-word threshold and stamps its chain depth + `wait_secs` on its `audio-start`; the
  Anamanti Core feeds recent conversation history into that turn's prompt and sizes its
  no-speech window to `wait_secs`. The no-speech window only counts down while `speech_started`
  is false; because the drain gate frees when the *software* playback ring empties (not the
  OS/hardware output buffer), the reopened mic can still catch a brief residual TTS tail or
  room echo, so the Anamanti Core **debounces the speech onset** — `speech_started` latches
  only after `MIN_SPEECH_ONSET` (250 ms) of *consecutive* voiced audio (`voiced_onset_step`),
  and any non-voiced chunk resets the run. This stops a lone transient from collapsing the
  full `wait_secs` window into the short `end_silence` finalize and cutting the user off
  (observed in production as a follow-up that closed after ~1 s with an empty transcript).
  If the user says nothing in that window the
  Anamanti Core ends the turn (empty transcript + `audio-stop`) and the device **sleeps**;
  the chain continues only while the user keeps responding (bounded by the optional
  `follow_up.max_chain`, default unlimited). See `Plan.MD` (Follow-up listening).
- **In-app AEC:** not shipped. Investigated on real hardware (Echo Show 8): the platform
  `VOICE_COMMUNICATION` AEC preset is reachable but doesn't actually cancel on this
  device, and a software NLMS canceller is net-negative for this flush-on-wake barge-in
  (there's no simultaneous echo to cancel once playback is flushed). The interim
  mitigation remains **raising the wake-word confidence threshold while active**. Real
  in-app AEC would require a true keep-playing-while-listening full-duplex redesign — see
  `Plan.MD` §4 / `TODO.md`.
- **On-device AEC shim — REQUIRED on every Echo Show 8 (1st gen, codename `crown`).**
  Echo cancellation is provided **outside this codebase** by a vendor audio-HAL shim
  `LD_PRELOAD`ed into `android.hardware.audio.service`. It taps the FPGA capture stream
  `pcmC0D22c` (6-ch 16 kHz — mics on ch0–3, a **sample-aligned DAC loopback** on ch4–5)
  and runs SpeexDSP linear AEC on the mic before `AudioRecord`/AudioFlinger ever see it,
  giving talker-preserving barge-in (~16 dB, measured). **Any Echo Show 8 gen-1 this app
  is deployed to must have the shim installed and activated** (`persist.vendor.amznaec.enable=1`,
  loaded via the audio-HAL init rc) — it is a **device prerequisite**, not part of this
  build, and survives reboots. Source + reversible adb install/uninstall:
  <https://github.com/Brutus-GTC6245/EchoShow8gen1-aec-shim>.
- Return to **IDLE** on playback completion, timeout, or reset.

### Wire format

- Newline-delimited (`\n`) JSON control frames for metadata and events.
- Raw audio chunks streamed immediately after the setup/metadata frame.
- The same socket carries outbound audio and inbound `transcript`, streamed
  `reply-token` (per-LLM-token text for on-screen rendering), and synthesized audio
  frames.
- **`anamanti-interrupt`** (device → Anamanti Core): a project-local barge-in frame that
  tells the Anamanti Core to abort the in-flight LLM generation + TTS at once, rather
  than only learning of the interruption when the socket drops.
- **`anamanti-timer`** (Anamanti Core → device): a project-local **device-action** frame
  (`data.action` = `start`/`cancel`; `duration_secs` + optional `label`). Emitted when
  an LLM **tool** the model called on the Mac (`set_timer` / `cancel_timer`) asks to
  act on the device. The device owns the resulting state — unlimited concurrent
  countdowns, the on-screen UI, and the alarm — so a timer keeps running after the
  turn's socket closes and even if the Mac disconnects. When a timer fires it rings a
  two-strike **bell** locally, then (if the Mac is reachable) requests the spoken
  announcement over **`anamanti-speak`**; bell-only when offline. See §8.
- **`anamanti-speak`** (device → Anamanti Core): a project-local frame (`data.text`)
  asking the Anamanti Core to synthesize the text with Piper and stream the audio
  (`audio-start`/`audio-chunk`…/`audio-stop`) straight back on the same socket. A
  fired timer uses it to voice "Time's up for {name}" in the real assistant voice; the
  device opens a fresh socket (same mDNS path a turn uses), outside any voice turn, and
  plays the returned audio through the shared `PlaybackSink` right after the bell.
- **`anamanti-recipe`** (Anamanti Core → device): a project-local **device-action** frame
  (`data.action` = `show`/`dismiss`; for `show`, `data.recipe` is the structured recipe).
  Emitted by the `recipe_lookup` tool; the device owns the 3-tab recipe screen until
  dismissed by voice (`close_recipe`) or touch. See `RecipePlan.md`.
- **`anamanti-weather`** (Anamanti Core → device): a project-local **device-action** frame
  (`data.action` = `show`/`current`/`dismiss`; for `show`/`current`, `data.weather` is the
  structured `WeatherReport` — `location_label`, `units`, `current{…}`, `daily[7]`). It
  rides **two transports**: `show`/`dismiss` on the per-turn voice socket, emitted by the
  `weather_lookup` / `close_weather` tools (`DeviceAction::{ShowWeather,DismissWeather}`),
  drive the **full-screen forecast** (today's conditions + a 7-day row); `current` is
  broadcast periodically on the persistent channel (below) by the Anamanti Core's
  `WeatherService` to refresh the **small icon + temperature beside the idle clock**
  without a voice turn. Data comes from the keyless **Open-Meteo** API behind a
  `WeatherProvider` trait. See `WeatherPlan.md`.
- **`anamanti-listen`** (Anamanti Core → device): a project-local **follow-up-listen**
  frame (`data.depth` + `data.wait_secs`). After **every** reply (gated by
  `follow_up.enabled`) the Anamanti Core sends this frame **just before** the turn's
  final `audio-stop` — it must precede that stop because the device ends its turn on the
  first `audio-stop`. It tells the device to **reopen the mic and start a fresh turn
  with no wake word** once the reply audio drains (see the follow-up bullet in §4's
  state machine). `wait_secs` is how long to keep the mic open for input before
  sleeping — **10 s after a question (`?`), 5 s otherwise** (`follow_up.question_wait_secs`
  / `reply_wait_secs`). The device **echoes `depth` + `wait_secs`** on the follow-up
  turn's `audio-start` (`followup: true`, `followup_depth`, `wait_secs`); the
  Anamanti Core uses `wait_secs` to size that turn's **no-speech VAD window** (the only
  place silence can be told from a user mid-sentence — the device runs no VAD), and
  `depth` against the optional `follow_up.max_chain` ceiling (**default 0 = unlimited**;
  the loop is normally ended by silence). On silence the Anamanti Core relays an empty
  transcript **and** an `audio-stop` so the device sleeps promptly, and sends no new
  `anamanti-listen`. A follow-up turn's LLM prompt is seeded with the **recent
  conversation history** (the last `follow_up.history_turns` turns within
  `follow_up.history_window_secs`, from the chat log); ordinary wake-word turns stay
  single-turn.
- **`anamanti-recipe`** (Anamanti Core → device): a project-local **device-action** frame
  driving the guided-recipe screen (`data.action` = `show` + `recipe` object /
  `dismiss` / `navigate` + `target` tab / `scroll` + `direction`). Emitted by the LLM
  **tools** the model calls on the Mac — `recipe_lookup` (fetch + parse → `show`),
  `close_recipe` (`dismiss`), and `recipe_control` (`navigate` to Overview/Ingredients/
  Steps, or `scroll` up/down/top/bottom). The device owns the resulting screen state —
  it persists across turns and idle while you cook — so, like timers, this is
  fire-and-forget from the Mac. See `plans/RecipePlan.md`.
- **Display context on `audio-start`** (device → Anamanti Core): a **general,
  extensible** mechanism that tells the Core **what the display is currently showing**, so
  the model can *drive that screen by voice*. The device stamps a `data.screen` block on
  every turn's `audio-start`, discriminated by a `kind` string, with a per-kind payload.
  Two kinds ship today:
  - **recipe:** `{kind:"recipe", recipe:{title, tab, at_top, at_bottom,
    ingredient_count, step_count}}` — the model can `recipe_control` (switch tab / scroll)
    or `close_recipe`.
  - **weather:** `{kind:"weather", weather:{location, units, temp, description}}` — the
    model knows the forecast is up and for where, so it answers follow-ups in context
    ("what about tomorrow" → `weather_lookup` for the same place) and closes it on "close
    it" (`close_weather`).

  Absent on an idle screen (and the small weather clock chip is **not** a screen — only
  the full-screen forecast sets weather context). The orchestrator parses the block
  (`protocol::display_context` → a `DisplayContext` enum) and injects a one-line
  description into that turn's system prompt (`orchestrator::display_context_line`, one
  arm per screen). **Adding a new voice-controllable screen** (music, photos) is a
  `DisplayContext` variant + a prompt-line arm + a device-side `set_<screen>_context`
  setter — the transport is unchanged; an unknown `kind` is ignored, so a newer device
  never breaks an older Core.

  **How context rides with spoken input.** This is the **only device→Core context
  channel** and it *piggybacks on the turn's own `audio-start`* — the first frame the
  device sends when the user speaks — alongside the follow-up `depth`/`wait_secs` markers.
  There is no separate context uplink and no conversation "context window" pushed ahead of
  time: what the user is looking at travels *with* their utterance, so the model always
  sees the exact on-screen state for the turn it is answering, and nothing goes stale
  between turns. (The other half of a turn's context — long-term memory recall and, for a
  follow-up turn, recent chat history — is assembled **Core-side** into the same system
  prompt; the display-context line is appended to it.) On the device, Rust holds the
  current block in a global slot (`engine::display_context` / `set_display_context`) set by
  the FRB layer (`set_recipe_context` / `set_weather_context`) whenever a screen opens,
  closes, or changes, and stamps it in `WyomingConnection::send_audio_start`.

### Proactive notifications (Approach A — persistent device-dialed channel)

Every frame above rides a socket the **device** opened for a voice turn, so the
Anamanti Core can only ever *reply*. Proactive notifications add the one path where the
Mac reaches the display **unprompted** — a meeting reminder, an alert — without the
user speaking first. The chosen design keeps the device as the dialer (no reverse
connection, no device-side listener, nothing new for NAT/firewalls, and the same
mDNS + `instance_id` pin the voice path uses):

- The device opens a **second, long-lived Wyoming connection** to the pinned
  Anamanti Core and holds it open, reconnecting with capped exponential backoff
  (`rust/src/wyoming/notify.rs`; started by `start_notify_channel` on its own thread,
  independent of the voice engine). This is a *sidecar* — the per-turn voice socket and
  its state machine are untouched.
- **`anamanti-hello`** (device → Anamanti Core): sent right after the notify socket
  opens (`data.role="notify"`, `data.device_id`). It registers the connection with the
  Anamanti Core's `NotificationService`, which parks the read loop and holds the socket
  to push down.
- **`anamanti-notify`** (Anamanti Core → device): a proactive notification
  (`data.id`, `data.priority` = `info`|`reminder`|`alert`, `data.title`, `data.body`).
  The device decodes it to a `NotifyEvent` on a dedicated FRB stream; Flutter's
  `NotificationController` shows a banner on the idle screen (top layer, tap or
  auto-expire to dismiss). **Visual-only in this phase** — no spoken output, no
  state-machine interaction.
- **`anamanti-notify-ack`** (device → Anamanti Core): reserved for a later
  delivery-tracking phase (store-and-forward across reconnects); the constructor exists
  in both crates but nothing sends it yet.
- The **ambient weather push** reuses this same persistent-channel design: the device
  opens a *third* long-lived connection (`rust/src/wyoming/weather.rs`, started by
  `start_weather_channel`) with an `anamanti-hello` carrying `data.role="weather"`, which
  the server registers with the `WeatherService` instead of the notify registry. The Core
  then pushes `anamanti-weather` `current` frames down it on a timer; the device surfaces
  them as a `WeatherPush` FRB stream that refreshes the clock's weather indicator. Same
  dialer/backoff/pin model as notify — see `WeatherPlan.md`.

Producers enqueue via `NotificationService::notify`, which fans a notification out to
every connected device and prunes dead channels. The first producer is the config
page's **Notify** tab (`POST /notifications/test`); if no device holds a channel open
the notification is simply dropped (delivered-count 0). Because the channel is pinned
to the selected `instance_id`, a pinned display only accepts pushes from its Mac.
Deferred (see `TODO.md`): spoken notifications + barge-in, ack/store-and-forward/TTL,
per-device targeting, quiet hours, and a paired/TLS control channel (the LAN hop is
currently unauthenticated, so this widens the same attack surface the voice/control
frames already have).

---

## 5. Discovery & networking

- **mDNS / Zeroconf**: the device browses `_wyoming._tcp`, resolves the Mac's
  host + port, and caches it for fast reconnect.
- No static IP configuration required.
- **Anamanti Core TXT records**: each Anamanti Core advertises TXT
  `role=core`, a friendly `name` (from `ANAMANTI_SERVICE_NAME`), and a
  **stable** `instance_id`. The `instance_id` resolves from `ANAMANTI_INSTANCE_ID` →
  the working directory's **git branch code** (so a copy run from a test
  branch/worktree names itself by its branch) → the sanitized service name. The
  device **filters browse results by `role=core`**, so the raw
  off-the-shelf Whisper (STT) and Piper (TTS) servers — which also advertise
  `_wyoming._tcp` — are never resolved or connected to.
- **Deployment layout**: the "local production" Anamanti Core is a copied release
  binary at `/Volumes/External/DeveloperSupport/Anamanti Core/`, run outside
  any git checkout with `ANAMANTI_INSTANCE_ID` set in `~/.zshenv`; test copies run
  from their git worktree and fall through to the branch-code default.
- **Multiple displays → one Anamanti Core**: the Anamanti Core's accept loop already
  spawns a task per connection and every reply is written back on that
  connection's own socket, so N displays share one Anamanti Core with no
  cross-routing. Memory + settings are a single shared household pool (speaker ID
  scopes per person, not per device).
- **Multiple Anamanti Cores → device picks one**: a display can run a "production"
  Anamanti Core plus short-lived test instances (each launched with a distinct
  `ANAMANTI_SERVICE_NAME` / `ANAMANTI_INSTANCE_ID` / `ANAMANTI_BIND_ADDR` /
  `ANAMANTI_CONFIG_ADDR`; `mdns-sd` does not auto-rename on collision, so the names
  must differ). The settings screen shows a device-local **Anamanti Core** dropdown
  (populated by a full-window mDNS enumeration, `list_orchestrators`), persisted
  by stable `instance_id` key in `AppSettings`. The selection is **strict**: a
  display pinned to one Anamanti Core resolves *only* that `instance_id` and stays
  **offline** if it is unreachable, never silently switching Macs; the `"Auto"`
  entry (empty key) restores first-responder behavior. The key steers **both**
  discovery paths — voice turns (`WakeWordConfig.orchestrator_key` → `Shared` →
  `resolve(preferred)`) and the settings/control calls (`api/settings.rs` →
  `control` → `resolve(preferred)`) — so both hit the same Mac. Shared backend data
  across instances relies on SQLite **WAL** mode; two processes writing the same
  embedded HelixDB graph is unsupported, so a test instance sharing prod data uses
  `ANAMANTI_MEMORY_BACKEND=sqlite` (see `Plan.MD`).
- **Resilience**: **auto-reconnect** with exponential backoff via mDNS when the
  Mac is unreachable. The idle photo slideshow keeps running; a subtle
  **disconnected** indicator reflects status; wake words queue until the socket
  is restored.

---

## 6. Resource & performance constraints

- **RAM (~1 GB on the Echo Show):** pre-allocated ring buffer; wake-word model
  kept small; avoid per-frame heap churn; single audio engine for capture +
  playback.
- **Latency:** wake word runs on-device; audio streams (not batched); reply tokens
  render as they arrive; **TTS is synthesized and streamed sentence-by-sentence** so
  playback starts after the first sentence, not the whole reply.
- **Privacy:** no audio leaves the device until the wake word fires.

---

## 7. Key design decisions

| Decision | Rationale |
| --- | --- |
| Rust owns all audio + networking | Deterministic memory, no GC, fits 1 GB budget |
| FRB v2 stream sinks over polling | Push-based, smooth real-time UI updates |
| Wyoming Protocol | Standard, streaming-friendly, pairs STT + Piper TTS cleanly |
| mDNS discovery | Robust to IP changes; zero manual config |
| Device-selectable Anamanti Core (TXT `instance_id`, strict offline) | Multi-Mac homes (prod + test instances); pin by stable key so a display never silently switches Macs and survives IP changes; `role=core` filtering also excludes raw STT/TTS servers |
| Multiple displays share one Anamanti Core | Per-connection reply routing already isolates devices; one shared household memory/settings pool (speaker ID scopes per person) |
| openWakeWord via tract-onnx | Pre-trained models, minimal deps, offline |
| Pluggable LLM behind a trait | Swap local/cloud without touching the pipeline |
| Optional System-1 fast-decision stage (pluggable, before recall+LLM) | Resolves common intents in one non-autoregressive forward pass, skipping the blocking embedding recall and the rig+tools full-completion; defers hard turns to System-2. Same trait pattern as the LLM; shared `/v1/systemone` contract serves local `laya-serve` (default) or Jev/OpenRouter (fallback); default off. See [`system1-fast-decisions.md`](./system1-fast-decisions.md) |
| Rust-side playback | One audio layer, symmetric with capture |
| Wake-word barge-in (flush-on-wake + `anamanti-interrupt`) | Natural interruption without full-duplex complexity; in-app AEC deferred, but a **required device-side HAL AEC shim** delivers echo cancellation on Echo Show 8 gen-1 (see §4) |
| Streaming sentence-chunked TTS | First-audio at first-sentence latency, not full-reply; coalesced to one device audio stream |
| VAD in the Anamanti Core | Device does no VAD; neither STT engine does streaming VAD, so the Mac runs energy VAD and finalizes the transcriber |
| STT engine behind a `Transcriber` seam (`wyoming` \| `whisper-rs`) | Default dials `wyoming-faster-whisper`; `whisper-rs` runs whisper.cpp **in-process** (no Python STT server) for deploy simplicity — same Whisper-class text, engine chosen by config (`plans/python-to-rust-whisper.md`) |
| SQLite is the memory store of record | Simple, debuggable; holds explicit+inferred facts and the settings-list/voice management |
| Recall defaults to embedded HelixDB GraphRAG | Vector KNN + graph hop beats keyword FTS for context; in-process (no server/Docker); needs `OPENAI_API_KEY`, falls back to SQLite FTS if absent |
| Per-person speaker ID (local, opt-in) | Local ECAPA voiceprint (passive + auto-cluster) keeps voice on the LAN and scopes memory + prompt per person for better context; no raw audio leaves the device |
| On-device OAuth for photos | Device displays directly; no Mac proxy needed |
| Auto-reconnect + status | Robust to Mac downtime; slideshow stays up |
| Idle photo slideshow (Google) | Ambient value when idle; user picks the folder |
| Proactive notifications via a persistent **device-dialed** channel (Approach A) | Mac reaches the display unprompted while keeping the device the dialer — reuses the existing mDNS + `instance_id` pin and the blessed auto-reconnect model; avoids a reverse connection / device listener / new trust direction. A doorbell (device advertises, Mac nudges) was considered but only wins idle-socket cost, which is free on a mains-powered display, at the price of lossy triggers. Shipped visual-only first (see §4) |

---

## 8. Extending the assistant with tools & device actions

The assistant does more than chat: the LLM can call **tools** to fetch information
and to take **actions** on the device. This section is the canonical process for
adding a new capability.

### 8.1 How tool calling works

Tool calling lives in the **rig engine** (`anamanti-core/src/llm/rig.rs`), the default LLM
engine (`ANAMANTI_LLM_ENGINE=rig`; `native` opts out to the tool-less HTTP backends).
Each turn the Anamanti Core advertises a set of **tool definitions** (name +
description + JSON-schema for the arguments) to the model. If the model asks to call
one, `RigBackend::respond` runs a short **tool-negotiation loop** (up to
`MAX_TOOL_ROUNDS`): it executes the tool, appends the result to the conversation, and
completes again — until the model answers with no further tool calls. Because a live
`ollama` streaming parser drops non-final tool calls, tool turns use a **non-streaming
completion**; the reply is still spoken whole (and rendered) via the sentence-chunked
TTS path, so there is no user-visible regression.

There are **two flavors** of tool:

- **Info tool** — runs entirely on the Mac and returns text the model speaks
  (e.g. `internet_search`, `calendar_lookup`). Adding one is self-contained in
  `rig.rs`, though it may lean on a supporting module: `calendar_lookup` (read-only
  web iCalendar) lives in `anamanti-core/src/calendar/`, holds an injected
  `Arc<dyn CalendarSource>` (mirroring how `internet_search` holds a
  `SearchProvider`), and is advertised only when `ANAMANTI_CALENDARS` lists one or
  more `.ics` subscription URLs. It fetches the feeds live per query (with a short
  TTL cache to dedupe fetches inside one tool-negotiation loop), expands recurring
  events (`RRULE`) within the requested window, and filters by time / person
  (fuzzy name match) / free text. A CalDAV or macOS-EventKit source can drop in
  later behind the same `CalendarSource` trait. The **`shopping_list_add`** tool has
  the same shape but acts on an external service: it holds an injected
  `Arc<dyn GroceryController>` (`anamanti-core/src/cadora/`) and, when the household
  shopping list is linked, POSTs the add to **Cadora's voice API** (the shared family
  backend behind NextHaul et al.) and speaks back its confirmation — mirroring how
  `spotify_control` holds a `SpotifyController`. Linking: NextHaul (Settings → Voice
  & Integrations) mints a one-time **6-digit code** (`/voice/links/pair`); enter it on
  the config page (Household tab) and the Anamanti Core redeems it
  (`/voice/links/redeem`) for a durable voice-link token stored in the 0600 settings
  file. It never touches Supabase directly — Cadora owns list creation, member
  attribution, and dedupe.
- **Action tool** — additionally causes an effect on the Echo Show (e.g.
  `set_timer`). The tool cannot reach the device directly (it runs inside the LLM
  loop, which has no socket), so it emits a **`DeviceAction`** onto a per-turn
  channel; the Anamanti Core's reply loop drains that channel and writes a
  project-local **`anamanti-*`** frame to the device on the same turn socket. The
  device owns whatever state results.

### 8.2 Adding an **info** tool (Mac-only)

1. In `anamanti-core/src/llm/rig.rs`, add a tool `NAME`, a `Deserialize` args struct, a
   `…_definition() -> ToolDefinition` (JSON-schema for the args), and an
   `…_invoke(args) -> Result<String>` that does the work and returns a short,
   speakable result.
2. Register it: add its definition in `Tools::new` and a match arm in
   `Tools::dispatch`.
3. Nudge the model: extend `tool_guidance` so the preamble tells the model when to
   call it (only when the tool is actually advertised).
4. Test it in `rig.rs` (a unit test for `…_invoke`, and an end-to-end
   `serve_sequence` test that has a fake model call the tool then answer).

No protocol, device, or Flutter changes are needed — build + test with
`cargo test --manifest-path anamanti-core/Cargo.toml`.

### 8.3 Adding a **device action** (spans all layers)

Worked example: **timers** (`set_timer` / `cancel_timer`). The flow is
model → Mac tool → `DeviceAction` → `anamanti-timer` frame → device timer manager →
FRB event → Flutter UI. To add a new action, mirror these steps:

1. **Mac tool** (`anamanti-core/src/llm/rig.rs`): as in §8.2, but `…_invoke` takes the per-turn
   `Option<&ActionSink>` and `send`s a new `DeviceAction` variant
   (`anamanti-core/src/llm/mod.rs`) instead of doing the work locally. Action tools are
   registered unconditionally (they need no config).
2. **Relay** (`anamanti-core/src/orchestrator.rs`): the reply loop already drains the per-turn
   action channel (`drain_device_actions`) and writes the frame; add a match arm
   mapping the new `DeviceAction` to its `WyomingEvent` constructor.
3. **Wire frame** (`anamanti-core/src/wyoming/protocol.rs` **and** `anamanti-display/rust/src/wyoming/protocol.rs`,
   kept **byte-identical** with round-trip tests in both crates): add an `anamanti-*`
   `type` const, constructor(s), and — on the device side — a decoder
   (e.g. `timer_command()`).
4. **Device intake** (`anamanti-display/rust/src/wyoming/client.rs`): add a `TurnUpdate` variant and a
   branch in `handle_server_event` that decodes the frame into it. The frame only
   arrives **mid-turn** (the device is always the Wyoming *client*; the socket exists
   only during a turn), which is fine — device actions are triggered by the very turn
   that carries them.
5. **Device state** (`anamanti-display/rust/src/engine/`): own the resulting state in a manager held by
   the long-lived `Network`/`Shared` (so it outlives the turn socket), applied from
   the `on_update` closure in `net.rs`. Reuse the shared `PlaybackSink` for any sound
   (no second audio path). See `anamanti-display/rust/src/engine/timer.rs`.
6. **FRB event** (`anamanti-display/rust/src/api/engine.rs`): add `WakeWordEventKind` variant(s) +
   neutral-default payload fields to the flat `WakeWordEvent`, plus `pub(crate)`
   builders; the manager emits them via the `StreamSink`. Then **regenerate**:
   `flutter_rust_bridge_codegen generate` (never hand-edit `frb_generated.rs` or
   `anamanti-display/lib/src/rust/`).
7. **Flutter** (`lib/`): fold the new event kind into `AssistantState` in
   `AssistantController._onEvent` (the exhaustive `switch` forces you to handle it),
   and render it. See `anamanti-display/lib/src/ui/timers_overlay.dart`, mounted in the always-visible
   `Stack` in `ambient_screen.dart` so it shows during both idle slideshow and a live
   turn. Timers render in two presentations: a big, screen-filling display on the idle
   screen (single / side-by-side / grid by count, each with a name, large mm:ss readout,
   and a draining circular ring), and a compact chip row top-center while a turn is active,
   so the conversation wins the screen and the timers stay glanceable.

Build/verify each layer independently: `cargo test` for `anamanti-core/` and `anamanti-display/rust/`,
`flutter test` + `flutter analyze` for the UI.

---

## 9. Cross-references

- Delivery phases & open questions → [`Plan.MD`](./Plan.MD)
- Contributor / AI-agent build guidance → [`agents.md`](../agents.md)
- Product overview & setup → [`README.md`](../README.md)
- **Music (planned, pre-implementation):** house-wide music is a **Mac-hosted,
  Snapcast-routed** capability that is deliberately **separate from the Wyoming/TTS
  audio path and the device's cpal/oboe path** — music PCM never enters
  `anamanti_core`. Sources (librespot + web-URL player) write raw PCM to
  snapfifos on the Mac; snapserver (on the Mac) fans out to snapclient speakers;
  the Anamanti Core only controls (Web API / mpv IPC / snapserver JSON-RPC ducking).
  Design → [`snapcast_routing_plan.md`](./snapcast_routing_plan.md) (transport) +
  [`MusicPlan.md`](./MusicPlan.md) (Spotify source + control plane).
