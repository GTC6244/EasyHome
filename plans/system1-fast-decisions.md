# system1-fast-decisions.md

A **System-1 fast-decision stage** for the Anamanti Core turn pipeline: a pluggable,
non-autoregressive decision engine that runs *before* memory recall and the LLM, resolves
common intents in a single forward pass, and only **defers** the hard turns to today's
"System-2" path (GraphRAG recall → rig+tools LLM → Piper).

Read with [`agents.md`](../agents.md) (locked decisions), [`architecture.md`](./architecture.md)
(design, §4 state machine / turn flow), and [`Plan.MD`](./Plan.MD) (decision table). Feature
sibling of [`WeatherPlan.md`](./WeatherPlan.md) (the `weather_lookup` tool + `anamanti-weather`
frame this reuses).

> **Status:** implemented through M2 (HTTP `laya-serve`/`jev` backends + **live config-page
> selection & persistence**) plus the M3 timer intent (see §10). Default remains **disabled**
> (`system1.backend = "none"`), so builds are unchanged until opted in (via `anamanti.json` or the
> `/system1` config page). The remaining arg-heavy intents are documented follow-ups. Adding a
> pipeline stage is a structural change — see *Locked-decisions check* below.

---

## 1. Problem — the two latency bottlenecks this targets

Traced from the turn pipeline (`anamanti-core/src/orchestrator.rs`):

- **Bottleneck #1 — the default `rig` engine defeats token streaming.** Tools are seeded on
  every turn (`src/llm/rig.rs:1424`: timers always; weather/web-search enabled by default), and
  with *any* tool present, `respond` takes the **non-streaming** branch
  (`src/llm/rig.rs:1840`): it runs a full blocking `model.completion()` and yields the entire
  reply at once, looping up to `MAX_TOOL_ROUNDS = 4` (`src/llm/rig.rs:49`). Net effect:
  **time-to-first-audio ≈ full LLM generation time**, not first-token latency.
- **Bottleneck #3 — memory recall blocks before the LLM even starts.** `respond_and_speak`
  awaits `build_context` (`src/orchestrator.rs:642`) → `recall.recall(...)`
  (`src/orchestrator.rs:955`) → `HelixRecall::recall`, whose first step is a **network OpenAI
  embedding round-trip** (`src/memory/backend.rs:97`, `text-embedding-3-small`) before any vector
  search. A round-trip to OpenAI sits on the critical path of *every* turn.

For a huge share of real utterances ("show me the weather", "set a 10-minute timer", "next
step", "add milk to the list") neither cost buys anything: the answer is a known action, not a
memory-grounded generation. System-1 collapses those turns.

**What this does *not* fix** (out of scope here; tracked separately): the ~700 ms VAD end-silence
hangover, Whisper non-streaming transcription, and cold-start warmth — all of which happen
*before* a transcript exists, where a decision engine cannot help.

---

## 2. Goals

1. **Cut perceived latency on routine turns** by skipping both bottleneck #1 and #3 when a fast,
   high-confidence decision is available — ideally emitting the visual widget the instant the
   intent resolves, before speech.
2. **Make the decision engine pluggable and selectable, exactly like the LLM backend** — a trait
   with a config-chosen implementation (local vs cloud), runtime-swappable from the config page,
   defaulting to off.
3. **Support both engines the user runs**: the in-process **Laya-Decision** Rust crate
   (candle/Metal, no server) and **Jev** via OpenRouter (`typesafe/jev-1.13`) — which share one
   wire contract (`/v1/systemone`), so a single local `laya-serve` sidecar is a third, free
   option.
4. **Reuse existing machinery**: resolved intents drive the *existing* tools and
   `DeviceAction`s (e.g. `weather_lookup` → `DeviceAction::ShowWeather`,
   `src/llm/mod.rs:52`) and the *existing* action-relay loop (`src/orchestrator.rs:689`). No new
   device protocol.
5. **Preserve conversational coherence**: resolved turns still write the chat log, run inferred
   memory capture, and emit the follow-up `listen` window.
6. **Zero-regression opt-in**: `system1.backend = "none"` (default) reproduces today's behavior
   byte-for-byte; the whole feature is gated.

## 3. Non-goals

- Not replacing the LLM. System-1 **routes**; System-2 still answers everything it can't.
- Not a slot/entity extractor. Jev/Laya return **typed** answers (choice/noul/score), not
  free-form text — so open-ended arguments (an arbitrary city, a long freeform request) cause a
  **Defer**, not a guess.
- Not on the device. It runs on the Mac Core (respects the "Rust owns real-time on the 1 GB
  device" boundary; the Core is where STT/LLM/TTS already live).
- No custom model training in v1 (mirrors the wake-word "no custom training in v1" decision).
- Does not touch the VAD/STT/warmth latencies (separate work).

## 4. Outcomes & success metrics

- **Resolve path latency:** for a resolved intent, transcript→(widget + first audio) is bounded
  by one System-1 forward pass + one tool call + first-sentence Piper — **no** embedding
  round-trip, **no** rig completion. Target: local Laya decision < ~50 ms on M4 Metal (single
  non-autoregressive forward pass; to be measured).
- **Coverage:** ≥ N% of turns resolved by System-1 for the seeded intent set (weather, timers,
  recipe nav, music transport, shopping-list add, simple smalltalk), measured from the chat log.
- **Precision:** mis-resolve rate below a strict bar (mis-resolves are worse than a slow-correct
  answer). Gated by a calibrated confidence threshold; ambiguous → Defer.
- **No regression:** with `none`, all existing `cargo test`/pipeline integration tests pass
  unchanged.
- **Selectable like models:** `system1.backend` switches between `none | laya-embedded |
  http (laya-serve | jev-openrouter) | mock` at boot and from the config page, persisted to
  settings, same lifecycle as `llm.backend`.

---

## 5. Background — Jev and Laya-Decision (what they are, one contract)

Both are "System-1" decision models: given a **state** and a set of **typed questions**, they
return **typed, probabilistic** answers in a single shot — no generation, nothing to parse, no
reasoning trace.

**Question types** (identical across both):
- `choice` — pick one of N labels; returns the label + per-label probabilities.
- `noul` — boolean; returns `P(true)`.
- `score` — ordinal `0..N-1`; returns the expected value.
- Every answer carries `answer_confidence` (calibrated `max(p)` across all types) → **gate on one
  threshold**.

**Wire contract** (shared — Laya is explicitly "Jev-compatible"):
```
POST /v1/systemone
{
  "state":     { "message": "show me the weather" },
  "questions": { "intent": { "type": "choice", "instructions": "...", "criteria": { ... } } }
}
→ { "model": "...", "answers": { "intent": { "label": "...", "probabilities": {...},
     "answer_confidence": 0.97 } }, "usage": {...}, "routing": {...} }
```

**Jev (cloud):** `typesafe/jev-1.13` on OpenRouter's Decisions API. Needs an OpenRouter API key;
billed on **input tokens only** (output free). A network hop — fine as a fallback / no-GPU
option, but not the low-latency default for a home device.

**Laya-Decision (the user's repo, `GTC6244/Laya-Decision`, Apache-2.0):** a pure-Rust,
non-autoregressive port (candle; ModernBERT-large EN / mmBERT multilingual), matches upstream
PyTorch to `1e-4`. Three integration modes:
1. **In-process library** — crate `laya-decision` (imported as `laya`):
   ```rust
   use laya::agent::{Agent, LoadOptions};
   use laya::{triage_questions, State};
   let agent = Agent::load("convaiinnovations/laya",
                           LoadOptions { device: Some("metal".into()), ..Default::default() })?;
   let result = agent.system_one(&state, &questions)?; // SystemOneResult { answers, .. }
   ```
   or the auto-checkpoint `Router::with_defaults()?.predict(&state, &questions, &hints)`.
   Runs on **Apple-Silicon Metal (~4× vs CPU on an M4)** via the crate's `metal` feature;
   downloads the checkpoint from HF on first use. **No server, no Docker** — same ethos as the
   embedded HelixDB store.
2. **`laya-serve`** — a local HTTP sidecar exposing `POST /v1/systemone` + `GET /health`, env
   config (`LAYA_HOST/PORT/DEVICE/PRELOAD/MODELS/API_KEY/...`).
3. Since (2) is Jev-wire-compatible, **one HTTP client covers both** `laya-serve` and OpenRouter
   Jev by swapping base URL + auth + model id.

**Design consequence:** this is the same "local-or-cloud behind one trait" split you already have
for the LLM (ollama vs Claude/OpenAI). The shipped default (decided 2026-09-25) is the local
**`laya-serve` sidecar** — local + low-latency, keeps candle out of the Core binary, and the same
HTTP client points at OpenRouter Jev as the cloud fallback. In-process Laya-on-Metal is an
optional lower-latency escape hatch.

---

## 6. Architecture

### 6.1 The trait (mirrors `LlmBackend`)

New module `anamanti-core/src/system1/` with a trait shaped like `LlmBackend`
(`src/llm/mod.rs:135`) and held in `RuntimeSettings` next to `llm` (`src/settings.rs:694`):

```rust
// system1/mod.rs
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// Short label for logs/settings, e.g. "laya-embedded", "jev", "none".
    fn name(&self) -> &str;

    /// Score a fixed question set over the turn state in one shot.
    /// Thin transport over the shared `/v1/systemone` contract — engines stay dumb;
    /// the routing POLICY (questions, thresholds, intent→handler) lives in the orchestrator.
    async fn decide(&self, state: &DecisionState, questions: &QuestionSet) -> Result<Answers>;
}

pub struct DecisionState {           // → the `state` object
    pub message: String,             // the transcript
    pub screen: Option<String>,      // display context label (so "next step" routes vs recipe)
    pub history: Vec<(String, String)>, // recent turns, for follow-up disambiguation
}
pub type QuestionSet = /* id → { type, instructions, criteria } */;
pub struct Answers   { /* id → { kind, label/p_true/expected, answer_confidence } */ }
```

Implementations:
- `system1/http.rs` — one `/v1/systemone` reqwest client; config picks base URL + auth +
  model id. Serves **both** a local **`laya-serve`** (`http://127.0.0.1:8000`,
  keyless/`LAYA_API_KEY`) and **Jev on OpenRouter** (`typesafe/jev-1.13`, `OPENROUTER_API_KEY`).
  Reuses the existing reqwest/secret-env plumbing. **`laya-serve` is the recommended enabled
  backend** (decided 2026-09-25): keeps candle out of the Core binary, still local + low-latency,
  and the same client trivially points at OpenRouter Jev as the no-GPU/cloud fallback.
- `system1/laya_embedded.rs` — wraps the `laya` crate's `Agent`/`Router` in-process (Metal).
  Loaded once at boot, shared behind `Arc` (like the LLM backends). Optional; lowest latency but
  compiles candle into the Core binary — kept as an escape hatch, not the default.
- `system1/mock.rs` — deterministic fixtures for tests (same role as `llm/mock.rs`).
- `NoDecision` (`"none"`) — always returns "defer"; the **safe merge default**, so the feature is
  a no-op until explicitly enabled.

### 6.2 Where it plugs in (the exact seam)

In `respond_and_speak` (`src/orchestrator.rs:586`), immediately **after** the existing
`parse_command` remember/forget fast-path (`:605`, which already proves the "resolve-and-return
before the LLM" shape) and **before** `build_context` (`:642`):

```
transcript
  → parse_command                       (existing remember/forget short-circuit, :605)
  → system1.decide(state, ROUTING_Qs)   (NEW; only if system1.backend != none)
       ├─ Resolve(intent) if answer_confidence ≥ system1.min_confidence
       │     run intent handler → reuse existing tool + DeviceAction (:52, :689),
       │     speak a short templated line, write chatlog + infer memories + emit
       │     follow-up `listen`, then RETURN.               ← skips #1 AND #3
       └─ Defer                                              ← unchanged: build_context (:642)
                                                               → rig+tools LLM (:704)
```

### 6.3 The routing question set (policy, provider-agnostic)

System-1 is a **router**, so we ask a small fixed `QuestionSet` and interpret typed answers.
Sketch:

```json
{
  "intent": { "type": "choice",
    "instructions": "What does the user want in `message`?",
    "criteria": {
      "weather":       "current conditions or forecast",
      "timer":         "start/cancel a timer or alarm",
      "recipe_nav":    "navigate/scroll the recipe already on screen",
      "music":         "play/pause/next/previous/volume",
      "shopping_add":  "add an item to the shopping list",
      "smalltalk":     "greeting/thanks/acknowledgement, no data needed",
      "other":         "anything that needs reasoning, memory, or open-ended understanding"
    }},
  "needs_full_understanding": { "type": "noul",
    "instructions": "Does answering require open-ended reasoning, personal memory, or details not implied by a simple command?" }
}
```

**Decision rule (in the orchestrator):**
- `Resolve(intent)` iff `intent != other`, `needs_full_understanding` is false, **and**
  `answer_confidence ≥ min_confidence` for both questions.
- Otherwise `Defer`.
- Open-ended slots (e.g. weather for a *non-home* city) are exactly what `needs_full_understanding`
  catches → Defer. Home-location weather ("show me the weather") resolves.

Each `intent` maps to a small handler that reuses existing code — e.g. `weather` →
`weather_lookup(home_location)` → `DeviceAction::ShowWeather` + a templated spoken summary; `timer`
→ parse duration deterministically (or Defer if unparseable) → `DeviceAction::StartTimer`;
`recipe_nav` → `DeviceAction::RecipeControl`; etc.

### 6.4 Worked example — "show me the weather"

1. Transcript lands; `parse_command` → no match.
2. `system1.decide` → `intent = weather (p≈0.97)`, `needs_full_understanding = false (p_true≈0.02)`.
   Both clear `min_confidence` → **Resolve(weather)**.
3. Handler calls the existing `weather_lookup` for `home_location`, emits
   `DeviceAction::ShowWeather(report)` on the existing action sink → the `anamanti-weather` frame
   opens the widget **immediately** (before speech).
4. Speak a short templated line ("Here's your forecast — 18 and clear."); write chatlog; emit the
   follow-up `listen` window.
5. **Skipped:** the OpenAI embedding recall (#3) and the full rig+tools completion (#1).

---

## 7. Config & selection (identical lifecycle to the LLM backend)

Mirror the LLM config surface: `LlmChoice` (`src/config.rs:110`), `build_llm()`
(`src/config.rs:1142`), and the settings/config-page persistence.

New JSON block in `anamanti.json` (all optional, defaults shown; add to `anamanti.example.json`):
```jsonc
"system1": {
  "backend": "none",              // none (default) | laya-serve | jev | laya-embedded
  "base_url": "http://127.0.0.1:8000",  // http backends: laya-serve (default host) or OpenRouter
  "openrouter_model": "typesafe/jev-1.13",  // jev backend only
  "device":  "metal",             // laya-embedded only: cpu | metal
  "model":   "convaiinnovations/laya",  // laya-embedded only: checkpoint / hub id
  "min_confidence": 0.85,         // strict; precision over recall
  "intents": ["weather","timer","recipe_nav","music","shopping_add","smalltalk"]
}
```

- New factory `build_system1() -> Result<Arc<dyn DecisionEngine>>` beside `build_llm()`; result
  stored on `RuntimeSettings` next to `llm`; **runtime-swappable + config-page selectable**,
  persisted to `anamanti_settings.json` (a persisted value wins over the JSON seed, same rule as
  the LLM fields).
- **Secret:** `OPENROUTER_API_KEY` — a new env-var secret for the Jev backend, following the
  existing pattern (boot seed + masked config-page entry, persisted to the `0600` settings file;
  never in either JSON file). Add it to the secrets list in `agents.md`.
- **Default `none`** ⇒ no behavior change on merge.

---

## 8. Coherence, safety, precision

- **Precision gate:** strict `min_confidence`; anything ambiguous defers. This is the primary
  safety knob. Start conservative, loosen with measured data.
- **Coherence:** resolved turns must still (a) append to `anamanti_chatlog.jsonl` for the GraphRAG
  ingester, (b) run `infer_memories` capture (cheap, local), and (c) emit the follow-up `listen`
  window — or conversation state drifts.
- **Barge-in:** resolved replies are **not** wake-word-interruptible in v1 (decided
  2026-09-25) — the templated line is a fraction of a second, so the interruptible window is
  negligible and not worth the complexity. This matches the existing canned remember/forget reply,
  which is also not interruptible. (Wake-word barge-in still works normally on the System-2 path.)
- **Failure = defer:** any `decide` error, timeout, or low confidence falls through to System-2.
  The engine never blocks a turn; a small per-decision timeout guards the HTTP backends.

## 9. Latency budget

- **laya-serve (default):** local HTTP round-trip to a sidecar doing one non-autoregressive
  forward pass (CPU or Metal via `LAYA_DEVICE`). Loopback + a single forward pass; measure, but
  expected to be small relative to what it replaces — an OpenAI embedding round-trip **plus** a
  full rig completion of up to 4 tool rounds. Keeps candle out of the Core binary.
- **laya-embedded (optional):** in-process forward pass on Metal (M4), no network, no extra
  process — the lowest-latency option; target < ~50 ms (measure). Escape hatch when the extra
  process isn't wanted, at the cost of compiling candle into the Core.
- **jev (OpenRouter):** one internet round-trip, input-token billed. Still far cheaper than the
  full System-2 path, but a WAN hop — the no-GPU / cloud fallback, not the home default.
- **Speculative option (later):** run `decide` in parallel with STT finalization / warm-up so its
  latency overlaps dead time.

## 10. Phasing & status

1. **M0 — scaffold, no-op. ✅ DONE.** Trait + `system1/` module + `NoDecision` default + config
   plumbing + `mock`; seam in `respond_and_speak` behind `backend != none`. 278 lib tests green,
   clippy clean, zero behavior change.
2. **M1 — `laya-serve` HTTP backend + weather end-to-end. ✅ DONE.** `system1/http.rs`
   (`/v1/systemone` client, pure `build_request`/`interpret` unit-tested) covers `laya-serve`
   (default) and `jev`; routing question set (`intent` choice + `needs_full_understanding` noul)
   under `min_confidence`; the `weather` handler fetches Open-Meteo, emits `DeviceAction::ShowWeather`
   **before** speech, speaks a templated summary, and invites a follow-up — skipping recall + the
   LLM. Integration test drives it with a **panicking LLM** to prove System-2 is skipped.
3. **M2 — Jev/OpenRouter backend + config-page integration. ✅ DONE.** The same HTTP client serves
   Jev via `base_url` (OpenRouter API root) + `openrouter_model` + `OPENROUTER_API_KEY`. System-1
   is now **runtime-swappable and persisted** like the LLM backend: the live selection lives in
   `RuntimeSettings.system1` (a `System1Runtime` bundle), rebuilt via `SharedSettings::apply_system1`
   (mirroring `apply_drive`/`apply_household`, so it never touches the LLM `apply()`), persisted in
   `anamanti_settings.json`, and seeded at boot in `shared_settings` (config seed → persisted
   overlay, with the OpenRouter key from env). A dedicated **config page** (`/system1`, nav
   "System-1") picks the backend/base-URL/model/min-confidence/key live via
   `/system1/status.json` + `/system1/save`. `laya-embedded` (in-process candle) remains the
   optional escape hatch and bails at boot.
4. **M3 — expand intents. ✅ weather + timer DONE; others DEFERRED with rationale.** Added the
   `timer` intent: a conservative, unit-tested duration parser (`system1::parse_duration_secs`)
   resolves a clear "set a timer for N …" to `DeviceAction::StartTimer` + a spoken confirmation, and
   **defers** cancels / free-form / out-of-range to System-2's timer tool. **Key finding that bounds
   the rest:** Jev/Laya are *typed classifiers* — they return an intent label, **not** free-form
   arguments. So intents whose arguments are **defaulted or absent** are System-1-eligible (weather →
   home location; timer → parsed number+unit), but intents needing free-form slot extraction
   (`shopping_add` → item text) or open sub-intent disambiguation + live device/screen state
   (`recipe_nav`, `music`) are **not** a good fit for a single classifier pass and are left to
   System-2. `smalltalk` is deferred too (the label alone can't tell "thanks" from "hello"; a canned
   reply would feel wrong). These could be revisited with additional typed sub-questions (e.g. a
   `music_action` choice) **plus** on-device QA — enumerated as follow-ups, not shipped blind.
5. **M4 — tuning + docs. ✅ Knobs in place; docs updated.** `min_confidence` and the intent list are
   config-tunable; per-intent thresholds and resolve-rate/precision telemetry remain open (§13).

## 11. Testing

- Mirror the LLM/TTS test style (in-memory fixtures, the duplex-pipe pattern used for the Wyoming
  clients).
- `mock` engine returns scripted `Answers` → assert Resolve vs Defer routing and that resolved
  turns still log/infer/emit-follow-up.
- Golden decisions for the seeded intents; a precision fixture set (utterances that **must**
  defer).
- An `--ignored` integration test against a running `laya-serve` / real checkpoint (like the
  existing model-dependent tests), off by default.

## 12. Locked-decisions check (read before coding)

`agents.md` locks the pipeline shape and says *"If a task seems to require changing one of these,
stop and confirm first"* and *"Do not add a second … interop mechanism … without confirmation"*,
plus *"record it in `Plan.MD` (decision table) and reflect structural changes in
`architecture.md`."* This feature:
- **Adds a pipeline stage** (System-1 before recall/LLM) — a structural change → needs a Plan.MD
  decision-table entry and explicit sign-off before implementation.
- Is **consistent with** the locked "pluggable behind a trait" decision (it copies that pattern)
  and does not add a second *audio* path or an IP-discovery fallback.
- Introduces a **new secret** (`OPENROUTER_API_KEY`) and a **new dependency** (`laya-decision`
  crate for the embedded backend) — both to be recorded.

Update on landing: `Plan.MD` (decision table + phase), `architecture.md` §4 (turn flow: the new
Resolve/Defer branch), `agents.md` (secrets list + the `system1` config block), and
`anamanti.example.json`.

## 13. Decisions made & open questions

**Decided (2026-09-25):**
- **Default backend = `laya-serve`** (local HTTP sidecar), with `jev`/OpenRouter as the cloud
  fallback and `laya-embedded` as an optional escape hatch. (Safe merge default stays `none`.)
- **Resolved replies are not barge-in-interruptible in v1** — the templated line is sub-second, so
  the interruptible window is negligible; matches the existing canned-reply behavior.

**Open:**
1. **Checkpoint choice / size** for the home intent set (English vs multilingual; the
   `typed-decisions` checkpoint) and its footprint in the `laya-serve` process.
2. **`laya-serve` supervision** — do we auto-start/supervise the sidecar at boot (like the music
   snapserver/librespot supervision), or assume it's run separately?
3. **Timer/duration parsing** — deterministic parser on the Resolve path, or defer any non-trivial
   duration to System-2?
4. **Confidence calibration** — is one global `min_confidence` enough, or per-intent thresholds?
5. **Telemetry** — reuse `anamanti_promptlog`/chatlog, or a dedicated decision log for
   resolve-rate/precision tuning?

## 14. Testing & QA notes

### 14.1 Automated coverage (already green)
Run from the repo root (the `rust-toolchain.toml` selects rustc 1.92 automatically; set
`CARGO_TARGET_DIR` to a writable dir if the external build drive isn't mounted):

```bash
cargo clippy --manifest-path anamanti-core/Cargo.toml --all-targets -- -D warnings
cargo test  --manifest-path anamanti-core/Cargo.toml
```

What's covered:
- **Wire codec** (`src/system1/http.rs`): `build_request` shape (state + intent choice + `other` +
  `needs_full_understanding` noul; model omitted for laya-serve) and `interpret` — resolve on
  confident closed intent, defer on low confidence / needs-full / `other` / unknown / malformed.
- **Duration parser** (`src/system1/mod.rs::parse_duration_secs`): digits + number words, `a`/`an`,
  `half`/`quarter`, compound ("1 hour 30 minutes"); defers on no-unit / cancel / empty.
- **NoDecision + MockDecider** behaviour.
- **Pipeline fast paths** (`tests/pipeline.rs`): weather and timer each drive a full turn with a
  **panicking LLM** — proving System-2 is never called on a resolve — and assert the
  `anamanti-weather` / `anamanti-timer` frame + templated reply + final `audio-stop`.
- **Settings swap** (`src/settings.rs::apply_system1_*`): `none → laya-serve → none`, unknown
  backend rejected leaves the engine untouched, `system1_view` reports the choice.
- **Config page** (`src/webconfig.rs`): `/system1` renders with nav; `/system1/status.json` reports
  the default disabled engine.

### 14.2 Manual QA — NOT yet verified (do these before shipping)
These need a real sidecar / device and could not be exercised unattended:

1. **`laya-serve` end-to-end.** Install + run the sidecar, then point the Core at it:
   ```bash
   cargo install laya-decision-serve
   LAYA_PORT=8000 LAYA_DEVICE=metal LAYA_PRELOAD=1 laya-serve
   # sanity-check the contract directly:
   curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
     "state":{"message":"show me the weather"},
     "questions":{"intent":{"type":"choice","instructions":"...","criteria":{"weather":"forecast","other":"else"}},
                  "needs_full_understanding":{"type":"noul","instructions":"..."}}}'
   ```
   Set `system1.backend="laya-serve"` in `anamanti.json` (or via the config page), set a home
   location, then say **"what's the weather"** and **"set a timer for ten minutes"**. Confirm the
   log shows `system1 (laya-serve) resolved intent ...` and the widget/timer appears.
   **Verify the response JSON keys actually match** what `interpret` reads
   (`answers.intent.choice`, `answers.<q>.answer_confidence`, `answers.needs_full_understanding.noul`)
   against your checkpoint — the parser was written from the README/source, not a live server.

2. **Config page (`http://<config_addr>/system1`).** Switch backend none↔laya-serve↔jev; confirm
   min-confidence/base-URL/model persist across a Core restart (written to
   `anamanti_settings.json`), the OpenRouter key field never echoes back, and an invalid backend
   surfaces the error inline without changing the live engine. Confirm the model/key rows show only
   for `jev`.

3. **Jev / OpenRouter. ✅ VERIFIED LIVE 2026-09-26.** Set `system1.backend="jev"`,
   `system1.base_url="https://openrouter.ai/api"`, `OPENROUTER_API_KEY=...`. The endpoint is the
   shared **System One** API: `POST https://openrouter.ai/api/v1/systemone` with `model:
   "typesafe/jev-1.13"` + bearer auth — confirmed against the live server and OpenRouter's OpenAPI
   spec (it is NOT the OpenAI chat-completions route). Two corrections landed from this verification:
   - **Response fields.** A `choice` answer carries `choice` + **`confidence`** + `probabilities`; a
     `noul` answer carries **only `noul`** (0..1), no confidence field. The parser previously read a
     nonexistent `answer_confidence`, so every turn scored 0.0 and silently deferred. Fixed in
     `system1/http.rs` (`choice_confidence` still falls back to `answer_confidence` for a laya-serve
     build that emits the older field).
   - **Weather needed a place.** Bare "what's the weather" scores `needs_full_understanding` noul
     ~0.94 (defer); "what's the weather **in <place>**" scores ~0.30 (resolve). Putting the location
     in a `state` field alone does **not** help (~0.87) — it must be in the question text. So the
     HTTP engine now retries once: a confident but deferred [`LOCATION_INTENTS`] intent (weather) is
     re-asked with the home location folded into the transcript. Two fast System One calls still beat
     a System-2 turn. Home location comes from `DecisionRequest.location` (`LiveHomeLocation`).

   Watch for a 4xx in the Core log; any error defers to System-2, so a misconfig degrades gracefully.

4. **On-device (Echo Show).** Confirm: the weather widget opens **before** the spoken line; the
   timer starts and counts down; the follow-up mic reopens after the reply (the fast path shares
   `emit_follow_up_and_stop`); and there is **no double-speak** if a handler's precondition fails
   mid-turn (e.g. weather with no home location should defer *before* anything is emitted).

5. **Deferral / precision spot-checks.** With a backend enabled, confirm these still go to
   System-2 (no wrong fast answer): "what's the weather in Tokyo" (non-home → `needs_full`),
   "cancel my timer" (no duration), "set a timer" (no duration), open-ended questions, and anything
   below `min_confidence`. Tune `min_confidence` up if you see mis-resolves.

### 14.3 Regression / safety
- **Default off:** with `system1.backend="none"` (the default), the seam is skipped entirely — a
  quick way to confirm no behaviour change is that the full suite passes and a normal turn is
  byte-identical to before.
- **Timeout:** `HttpDecider` uses a 4 s client timeout; a hung/absent sidecar defers rather than
  stalling the turn. Worth confirming by pointing `base_url` at a dead port and checking the turn
  still completes via System-2.
- **Latency:** measure a resolved turn vs a System-2 turn (transcript → first audio) to confirm the
  win; log timestamps around the seam if needed.
