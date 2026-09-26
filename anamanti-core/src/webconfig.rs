//! Local HTTP **config page** served from the orchestrator (rollout scope
//! addition, `memory_and_provider_rollout.md`).
//!
//! Phase 6 already made the LLM backend / voice runtime-swappable via Wyoming
//! control frames from the device (`control.rs`). This adds a second front door
//! for the same [`SharedSettings`]: a tiny web page you can open from any browser
//! on the LAN to see and change the live settings without the device.
//!
//! **No authentication.** This is a convenience surface for a trusted home
//! network; bind it to loopback (the default) unless you understand the exposure.
//! It is deliberately hand-rolled over `tokio` TCP — the same style as the
//! Wyoming protocol here — so it pulls in no HTTP framework dependency.
//!
//! It also serves a small set of **read-only debug pages** over the same socket
//! ([`DebugSources`]) so you can inspect what the assistant is doing from a
//! browser: the chat log, the exact prompt sent to the LLM, the SQLite memory
//! store, and the HelixDB GraphRAG store. These are strictly read-only.
//!
//! Routes:
//! - `GET /`         → the HTML config page.
//! - `GET /config`   → the live [`SettingsView`](crate::settings::SettingsView) as JSON.
//! - `GET /models`   → the selectable LLM models for the model dropdown.
//! - `GET /voices`   → the installed Piper voices for the TTS voice dropdown.
//! - `POST /config`  → apply a `{llm_backend?, llm_model?, tts_voice?}` change
//!   (same JSON shape as the `ambient-set-settings` control frame; a `tts_voice`
//!   of `null` clears the voice) and return the resulting settings.
//! - `GET /household`   → the household editor page; `GET /household/status.json`
//!   → the canonical home location + units + people; `POST /household/save` replaces
//!   the whole record (location, units, roster).
//! - `GET /chatlog`     → debug page; `GET /chatlog.json?limit=N` → recent turns.
//! - `GET /prompts`     → debug page; `GET /prompts.json?limit=N` → recent LLM prompts.
//! - `GET /sqlite`      → debug page; `GET /sqlite.json`  → the memory store rows.
//! - `GET /helix`       → debug page; `GET /helix.json`   → GraphRAG node stats + sample.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::llm::anthropic_auth::AnthropicAuth;
use crate::llm::catalog::ModelCatalog;
use crate::memory::{chatlog, promptlog, GraphView, MemoryStore};
use crate::music::{ManagedProc, MusicHub};
use crate::notify::{Notification, NotificationService};
use crate::orchestrator::ServiceConnector;
use crate::settings::{
    CadoraUpdate, DirectionsUpdate, DriveUpdate, Household, HouseholdMember, LlmEngine,
    SettingsUpdate, SharedSettings, SpotifyUpdate, System1Update,
};

/// Read-only data sources the debug pages render (chat log, prompts, SQLite,
/// HelixDB). Cheaply cloneable — everything is behind an `Arc` or a small `PathBuf`.
#[derive(Clone)]
pub struct DebugSources {
    /// The SQLite memory store (rendered by `/sqlite`).
    pub memory: Arc<MemoryStore>,
    /// Path to the append-only chat log JSONL (rendered by `/chatlog`).
    pub chatlog_path: PathBuf,
    /// Path to the append-only prompt log JSONL (rendered by `/prompts`).
    pub promptlog_path: PathBuf,
    /// The SQLite database path (shown for context on `/sqlite`).
    pub db_path: PathBuf,
    /// The HelixDB store root (shown for context on `/helix`).
    pub helix_path: PathBuf,
    /// Active memory retrieval backend label (`"sqlite"` or `"helix"`).
    pub memory_backend: String,
    /// Read-only view of the graph store when GraphRAG is live; `None` otherwise
    /// (feature off, SQLite backend, or init failed → `/helix` reports disabled).
    pub graph: Option<Arc<dyn GraphView>>,
}

// Build metadata for the About tab, captured at compile time by `build.rs` so a
// running orchestrator can report exactly which build it is.
const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_SHA: &str = env!("ANAMANTI_GIT_SHA");
const GIT_BRANCH: &str = env!("ANAMANTI_GIT_BRANCH");
const BUILD_TIME: &str = env!("ANAMANTI_BUILD_TIME");

/// `/about` body — the running build's version, git commit, and build time, so you
/// can confirm what's actually deployed. Values are compile-time constants.
fn about_body() -> String {
    let cell = |v: &str| {
        if v.is_empty() {
            "—".to_string()
        } else {
            v.to_string()
        }
    };
    format!(
        "<p class=\"sub\">The running orchestrator build — use this to confirm what's deployed.</p>\
         <div class=\"card\"><h2>Build</h2>\
         <p class=\"desc\">Version and provenance of the binary currently serving this page.</p>\
         <table><tbody>\
         <tr><th>Version</th><td>{version}</td></tr>\
         <tr><th>Git branch</th><td>{branch}</td></tr>\
         <tr><th>Git commit</th><td><code>{sha}</code></td></tr>\
         <tr><th>Built (UTC)</th><td>{built}</td></tr>\
         <tr><th>Drive support</th><td>yes — Photos tab + <code>ambient-get-drive-token</code></td></tr>\
         </tbody></table></div>",
        version = cell(VERSION),
        branch = cell(GIT_BRANCH),
        sha = cell(GIT_SHA),
        built = cell(BUILD_TIME),
    )
}

/// `/about.json` — machine-readable build metadata (same values as the About page).
fn about_json() -> String {
    json!({
        "ok": true,
        "version": VERSION,
        "git_branch": GIT_BRANCH,
        "git_sha": GIT_SHA,
        "build_time": BUILD_TIME,
    })
    .to_string()
}

/// Default cap on how many log records a debug page returns per request.
const DEFAULT_LOG_LIMIT: usize = 100;
/// Hard cap, so a hand-typed `?limit=` can't ask the server to buffer the world.
const MAX_LOG_LIMIT: usize = 1000;

/// The Config page body. Authored as a real HTML file (with proper tooling) and
/// compiled into the binary via `include_str!` — no runtime asset files, so the
/// single copied release binary still serves the whole admin UI with nothing beside
/// it. Same rationale for every asset below.
const CONFIG_BODY: &str = include_str!("webconfig/config.html");

/// The shared admin-panel design system (split-pane layout, cards, form controls,
/// pill toggles, tables, danger zone). Used by every page, Config included.
const PANEL_CSS: &str = include_str!("webconfig/panel.css");

/// Client-side helpers shared by every page (`esc`/`fmtTime`/`getJSON`).
const SHELL_SCRIPT: &str = include_str!("webconfig/shell.js");

/// The left navigation for the split-pane shell: a brand block plus grouped links,
/// with `active` highlighted. Each item carries a small inline icon. The link
/// `href`s are kept verbatim (`href="/music"` etc.) — the device/tests rely on them.
fn sidebar_html(active: &str) -> String {
    // (href, label, inline-svg-body) for one nav link, grouped under a section title.
    type Link = (&'static str, &'static str, &'static str);
    type Group = (&'static str, &'static [Link]);
    const GROUPS: [Group; 4] = [
        (
            "Settings",
            &[
                (
                    "/",
                    "Config",
                    r##"<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M1 14h6"/><path d="M9 8h6"/><path d="M17 16h6"/>"##,
                ),
                (
                    "/household",
                    "Household",
                    r##"<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>"##,
                ),
                (
                    "/tools",
                    "Tools",
                    r##"<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.3 2.3-2-2z"/>"##,
                ),
                (
                    "/system1",
                    "System-1",
                    r##"<path d="M13 2L3 14h9l-1 8 10-12h-9z"/>"##,
                ),
                (
                    "/notifications",
                    "Notify",
                    r##"<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/>"##,
                ),
                (
                    "/music",
                    "Music",
                    r##"<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>"##,
                ),
                (
                    "/drive",
                    "Photos",
                    r##"<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/>"##,
                ),
            ],
        ),
        (
            "Memory",
            &[
                (
                    "/sqlite",
                    "SQLite",
                    r##"<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>"##,
                ),
                (
                    "/helix",
                    "HelixDB",
                    r##"<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4"/>"##,
                ),
            ],
        ),
        (
            "Logs",
            &[
                (
                    "/chatlog",
                    "Chat log",
                    r##"<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>"##,
                ),
                (
                    "/prompts",
                    "Prompts",
                    r##"<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h8"/>"##,
                ),
            ],
        ),
        (
            "System",
            &[(
                "/about",
                "About",
                r##"<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>"##,
            )],
        ),
    ];
    let mut out = String::from(
        "<aside class=\"sidebar\">\
         <div class=\"brand\"><span class=\"title\">Anamanti Core</span>\
         <span class=\"tag\">Admin console</span></div><nav class=\"nav\">",
    );
    for (group, links) in GROUPS {
        out.push_str(&format!("<div class=\"group\">{group}</div>"));
        for (href, label, icon) in links.iter() {
            let cls = if *href == active { " active" } else { "" };
            out.push_str(&format!(
                "<a class=\"navlink{cls}\" href=\"{href}\">\
                 <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke-width=\"2\" \
                 stroke-linecap=\"round\" stroke-linejoin=\"round\">{icon}</svg>{label}</a>"
            ));
        }
    }
    out.push_str("</nav></aside>");
    out
}

/// Wrap a page `body` in the shared HTML shell: `<head>` + design-system CSS, then a
/// split-pane `<div class="layout">` with the sidebar on the left and a scrolling
/// content pane on the right (page header + shared helper script + the page body).
/// The shared helper script (`esc`/`fmtTime`/`getJSON`) is emitted **before** the
/// body so a body's inline `load()` (which runs as it is parsed) can rely on it.
fn page(active: &str, title: &str, body: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>Ambient — {title}</title><style>{PANEL_CSS}</style></head>\
         <body><div class=\"layout\">{nav}\
         <main class=\"content\"><div class=\"content-inner\">\
         <div class=\"page-head\"><h1>{title}</h1></div>\
         <script>{SHELL_SCRIPT}</script>{body}\
         </div></main></div></body></html>",
        nav = sidebar_html(active),
    )
}

/// `/chatlog` body — every completed turn, newest first.
const CHATLOG_BODY: &str = include_str!("webconfig/chatlog.html");

/// `/prompts` body — the exact assembled LLM prompt per turn, newest first.
const PROMPTS_BODY: &str = include_str!("webconfig/prompts.html");

/// `/sqlite` body — the persistent memory store rows.
const SQLITE_BODY: &str = include_str!("webconfig/sqlite.html");

/// `/helix` body — GraphRAG node counts + a sample of nodes per label.
const HELIX_BODY: &str = include_str!("webconfig/helix.html");

/// `/music` body — start/stop the music sibling processes and see snapserver status.
const MUSIC_BODY: &str = include_str!("webconfig/music.html");

/// `/drive` body — link a Google Drive folder for the idle photo slideshow. The
/// orchestrator runs the one-time OAuth consent here (a browser opens on the Mac);
/// the tablet then pulls the token over Wyoming. Uses a private `apiJSON` helper —
/// NOT the shared `getJSON`, which the page shell redefines (single-arg, GET-only)
/// in a script appended after this one, so calling it here would silently downgrade
/// our POSTs to GET.
const DRIVE_BODY: &str = include_str!("webconfig/drive.html");

/// `/tools` body — credentials for the LLM tools that need a secret. v1 hosts the
/// **Mapbox token** for the `directions_lookup` tool; a saved token rebuilds the tool
/// set live (no restart). Uses a private `apiJSON` helper (not the shell's GET-only
/// `getJSON`, which is appended after this script).
const TOOLS_BODY: &str = include_str!("webconfig/tools.html");

/// `/notifications` body — push a proactive (visual-only) notification to the
/// display for testing, and see how many device notify channels are connected. Uses
/// a private `apiJSON` helper (not the shell's GET-only `getJSON`).
const NOTIFY_BODY: &str = include_str!("webconfig/notifications.html");

/// `/household` body — edit the canonical household + home information: the home
/// location + units (grounds "here" for weather/nearby questions) and the roster of
/// people who live here with their emails + phone numbers. A full-record save. Uses
/// a private `apiJSON` helper (not the shell's GET-only `getJSON`).
const HOUSEHOLD_BODY: &str = include_str!("webconfig/household.html");
const SYSTEM1_BODY: &str = include_str!("webconfig/system1.html");

/// Cap on request bytes we buffer before the body — a config request is tiny; this
/// just bounds a misbehaving/hostile client on the (unauthenticated) socket.
const MAX_REQUEST_BYTES: usize = 64 * 1024;

/// Accept config-page connections forever, one task per connection. Returns only if
/// the listener itself fails.
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    listener: TcpListener,
    settings: Arc<SharedSettings>,
    catalog: Arc<ModelCatalog>,
    connector: Arc<dyn ServiceConnector>,
    voices_dir: Option<PathBuf>,
    debug: DebugSources,
    music: Option<MusicHub>,
    notify: Arc<NotificationService>,
) -> Result<()> {
    loop {
        let (stream, peer) = listener.accept().await?;
        let settings = settings.clone();
        let catalog = catalog.clone();
        let connector = connector.clone();
        let voices_dir = voices_dir.clone();
        let debug = debug.clone();
        let music = music.clone();
        let notify = notify.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(
                stream, settings, catalog, connector, voices_dir, debug, music, notify,
            )
            .await
            {
                log::debug!("config page connection {peer} ended: {e:#}");
            }
        });
    }
}

/// Read one HTTP request, route it, write one response, close. One request per
/// connection (`Connection: close`) — ample for a settings page.
#[allow(clippy::too_many_arguments)]
async fn handle(
    mut stream: TcpStream,
    settings: Arc<SharedSettings>,
    catalog: Arc<ModelCatalog>,
    connector: Arc<dyn ServiceConnector>,
    voices_dir: Option<PathBuf>,
    debug: DebugSources,
    music: Option<MusicHub>,
    notify: Arc<NotificationService>,
) -> Result<()> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];

    // Read until the end of the header block.
    let header_end = loop {
        if let Some(pos) = find(&buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > MAX_REQUEST_BYTES {
            return write_response(
                &mut stream,
                "413 Payload Too Large",
                "text/plain",
                b"too large",
            )
            .await;
        }
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(()); // client closed before sending a full request
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = normalize_target(parts.next().unwrap_or("/"));

    let content_length = lines
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
        .min(MAX_REQUEST_BYTES);

    // Read the body (already-buffered bytes plus whatever remains on the socket).
    let body_start = header_end + 4;
    let mut body = buf[body_start..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);

    // The model list needs an async catalog fetch and the voice list an async Piper
    // `describe`, so both are handled here rather than in the pure `route` function.
    let path = target.split(['?', '#']).next().unwrap_or(&target);
    if method == "GET" && path == "/models" {
        let payload = models_json(&catalog).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "GET" && path == "/voices" {
        // Reuse the exact device-facing logic: Piper's catalog intersected with the
        // installed voices when a voices dir is configured.
        let ev = crate::control::voices_response(connector.as_ref(), voices_dir.as_deref()).await;
        let payload = ev.data.to_string().into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Google Drive photo linkage status (booleans + folder ids; never secrets).
    if method == "GET" && path == "/drive/status.json" {
        let payload = drive_status_json(&settings).into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Directions tool (Mapbox) status — only a token-set boolean, never the token.
    if method == "GET" && path == "/tools/status.json" {
        let payload = directions_status_json(&settings).into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Save the Mapbox token for the directions tool (rebuilds the tool set live).
    if method == "POST" && path == "/tools/save" {
        let payload = directions_save_json(&settings, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }

    // System-1 fast-decision engine: current selection + a live swap (rebuilds the
    // engine, persists, takes effect on the next turn).
    if method == "GET" && path == "/system1/status.json" {
        let payload = system1_status_json(&settings).into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/system1/save" {
        let payload = system1_save_json(&settings, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }

    // Proactive notifications: how many device notify channels are connected, and a
    // button to push a test notification down them (Approach A, visual-only).
    if method == "GET" && path == "/notifications/status.json" {
        let payload = json!({ "connected": notify.connected() }).to_string();
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    if method == "POST" && path == "/notifications/test" {
        let payload = notifications_test_json(&notify, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }

    // Debug/inspection data endpoints — need I/O and the debug sources, so they're
    // handled here (like `/models`) rather than in the pure `route` function.
    if method == "GET" {
        let json = match path {
            "/chatlog.json" => Some(chatlog_json(&debug, limit_param(&target))),
            "/prompts.json" => Some(prompts_json(&debug, limit_param(&target))),
            "/sqlite.json" => Some(sqlite_json(&debug)),
            "/helix.json" => Some(helix_json(&debug).await),
            _ => None,
        };
        if let Some(body) = json {
            return write_response(&mut stream, "200 OK", "application/json", body.as_bytes())
                .await;
        }
    }

    // The only debug mutation: rename a GraphRAG `Entity` (fix a misspelled fact).
    // Needs the async graph handle + request body, so it's handled here.
    if method == "POST" && path == "/helix/rename-entity" {
        let payload = helix_rename_json(&debug, &body).await;
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }

    // Music control endpoints — need async I/O (process spawn, snapserver JSON-RPC,
    // mpv IPC) and the `MusicHub`, so they're handled here like `/models`.
    if method == "GET" && path == "/music/status.json" {
        let payload = music_status_json(music.as_ref()).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/music/proc" {
        let payload = music_proc_json(music.as_ref(), &body).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/music/play" {
        let payload = music_play_json(music.as_ref(), &body).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/music/stopweb" {
        let payload = music_stopweb_json(music.as_ref()).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/music/startall" {
        let payload = music_startall_json(music.as_ref()).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    if method == "POST" && path == "/music/stopall" {
        let payload = music_stopall_json(music.as_ref()).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // List the linked account's Drive folders for the Photos-page picker. Needs
    // async I/O (mints an access token, calls the Drive API), so it lives here.
    if method == "GET" && path == "/drive/folders.json" {
        let payload = drive_folders_json(&settings).await.into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Save the Drive OAuth client credentials / folder ids (no consent yet).
    if method == "POST" && path == "/drive/save" {
        let payload = drive_save_json(&settings, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    // Run the one-time OAuth consent (opens a browser on the Mac) and store the
    // resulting refresh token. Blocks this connection until consent completes or
    // times out — fine for a single admin request on the loopback page.
    if method == "POST" && path == "/drive/link" {
        let payload = drive_link_json(&settings).await;
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    // Spotify voice-control linkage status (booleans + device name; never secrets).
    if method == "GET" && path == "/spotify/status.json" {
        let payload = spotify_status_json(&settings).into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Save the Spotify app credentials + device name (no consent yet).
    if method == "POST" && path == "/spotify/save" {
        let payload = spotify_save_json(&settings, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    // Run the one-time Spotify OAuth consent (opens a browser on the Mac) and store
    // the resulting refresh token. Blocks this connection until consent completes or
    // times out — fine for a single admin request on the loopback page.
    if method == "POST" && path == "/spotify/link" {
        let payload = spotify_link_json(&settings).await;
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    // Cadora shopping-list linkage status (booleans + base URL; never the token).
    if method == "GET" && path == "/cadora/status.json" {
        let payload = cadora_status_json(&settings).into_bytes();
        return write_response(&mut stream, "200 OK", "application/json", &payload).await;
    }
    // Set the Cadora app-server base URL (no linking yet).
    if method == "POST" && path == "/cadora/save" {
        let payload = cadora_save_json(&settings, &body);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }
    // Redeem a spoken 6-digit pairing code for a durable voice-link token. Makes a
    // network call to the Cadora server; blocks this loopback connection until it
    // returns — fine for a single admin request.
    if method == "POST" && path == "/cadora/link" {
        let payload = cadora_link_json(&settings, &body).await;
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.as_bytes(),
        )
        .await;
    }

    let (status, content_type, payload) = route(&method, &target, &body, &settings);
    write_response(&mut stream, status, content_type, &payload).await
}

/// The selectable models as a JSON string (`{ "ok": true, "models": [...] }`).
async fn models_json(catalog: &ModelCatalog) -> String {
    let models: Vec<Value> = catalog
        .models()
        .await
        .into_iter()
        .map(|m| json!({ "provider": m.provider, "id": m.id, "label": m.label }))
        .collect();
    json!({ "ok": true, "models": models }).to_string()
}

/// Parse a `?limit=N` query parameter, clamped to `[1, MAX_LOG_LIMIT]`; absent or
/// unparseable falls back to [`DEFAULT_LOG_LIMIT`].
fn limit_param(target: &str) -> usize {
    target
        .split(['?', '#'])
        .nth(1)
        .into_iter()
        .flat_map(|q| q.split('&'))
        .find_map(|pair| pair.strip_prefix("limit=")?.parse::<usize>().ok())
        .map(|n| n.clamp(1, MAX_LOG_LIMIT))
        .unwrap_or(DEFAULT_LOG_LIMIT)
}

/// `{ ok, records: [ChatLogRecord…] }` for the newest `limit` turns.
fn chatlog_json(debug: &DebugSources, limit: usize) -> String {
    match chatlog::read_tail(&debug.chatlog_path, limit) {
        Ok(records) => json!({ "ok": true, "records": records }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `{ ok, records: [PromptLogRecord…] }` for the newest `limit` prompts.
fn prompts_json(debug: &DebugSources, limit: usize) -> String {
    match promptlog::read_tail(&debug.promptlog_path, limit) {
        Ok(records) => json!({ "ok": true, "records": records }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `{ ok, count, db_path, memory_backend, memories: [...] }` — the whole memory store.
fn sqlite_json(debug: &DebugSources) -> String {
    match debug.memory.list() {
        Ok(items) => {
            let memories: Vec<Value> = items
                .iter()
                .map(|m| {
                    json!({
                        "id": m.id,
                        "kind": m.kind.as_str(),
                        "content": m.content,
                        "source": m.source.as_str(),
                        "created_at": m.created_at,
                        "speaker_id": m.speaker_id,
                    })
                })
                .collect();
            json!({
                "ok": true,
                "count": memories.len(),
                "db_path": debug.db_path.display().to_string(),
                "memory_backend": debug.memory_backend,
                "memories": memories,
            })
            .to_string()
        }
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `{ ok, enabled, helix_path, stats, nodes }` — GraphRAG store overview, or
/// `{ ok, enabled: false, message }` when the graph backend is not live. The
/// message distinguishes "not configured for HelixDB" from "configured for HelixDB
/// but init failed, so recall fell back to SQLite FTS" (commonly a missing
/// `OPENAI_API_KEY`), so the page doesn't tell you to set a value that is already set.
async fn helix_json(debug: &DebugSources) -> String {
    let Some(graph) = &debug.graph else {
        let message = if debug.memory_backend.eq_ignore_ascii_case("helix") {
            "GraphRAG (HelixDB) is configured (memory_backend=\"helix\") but failed to \
             initialize, so recall fell back to SQLite FTS. The usual cause is a missing \
             OPENAI_API_KEY (required for embeddings); a HelixDB store that could not be \
             opened — e.g. two orchestrators sharing one helix_path — does this too. Check \
             the orchestrator log for \"GraphRAG init failed\"."
        } else {
            "GraphRAG (HelixDB) is not active. Set memory_backend=\"helix\" in \
             anamanti.json to enable it."
        };
        return json!({
            "ok": true,
            "enabled": false,
            "message": message,
        })
        .to_string();
    };
    let stats = match graph.stats().await {
        Ok(v) => v,
        Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    };
    let nodes = match graph.sample(50).await {
        Ok(v) => v,
        Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    };
    json!({
        "ok": true,
        "enabled": true,
        "helix_path": debug.helix_path.display().to_string(),
        "stats": stats,
        "nodes": nodes,
    })
    .to_string()
}

/// POST `/helix/rename-entity` — body `{ "old_name": "...", "new_name": "..." }`.
/// Fixes a misspelled entity everywhere: the `Entity` node's `name` (id + edges
/// preserved) plus whole-word occurrences in `Turn.text` / `Memory.content`.
/// Returns `{ ok: true, entities, turns, memories, total }`, or `{ ok: false,
/// message }` when the graph is inactive, the request is malformed, the name was
/// found nowhere, or the target already names a different entity.
async fn helix_rename_json(debug: &DebugSources, body: &[u8]) -> String {
    let Some(graph) = &debug.graph else {
        return json!({ "ok": false, "message": "GraphRAG (HelixDB) is not active." }).to_string();
    };
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let old_name = data
        .get("old_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let new_name = data
        .get("new_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if old_name.is_empty() || new_name.is_empty() {
        return json!({
            "ok": false,
            "message": "rename requires non-empty `old_name` and `new_name`",
        })
        .to_string();
    }
    match graph.rename_entity(old_name, new_name).await {
        Ok(counts) => {
            let total = counts.get("total").and_then(Value::as_u64).unwrap_or(0);
            if total == 0 {
                json!({ "ok": false, "message": format!("no occurrences of {old_name:?} found") })
                    .to_string()
            } else {
                json!({
                    "ok": true,
                    "entities": counts.get("entities").cloned().unwrap_or(json!(0)),
                    "turns": counts.get("turns").cloned().unwrap_or(json!(0)),
                    "memories": counts.get("memories").cloned().unwrap_or(json!(0)),
                    "total": total,
                })
                .to_string()
            }
        }
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `GET /music/status.json` — supervised process state + live snapserver status.
/// `{ ok, enabled, snapserver_addr, procs: [...], snapserver: { reachable, groups } }`.
async fn music_status_json(music: Option<&MusicHub>) -> String {
    let Some(hub) = music else {
        return json!({ "ok": true, "enabled": false }).to_string();
    };
    let procs = serde_json::to_value(hub.supervisor.status().await).unwrap_or_else(|_| json!([]));
    let snapserver = match hub.snapcast.get_status().await {
        Ok(status) => {
            let groups: Vec<Value> = status
                .groups
                .iter()
                .map(|g| {
                    let clients: Vec<Value> = g
                        .clients
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "name": c.host.name,
                                "volume": c.volume_percent(),
                                "muted": c.volume_muted(),
                            })
                        })
                        .collect();
                    json!({ "id": g.id, "stream": g.stream_id, "muted": g.muted, "clients": clients })
                })
                .collect();
            json!({ "reachable": true, "groups": groups })
        }
        Err(_) => json!({ "reachable": false }),
    };
    json!({
        "ok": true,
        "enabled": true,
        "snapserver_addr": hub.snapserver_addr,
        "procs": procs,
        "snapserver": snapserver,
    })
    .to_string()
}

/// `GET /drive/status.json` — the Drive photo linkage state for the `/drive` page.
/// Reports only booleans + folder ids + scope; the client secret and refresh token
/// are never included (they leave the orchestrator only over the device Wyoming hop).
fn drive_status_json(settings: &SharedSettings) -> String {
    let d = settings.drive();
    json!({
        "ok": true,
        "configured": d.configured(),
        "linked": d.linked(),
        "client_id_set": d.client_id.as_deref().is_some_and(|s| !s.is_empty()),
        "client_secret_set": d.client_secret.as_deref().is_some_and(|s| !s.is_empty()),
        "has_refresh_token": d.refresh_token.as_deref().is_some_and(|s| !s.is_empty()),
        "folder_ids": d.folder_ids,
        "scope": d.scope.unwrap_or_default(),
    })
    .to_string()
}

/// `GET /tools/status.json` — the directions tool state for the `/tools` page.
/// Reports only whether a Mapbox token is set (never the token) and whether the
/// `directions_lookup` tool is therefore active (v1 provider is always Mapbox).
fn directions_status_json(settings: &SharedSettings) -> String {
    let token_set = settings.mapbox_token_set();
    json!({
        "ok": true,
        "token_set": token_set,
        "tool_active": token_set,
    })
    .to_string()
}

/// `POST /tools/save` — set the Mapbox token for the directions tool. A blank/absent
/// token is left unchanged (a page reload never wipes the stored token); a non-empty
/// value sets it and rebuilds the backend so `directions_lookup` activates live.
fn directions_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let mapbox_token = match data.get("mapbox_token").and_then(Value::as_str) {
        Some(s) if !s.trim().is_empty() => Some(Some(s.trim().to_string())),
        _ => None,
    };
    settings.apply_directions(&DirectionsUpdate { mapbox_token });
    directions_status_json(settings)
}

/// Current System-1 selection for the config page. Never returns the OpenRouter key,
/// only whether one is set.
fn system1_status_json(settings: &SharedSettings) -> String {
    let v = settings.system1_view();
    json!({
        "ok": true,
        "backend": v.backend,
        "base_url": v.base_url,
        "model": v.model,
        "min_confidence": v.min_confidence,
        "openrouter_key_set": v.openrouter_key_set,
        "intents": v.intents,
        "active": v.backend != "none",
    })
    .to_string()
}

/// Apply a System-1 selection from the config page (rebuilds the engine live + persists).
/// A blank `openrouter_api_key` means "leave unchanged" (never shown back). On a build
/// failure (e.g. unknown backend) the current engine is left untouched and the error is
/// returned.
fn system1_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let opt_str = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
    };
    let update = System1Update {
        backend: opt_str("backend").filter(|s| !s.is_empty()),
        base_url: opt_str("base_url").filter(|s| !s.is_empty()),
        model: opt_str("model").filter(|s| !s.is_empty()),
        min_confidence: data.get("min_confidence").and_then(Value::as_f64),
        // Blank = keep the current key (it is never echoed back to the page).
        openrouter_api_key: match data.get("openrouter_api_key").and_then(Value::as_str) {
            Some(s) if !s.trim().is_empty() => Some(Some(s.trim().to_string())),
            _ => None,
        },
        intents: None,
    };
    match settings.apply_system1(&update) {
        Ok(_) => system1_status_json(settings),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `GET /household/status.json` — the canonical household record (home location +
/// units + roster). Contact details are shown so they can be edited on the page;
/// this surface is loopback + unauthenticated by design (same as the rest).
fn household_status_json(settings: &SharedSettings) -> String {
    let h = settings.household();
    let members: Vec<Value> = h
        .members
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "emails": m.emails,
                "phones": m.phones,
                "relationship": m.relationship,
            })
        })
        .collect();
    json!({
        "ok": true,
        "location": h.location,
        "weather_units": h.weather_units,
        "members": members,
    })
    .to_string()
}

/// `POST /household/save` — replace the whole household record (location, units,
/// roster). The value is sanitized on apply (trim, drop blanks / nameless members).
fn household_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let opt_str = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let str_list = |v: Option<&Value>| {
        v.and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let members = data
        .get("members")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|m| HouseholdMember {
                    name: opt_str(m.get("name")).unwrap_or_default(),
                    emails: str_list(m.get("emails")),
                    phones: str_list(m.get("phones")),
                    relationship: opt_str(m.get("relationship")),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let household = Household {
        location: opt_str(data.get("location")),
        weather_units: opt_str(data.get("weather_units")),
        members,
    };
    settings.apply_household(&household);
    household_status_json(settings)
}

/// `POST /music/proc` — body `{ proc, action }` starts/stops a managed process.
async fn music_proc_json(music: Option<&MusicHub>, body: &[u8]) -> String {
    let Some(hub) = music else {
        return json!({ "ok": false, "message": "music routing is disabled" }).to_string();
    };
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let Some(proc) = data
        .get("proc")
        .and_then(Value::as_str)
        .and_then(ManagedProc::from_key)
    else {
        return json!({ "ok": false, "message": "unknown proc" }).to_string();
    };
    let action = data.get("action").and_then(Value::as_str).unwrap_or("");
    let res = match action {
        "start" => hub.supervisor.start(proc).await,
        "stop" => hub.supervisor.stop(proc).await,
        other => Err(anyhow::anyhow!("unknown action {other:?}")),
    };
    match res {
        Ok(()) => json!({ "ok": true }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `POST /notifications/test` — push a proactive notification to every connected
/// device notify channel. Body: `{ title?, body?, priority? }` (all optional; sane
/// defaults). Returns `{ ok, delivered }` — the number of channels it reached (0 if
/// the device isn't currently holding a notify channel open).
fn notifications_test_json(notify: &NotificationService, body: &[u8]) -> String {
    let data: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let field = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let note = Notification {
        id: notify.new_id(),
        priority: field("priority").unwrap_or_else(|| "info".to_string()),
        title: field("title").unwrap_or_else(|| "Test notification".to_string()),
        body: field("body").unwrap_or_else(|| "Hello from the orchestrator.".to_string()),
    };
    let delivered = notify.notify(&note);
    json!({ "ok": true, "delivered": delivered }).to_string()
}

/// `POST /music/play` — body `{ url }` loads a URL in the mpv web player.
async fn music_play_json(music: Option<&MusicHub>, body: &[u8]) -> String {
    let Some(hub) = music else {
        return json!({ "ok": false, "message": "music routing is disabled" }).to_string();
    };
    let Some(mpv) = &hub.mpv else {
        return json!({ "ok": false, "message": "no mpv IPC configured (music.web_ipc)" })
            .to_string();
    };
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let url = data.get("url").and_then(Value::as_str).unwrap_or("").trim();
    if url.is_empty() {
        return json!({ "ok": false, "message": "empty url" }).to_string();
    }
    match mpv.load(url).await {
        Ok(()) => json!({ "ok": true }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `POST /music/stopweb` — stop the mpv web player.
async fn music_stopweb_json(music: Option<&MusicHub>) -> String {
    let Some(hub) = music else {
        return json!({ "ok": false, "message": "music routing is disabled" }).to_string();
    };
    let Some(mpv) = &hub.mpv else {
        return json!({ "ok": false, "message": "no mpv IPC configured (music.web_ipc)" })
            .to_string();
    };
    match mpv.stop().await {
        Ok(()) => json!({ "ok": true }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `POST /music/startall` — start all managed processes (skips a snapserver that is
/// already running).
async fn music_startall_json(music: Option<&MusicHub>) -> String {
    match music {
        Some(hub) => {
            hub.start_all().await;
            json!({ "ok": true }).to_string()
        }
        None => json!({ "ok": false, "message": "music routing is disabled" }).to_string(),
    }
}

/// `POST /music/stopall` — stop every process this orchestrator started.
async fn music_stopall_json(music: Option<&MusicHub>) -> String {
    match music {
        Some(hub) => {
            hub.stop_all().await;
            json!({ "ok": true }).to_string()
        }
        None => json!({ "ok": false, "message": "music routing is disabled" }).to_string(),
    }
}

/// `POST /drive/save` — set the Drive OAuth client id/secret and folder ids. A blank
/// or absent credential is left unchanged (a page reload never wipes a stored
/// secret); `folder_ids` (array or comma string) always replaces the stored list.
fn drive_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    // Blank/absent = leave unchanged; a non-empty string sets the value.
    let opt_set = |key: &str| match data.get(key).and_then(Value::as_str) {
        Some(s) if !s.trim().is_empty() => Some(Some(s.trim().to_string())),
        _ => None,
    };
    let folder_ids = data.get("folder_ids").and_then(|v| {
        if let Some(arr) = v.as_array() {
            Some(
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect::<Vec<_>>(),
            )
        } else {
            v.as_str().map(|s| {
                s.split(',')
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty())
                    .collect()
            })
        }
    });
    settings.apply_drive(&DriveUpdate {
        client_id: opt_set("client_id"),
        client_secret: opt_set("client_secret"),
        folder_ids,
        ..Default::default()
    });
    drive_status_json(settings)
}

/// `GET /drive/folders.json` — list the linked account's Drive folders so the Photos
/// page can offer a picker instead of hand-typed folder ids. Mints an access token
/// from the stored refresh token; never returns secrets.
async fn drive_folders_json(settings: &SharedSettings) -> String {
    let d = settings.drive();
    let (Some(cid), Some(secret), Some(rt)) = (
        d.client_id.clone().filter(|s| !s.is_empty()),
        d.client_secret.clone().filter(|s| !s.is_empty()),
        d.refresh_token.clone().filter(|s| !s.is_empty()),
    ) else {
        return json!({
            "ok": false,
            "message": "Link Google Drive first (set the client id/secret, then Link).",
        })
        .to_string();
    };
    let token = match crate::drive_consent::mint_access_token(&cid, &secret, &rt).await {
        Ok(t) => t,
        Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    };
    match crate::drive_consent::list_folders(&token, 500).await {
        Ok(folders) => json!({ "ok": true, "folders": folders }).to_string(),
        Err(e) => json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    }
}

/// `POST /drive/link` — run the one-time OAuth consent using the stored client
/// credentials (opens a browser on the Mac), store the resulting refresh token, and
/// verify the configured folders with the immediate access token.
async fn drive_link_json(settings: &SharedSettings) -> String {
    let d = settings.drive();
    let (Some(cid), Some(secret)) = (
        d.client_id.clone().filter(|s| !s.is_empty()),
        d.client_secret.clone().filter(|s| !s.is_empty()),
    ) else {
        return json!({
            "ok": false,
            "message": "Set the Drive client id and secret first (a 'Desktop app' OAuth client).",
        })
        .to_string();
    };
    let scope = d
        .scope
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| crate::config::DEFAULT_DRIVE_SCOPE.to_string());
    let outcome = match crate::drive_consent::run_consent(
        &cid,
        &secret,
        &scope,
        std::time::Duration::from_secs(180),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    };
    settings.apply_drive(&DriveUpdate {
        refresh_token: Some(Some(outcome.refresh_token.clone())),
        scope: Some(Some(outcome.scope.clone())),
        ..Default::default()
    });
    // Best-effort verification of each folder with the freshly minted access token.
    let mut verify = Vec::new();
    if let Some(at) = &outcome.access_token {
        for fid in &d.folder_ids {
            match crate::drive_consent::verify_folder(at, fid).await {
                Ok(n) => verify.push(json!({ "folder_id": fid, "images": n })),
                Err(e) => verify.push(json!({ "folder_id": fid, "error": format!("{e:#}") })),
            }
        }
    }
    let now = settings.drive();
    json!({
        "ok": true,
        "message": "Linked. The tablet picks this up on its next refresh (or reboot).",
        "linked": now.linked(),
        "folder_ids": now.folder_ids,
        "verify": verify,
    })
    .to_string()
}

/// `GET /spotify/status.json` — the Spotify voice-control linkage state for the
/// `/music` page. Reports only booleans + the device name; the client secret and
/// refresh token are never included.
fn spotify_status_json(settings: &SharedSettings) -> String {
    let s = settings.spotify();
    json!({
        "ok": true,
        "configured": s.configured(),
        "linked": s.linked(),
        "client_id_set": s.client_id.as_deref().is_some_and(|v| !v.is_empty()),
        "client_secret_set": s.client_secret.as_deref().is_some_and(|v| !v.is_empty()),
        "has_refresh_token": s.refresh_token.as_deref().is_some_and(|v| !v.is_empty()),
        "device_name": s.device_label(),
    })
    .to_string()
}

/// `POST /spotify/save` — set the Spotify app client id/secret and device name. A
/// blank or absent credential is left unchanged (a page reload never wipes a stored
/// secret). Applying rebuilds the LLM so the `spotify_control` tool tracks linkage.
fn spotify_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    // Blank/absent = leave unchanged; a non-empty string sets the value.
    let opt_set = |key: &str| match data.get(key).and_then(Value::as_str) {
        Some(s) if !s.trim().is_empty() => Some(Some(s.trim().to_string())),
        _ => None,
    };
    settings.apply_spotify(&SpotifyUpdate {
        client_id: opt_set("client_id"),
        client_secret: opt_set("client_secret"),
        device_name: opt_set("device_name"),
        ..Default::default()
    });
    spotify_status_json(settings)
}

/// `POST /spotify/link` — run the one-time Spotify OAuth consent using the stored
/// client credentials (opens a browser on the Mac) and store the refresh token.
/// Requires the redirect `http://127.0.0.1:8888/callback` to be registered in the
/// Spotify app. On success the `spotify_control` tool activates immediately.
async fn spotify_link_json(settings: &SharedSettings) -> String {
    let s = settings.spotify();
    let (Some(cid), Some(secret)) = (
        s.client_id.clone().filter(|v| !v.is_empty()),
        s.client_secret.clone().filter(|v| !v.is_empty()),
    ) else {
        return json!({
            "ok": false,
            "message": "Set the Spotify client id and secret first (Save), then Connect.",
        })
        .to_string();
    };
    let outcome = match crate::spotify_consent::run_consent(
        &cid,
        &secret,
        crate::spotify_consent::DEFAULT_CONSENT_PORT,
        crate::spotify_consent::SPOTIFY_SCOPE,
        std::time::Duration::from_secs(180),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
    };
    settings.apply_spotify(&SpotifyUpdate {
        refresh_token: Some(Some(outcome.refresh_token)),
        scope: Some(Some(outcome.scope)),
        ..Default::default()
    });
    let now = settings.spotify();
    json!({
        "ok": true,
        "message": "Spotify linked — the voice tool is live. Try \"play some Radiohead\".",
        "linked": now.linked(),
        "device_name": now.device_label(),
    })
    .to_string()
}

/// `GET /cadora/status.json` — the Cadora shopping-list linkage state for the
/// `/household` page. Reports only booleans + the base URL; the voice-link token is
/// never included.
fn cadora_status_json(settings: &SharedSettings) -> String {
    let c = settings.cadora();
    json!({
        "ok": true,
        "linked": c.linked(),
        "base_url": c.base_url_or_default(),
    })
    .to_string()
}

/// `POST /cadora/save` — set the Cadora app-server base URL (a blank value resets it
/// to the default). Does not link. Applying rebuilds the LLM so the tool tracks state.
fn cadora_save_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    // A present `base_url` (even blank → clear to default) updates it; absent leaves it.
    let base_url = data.get("base_url").and_then(Value::as_str).map(|s| {
        let s = s.trim();
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    });
    settings.apply_cadora(&CadoraUpdate {
        base_url,
        ..Default::default()
    });
    cadora_status_json(settings)
}

/// `POST /cadora/link` — link the shopping list. Two inputs are accepted: a **6-digit
/// pairing `code`** (the primary path — NextHaul → Settings → Voice & Integrations
/// mints one via `/voice/links/pair`; we redeem it here via `/voice/links/redeem` for
/// a durable token, exactly as the "speaker" is meant to), or a directly-pasted
/// **`vl_…` token** as a fallback. On success the `shopping_list_add` tool activates
/// immediately.
async fn cadora_link_json(settings: &SharedSettings, body: &[u8]) -> String {
    let data: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return json!({ "ok": false, "message": format!("invalid JSON: {e}") }).to_string()
        }
    };
    let str_field = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
    };
    let token = str_field("token");
    let code = str_field("code");

    // An optional base_url in the same request lets the user point at a non-default
    // server before linking; persist it first so a redeem is minted against it.
    let base = str_field("base_url");
    if !base.is_empty() {
        settings.apply_cadora(&CadoraUpdate {
            base_url: Some(Some(base.to_string())),
            ..Default::default()
        });
    }

    // Resolve the durable token: redeem a 6-digit code (the NextHaul flow), else
    // accept a directly-pasted `vl_…` token (fallback / older app builds).
    let resolved = if !code.is_empty() {
        let base_url = settings.cadora().base_url_or_default();
        match crate::cadora::redeem_pairing_code(&base_url, code).await {
            Ok(t) => t,
            Err(e) => return json!({ "ok": false, "message": format!("{e:#}") }).to_string(),
        }
    } else if !token.is_empty() {
        token.to_string()
    } else {
        return json!({
            "ok": false,
            "message": "Enter the 6-digit code from NextHaul (Settings → Voice & Integrations), or paste a vl_… token.",
        })
        .to_string();
    };

    settings.apply_cadora(&CadoraUpdate {
        link_token: Some(Some(resolved)),
        ..Default::default()
    });
    let now = settings.cadora();
    json!({
        "ok": true,
        "message": "Shopping list linked — try \"add milk to the shopping list\".",
        "linked": now.linked(),
        "base_url": now.base_url_or_default(),
    })
    .to_string()
}

/// Pure request router: maps `(method, target, body)` to a response. Kept free of
/// I/O so it is unit-testable against a [`SharedSettings`].
fn route(
    method: &str,
    target: &str,
    body: &[u8],
    settings: &SharedSettings,
) -> (&'static str, &'static str, Vec<u8>) {
    let path = target.split(['?', '#']).next().unwrap_or(target);
    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/", "Anamanti Core", CONFIG_BODY).into_bytes(),
        ),
        ("GET", "/household") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/household", "Household", HOUSEHOLD_BODY).into_bytes(),
        ),
        ("GET", "/household/status.json") => (
            "200 OK",
            "application/json",
            household_status_json(settings).into_bytes(),
        ),
        ("POST", "/household/save") => (
            "200 OK",
            "application/json",
            household_save_json(settings, body).into_bytes(),
        ),
        ("GET", "/drive") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/drive", "Photos", DRIVE_BODY).into_bytes(),
        ),
        ("GET", "/tools") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/tools", "Tools", TOOLS_BODY).into_bytes(),
        ),
        ("GET", "/system1") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/system1", "System-1", SYSTEM1_BODY).into_bytes(),
        ),
        ("GET", "/notifications") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/notifications", "Notify", NOTIFY_BODY).into_bytes(),
        ),
        ("GET", "/chatlog") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/chatlog", "Chat log", CHATLOG_BODY).into_bytes(),
        ),
        ("GET", "/prompts") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/prompts", "Prompts", PROMPTS_BODY).into_bytes(),
        ),
        ("GET", "/sqlite") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/sqlite", "SQLite", SQLITE_BODY).into_bytes(),
        ),
        ("GET", "/helix") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/helix", "HelixDB", HELIX_BODY).into_bytes(),
        ),
        ("GET", "/music") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/music", "Music", MUSIC_BODY).into_bytes(),
        ),
        ("GET", "/about") => (
            "200 OK",
            "text/html; charset=utf-8",
            page("/about", "About", &about_body()).into_bytes(),
        ),
        ("GET", "/about.json") => ("200 OK", "application/json", about_json().into_bytes()),
        ("GET", "/config") => (
            "200 OK",
            "application/json",
            view_json(settings, true, None).into_bytes(),
        ),
        ("POST", "/config") => match serde_json::from_slice::<Value>(body) {
            Ok(data) => match settings.apply(&parse_update(&data)) {
                Ok(_) => (
                    "200 OK",
                    "application/json",
                    view_json(settings, true, Some("settings applied")).into_bytes(),
                ),
                Err(e) => (
                    "200 OK",
                    "application/json",
                    view_json(settings, false, Some(&format!("{e:#}"))).into_bytes(),
                ),
            },
            Err(e) => (
                "400 Bad Request",
                "application/json",
                json!({ "ok": false, "message": format!("invalid JSON: {e}") })
                    .to_string()
                    .into_bytes(),
            ),
        },
        _ => ("404 Not Found", "text/plain", b"not found".to_vec()),
    }
}

/// The current settings as a JSON string, with an `ok`/`message` envelope that
/// mirrors the Wyoming `ambient-settings` response.
fn view_json(settings: &SharedSettings, ok: bool, message: Option<&str>) -> String {
    let v = settings.view();
    let engine = match v.engine {
        LlmEngine::Native => "native",
        LlmEngine::Rig => "rig",
    };
    json!({
        "ok": ok,
        "message": message,
        "llm_backend": v.llm_backend,
        "llm_model": v.llm_model,
        "anthropic_key_set": v.anthropic_key_set,
        "openai_key_set": v.openai_key_set,
        "anthropic_oauth_token_set": v.anthropic_oauth_token_set,
        "anthropic_auth": v.anthropic_auth.as_str(),
        "tts_voice": v.tts_voice,
        "engine": engine,
        "web_search": v.web_search,
        "search_provider": v.search_provider,
        "search_key_set": v.search_key_set,
        "end_silence_ms": v.end_silence_ms,
        "voice_rms_threshold": v.voice_rms_threshold,
    })
    .to_string()
}

/// Parse a [`SettingsUpdate`] from a JSON object. Identical semantics to the
/// Wyoming control path (`control::parse_update`): an absent key leaves that
/// setting unchanged; `tts_voice: null` clears the voice.
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
    // Key: absent/empty = leave unchanged (so a page reload never wipes it);
    // explicit JSON null = clear.
    let search_api_key = match data.get("search_api_key") {
        None => None,
        Some(Value::Null) => Some(None),
        Some(v) => match v.as_str() {
            Some(s) if !s.is_empty() => Some(Some(s.to_string())),
            _ => None,
        },
    };
    let anthropic_auth = data
        .get("anthropic_auth")
        .and_then(Value::as_str)
        .map(AnthropicAuth::from_label);
    // Provider API keys, same tri-state as the search key: absent/empty = leave
    // unchanged (a page reload never wipes a stored key); explicit JSON null = clear.
    let key_field = |key: &str| match data.get(key) {
        None => None,
        Some(Value::Null) => Some(None),
        Some(v) => match v.as_str() {
            Some(s) if !s.is_empty() => Some(Some(s.to_string())),
            _ => None,
        },
    };
    SettingsUpdate {
        llm_backend: string_field("llm_backend"),
        llm_model: string_field("llm_model"),
        anthropic_api_key: key_field("anthropic_api_key"),
        openai_api_key: key_field("openai_api_key"),
        anthropic_oauth_token: key_field("anthropic_oauth_token"),
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

/// Write a minimal HTTP/1.1 response and close the connection.
async fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {len}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\
         \r\n",
        len = body.len(),
    );
    stream
        .write_all(header.as_bytes())
        .await
        .context("writing config response header")?;
    stream
        .write_all(body)
        .await
        .context("writing config response body")?;
    stream.flush().await.context("flushing config response")?;
    Ok(())
}

/// Reduce a request target to origin-form (`/path?query`). Clients behind an HTTP
/// **proxy** send the **absolute-form** target (RFC 7230 §5.3.2), e.g.
/// `POST http://127.0.0.1:8731/drive/save HTTP/1.1`; a server MUST accept it. Our
/// routing matches on the path, so strip any `scheme://authority` prefix down to the
/// first `/` (an absolute URI with no path becomes `/`). Origin-form is returned
/// unchanged.
fn normalize_target(target: &str) -> String {
    let rest = target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"));
    match rest {
        Some(after_scheme) => match after_scheme.find('/') {
            Some(i) => after_scheme[i..].to_string(),
            None => "/".to_string(),
        },
        None => target.to_string(),
    }
}

/// First index of `needle` in `haystack`, or `None`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::mock::MockLlm;

    fn settings() -> Arc<SharedSettings> {
        SharedSettings::fixed(Arc::new(MockLlm::default()), "mock", None)
    }

    fn catalog() -> Arc<ModelCatalog> {
        // No keys → the static fallback list, so the test needs no network.
        Arc::new(ModelCatalog::new(
            "http://unused",
            None,
            "http://unused",
            None,
        ))
    }

    fn notify() -> Arc<NotificationService> {
        Arc::new(NotificationService::new())
    }

    fn debug() -> DebugSources {
        DebugSources {
            memory: Arc::new(MemoryStore::open_in_memory().unwrap()),
            chatlog_path: std::env::temp_dir().join("wc_test_chatlog.jsonl"),
            promptlog_path: std::env::temp_dir().join("wc_test_promptlog.jsonl"),
            db_path: std::path::PathBuf::from(":memory:"),
            helix_path: std::path::PathBuf::from("anamanti_helix"),
            memory_backend: "sqlite".to_string(),
            graph: None,
        }
    }

    /// A connector whose `connect_tts` answers a Wyoming `describe` with a fixed
    /// voice catalog, so the `/voices` route can be exercised without a real Piper.
    struct VoiceConnector;

    #[async_trait::async_trait]
    impl ServiceConnector for VoiceConnector {
        async fn connect_stt(&self) -> Result<crate::wyoming::DynConnection> {
            anyhow::bail!("stt not used in config-page tests")
        }

        async fn connect_tts(&self) -> Result<crate::wyoming::DynConnection> {
            use crate::wyoming::protocol::{types, write_event, WyomingEvent};
            let (client, server) = tokio::io::duplex(64 * 1024);
            tokio::spawn(async move {
                let (r, w) = tokio::io::split(server);
                let mut reader = tokio::io::BufReader::new(r);
                let mut writer = w;
                let _ = crate::wyoming::protocol::read_event(&mut reader).await;
                let info = WyomingEvent::with_data(
                    types::INFO,
                    json!({ "tts": [{ "voices": [
                        { "name": "en_US-amy-medium", "languages": ["en_US"], "description": "amy (medium)" },
                        { "name": "en_US-lessac-medium", "languages": ["en_US"], "description": "lessac (medium)" },
                    ]}]}),
                );
                let _ = write_event(&mut writer, &info).await;
            });
            let (r, w) = tokio::io::split(client);
            Ok(crate::wyoming::DynConnection::from_io(r, w))
        }
    }

    fn connector() -> Arc<dyn ServiceConnector> {
        Arc::new(VoiceConnector)
    }

    #[test]
    fn get_root_serves_html() {
        let (status, ctype, body) = route("GET", "/", b"", &settings());
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        assert!(String::from_utf8_lossy(&body).contains("Anamanti Core"));
    }

    #[test]
    fn music_page_renders_with_nav() {
        let (status, ctype, body) = route("GET", "/music", b"", &settings());
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("Web-URL player"), "music page body");
        assert!(html.contains("href=\"/music\""), "nav links music");
    }

    #[tokio::test]
    async fn music_status_reports_disabled_without_a_hub() {
        let v: Value = serde_json::from_str(&music_status_json(None).await).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["enabled"], false);
    }

    #[test]
    fn get_config_reports_current_backend() {
        let (status, ctype, body) = route("GET", "/config?_=1", b"", &settings());
        assert_eq!(status, "200 OK");
        assert_eq!(ctype, "application/json");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["llm_backend"], "mock");
    }

    #[test]
    fn post_config_applies_tts_voice_and_reports_it_back() {
        let s = settings();
        let body = br#"{"tts_voice":"en_US-amy-medium"}"#;
        let (status, _ctype, out) = route("POST", "/config", body, &s);
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["tts_voice"], "en_US-amy-medium");
        assert_eq!(s.view().tts_voice.as_deref(), Some("en_US-amy-medium"));
    }

    #[test]
    fn post_config_null_voice_clears_it() {
        let s = settings();
        route("POST", "/config", br#"{"tts_voice":"amy"}"#, &s);
        route("POST", "/config", br#"{"tts_voice":null}"#, &s);
        assert_eq!(s.view().tts_voice, None);
    }

    #[test]
    fn post_config_rejects_unknown_backend_without_dropping_settings() {
        let s = settings();
        let before = s.view();
        let (status, _c, out) = route("POST", "/config", br#"{"llm_backend":"gpt"}"#, &s);
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(s.view(), before, "a rejected change leaves settings intact");
    }

    #[test]
    fn post_config_toggles_engine_and_web_search() {
        let s = settings();
        let (status, _c, out) = route(
            "POST",
            "/config",
            br#"{"engine":"rig","web_search":true}"#,
            &s,
        );
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["engine"], "rig");
        assert_eq!(v["web_search"], true);
        assert!(s.view().web_search);
    }

    #[test]
    fn post_config_sets_search_provider_and_key_without_leaking_it() {
        let s = settings();
        let (status, _c, out) = route(
            "POST",
            "/config",
            br#"{"search_provider":"tavily","search_api_key":"tvly-secret"}"#,
            &s,
        );
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["search_provider"], "tavily");
        assert_eq!(v["search_key_set"], true);
        // The key value must never appear in a POST response or a later GET.
        assert!(!String::from_utf8_lossy(&out).contains("tvly-secret"));
        let (_s, _c, g) = route("GET", "/config", b"", &s);
        assert!(!String::from_utf8_lossy(&g).contains("tvly-secret"));
        assert!(s.view().search_key_set);
    }

    #[test]
    fn post_config_sets_anthropic_key_and_selects_the_backend_without_leaking_it() {
        let s = settings(); // mock backend, no env anthropic key
        assert!(!s.view().anthropic_key_set);
        // A runtime key + backend switch in one POST enables the cloud backend with
        // no restart (mirrors entering the key on the page and choosing anthropic).
        let (status, _c, out) = route(
            "POST",
            "/config",
            br#"{"llm_backend":"anthropic","anthropic_api_key":"sk-secret"}"#,
            &s,
        );
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["llm_backend"], "anthropic");
        assert_eq!(v["anthropic_key_set"], true);
        // The key must never appear in a POST response or a later GET.
        assert!(!String::from_utf8_lossy(&out).contains("sk-secret"));
        let (_s, _c, g) = route("GET", "/config", b"", &s);
        assert!(!String::from_utf8_lossy(&g).contains("sk-secret"));
        assert!(s.view().anthropic_key_set);
    }

    #[test]
    fn about_page_and_json_report_the_build() {
        let (status, ctype, body) = route("GET", "/about", b"", &settings());
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("Version"));
        assert!(html.contains(env!("CARGO_PKG_VERSION")));
        assert!(
            html.contains("href=\"/drive\""),
            "About page missing shared nav"
        );

        let (status, ctype, body) = route("GET", "/about.json", b"", &settings());
        assert_eq!(status, "200 OK");
        assert_eq!(ctype, "application/json");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert!(v.get("git_sha").is_some());
        assert!(v.get("build_time").is_some());
    }

    #[test]
    fn get_drive_page_renders_with_nav() {
        let (status, ctype, body) = route("GET", "/drive", b"", &settings());
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("Link Google Drive"));
        assert!(html.contains("href=\"/helix\""), "missing shared nav");
    }

    #[test]
    fn get_system1_page_renders_with_nav_and_status_reports_default() {
        let s = settings();
        let (status, ctype, body) = route("GET", "/system1", b"", &s);
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        let html = String::from_utf8_lossy(&body);
        assert!(html.contains("Decision engine"), "system1 page body");
        assert!(html.contains("href=\"/system1\""), "nav links system1");
        // The status endpoint reports the default disabled engine.
        let json = system1_status_json(&s);
        assert!(json.contains("\"backend\":\"none\""), "status: {json}");
        assert!(json.contains("\"active\":false"), "status: {json}");
    }

    #[test]
    fn drive_status_reports_unconfigured_by_default() {
        let v: Value = serde_json::from_str(&drive_status_json(&settings())).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["configured"], false);
        assert_eq!(v["linked"], false);
        assert_eq!(v["client_secret_set"], false);
        assert_eq!(v["folder_ids"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn drive_folders_json_refuses_when_not_linked() {
        // Unlinked: the guard returns before any network I/O, so this is offline-safe.
        let v: Value = serde_json::from_str(&drive_folders_json(&settings()).await).unwrap();
        assert_eq!(v["ok"], false);
        assert!(
            v["message"].as_str().unwrap().contains("Link Google Drive"),
            "expected a link-first hint, got {v}"
        );
    }

    #[test]
    fn drive_save_sets_creds_and_folders_without_leaking_the_secret() {
        let s = settings();
        let body = br#"{"client_id":"cid.apps","client_secret":"gocspx-secret","folder_ids":"1AbC, 1XyZ"}"#;
        let out = drive_save_json(&s, body);
        // The status echo must never contain the secret.
        assert!(!out.contains("gocspx-secret"), "secret leaked: {out}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["configured"], true, "client id + secret now set");
        assert_eq!(v["client_secret_set"], true);
        assert_eq!(v["linked"], false, "no refresh token yet");
        assert_eq!(v["folder_ids"][0], "1AbC");
        assert_eq!(v["folder_ids"][1], "1XyZ");
        // The live settings hold the real secret, but a status read never exposes it.
        let d = s.drive();
        assert_eq!(d.client_secret.as_deref(), Some("gocspx-secret"));
        assert!(!drive_status_json(&s).contains("gocspx-secret"));
    }

    #[test]
    fn drive_save_blank_credential_leaves_the_stored_one_intact() {
        let s = settings();
        drive_save_json(&s, br#"{"client_id":"cid.apps","client_secret":"keep-me"}"#);
        // A later save with a blank secret (page reload) must not wipe it.
        drive_save_json(
            &s,
            br#"{"client_id":"cid.apps","client_secret":"","folder_ids":["1AbC"]}"#,
        );
        assert_eq!(s.drive().client_secret.as_deref(), Some("keep-me"));
        assert_eq!(s.drive().folder_ids, vec!["1AbC".to_string()]);
    }

    #[test]
    fn spotify_status_reports_unconfigured_by_default() {
        let v: Value = serde_json::from_str(&spotify_status_json(&settings())).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["configured"], false);
        assert_eq!(v["linked"], false);
        assert_eq!(v["has_refresh_token"], false);
        // The device name defaults to the librespot device name.
        assert_eq!(v["device_name"], "Ambient");
    }

    #[test]
    fn spotify_save_sets_creds_and_device_without_leaking_the_secret() {
        let s = settings();
        let body = br#"{"client_id":"spcid","client_secret":"sp-secret","device_name":"Kitchen"}"#;
        let out = spotify_save_json(&s, body);
        assert!(!out.contains("sp-secret"), "secret leaked: {out}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["configured"], true, "client id + secret now set");
        assert_eq!(v["client_secret_set"], true);
        assert_eq!(v["linked"], false, "no refresh token yet");
        assert_eq!(v["device_name"], "Kitchen");
        // The live settings hold the real secret, but a status read never exposes it.
        assert_eq!(s.spotify().client_secret.as_deref(), Some("sp-secret"));
        assert!(!spotify_status_json(&s).contains("sp-secret"));
    }

    #[test]
    fn spotify_save_blank_credential_leaves_the_stored_one_intact() {
        let s = settings();
        spotify_save_json(&s, br#"{"client_id":"spcid","client_secret":"keep-me"}"#);
        // A page reload posting a blank secret must not wipe the stored one.
        spotify_save_json(
            &s,
            br#"{"client_id":"spcid","client_secret":"","device_name":"Den"}"#,
        );
        assert_eq!(s.spotify().client_secret.as_deref(), Some("keep-me"));
        assert_eq!(s.spotify().device_label(), "Den");
    }

    #[test]
    fn tools_save_sets_mapbox_token_without_leaking_it() {
        let s = settings();
        let before: Value = serde_json::from_str(&directions_status_json(&s)).unwrap();
        assert_eq!(before["token_set"], false);
        assert_eq!(before["tool_active"], false);
        let out = directions_save_json(&s, br#"{"mapbox_token":"pk.secret-token"}"#);
        assert!(!out.contains("pk.secret-token"), "token leaked: {out}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["token_set"], true);
        assert_eq!(v["tool_active"], true);
        // The live settings hold the real token, but a status read never exposes it.
        assert!(!directions_status_json(&s).contains("pk.secret-token"));
        // A page reload posting a blank token must not wipe the stored one.
        directions_save_json(&s, br#"{"mapbox_token":""}"#);
        assert!(s.mapbox_token_set());
    }

    #[test]
    fn post_config_sets_anthropic_oauth_token_without_leaking_it() {
        let s = settings();
        assert!(!s.view().anthropic_oauth_token_set);
        let (status, _c, out) = route(
            "POST",
            "/config",
            br#"{"anthropic_oauth_token":"oauth-secret"}"#,
            &s,
        );
        assert_eq!(status, "200 OK");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["anthropic_oauth_token_set"], true);
        // The token must never appear in a POST response or a later GET.
        assert!(!String::from_utf8_lossy(&out).contains("oauth-secret"));
        let (_s, _c, g) = route("GET", "/config", b"", &s);
        assert!(!String::from_utf8_lossy(&g).contains("oauth-secret"));
        assert!(s.view().anthropic_oauth_token_set);
    }

    #[tokio::test]
    async fn spotify_link_without_credentials_is_a_clear_error() {
        // No client id/secret set → link returns a helpful message, never runs consent.
        let s = settings();
        let out = spotify_link_json(&s).await;
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["message"].as_str().unwrap().contains("client id"));
    }

    #[test]
    fn cadora_status_reports_unlinked_by_default() {
        let v: Value = serde_json::from_str(&cadora_status_json(&settings())).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["linked"], false);
        // Falls back to the built-in Cadora server URL when unset.
        assert_eq!(v["base_url"], "https://cadora-server.fly.dev");
    }

    #[test]
    fn cadora_save_sets_base_url() {
        let s = settings();
        cadora_save_json(&s, br#"{"base_url":"http://localhost:8080"}"#);
        assert_eq!(s.cadora().base_url_or_default(), "http://localhost:8080");
        // A blank base_url resets to the default.
        cadora_save_json(&s, br#"{"base_url":""}"#);
        assert_eq!(
            s.cadora().base_url_or_default(),
            "https://cadora-server.fly.dev"
        );
    }

    #[tokio::test]
    async fn cadora_link_without_input_is_a_clear_error() {
        let s = settings();
        let out = cadora_link_json(&s, br#"{}"#).await;
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert!(v["message"].as_str().unwrap().contains("6-digit code"));
        assert!(!s.cadora().linked());
    }

    #[tokio::test]
    async fn cadora_link_with_a_pasted_token_links_without_network() {
        // The primary path: the Cadora app mints a full `vl_…` token to copy/paste, so
        // linking stores it directly with no server round-trip.
        let s = settings();
        let out = cadora_link_json(&s, br#"{"token":"vl_pasted"}"#).await;
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["linked"], true);
        assert!(s.cadora().linked());
        // The token must never appear in the response.
        assert!(!out.contains("vl_pasted"));
    }

    #[test]
    fn invalid_json_is_a_400() {
        let (status, _c, _b) = route("POST", "/config", b"not json", &settings());
        assert_eq!(status, "400 Bad Request");
    }

    #[test]
    fn household_page_and_status_are_served() {
        let s = settings();
        let (status, ctype, body) = route("GET", "/household", b"", &s);
        assert_eq!(status, "200 OK");
        assert!(ctype.starts_with("text/html"));
        assert!(String::from_utf8_lossy(&body).contains("Household"));

        let v: Value = serde_json::from_str(&household_status_json(&s)).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["members"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn household_save_round_trips_through_the_route() {
        let s = settings();
        let body = br#"{"location":"  Austin, TX  ","weather_units":"imperial",
            "members":[
              {"name":" Alice ","emails":["alice@example.com"," "],"phones":["+1 555 0001"],"relationship":" parent "},
              {"name":"   ","emails":["ghost@example.com"]}
            ]}"#;
        let (status, ctype, out) = route("POST", "/household/save", body, &s);
        assert_eq!(status, "200 OK");
        assert_eq!(ctype, "application/json");
        let v: Value = serde_json::from_slice(&out).unwrap();
        // Location trimmed, the nameless member dropped, blank email dropped.
        assert_eq!(v["location"], "Austin, TX");
        assert_eq!(v["weather_units"], "imperial");
        let members = v["members"].as_array().unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0]["name"], "Alice");
        assert_eq!(members[0]["emails"][0], "alice@example.com");
        assert_eq!(members[0]["emails"].as_array().unwrap().len(), 1);
        assert_eq!(members[0]["relationship"], "parent");

        // It landed in the live settings, so the per-turn prompt snapshot sees it.
        let h = s.household();
        assert_eq!(h.location.as_deref(), Some("Austin, TX"));
        assert_eq!(h.members.len(), 1);
    }

    #[test]
    fn normalize_target_reduces_absolute_form_to_path() {
        // Origin-form is unchanged.
        assert_eq!(normalize_target("/drive/save"), "/drive/save");
        assert_eq!(normalize_target("/config?_=1"), "/config?_=1");
        assert_eq!(normalize_target("/"), "/");
        // Absolute-form (proxied client) is reduced to origin-form.
        assert_eq!(
            normalize_target("http://127.0.0.1:8731/drive/save"),
            "/drive/save"
        );
        assert_eq!(
            normalize_target("https://host:8731/config?_=1"),
            "/config?_=1"
        );
        // Absolute URI with no path → root.
        assert_eq!(normalize_target("http://127.0.0.1:8731"), "/");
    }

    #[tokio::test]
    async fn serves_a_proxied_absolute_form_post() {
        // A browser behind an HTTP proxy sends the absolute-form request target;
        // the server must route it the same as origin-form (regression: this used to
        // fall through to a 404 "not found", which broke the Drive config page).
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let s = settings();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle(
                stream,
                s,
                catalog(),
                connector(),
                None,
                debug(),
                None,
                notify(),
            )
            .await
            .unwrap();
        });

        let mut client = TcpStream::connect(addr).await.unwrap();
        let body = br#"{"tts_voice":"en_US-amy-medium"}"#;
        // Absolute-form target, exactly as an HTTP proxy forwards it.
        let req = format!(
            "POST http://127.0.0.1:{}/config HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
            addr.port(),
            body.len()
        );
        client.write_all(req.as_bytes()).await.unwrap();
        client.write_all(body).await.unwrap();

        let mut resp = String::new();
        client.read_to_string(&mut resp).await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp: {resp}");
        assert!(resp.contains("en_US-amy-medium"), "resp: {resp}");
    }

    #[test]
    fn unknown_route_is_404() {
        let (status, ..) = route("GET", "/nope", b"", &settings());
        assert_eq!(status, "404 Not Found");
    }

    #[test]
    fn debug_pages_render_with_nav() {
        for (path, marker) in [
            ("/chatlog", "Chat log"),
            ("/prompts", "System prompt"),
            ("/sqlite", "Persistent memory"),
            ("/helix", "GraphRAG"),
        ] {
            let (status, ctype, body) = route("GET", path, b"", &settings());
            assert_eq!(status, "200 OK", "{path}");
            assert!(ctype.starts_with("text/html"), "{path}");
            let html = String::from_utf8_lossy(&body);
            assert!(html.contains(marker), "{path} missing {marker}");
            // Every debug page carries the shared nav linking the others.
            assert!(html.contains("href=\"/helix\""), "{path} missing nav");
        }
    }

    #[test]
    fn limit_param_parses_and_clamps() {
        assert_eq!(limit_param("/chatlog.json"), DEFAULT_LOG_LIMIT);
        assert_eq!(limit_param("/chatlog.json?limit=25"), 25);
        assert_eq!(limit_param("/chatlog.json?limit=0"), 1);
        assert_eq!(limit_param("/chatlog.json?limit=99999"), MAX_LOG_LIMIT);
        assert_eq!(limit_param("/chatlog.json?x=1&limit=7"), 7);
    }

    #[test]
    fn sqlite_json_reports_memory_rows() {
        use crate::memory::{MemoryKind, MemorySource};
        let d = debug();
        d.memory
            .add(
                MemoryKind::Fact,
                "The user likes tea",
                MemorySource::Explicit,
            )
            .unwrap();
        let v: Value = serde_json::from_str(&sqlite_json(&d)).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["count"], 1);
        assert_eq!(v["memory_backend"], "sqlite");
        assert_eq!(v["memories"][0]["content"], "The user likes tea");
        assert_eq!(v["memories"][0]["kind"], "fact");
    }

    #[tokio::test]
    async fn helix_json_reports_disabled_without_a_graph() {
        let v: Value = serde_json::from_str(&helix_json(&debug()).await).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["enabled"], false);
        assert!(v["message"].is_string());
        // memory_backend="sqlite" → tell the user how to turn HelixDB on.
        assert!(v["message"]
            .as_str()
            .unwrap()
            .contains("Set memory_backend"));
    }

    #[tokio::test]
    async fn helix_json_explains_the_fallback_when_configured_for_helix() {
        // memory_backend="helix" but no graph → init failed and recall fell back.
        // The message must NOT tell the user to set a value that is already set;
        // it should point at the likely cause (OPENAI_API_KEY) and the log line.
        let mut d = debug();
        d.memory_backend = "helix".to_string();
        let v: Value = serde_json::from_str(&helix_json(&d).await).unwrap();
        assert_eq!(v["enabled"], false);
        let msg = v["message"].as_str().unwrap();
        assert!(msg.contains("fell back to SQLite FTS"), "got: {msg}");
        assert!(msg.contains("OPENAI_API_KEY"), "got: {msg}");
        assert!(!msg.contains("Set memory_backend"), "got: {msg}");
    }

    #[tokio::test]
    async fn helix_rename_reports_disabled_without_a_graph() {
        let body = br#"{"old_name":"Portlnad","new_name":"Portland"}"#;
        let v: Value = serde_json::from_str(&helix_rename_json(&debug(), body).await).unwrap();
        assert_eq!(v["ok"], false, "no graph backend → rename cannot apply");
        assert!(v["message"].is_string());
    }

    /// End-to-end over a real socket: exercises the HTTP request parsing in
    /// `handle` (header split, Content-Length body read), not just `route`.
    #[tokio::test]
    async fn serves_a_post_over_a_real_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let s = settings();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle(
                stream,
                s,
                catalog(),
                connector(),
                None,
                debug(),
                None,
                notify(),
            )
            .await
            .unwrap();
        });

        let mut client = TcpStream::connect(addr).await.unwrap();
        let body = br#"{"tts_voice":"en_US-amy-medium"}"#;
        let req = format!(
            "POST /config HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        client.write_all(req.as_bytes()).await.unwrap();
        client.write_all(body).await.unwrap();

        let mut resp = String::new();
        client.read_to_string(&mut resp).await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp: {resp}");
        assert!(resp.contains("en_US-amy-medium"), "resp: {resp}");
    }

    /// `GET /models` returns the catalog JSON (static fallback here, no network).
    #[tokio::test]
    async fn serves_the_model_catalog() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let s = settings();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle(
                stream,
                s,
                catalog(),
                connector(),
                None,
                debug(),
                None,
                notify(),
            )
            .await
            .unwrap();
        });

        let mut client = TcpStream::connect(addr).await.unwrap();
        client
            .write_all(b"GET /models HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();

        let mut resp = String::new();
        client.read_to_string(&mut resp).await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp: {resp}");
        assert!(resp.contains("claude-opus-5"), "resp: {resp}");
        assert!(resp.contains("gpt-4o-mini"), "resp: {resp}");
    }

    /// `GET /voices` returns the installed-voice list, filtered to a voices dir.
    #[tokio::test]
    async fn serves_the_voice_catalog_filtered_to_installed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // A voices dir with only amy installed → lessac is filtered out even though
        // the mock Piper advertises both.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ambient-webcfg-voices-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("en_US-amy-medium.onnx"), b"").unwrap();

        let s = settings();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let dir_for_task = dir.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle(
                stream,
                s,
                catalog(),
                connector(),
                Some(dir_for_task),
                debug(),
                None,
                notify(),
            )
            .await
            .unwrap();
        });

        let mut client = TcpStream::connect(addr).await.unwrap();
        client
            .write_all(b"GET /voices HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();

        let mut resp = String::new();
        client.read_to_string(&mut resp).await.unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp: {resp}");
        assert!(resp.contains("en_US-amy-medium"), "resp: {resp}");
        assert!(!resp.contains("en_US-lessac-medium"), "resp: {resp}");
    }
}
