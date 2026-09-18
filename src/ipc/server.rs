use crate::audio::{AudioChunk, AudioSink};
use crate::config::Config;
use crate::dsp::{AudioMaster, AudioResampler};
use crate::engine::KokoroEngine;
use crate::ipc::protocol::{IpcRequest, IpcResponse};
use crate::text::SentenceChunker;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// RAII Drop guard to ensure Unix domain socket and lockfile are unlinked when daemon terminates.
struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = fs::remove_file(&self.0);
            debug!("Unlinked daemon socket at {}", self.0.display());
        }
        let lock_path = self.0.with_extension("lock");
        if lock_path.exists() {
            let _ = fs::remove_file(&lock_path);
            debug!("Unlinked daemon lockfile at {}", lock_path.display());
        }
    }
}

enum WorkerMessage {
    SynthesizeAndPlay {
        connection_id: u64,
        id: u64,
        text: String,
    },
    StopSession {
        connection_id: u64,
    },
    StopAll,
    SessionDisconnect {
        connection_id: u64,
    },
    SetVoice {
        voice: String,
    },
    SetSpeed {
        speed: f32,
    },
    Shutdown,
}

/// Daemon server handling Unix domain socket requests and managing multi-session synthesis/playback.
pub struct DaemonServer {
    config: Config,
    _audio_sink: Arc<AudioSink>,
    tx_worker: Sender<WorkerMessage>,
    current_request_id: Arc<AtomicU64>,
    next_connection_id: Arc<AtomicU64>,
    active_clients: Arc<AtomicUsize>,
    is_busy: Arc<AtomicBool>,
    current_voice: Arc<RwLock<String>>,
    current_speed: Arc<RwLock<f32>>,
    model_name: String,
}

impl DaemonServer {
    /// Initializes the daemon server and background multi-session synthesis worker.
    pub fn new(config: Config) -> Result<Self> {
        let audio_sink = Arc::new(AudioSink::new(config.target_sample_rate)?);
        let (tx_worker, rx_worker): (Sender<WorkerMessage>, Receiver<WorkerMessage>) =
            crossbeam_channel::unbounded();

        let mut engine = KokoroEngine::new(
            &config.kokoro_model,
            &config.kokoro_voices,
            &config.kokoro_voice,
            config.kokoro_speed,
        )?;
        let native_rate = engine.sample_rate();

        let model_name = "Kokoro-82M (Expressive Neural)".to_string();
        let current_voice = Arc::new(RwLock::new(config.kokoro_voice.clone()));
        let current_speed = Arc::new(RwLock::new(config.kokoro_speed));

        let mut resampler = AudioResampler::new(native_rate, config.target_sample_rate)?;
        let master = AudioMaster::default();

        let worker_sink = Arc::clone(&audio_sink);
        let current_request_id = Arc::new(AtomicU64::new(0));
        let next_connection_id = Arc::new(AtomicU64::new(1));
        let active_clients = Arc::new(AtomicUsize::new(0));
        let is_busy = Arc::new(AtomicBool::new(false));
        let is_busy_clone = Arc::clone(&is_busy);
        let model_disp = model_name.clone();

        // Spawn dedicated Producer Worker thread for TTS Inference & SIMD DSP
        std::thread::Builder::new()
            .name("pi-speak-producer".to_string())
            .spawn(move || {
                info!("Multi-session synthesis & DSP worker thread running ({})", model_name);

                struct WorkerState {
                    session_queues: HashMap<u64, VecDeque<(u64, String)>>,
                    session_order: VecDeque<u64>,
                    active_session: Option<u64>,
                }

                impl WorkerState {
                    fn new() -> Self {
                        Self {
                            session_queues: HashMap::new(),
                            session_order: VecDeque::new(),
                            active_session: None,
                        }
                    }

                    fn process_msg(
                        &mut self,
                        msg: WorkerMessage,
                        engine: &mut KokoroEngine,
                        worker_sink: &AudioSink,
                        is_busy: &AtomicBool,
                    ) -> bool {
                        match msg {
                            WorkerMessage::Shutdown => return true,
                            WorkerMessage::StopAll => {
                                self.session_queues.clear();
                                self.session_order.clear();
                                self.active_session = None;
                                worker_sink.stop();
                                is_busy.store(false, Ordering::SeqCst);
                            }
                            WorkerMessage::StopSession { connection_id } => {
                                self.session_queues.remove(&connection_id);
                                self.session_order.retain(|&id| id != connection_id);
                                if self.active_session == Some(connection_id) {
                                    worker_sink.stop();
                                    self.active_session = None;
                                }
                            }
                            WorkerMessage::SessionDisconnect { connection_id } => {
                                self.session_queues.remove(&connection_id);
                                self.session_order.retain(|&id| id != connection_id);
                                if self.active_session == Some(connection_id) {
                                    worker_sink.stop();
                                    self.active_session = None;
                                }
                            }
                            WorkerMessage::SetVoice { voice } => {
                                engine.set_voice(&voice);
                                info!("Engine switched voice to '{}'", voice);
                            }
                            WorkerMessage::SetSpeed { speed } => {
                                engine.set_speed(speed);
                                info!("Engine set speaking speed to {:.2}x", speed);
                            }
                            WorkerMessage::SynthesizeAndPlay {
                                connection_id,
                                id,
                                text,
                            } => {
                                self.session_queues
                                    .entry(connection_id)
                                    .or_default()
                                    .push_back((id, text));
                                if !self.session_order.contains(&connection_id) {
                                    self.session_order.push_back(connection_id);
                                }
                            }
                        }
                        false
                    }
                }

                let mut state = WorkerState::new();

                loop {
                    // 1. Drain all pending messages in channel non-blockingly
                    while let Ok(msg) = rx_worker.try_recv() {
                        if state.process_msg(msg, &mut engine, &worker_sink, &is_busy_clone) {
                            return;
                        }
                    }

                    // 2. Determine active session
                    if let Some(conn_id) = state.active_session {
                        let has_pending = state
                            .session_queues
                            .get(&conn_id)
                            .map(|q| !q.is_empty())
                            .unwrap_or(false);
                        if !has_pending {
                            let next_available = state
                                .session_order
                                .iter()
                                .find(|&&id| {
                                    id != conn_id
                                        && state
                                            .session_queues
                                            .get(&id)
                                            .map(|q| !q.is_empty())
                                            .unwrap_or(false)
                                })
                                .copied();

                            if let Some(next_id) = next_available {
                                if worker_sink.is_playing() {
                                    std::thread::sleep(Duration::from_millis(20));
                                    continue;
                                }
                                state.active_session = Some(next_id);
                            } else {
                                state.active_session = None;
                            }
                        }
                    }

                    if state.active_session.is_none() {
                        while let Some(candidate) = state.session_order.front().copied() {
                            let is_empty = state
                                .session_queues
                                .get(&candidate)
                                .map(|q| q.is_empty())
                                .unwrap_or(true);
                            if is_empty {
                                state.session_order.pop_front();
                            } else {
                                state.active_session = Some(candidate);
                                break;
                            }
                        }
                    }

                    // 3. If a session is active and has work, synthesize next sentence
                    let next_task = state.active_session.and_then(|conn_id| {
                        state
                            .session_queues
                            .get_mut(&conn_id)
                            .and_then(|q| q.pop_front())
                            .map(|task| (conn_id, task.0, task.1))
                    });

                    if let Some((conn_id, id, text)) = next_task {
                        is_busy_clone.store(true, Ordering::SeqCst);
                        let t0 = std::time::Instant::now();

                        match engine.synthesize(&text) {
                            Ok(pcm) => {
                                if !pcm.is_empty() {
                                    match resampler.resample_pcm16_to_stereo_f32(&pcm) {
                                        Ok(mut stereo_samples) => {
                                            master.apply_limiting(&mut stereo_samples);
                                            master.apply_edge_fades(&mut stereo_samples, 240, 2);

                                            let elapsed = t0.elapsed();
                                            let audio_secs = (stereo_samples.len() / 2) as f64 / 48000.0;
                                            let rtf = elapsed.as_secs_f64() / audio_secs.max(0.001);

                                            debug!(
                                                "[Session {}] Synthesized chunk #{} in {:.2?} (audio: {:.2}s, RTF: {:.3})",
                                                conn_id, id, elapsed, audio_secs, rtf
                                            );

                                            let chunk = AudioChunk {
                                                samples: stereo_samples,
                                                id,
                                                connection_id: conn_id,
                                            };

                                            if let Err(e) = worker_sink.play_chunk(chunk) {
                                                error!("Failed to enqueue audio chunk: {}", e);
                                            }
                                        }
                                        Err(e) => {
                                            error!("DSP Resampling error: {}", e);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Synthesis error for text {:?}: {}", text, e);
                            }
                        }
                        continue;
                    }

                    // 4. No work pending: update busy flag and wait for next message
                    let still_playing = worker_sink.is_playing();
                    is_busy_clone.store(still_playing, Ordering::SeqCst);

                    if still_playing {
                        if let Ok(msg) = rx_worker.recv_timeout(Duration::from_millis(30))
                            && state.process_msg(msg, &mut engine, &worker_sink, &is_busy_clone)
                        {
                            return;
                        }
                    } else {
                        match rx_worker.recv() {
                            Ok(msg) => {
                                if state.process_msg(msg, &mut engine, &worker_sink, &is_busy_clone) {
                                    return;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
            })
            .context("failed to spawn synthesis worker thread")?;

        Ok(Self {
            config,
            _audio_sink: audio_sink,
            tx_worker,
            current_request_id,
            next_connection_id,
            active_clients,
            is_busy,
            current_voice,
            current_speed,
            model_name: model_disp,
        })
    }

    /// Starts listening for IPC connections on the Unix domain socket.
    /// Intelligently checks for an already-running daemon to prevent duplicate instances.
    pub async fn run(&self) -> Result<()> {
        let sock_path = &self.config.socket_path;

        // 1. Check if an active daemon is already listening on this socket
        if sock_path.exists() {
            if let Ok(mut stream) = UnixStream::connect(sock_path).await {
                let status_req = serde_json::to_string(&IpcRequest::Status)? + "\n";
                if stream.write_all(status_req.as_bytes()).await.is_ok() {
                    info!(
                        "pi-speak daemon is already running and responsive on {}",
                        sock_path.display()
                    );
                    return Ok(());
                }
            }
            // Socket exists but cannot connect -> stale socket file
            let _ = fs::remove_file(sock_path);
        }

        // 2. Ensure parent directory exists
        if let Some(parent) = sock_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create socket directory {}", parent.display())
            })?;
        }

        // 3. Acquire advisory file lock to guard against racing starts
        let lock_path = sock_path.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("failed to open daemon lockfile {}", lock_path.display()))?;

        let res = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if res != 0 {
            info!(
                "Another pi-speak daemon holds lockfile on {}, exiting duplicate process",
                lock_path.display()
            );
            return Ok(());
        }

        if sock_path.exists() {
            let _ = fs::remove_file(sock_path);
        }

        let listener = UnixListener::bind(sock_path)
            .with_context(|| format!("failed to bind unix socket at {}", sock_path.display()))?;

        info!("pi-speak daemon listening on {}", sock_path.display());

        let _guard = SocketGuard(sock_path.clone());
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        let mut sigterm =
            signal(SignalKind::terminate()).context("failed to register SIGTERM handler")?;
        let mut sigint =
            signal(SignalKind::interrupt()).context("failed to register SIGINT handler")?;

        loop {
            tokio::select! {
                accept_res = listener.accept() => {
                    match accept_res {
                        Ok((stream, _)) => {
                            let tx_worker = self.tx_worker.clone();
                            let chunker = SentenceChunker::new(
                                self.config.summarize_code_blocks,
                                self.config.sustained_words,
                            );
                            let req_id = Arc::clone(&self.current_request_id);
                            let is_busy = Arc::clone(&self.is_busy);
                            let sample_rate = self.config.target_sample_rate;
                            let model_name = self.model_name.clone();
                            let current_voice = Arc::clone(&self.current_voice);
                            let current_speed = Arc::clone(&self.current_speed);
                            let shutdown_tx = shutdown_tx.clone();
                            let connection_id = self.next_connection_id.fetch_add(1, Ordering::SeqCst);
                            let active_clients = Arc::clone(&self.active_clients);
                            active_clients.fetch_add(1, Ordering::SeqCst);

                            debug!("[Session {}] Connected to pi-speak daemon (active clients: {})", connection_id, active_clients.load(Ordering::Relaxed));

                            let handler = ClientHandler {
                                connection_id,
                                active_clients,
                                tx_worker,
                                chunker,
                                req_id,
                                is_busy,
                                sample_rate,
                                model_name,
                                current_voice,
                                current_speed,
                                shutdown_tx,
                            };

                            tokio::spawn(async move {
                                if let Err(e) = handler.handle(stream).await {
                                    debug!("Client connection closed: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            warn!("Error accepting connection: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("Received shutdown request, terminating pi-speak daemon");
                        break;
                    }
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, gracefully shutting down pi-speak daemon");
                    break;
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT, gracefully shutting down pi-speak daemon");
                    break;
                }
            }
        }

        let _ = self.tx_worker.send(WorkerMessage::Shutdown);
        info!("pi-speak daemon stopped cleanly and unlinked socket");
        Ok(())
    }
}

struct ClientHandler {
    connection_id: u64,
    active_clients: Arc<AtomicUsize>,
    tx_worker: Sender<WorkerMessage>,
    chunker: SentenceChunker,
    req_id: Arc<AtomicU64>,
    is_busy: Arc<AtomicBool>,
    sample_rate: u32,
    model_name: String,
    current_voice: Arc<RwLock<String>>,
    current_speed: Arc<RwLock<f32>>,
    shutdown_tx: watch::Sender<bool>,
}

impl ClientHandler {
    async fn handle(mut self, stream: UnixStream) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            let request: IpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp = IpcResponse::Error {
                        error: format!("Invalid JSON request: {}", e),
                    };
                    let resp_str = serde_json::to_string(&resp)? + "\n";
                    writer.write_all(resp_str.as_bytes()).await?;
                    continue;
                }
            };

            match request {
                IpcRequest::Feed { text } => {
                    let chunks = self.chunker.feed(&text);
                    for chunk in chunks {
                        let id = self.req_id.fetch_add(1, Ordering::SeqCst);
                        let _ = self.tx_worker.send(WorkerMessage::SynthesizeAndPlay {
                            connection_id: self.connection_id,
                            id,
                            text: chunk,
                        });
                    }
                    let resp = IpcResponse::Ok { message: None };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::Flush => {
                    if let Some(remaining) = self.chunker.flush() {
                        let id = self.req_id.fetch_add(1, Ordering::SeqCst);
                        let _ = self.tx_worker.send(WorkerMessage::SynthesizeAndPlay {
                            connection_id: self.connection_id,
                            id,
                            text: remaining,
                        });
                    }
                    let resp = IpcResponse::Ok { message: None };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::Say { text } => {
                    self.chunker.reset();
                    let _ = self.tx_worker.send(WorkerMessage::StopSession {
                        connection_id: self.connection_id,
                    });
                    let id = self.req_id.fetch_add(1, Ordering::SeqCst);
                    let _ = self.tx_worker.send(WorkerMessage::SynthesizeAndPlay {
                        connection_id: self.connection_id,
                        id,
                        text,
                    });
                    let resp = IpcResponse::Ok {
                        message: Some("Queued for playback".to_string()),
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::Stop { all } => {
                    self.chunker.reset();
                    if all {
                        let _ = self.tx_worker.send(WorkerMessage::StopAll);
                    } else {
                        let _ = self.tx_worker.send(WorkerMessage::StopSession {
                            connection_id: self.connection_id,
                        });
                    }
                    let resp = IpcResponse::Ok {
                        message: Some("Stopped".to_string()),
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::StopAll => {
                    self.chunker.reset();
                    let _ = self.tx_worker.send(WorkerMessage::StopAll);
                    let resp = IpcResponse::Ok {
                        message: Some("All speech stopped".to_string()),
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::SetVoice { voice } => {
                    if let Ok(mut v) = self.current_voice.write() {
                        *v = voice.clone();
                    }
                    let _ = self.tx_worker.send(WorkerMessage::SetVoice {
                        voice: voice.clone(),
                    });
                    let resp = IpcResponse::Ok {
                        message: Some(format!("Voice switched to '{}'", voice)),
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::SetSpeed { speed } => {
                    if let Ok(mut s) = self.current_speed.write() {
                        *s = speed;
                    }
                    let _ = self.tx_worker.send(WorkerMessage::SetSpeed { speed });
                    let resp = IpcResponse::Ok {
                        message: Some(format!("Speaking speed set to {:.2}x", speed)),
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::Status => {
                    let client_count = self.active_clients.load(Ordering::Relaxed);
                    let voice = self.current_voice.read().ok().map(|v| v.clone());
                    let speed = self.current_speed.read().ok().map(|s| *s);
                    let resp = IpcResponse::Status {
                        playing: self.is_busy.load(Ordering::Relaxed),
                        model: self.model_name.clone(),
                        sample_rate: self.sample_rate,
                        client_count,
                        voice,
                        speed,
                    };
                    writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await?;
                }
                IpcRequest::Shutdown { force } => {
                    let clients = self.active_clients.load(Ordering::Relaxed);
                    if !force && clients > 1 {
                        let resp = IpcResponse::Error {
                            error: format!(
                                "Refusing shutdown: {} Pi sessions are currently connected. Run /speak off to mute this session, or /speak shutdown --force to terminate for all sessions.",
                                clients
                            ),
                        };
                        writer
                            .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                            .await?;
                        continue;
                    }
                    self.chunker.reset();
                    let _ = self.tx_worker.send(WorkerMessage::StopAll);
                    let resp = IpcResponse::Ok {
                        message: Some("Daemon shutting down".to_string()),
                    };
                    let _ = writer
                        .write_all((serde_json::to_string(&resp)? + "\n").as_bytes())
                        .await;
                    let _ = self.shutdown_tx.send(true);
                    break;
                }
            }
        }

        // Connection closed (EOF or error): cleanly unregister this session
        debug!("[Session {}] Disconnected from daemon", self.connection_id);
        let _ = self.tx_worker.send(WorkerMessage::SessionDisconnect {
            connection_id: self.connection_id,
        });
        self.active_clients.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}
