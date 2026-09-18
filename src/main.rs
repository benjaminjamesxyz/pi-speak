use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pi_speak::config::Config;
use pi_speak::ipc::{DaemonServer, IpcRequest, IpcResponse};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "pi-speak")]
#[command(
    about = "High-fidelity expressive neural TTS engine for Pi Coding Agent",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Custom path to unix domain socket
    #[arg(long, global = true)]
    socket: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the background TTS daemon
    Daemon {
        /// Target output sample rate in Hz (default: 48000)
        #[arg(long, default_value = "48000")]
        rate: u32,

        /// Default Kokoro voice (e.g. "jarvis", "af_heart", "am_adam", "bf_emma")
        #[arg(long, default_value = "jarvis")]
        voice: String,

        /// Default speaking speed multiplier (default: 1.15)
        #[arg(long, default_value = "1.15")]
        speed: f32,
    },
    /// Speak a given text sentence immediately
    Say {
        /// The text to speak
        text: String,
    },
    /// Change or view the active voice
    Voice {
        /// Voice identifier (e.g. af_heart, am_adam, bf_emma)
        name: Option<String>,
    },
    /// Change speaking speed
    Speed {
        /// Speed multiplier (e.g. 1.0, 1.2, 0.9)
        multiplier: f32,
    },
    /// Stop and silence any active speech playback
    Stop {
        /// Silence speech across all connected sessions
        #[arg(long)]
        all: bool,
    },
    /// Query the daemon status
    Status,
    /// Gracefully stop and shut down the background TTS daemon
    Shutdown {
        /// Force shutdown even if other sessions are active
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let mut config = Config::default();

    if let Some(socket_path) = cli.socket {
        config.socket_path = socket_path;
    }

    match cli.command {
        Commands::Daemon { rate, voice, speed } => {
            config.target_sample_rate = rate;
            config.kokoro_voice = voice;
            config.kokoro_speed = speed;

            info!(
                "Starting pi-speak daemon (voice: {}, speed: {:.2}x, rate: {} Hz)",
                config.kokoro_voice, config.kokoro_speed, rate
            );
            let server = DaemonServer::new(config)?;
            server.run().await?;
        }
        Commands::Say { text } => {
            send_client_request(&config.socket_path, IpcRequest::Say { text }).await?;
        }
        Commands::Voice { name } => {
            if let Some(voice) = name {
                send_client_request(&config.socket_path, IpcRequest::SetVoice { voice }).await?;
            } else {
                send_client_request(&config.socket_path, IpcRequest::Status).await?;
            }
        }
        Commands::Speed { multiplier } => {
            send_client_request(
                &config.socket_path,
                IpcRequest::SetSpeed { speed: multiplier },
            )
            .await?;
        }
        Commands::Stop { all } => {
            send_client_request(&config.socket_path, IpcRequest::Stop { all }).await?;
        }
        Commands::Status => {
            send_client_request(&config.socket_path, IpcRequest::Status).await?;
        }
        Commands::Shutdown { force } => {
            send_client_request(&config.socket_path, IpcRequest::Shutdown { force }).await?;
        }
    }

    Ok(())
}

async fn send_client_request(socket_path: &std::path::Path, req: IpcRequest) -> Result<()> {
    let stream = UnixStream::connect(socket_path).await.with_context(|| {
        format!(
            "failed to connect to pi-speak daemon at {}. Is 'pi-speak daemon' running?",
            socket_path.display()
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let req_str = serde_json::to_string(&req)? + "\n";
    writer.write_all(req_str.as_bytes()).await?;

    let mut lines = BufReader::new(reader).lines();
    if let Some(resp_line) = lines.next_line().await? {
        let resp: IpcResponse = serde_json::from_str(&resp_line)?;
        match resp {
            IpcResponse::Ok { message } => {
                if let Some(msg) = message {
                    println!("{}", msg);
                }
            }
            IpcResponse::Status {
                playing,
                model,
                sample_rate,
                client_count,
                voice,
                speed,
            } => {
                let voice_info = voice
                    .map(|v| format!(" | voice: {}", v))
                    .unwrap_or_default();
                let speed_info = speed
                    .map(|s| format!(" | speed: {:.2}x", s))
                    .unwrap_or_default();
                println!(
                    "Daemon Status: playing={}, clients={}, model={}, rate={}Hz{}{}",
                    playing, client_count, model, sample_rate, voice_info, speed_info
                );
            }
            IpcResponse::Error { error } => {
                eprintln!("Daemon Error: {}", error);
            }
        }
    }

    Ok(())
}
