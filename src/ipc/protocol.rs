use serde::{Deserialize, Serialize};

/// IPC Request message sent from Pi extension or CLI to pi-speak daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum IpcRequest {
    /// Feed a streaming token chunk from the assistant
    Feed { text: String },
    /// Flush any remaining buffered text at the end of the turn
    Flush,
    /// Speak a complete text directly
    Say { text: String },
    /// Immediately interrupt and halt playback for this session (or all sessions if all=true)
    Stop {
        #[serde(default)]
        all: bool,
    },
    /// Immediately interrupt and halt playback across ALL sessions (used by mic barge-in)
    StopAll,
    /// Switch voice dynamically (e.g. "af_heart", "am_adam", "bf_emma")
    SetVoice { voice: String },
    /// Set speaking speed multiplier (e.g. 0.8 to 2.0)
    SetSpeed { speed: f32 },
    /// Query status of the daemon
    Status,
    /// Gracefully shut down the daemon and unlink socket (requires force if other sessions connected)
    Shutdown {
        #[serde(default)]
        force: bool,
    },
}

/// IPC Response sent back from daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IpcResponse {
    Ok {
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    Status {
        playing: bool,
        model: String,
        sample_rate: u32,
        #[serde(default)]
        client_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        voice: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        speed: Option<f32>,
    },
    Error {
        error: String,
    },
}
