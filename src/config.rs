use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub socket_path: PathBuf,
    pub kokoro_model: PathBuf,
    pub kokoro_voices: PathBuf,
    pub kokoro_voice: String,
    pub kokoro_speed: f32,
    pub target_sample_rate: u32,
    pub summarize_code_blocks: bool,
    pub sustained_words: usize,
}

/// Locates the models directory across standard installation paths.
pub fn find_models_dir() -> PathBuf {
    // 1. Explicit environment override
    if let Ok(dir) = std::env::var("PI_SPEAK_MODELS_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return p;
        }
    }

    // 2. Relative to current working directory
    let cwd_models = PathBuf::from("models");
    if cwd_models.join("kokoro").exists() {
        return cwd_models;
    }

    // 3. Relative to executable location
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        let next_to_exe = parent.join("models");
        if next_to_exe.exists() {
            return next_to_exe;
        }
        let parent_models = parent.join("../models");
        if parent_models.exists() {
            return parent_models;
        }
        let target_root = parent.join("../../models");
        if target_root.exists() {
            return target_root;
        }
    }

    // 4. User data directories (XDG / ~/.local/share/pi-speak/models)
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME") {
        let p = PathBuf::from(xdg_data).join("pi-speak/models");
        if p.exists() {
            return p;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".local/share/pi-speak/models");
        if p.exists() {
            return p;
        }
    }

    // 5. System directories
    let sys_local = PathBuf::from("/usr/local/share/pi-speak/models");
    if sys_local.exists() {
        return sys_local;
    }
    let sys = PathBuf::from("/usr/share/pi-speak/models");
    if sys.exists() {
        return sys;
    }

    // Fallback default
    PathBuf::from("models")
}

impl Default for Config {
    fn default() -> Self {
        let models_dir = find_models_dir();

        let socket_path = std::env::var("PI_SPEAK_SOCKET")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
                    let runtime_sock = PathBuf::from(&runtime_dir).join("pi-speak.sock");
                    let tmp_sock = PathBuf::from("/tmp/pi-speak.sock");
                    if tmp_sock.exists() && !runtime_sock.exists() {
                        tmp_sock
                    } else {
                        runtime_sock
                    }
                } else {
                    PathBuf::from("/tmp/pi-speak.sock")
                }
            });

        let kokoro_voice = std::env::var("PI_SPEAK_VOICE").unwrap_or_else(|_| "jarvis".to_string());
        let kokoro_speed = std::env::var("PI_SPEAK_SPEED")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.15);

        Self {
            socket_path,
            kokoro_model: models_dir.join("kokoro/kokoro-v1.0.onnx"),
            kokoro_voices: models_dir.join("kokoro/voices-v1.0.bin"),
            kokoro_voice,
            kokoro_speed,
            target_sample_rate: 48000, // Native 48kHz audio sink
            summarize_code_blocks: true,
            sustained_words: 30,
        }
    }
}
