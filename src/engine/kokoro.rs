use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing::info;

use super::phonemizer::Phonemizer;
use super::vocab;
use super::voice_loader::{VoiceDatabase, blend_styles};

static ORT_INIT: OnceLock<bool> = OnceLock::new();

/// Automatically detects and initializes the ONNX Runtime dynamic library.
pub fn init_onnx_runtime() -> Result<()> {
    if ORT_INIT.get().is_some() {
        return Ok(());
    }

    let mut candidate_paths = Vec::new();

    // 1. Explicit environment override
    if let Ok(env_path) = std::env::var("ORT_DYLIB_PATH") {
        candidate_paths.push(PathBuf::from(env_path));
    }

    // 2. Relative to executable
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        candidate_paths.push(exe_dir.join("libonnxruntime.so"));
        candidate_paths.push(exe_dir.join("../lib/libonnxruntime.so"));
        candidate_paths.push(exe_dir.join("../../lib/libonnxruntime.so"));
    }

    // 3. Current working directory / repo root
    candidate_paths.push(PathBuf::from("lib/libonnxruntime.so"));

    // 4. User library paths
    if let Ok(home) = std::env::var("HOME") {
        let home_p = PathBuf::from(home);
        candidate_paths.push(home_p.join(".local/lib/libonnxruntime.so"));
        candidate_paths.push(home_p.join(".local/lib/libonnxruntime.dylib"));
    }

    // 5. System paths & Linux multiarch / macOS Homebrew
    candidate_paths.push(PathBuf::from("/usr/lib/libonnxruntime.so"));
    candidate_paths.push(PathBuf::from("/usr/local/lib/libonnxruntime.so"));
    candidate_paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime.so"));
    candidate_paths.push(PathBuf::from(
        "/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
    ));
    candidate_paths.push(PathBuf::from("/usr/lib64/libonnxruntime.so"));
    candidate_paths.push(PathBuf::from("/opt/homebrew/lib/libonnxruntime.dylib"));
    candidate_paths.push(PathBuf::from("/usr/local/lib/libonnxruntime.dylib"));

    for candidate in candidate_paths {
        if candidate.exists()
            && let Some(path_str) = candidate.to_str()
            && ort::init_from(path_str).is_ok()
        {
            info!(
                "Initialized ONNX Runtime successfully from {}",
                candidate.display()
            );
            ORT_INIT.set(true).ok();
            return Ok(());
        }
    }

    anyhow::bail!(
        "Could not find a valid libonnxruntime dynamic library.\n\
         Please install ONNX Runtime, run 'bash scripts/download-models.sh', or set ORT_DYLIB_PATH."
    )
}

/// Kokoro-82M neural TTS synthesis engine in 100% pure Rust.
/// Runs ONNX inference in-process via `ort` with zero external child processes.
pub struct KokoroEngine {
    session: ort::session::Session,
    voices_db: VoiceDatabase,
    phonemizer: Phonemizer,
    voice: String,
    speed: f32,
    sample_rate: u32,
    /// Cached 4-way British RP JARVIS blend (60% George, 30% Daniel, 5% Fable, 5% Lewis)
    jarvis_style: Option<Vec<f32>>,
    /// Pre-allocated tensor for JARVIS style to avoid allocations on every synthesis call
    cached_jarvis_tensor: Option<ort::value::Tensor<f32>>,
}

impl KokoroEngine {
    /// Initializes the native Kokoro engine with model weights and voice database.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        voices_path: P,
        voice: &str,
        speed: f32,
    ) -> Result<Self> {
        init_onnx_runtime()?;

        let model_path = model_path.as_ref();
        let voices_path = voices_path.as_ref();

        if !model_path.exists() {
            anyhow::bail!("Kokoro model file not found at: {}", model_path.display());
        }
        if !voices_path.exists() {
            anyhow::bail!("Kokoro voices file not found at: {}", voices_path.display());
        }

        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("failed to configure intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("failed to load Kokoro ONNX model: {e}"))?;

        let voices_db = VoiceDatabase::load_from_zip(voices_path)?;
        let phonemizer = Phonemizer::new("en-gb");

        // Precompute the J.A.R.V.I.S. 4-way British RP blend if voice components exist
        let jarvis_style = Self::compute_jarvis_blend(&voices_db);
        let cached_jarvis_tensor = jarvis_style
            .as_ref()
            .and_then(|j| ort::value::Tensor::from_array(([1usize, 256usize], j.clone())).ok());

        info!(
            "Initialized KokoroEngine (voice: {}, speed: {:.2}x @ 24,000 Hz, native in-process)",
            voice, speed
        );

        Ok(Self {
            session,
            voices_db,
            phonemizer,
            voice: voice.to_string(),
            speed,
            sample_rate: 24000,
            jarvis_style,
            cached_jarvis_tensor,
        })
    }

    /// Computes the J.A.R.V.I.S. 4-way British RP blend:
    /// 60% George, 30% Daniel, 5% Fable, 5% Lewis.
    fn compute_jarvis_blend(db: &VoiceDatabase) -> Option<Vec<f32>> {
        let v_george = db.get_style("bm_george", 50)?;
        let v_daniel = db.get_style("bm_daniel", 50)?;
        let v_fable = db.get_style("bm_fable", 50)?;
        let v_lewis = db.get_style("bm_lewis", 50)?;

        let blend = blend_styles(&[
            (v_george, 0.60),
            (v_daniel, 0.30),
            (v_fable, 0.05),
            (v_lewis, 0.05),
        ]);

        info!("Precomputed native 4-way British RP JARVIS blend vector (256-d)");
        Some(blend)
    }

    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub fn voice(&self) -> &str {
        &self.voice
    }

    pub fn set_voice(&mut self, voice: &str) {
        self.voice = voice.to_string();
    }

    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn set_speed(&mut self, speed: f32) {
        if (0.2..=4.0).contains(&speed) {
            self.speed = speed;
        }
    }

    /// Preconditions text for British cadence, expanding initialisms cleanly.
    fn enhance_cadence(text: &str) -> String {
        let mut t = text.replace(['—', '–'], " — ");
        t = t.replace("...", "…");

        // Initialisms pronounced cleanly
        let replacements = [
            (r"\bAI\b", "A.I."),
            (r"\bAPI\b", "A.P.I."),
            (r"\bUI\b", "U.I."),
            (r"\bCLI\b", "C.L.I."),
            (r"\bJARVIS\b", "Jarvis"),
        ];

        for (pattern, replacement) in replacements {
            if let Ok(re) = regex::Regex::new(pattern) {
                t = re.replace_all(&t, replacement).to_string();
            }
        }

        t.trim().to_string()
    }

    /// Detects situational tone (alert, inquiry, witty, standard) for dynamic speed modulation.
    fn detect_speed_modifier(text: &str) -> f32 {
        let lower = text.to_lowercase();
        if lower.contains("warning")
            || lower.contains("alert")
            || lower.contains("critical")
            || lower.contains("error")
            || lower.contains("emergency")
            || text.ends_with('!')
        {
            1.06 // Slightly crisper, urgent
        } else if text.ends_with('?')
            || lower.starts_with("shall i")
            || lower.starts_with("would you like")
        {
            0.98 // Deliberate, inquisitive
        } else if text.contains('…')
            || lower.starts_with("as always")
            || lower.starts_with("per your")
        {
            0.95 // Measured, witty
        } else {
            1.00 // Standard
        }
    }

    /// Synthesizes text into 16-bit mono PCM bytes at 24,000 Hz natively.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<u8>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let formatted = Self::enhance_cadence(text);
        let speed_modifier = Self::detect_speed_modifier(&formatted);
        let effective_speed = (self.speed * speed_modifier).clamp(0.5, 3.0);

        // Convert text to IPA phonemes natively
        let ipa = self.phonemizer.phonemize(&formatted)?;
        let tokens = vocab::tokenize_phonemes(&ipa);

        if tokens.len() <= 2 {
            // Only start and end tokens (no pronounceable content)
            return Ok(Vec::new());
        }

        // Split tokens if exceeding Kokoro's 510 token limit
        if tokens.len() <= 510 {
            self.synthesize_tokens(&tokens, effective_speed)
        } else {
            self.synthesize_long_tokens(&tokens, effective_speed)
        }
    }

    /// Synthesizes a single chunk of tokens (<= 510).
    fn synthesize_tokens(&mut self, tokens: &[i64], speed: f32) -> Result<Vec<u8>> {
        let phoneme_count = tokens.len().saturating_sub(2);

        // Obtain style vector
        let style_vec: &[f32] = if self.voice == "jarvis" {
            if let Some(ref j) = self.jarvis_style {
                j.as_slice()
            } else if let Some(style) = self.voices_db.get_style("bm_george", phoneme_count) {
                style
            } else {
                anyhow::bail!("Voice style not available for JARVIS");
            }
        } else if let Some(style) = self.voices_db.get_style(&self.voice, phoneme_count) {
            style
        } else if let Some(ref j) = self.jarvis_style {
            j.as_slice()
        } else {
            anyhow::bail!("Voice '{}' not found in voice database", self.voice);
        };

        let seq_len = tokens.len();
        let tokens_tensor = ort::value::Tensor::from_array(([1usize, seq_len], tokens.to_vec()))?;
        let speed_tensor = ort::value::Tensor::from_array(([1usize], vec![speed]))?;

        let temp_style_tensor;
        let style_ref: &ort::value::Tensor<f32> = if self.voice == "jarvis"
            && let Some(ref cached) = self.cached_jarvis_tensor
        {
            cached
        } else {
            temp_style_tensor =
                ort::value::Tensor::from_array(([1usize, 256usize], style_vec.to_vec()))?;
            &temp_style_tensor
        };

        let outputs = self.session.run(ort::inputs![
            "tokens" => tokens_tensor,
            "style" => style_ref,
            "speed" => speed_tensor,
        ])?;

        let audio_val = outputs
            .get("audio")
            .context("missing 'audio' output tensor")?;
        let (_shape, samples) = audio_val.try_extract_tensor::<f32>()?;

        // Convert f32 samples to 16-bit PCM bytes
        let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let s16 = (clamped * 32767.0) as i16;
            pcm_bytes.extend_from_slice(&s16.to_le_bytes());
        }

        Ok(pcm_bytes)
    }

    /// Synthesizes long sequences by partitioning tokens at punctuation/spaces.
    fn synthesize_long_tokens(&mut self, tokens: &[i64], speed: f32) -> Result<Vec<u8>> {
        let mut all_pcm = Vec::new();
        let mut start_idx = 1; // Skip leading start token (0)
        let end_idx = tokens.len() - 1; // Exclude trailing end token (0)

        while start_idx < end_idx {
            let chunk_max = (start_idx + 450).min(end_idx);
            // Look for clean boundary (punctuation or space)
            let mut split_at = chunk_max;
            if chunk_max < end_idx {
                for i in (start_idx..chunk_max).rev() {
                    let tok = tokens[i];
                    // Tokens for '.', '!', '?', ';', ',', or space
                    if tok == 4 || tok == 5 || tok == 6 || tok == 1 || tok == 3 || tok == 16 {
                        split_at = i + 1;
                        break;
                    }
                }
            }

            let mut sub_tokens = Vec::with_capacity((split_at - start_idx) + 2);
            sub_tokens.push(0); // Start token
            sub_tokens.extend_from_slice(&tokens[start_idx..split_at]);
            sub_tokens.push(0); // End token

            let pcm = self.synthesize_tokens(&sub_tokens, speed)?;
            all_pcm.extend_from_slice(&pcm);

            start_idx = split_at;
        }

        Ok(all_pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kokoro_full_synthesis() {
        let models = crate::config::find_models_dir();
        let model = models.join("kokoro/kokoro-v1.0.onnx");
        let voices = models.join("kokoro/voices-v1.0.bin");

        if model.exists() && voices.exists() {
            let mut engine = KokoroEngine::new(&model, &voices, "jarvis", 1.15)
                .expect("Failed to initialize KokoroEngine");

            let pcm = engine
                .synthesize("Good evening, sir. Native Rust neural synthesis is online.")
                .expect("Synthesis failed");

            assert!(!pcm.is_empty(), "PCM bytes should not be empty");
            assert_eq!(pcm.len() % 2, 0, "PCM bytes must be 16-bit aligned");
            assert_eq!(engine.sample_rate(), 24000);
            println!("Synthesized {} bytes of 24kHz PCM natively!", pcm.len());
        }
    }
}
