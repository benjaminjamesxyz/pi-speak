use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tracing::info;

/// Loaded voice style database for Kokoro-82M.
/// Each voice is an array of 510 rows x 256 dimensions (130,560 float32 values).
pub struct VoiceDatabase {
    voices: HashMap<String, Vec<f32>>,
}

impl VoiceDatabase {
    /// Loads all voice style vectors directly from the `voices-v1.0.bin` ZIP archive.
    pub fn load_from_zip<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open voices file at: {}", path.display()))?;

        let mut archive =
            zip::ZipArchive::new(file).context("failed to parse voices ZIP archive")?;
        let mut voices = HashMap::with_capacity(archive.len());

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();

            if name.ends_with(".npy") {
                let voice_id = name.trim_end_matches(".npy").to_string();
                let mut raw_bytes = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut raw_bytes)?;

                let floats = parse_npy_float32(&raw_bytes)
                    .with_context(|| format!("failed to parse .npy data for voice: {voice_id}"))?;

                if floats.len() == 510 * 256 {
                    voices.insert(voice_id, floats);
                }
            }
        }

        info!(
            "Loaded {} voices natively from {}",
            voices.len(),
            path.display()
        );
        Ok(Self { voices })
    }

    /// Retrieves the 256-dimensional style vector for a given voice and phoneme count.
    #[must_use]
    pub fn get_style(&self, voice_name: &str, phoneme_count: usize) -> Option<&[f32]> {
        let voice_data = self.voices.get(voice_name)?;
        let row = phoneme_count.clamp(1, 510) - 1;
        let start = row * 256;
        let end = start + 256;
        Some(&voice_data[start..end])
    }

    /// Checks if a voice exists in the database.
    #[must_use]
    pub fn contains_voice(&self, voice_name: &str) -> bool {
        self.voices.contains_key(voice_name)
    }
}

/// Parses raw bytes of a NumPy `.npy` file containing little-endian float32 values.
fn parse_npy_float32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        anyhow::bail!("invalid .npy magic header");
    }

    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_offset = 10 + header_len;

    if bytes.len() < data_offset {
        anyhow::bail!("unexpected EOF in .npy header");
    }

    let raw_payload = &bytes[data_offset..];
    let float_count = raw_payload.len() / 4;
    let mut floats = Vec::with_capacity(float_count);

    for chunk in raw_payload.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        floats.push(val);
    }

    Ok(floats)
}

/// Computes the standard deviation of a float slice.
fn calculate_std(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let mean = slice.iter().sum::<f32>() / slice.len() as f32;
    let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32;
    variance.sqrt()
}

/// Blends multiple 256-dimensional style vectors while preserving standard deviation variance.
#[must_use]
pub fn blend_styles(styles_with_weights: &[(&[f32], f32)]) -> Vec<f32> {
    if styles_with_weights.is_empty() {
        return Vec::new();
    }

    let dim = styles_with_weights[0].0.len();
    let total_w: f32 = styles_with_weights.iter().map(|(_, w)| *w).sum();
    let mut blended = vec![0.0f32; dim];
    let mut target_std = 0.0f32;

    for &(slice, w) in styles_with_weights {
        let norm_w = w / total_w;
        target_std += norm_w * calculate_std(slice);
        for i in 0..dim {
            blended[i] += norm_w * slice[i];
        }
    }

    let current_std = calculate_std(&blended);
    if current_std > 1e-6 {
        let scale = target_std / current_std;
        for val in &mut blended {
            *val *= scale;
        }
    }

    blended
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_voices_zip() {
        let models = crate::config::find_models_dir();
        let voices_path = models.join("kokoro/voices-v1.0.bin");

        if voices_path.exists() {
            let db = VoiceDatabase::load_from_zip(&voices_path).expect("failed to load voices");
            assert!(db.contains_voice("af_heart"));
            assert!(db.contains_voice("bm_george"));

            let style = db.get_style("bm_george", 50).expect("missing style");
            assert_eq!(style.len(), 256);
        }
    }

    #[test]
    fn test_blend_styles_preserves_variance() {
        let v1 = vec![0.5f32; 256];
        let v2 = vec![-0.5f32; 256];
        let blended = blend_styles(&[(&v1, 0.5), (&v2, 0.5)]);
        assert_eq!(blended.len(), 256);
    }
}
