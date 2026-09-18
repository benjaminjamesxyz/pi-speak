use anyhow::{Context, Result};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// High-fidelity SIMD-accelerated audio resampler.
/// Converts mono neural TTS output (e.g. 22.05kHz or 24kHz) to native PipeWire 48kHz stereo float32.
pub struct AudioResampler {
    resampler: Async<f32>,
    input_sample_rate: u32,
    output_sample_rate: u32,
    chunk_size_in: usize,
}

impl AudioResampler {
    /// Creates a new SIMD Sinc resampler.
    pub fn new(input_sample_rate: u32, output_sample_rate: u32) -> Result<Self> {
        let chunk_size_in = 1024;
        let ratio = output_sample_rate as f64 / input_sample_rate as f64;

        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: Some(0.95),
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 32,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = Async::<f32>::new_sinc(
            ratio,
            1.1,
            &params,
            chunk_size_in,
            1, // 1 channel input (mono)
            FixedAsync::Input,
        )
        .context("failed to initialize sinc resampler")?;

        Ok(Self {
            resampler,
            input_sample_rate,
            output_sample_rate,
            chunk_size_in,
        })
    }

    /// Resamples a raw 16-bit mono PCM byte slice into 48kHz stereo f32 samples.
    pub fn resample_pcm16_to_stereo_f32(&mut self, pcm_bytes: &[u8]) -> Result<Vec<f32>> {
        if pcm_bytes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Convert i16 LE bytes to normalized f32 samples (-1.0 to 1.0)
        let sample_count = pcm_bytes.len() / 2;
        let mut mono_f32 = Vec::with_capacity(sample_count);

        for chunk in pcm_bytes.chunks_exact(2) {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            mono_f32.push(sample_i16 as f32 / 32768.0);
        }

        // 2. Pad input to multiple of chunk_size_in if needed
        let total_samples = mono_f32.len();
        let remainder = total_samples % self.chunk_size_in;
        if remainder != 0 {
            let padding = self.chunk_size_in - remainder;
            mono_f32.resize(total_samples + padding, 0.0);
        }

        // 3. Process through sinc resampler block by block
        let max_output_frames = self.resampler.output_frames_max();
        let mut out_block = vec![0.0f32; max_output_frames];
        let mut resampled_mono = Vec::new();
        let indexing = Indexing::new();

        for chunk in mono_f32.chunks_exact(self.chunk_size_in) {
            let in_adapter = InterleavedSlice::new(chunk, 1, self.chunk_size_in)
                .context("failed to create input audio adapter")?;
            let mut out_adapter = InterleavedSlice::new_mut(&mut out_block, 1, max_output_frames)
                .context("failed to create output audio adapter")?;

            let (_frames_read, frames_written) = self
                .resampler
                .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
                .context("resampling block failed")?;

            resampled_mono.extend_from_slice(&out_block[..frames_written]);
        }

        // 4. Duplicate mono to stereo interleaved [L, R, L, R, ...]
        let mut stereo_out = Vec::with_capacity(resampled_mono.len() * 2);
        for &sample in &resampled_mono {
            stereo_out.push(sample);
            stereo_out.push(sample);
        }

        Ok(stereo_out)
    }

    #[must_use]
    pub fn input_rate(&self) -> u32 {
        self.input_sample_rate
    }

    #[must_use]
    pub fn output_rate(&self) -> u32 {
        self.output_sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_conversion() {
        let mut resampler = AudioResampler::new(22050, 48000).expect("failed to create resampler");

        // 1 second of 440Hz sine wave in i16
        let sample_count = 22050;
        let mut pcm_bytes = Vec::with_capacity(sample_count * 2);
        for i in 0..sample_count {
            let t = i as f32 / 22050.0;
            let val = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            let sample_i16 = (val * 30000.0) as i16;
            pcm_bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }

        let stereo = resampler
            .resample_pcm16_to_stereo_f32(&pcm_bytes)
            .expect("resampling failed");

        // Stereo has 2 channels, and 48000 samples per second -> approx 96000 samples
        assert!(!stereo.is_empty());
        assert!(stereo.len() >= 90000 && stereo.len() <= 105000);
        // Stereo has an even number of samples
        assert_eq!(stereo.len() % 2, 0);
    }
}
