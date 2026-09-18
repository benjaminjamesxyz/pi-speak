/// Audio mastering processor: applies soft-knee limiting and cross-fading.
#[derive(Debug, Clone)]
pub struct AudioMaster {
    threshold: f32,
    knee_width: f32,
}

impl Default for AudioMaster {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            knee_width: 0.15,
        }
    }
}

impl AudioMaster {
    #[must_use]
    pub fn new(threshold: f32, knee_width: f32) -> Self {
        Self {
            threshold,
            knee_width,
        }
    }

    /// In-place soft-knee limiter and anti-clipping filter.
    /// Ensures that audio peaks never clip harshly on headphones.
    pub fn apply_limiting(&self, samples: &mut [f32]) {
        let t = self.threshold;
        let w = self.knee_width;

        for sample in samples.iter_mut() {
            let x = *sample;
            let abs_x = x.abs();

            if abs_x <= t - w / 2.0 {
                // Linear region: pass through unchanged
            } else if abs_x <= t + w / 2.0 {
                // Knee region: smooth polynomial curve
                let delta = abs_x - t + w / 2.0;
                let compressed = abs_x - (delta * delta) / (2.0 * w);
                *sample = x.signum() * compressed;
            } else {
                // Beyond knee: soft compression asymptotically approaching 1.0
                let overshoot = abs_x - (t + w / 2.0);
                let compressed = t + (w / 2.0) + (1.0 - t - w / 2.0) * (1.0 - (-overshoot).exp());
                *sample = x.signum() * compressed.min(0.99);
            }
        }
    }

    /// Applies a smooth S-curve fade-in at the beginning and fade-out at the end
    /// to avoid any abrupt clicks or DC offset pops on chunk boundaries.
    pub fn apply_edge_fades(&self, samples: &mut [f32], fade_samples: usize, channels: usize) {
        if samples.len() < fade_samples * channels * 2 {
            return;
        }

        // Fade in
        for i in 0..fade_samples {
            let progress = i as f32 / fade_samples as f32;
            // Cosine S-curve: 0.5 * (1 - cos(pi * t))
            let factor = 0.5 * (1.0 - (std::f32::consts::PI * progress).cos());
            for ch in 0..channels {
                samples[i * channels + ch] *= factor;
            }
        }

        // Fade out
        let total_frames = samples.len() / channels;
        for i in 0..fade_samples {
            let frame = total_frames - fade_samples + i;
            let progress = (fade_samples - i) as f32 / fade_samples as f32;
            let factor = 0.5 * (1.0 - (std::f32::consts::PI * progress).cos());
            for ch in 0..channels {
                samples[frame * channels + ch] *= factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_limiter_clamps_overshoots() {
        let master = AudioMaster::default();
        let mut samples = vec![0.0, 0.5, 0.8, 0.95, 1.2, -1.5];
        master.apply_limiting(&mut samples);

        for &s in &samples {
            assert!(s.abs() <= 1.0, "Sample exceeds 1.0: {}", s);
        }
    }

    #[test]
    fn test_edge_fades() {
        let master = AudioMaster::default();
        let mut samples = vec![1.0; 200]; // 100 stereo frames
        master.apply_edge_fades(&mut samples, 10, 2);

        // First frame should be near 0
        assert!(samples[0] < 0.05);
        assert!(samples[1] < 0.05);

        // Middle frames should remain 1.0
        assert_eq!(samples[100], 1.0);

        // Last frame should be near 0
        assert!(samples[198] < 0.05);
        assert!(samples[199] < 0.05);
    }
}
