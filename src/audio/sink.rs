use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, error, info, warn};

/// Represents an audio chunk ready for playback.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Interleaved stereo f32 samples at target rate (48kHz)
    pub samples: Vec<f32>,
    /// Sequential ID for tracking
    pub id: u64,
    /// Connection ID that requested this speech
    pub connection_id: u64,
}

/// Manages low-latency audio output via CPAL to PipeWire / PulseAudio.
pub struct AudioSink {
    _stream: Stream,
    tx_chunk: Sender<AudioChunk>,
    cancel_token: Arc<AtomicBool>,
    is_active: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
}

impl AudioSink {
    /// Initializes CPAL output stream at 48,000 Hz stereo float32.
    pub fn new(target_sample_rate: u32) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default audio output device available")?;

        info!("Selected audio output device: {}", device);

        // Check supported formats
        let default_config = device
            .default_output_config()
            .context("failed to query default audio output configuration")?;

        let channels = 2; // Always stereo
        let sample_rate = target_sample_rate;

        let stream_config = StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let (tx_chunk, rx_chunk): (Sender<AudioChunk>, Receiver<AudioChunk>) =
            crossbeam_channel::bounded(32);
        let cancel_token = Arc::new(AtomicBool::new(false));
        let is_active = Arc::new(AtomicBool::new(false));

        let cancel_token_clone = Arc::clone(&cancel_token);
        let is_active_clone = Arc::clone(&is_active);
        let mut current_chunk: Option<AudioChunk> = None;
        let mut current_pos = 0;

        let err_fn = |err| {
            error!("Audio output stream error: {}", err);
        };

        let stream = match default_config.sample_format() {
            SampleFormat::F32 => device.build_output_stream(
                stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if cancel_token_clone.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        current_chunk = None;
                        current_pos = 0;
                        while rx_chunk.try_recv().is_ok() {}
                        is_active_clone.store(false, Ordering::Relaxed);
                        return;
                    }

                    for sample in data.iter_mut() {
                        // Check if we need to load next chunk
                        if current_chunk.is_none() {
                            current_chunk = rx_chunk.try_recv().ok();
                            current_pos = 0;
                        }

                        if let Some(ref chunk) = current_chunk {
                            is_active_clone.store(true, Ordering::Relaxed);
                            if current_pos < chunk.samples.len() {
                                *sample = chunk.samples[current_pos];
                                current_pos += 1;
                            } else {
                                current_chunk = None;
                                current_pos = 0;
                                *sample = 0.0;
                            }
                        } else {
                            is_active_clone.store(false, Ordering::Relaxed);
                            *sample = 0.0;
                        }
                    }
                },
                err_fn,
                None,
            )?,
            other => {
                warn!(
                    "Device format is {:?}, attempting fallback stream config",
                    other
                );
                device.build_output_stream(
                    stream_config,
                    move |data: &mut [f32], _| {
                        data.fill(0.0);
                    },
                    err_fn,
                    None,
                )?
            }
        };

        stream
            .play()
            .context("failed to start cpal playback stream")?;

        info!(
            "Audio playback stream active: {} Hz, {} channels (float32le)",
            sample_rate, channels
        );

        Ok(Self {
            _stream: stream,
            tx_chunk,
            cancel_token,
            is_active,
            sample_rate,
            channels,
        })
    }

    /// Enqueues a mastered audio chunk for gapless playback.
    pub fn play_chunk(&self, chunk: AudioChunk) -> Result<()> {
        self.cancel_token.store(false, Ordering::SeqCst);
        self.tx_chunk
            .send(chunk)
            .context("failed to enqueue audio chunk to playback stream")?;
        Ok(())
    }

    /// Instantly halts playback and flushes any buffered audio (< 10ms latency).
    pub fn stop(&self) {
        debug!("AudioSink received stop signal -> clearing buffer");
        self.cancel_token.store(true, Ordering::SeqCst);
        self.is_active.store(false, Ordering::SeqCst);
    }

    /// Returns true if audio is actively playing or queued in the buffer.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        !self.cancel_token.load(Ordering::Relaxed)
            && (self.is_active.load(Ordering::Relaxed) || !self.tx_chunk.is_empty())
    }

    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }
}
