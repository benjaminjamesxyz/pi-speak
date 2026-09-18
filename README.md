# pi-speak 🔊

Real-time neural text-to-speech for the [Pi coding agent](https://github.com/earendil-works/pi-coding-agent).
Speaks assistant responses **as they stream** — sentence by sentence, with barge-in interruption,
multi-session voice arbitration, and studio-quality DSP — powered by a Rust daemon running
**Kokoro-82M** fully in-process via ONNX Runtime.

```text
pi streams tokens ──► speak.ts extension ──► Unix socket ──► pi-speak daemon (Rust)
                                                             ├─ sentence chunker + sanitizer
                                                             ├─ Kokoro-82M ONNX inference (24 kHz)
                                                             ├─ sinc resampler → 48 kHz stereo
                                                             ├─ soft-knee limiter + edge fades
                                                             └─ cpal playback (PipeWire/ALSA)
```

## Features

- **Streaming speech** — first sentence is spoken while the LLM is still generating (~400 ms time-to-first-audio)
- **Barge-in** — start typing a new prompt and the current speech stops in < 10 ms
- **Multi-session aware** — several Pi sessions can talk through one daemon without interleaving; `/speak stop` silences only your session
- **Adaptive chunking** — fast first sentence, then 5–10 s prosodic chunks for natural cadence
- **Code-aware sanitizer** — markdown, code blocks, and ASCII diagrams are summarized or stripped instead of read aloud
- **17 voices** — including a blended "JARVIS" British RP voice (60% George / 30% Daniel / 5% Fable / 5% Lewis)
- **Tone-aware pacing** — subtle speed modulation for questions, warnings, and asides
- **Clean handoff DSP** — sinc resampling to 48 kHz, soft-knee limiting, S-curve edge fades (no clicks between chunks)

## Requirements

- Linux with a working audio output (PipeWire / PulseAudio / ALSA) — macOS mostly works, not battle-tested
- [Rust](https://rustup.rs) 1.85+ (edition 2024)
- `espeak-ng` on `PATH` (phonemization): `sudo apt install espeak-ng`
- ONNX Runtime shared library (see below)

## Install

### One command

```bash
curl -fsSL https://raw.githubusercontent.com/benjaminjamesxyz/pi-speak/main/install.sh | bash
```

Installs prerequisites (Rust, espeak-ng), builds the daemon, downloads ONNX Runtime and models
(~350 MB), and registers the extension with Pi. Idempotent — safe to re-run to update.

<details>
<summary>Manual install</summary>

```bash
git clone https://github.com/benjaminjamesxyz/pi-speak.git
cd pi-speak

# 1. Build the daemon
cargo build --release

# 2. Download models (~350 MB, git-ignored)
bash scripts/download-models.sh

# 3. Make sure ONNX Runtime is findable (any ONE of these works)
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so          # explicit
# …or install a package: apt install libonnxruntime-dev / brew install onnxruntime
# …or copy it to ~/.local/lib/libonnxruntime.so

# 4. (optional) Install the binary so any session can auto-spawn the daemon
cp target/release/pi-speak ~/.local/bin/
```

Then load the extension in Pi:

```bash
pi -e /path/to/pi-speak/extension/speak.ts     # try it out
pi install /path/to/pi-speak                    # or install as a package
```

The extension auto-spawns the daemon on first use and reuses it across all Pi sessions.

</details>

### Models

| Model | Size | Source |
|-------|------|--------|
| Kokoro-82M + voices | ~350 MB | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) |

Models are **not** committed to the repo — run `scripts/download-models.sh` after cloning.

## Usage

Speech starts automatically. While the assistant responds, you hear it sentence by sentence.

```text
/speak                toggle mute/unmute for this session
/speak on | off       explicit mute/unmute
/speak stop [all]     stop playback (this session, or every session)
/speak voice <name>   switch voice (jarvis, af_heart, bm_george, am_adam, …)
/speak speed <0.2-4>  speaking rate multiplier
/speak status         daemon status: playing, sessions, model, voice, speed
/speak shutdown       stop the daemon (refuses while other sessions are connected; --force overrides)
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PI_SPEAK_BIN` | Explicit path to the `pi-speak` binary |
| `PI_SPEAK_MODELS_DIR` | Models directory override (default: `./models`, then exe-relative, `~/.local/share/pi-speak/models`) |
| `PI_SPEAK_SOCKET` | Unix socket path (default: `$XDG_RUNTIME_DIR/pi-speak.sock`, else `/tmp/pi-speak.sock`) |
| `PI_SPEAK_VOICE` | Default voice (default: `jarvis`) |
| `PI_SPEAK_SPEED` | Default speed multiplier (default: `1.15`) |
| `ORT_DYLIB_PATH` | Path to `libonnxruntime.so` |

## Architecture

Two layers:

1. **`extension/speak.ts`** — Pi extension. Hooks `message_update` / `message_end` / `turn_end` to stream
   text deltas to the daemon over a line-delimited JSON Unix-socket protocol; hooks `before_agent_start`
   for barge-in; registers `/speak`. Resolves and auto-spawns the daemon binary if none is running.

2. **`src/` — Rust daemon** (`cargo build --release`):
   - `ipc/` — tokio Unix-socket server; per-connection IDs; `flock`-guarded single instance; graceful shutdown
   - `text/` — streaming sentence chunker (fast-start + sustained modes) and markdown/code sanitizer
   - `engine/` — Kokoro-82M ONNX inference in-process via `ort`; espeak-ng phonemizer with cache;
     ZIP-of-`.npy` voice style database; JARVIS blend precomputation
   - `dsp/` — rubato sinc resampler (24 kHz → 48 kHz), soft-knee limiter, S-curve edge fades
   - `audio/` — cpal output stream with lock-free-ish chunk queue and instant-cancel token

One dedicated producer thread owns the engine and a fair per-session queue map, so sessions never
interleave audio and cancellation is O(1).

## Development

```bash
cargo test        # 17 unit tests (chunker, sanitizer, DSP, engine)
cargo clippy
```

## License

MIT. Model weights keep their upstream licenses (Kokoro: Apache-2.0).
