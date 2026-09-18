#!/usr/bin/env bash
# Downloads the neural TTS models used by pi-speak into ./models/
#
# Kokoro-82M (default engine)  — Apache-2.0 — https://huggingface.co/hexgrad/Kokoro-82M
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p models/kokoro models/jarvis

fetch() {
	local url=$1 dest=$2
	if [[ -s "$dest" ]]; then
		echo "== $(basename "$dest") already present, skipping"
		return 0
	fi
	echo "== Downloading $(basename "$dest") ..."
	curl -fL --retry 3 --progress-bar "$url" -o "$dest"
}

# Kokoro-82M (required for the default engine)
# hexgrad/Kokoro-82M no longer hosts the ONNX export; kokoro-onnx releases do (ZIP of .npy voices,
# the exact format voice_loader.rs parses).
MODEL_BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1
fetch $MODEL_BASE/kokoro-v1.0.onnx models/kokoro/kokoro-v1.0.onnx
# voices-v1.0.bin is a ZIP of per-voice .npy style tensors; pi-speak loads it directly.
fetch $MODEL_BASE/voices-v1.0.bin models/kokoro/voices-v1.0.bin

echo
echo "Done. Models are git-ignored; re-run this script on fresh clones."
