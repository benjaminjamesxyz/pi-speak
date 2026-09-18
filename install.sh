#!/usr/bin/env bash
# pi-speak one-command installer
#
#   curl -fsSL https://raw.githubusercontent.com/benjaminjamesxyz/pi-speak/main/install.sh | bash
#
# What it does:
#   1. Installs prerequisites (Rust toolchain, espeak-ng)
#   2. Clones pi-speak to ~/.local/share/pi-speak
#   3. Builds the daemon and installs it to ~/.local/bin/pi-speak
#   4. Downloads ONNX Runtime to ~/.local/lib
#   5. Downloads TTS models (~350 MB)
#   6. Registers the extension with Pi
#
# Idempotent: safe to re-run; re-clones/pulls and rebuilds only what changed.
set -euo pipefail

REPO="https://github.com/benjaminjamesxyz/pi-speak.git"
DEST="${PI_SPEAK_HOME:-$HOME/.local/share/pi-speak}"
BIN_DIR="$HOME/.local/bin"
LIB_DIR="$HOME/.local/lib"
ORT_VERSION="1.20.1"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Prerequisites -------------------------------------------------------

if ! command -v git >/dev/null; then
	die "git is required. Install it via your package manager first."
fi

if ! command -v cargo >/dev/null; then
	log "Installing Rust toolchain ..."
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
	export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v espeak-ng >/dev/null; then
	log "Installing espeak-ng (phonemizer) ..."
	if command -v apt-get >/dev/null; then
		sudo apt-get update -qq && sudo apt-get install -y espeak-ng
	elif command -v dnf >/dev/null; then
		sudo dnf install -y espeak-ng
	elif command -v pacman >/dev/null; then
		sudo pacman -S --noconfirm espeak-ng
	elif command -v zypper >/dev/null; then
		sudo zypper install -y espeak-ng
	elif command -v brew >/dev/null; then
		brew install espeak-ng
	else
		die "espeak-ng not found and no known package manager detected. Install it manually, then re-run."
	fi
fi

# --- 2. Clone / update ------------------------------------------------------

if [[ -d "$DEST/.git" ]]; then
	log "Updating existing checkout at $DEST ..."
	git -C "$DEST" pull --ff-only
else
	log "Cloning pi-speak to $DEST ..."
	git clone --depth 1 "$REPO" "$DEST"
fi

# --- 3. Build daemon --------------------------------------------------------

log "Building daemon (release) — this takes a few minutes on first run ..."
cargo build --release --manifest-path "$DEST/Cargo.toml"

mkdir -p "$BIN_DIR" "$LIB_DIR"
install -m 0755 "$DEST/target/release/pi-speak" "$BIN_DIR/pi-speak"

# --- 4. ONNX Runtime --------------------------------------------------------

arch=$(uname -m)
case "$arch" in
	x86_64) ort_arch="x64" ;;
	aarch64 | arm64) ort_arch="aarch64" ;;
	*) die "Unsupported architecture: $arch (only x86_64 and aarch64 have prebuilt ONNX Runtime)" ;;
esac

if [[ -f "$LIB_DIR/libonnxruntime.so" ]]; then
	log "ONNX Runtime already present at $LIB_DIR/libonnxruntime.so"
else
	log "Downloading ONNX Runtime v$ORT_VERSION ($ort_arch) ..."
	ort_url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"
	ort_tmp="$(mktemp -d)"
	curl -fL --retry 3 --progress-bar "$ort_url" -o "$ort_tmp/ort.tgz"
	tar -xzf "$ort_tmp/ort.tgz" -C "$ort_tmp"
	install -m 0755 "$ort_tmp"/onnxruntime-*/lib/libonnxruntime.so "$LIB_DIR/libonnxruntime.so"
	rm -rf "$ort_tmp"
fi

# --- 5. Models ---------------------------------------------------------------

log "Downloading TTS models (~350 MB) ..."
bash "$DEST/scripts/download-models.sh"

# download-models.sh writes to ./models relative to repo root. The daemon
# searches ~/.local/share/pi-speak/models, so canonicalize there (no-op when
# DEST is already the default data dir).
if [[ ! -f "$DEST/models/kokoro/kokoro-v1.0.onnx" ]]; then
	die "Model download did not produce expected files"
fi
DATA_MODELS="$HOME/.local/share/pi-speak/models"
mkdir -p "$DATA_MODELS"
if [[ "$DEST/models" != "$DATA_MODELS" ]]; then
	log "Copying models to $DATA_MODELS ..."
	cp -Rn "$DEST/models/." "$DATA_MODELS/"
fi

# --- 6. Register with Pi -----------------------------------------------------

log "Registering extension with Pi ..."
export DEST
node -e '
const fs = require("fs"), path = require("path"), os = require("os");
const settingsPath = path.join(os.homedir(), ".pi", "agent", "settings.json");
fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
let settings = {};
if (fs.existsSync(settingsPath)) {
	settings = JSON.parse(fs.readFileSync(settingsPath, "utf8"));
}
const extPath = path.join(process.env.DEST, "extension", "speak.ts");
settings.extensions = settings.extensions || [];
if (!settings.extensions.includes(extPath)) settings.extensions.push(extPath);
fs.writeFileSync(settingsPath, JSON.stringify(settings, null, "\t") + "\n");
console.log("  added " + extPath);
'

# --- Done ---------------------------------------------------------------------

printf '\n\033[1;32mpi-speak installed.\033[0m\n'
cat <<EOF

  binary : $BIN_DIR/pi-speak
  models : $DATA_MODELS
  voices : default "jarvis" (change with /speak voice <name>)

Restart your Pi session, then just talk — speech starts automatically.
Manage with /speak (on | off | voice | speed | status | stop | shutdown).
EOF
