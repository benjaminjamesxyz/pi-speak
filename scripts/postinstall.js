#!/usr/bin/env node
/**
 * pi-speak npm postinstall: self-provisions the native side.
 *
 * Downloads into ~/.local/share/pi-speak/ (a path the daemon already searches):
 *   bin/pi-speak            — prebuilt daemon from GitHub Releases (falls back to cargo build)
 *   lib/libonnxruntime.so   — ONNX Runtime (~/.local/lib is also on the daemon's search path)
 *   models/…                — Kokoro-82M weights + voices from Hugging Face
 *
 * Never fails the npm install: any problem prints a warning and exits 0.
 * Set PI_SPEAK_SKIP_SETUP=1 to skip entirely.
 */
import { copyFileSync, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir, platform, arch } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const ROOT = path.join(homedir(), ".local", "share", "pi-speak");
const LIB = path.join(homedir(), ".local", "lib");
const REPO = "benjaminjamesxyz/pi-speak";
const ORT_VERSION = "1.20.1";

const log = (msg) => console.log(`pi-speak: ${msg}`);
const warn = (msg) => console.warn(`pi-speak: WARNING: ${msg}`);

function targets() {
	if (platform() === "linux" && arch() === "x64") return { bin: "linux-x64", ort: "linux-x64", ortLib: "libonnxruntime.so" };
	if (platform() === "linux" && arch() === "arm64") return { bin: "linux-aarch64", ort: "linux-aarch64", ortLib: "libonnxruntime.so" };
	if (platform() === "darwin" && arch() === "arm64") return { bin: "darwin-arm64", ort: "osx-arm64", ortLib: "libonnxruntime.dylib" };
	if (platform() === "darwin" && arch() === "x64") return { bin: "darwin-x64", ort: "osx-x64", ortLib: "libonnxruntime.dylib" };
	return null;
}

async function download(url, dest) {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
	const tmp = `${dest}.tmp`;
	await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
	renameSync(tmp, dest);
	return dest;
}

function untar(tarball, destDir) {
	mkdirSync(destDir, { recursive: true });
	execFileSync("tar", ["xzf", tarball, "-C", destDir]);
}

async function fetchDaemonBinary(t) {
	const dest = path.join(ROOT, "bin");
	mkdirSync(dest, { recursive: true });
	const out = path.join(dest, "pi-speak");
	try {
		const url = `https://github.com/${REPO}/releases/latest/download/pi-speak-${t.bin}`;
		await download(url, out);
		execFileSync("chmod", ["0755", out]);
		log(`daemon binary installed (${t.bin})`);
		return true;
	} catch (e) {
		warn(`no prebuilt binary: ${e.message}`);
	}
	return false;
}

async function fetchOnnxRuntime(t) {
	const libPath = path.join(LIB, t.ortLib);
	if (existsSync(libPath)) {
		log("ONNX Runtime already installed");
		return;
	}
	mkdirSync(LIB, { recursive: true });
	const name = `onnxruntime-${t.ort}-${ORT_VERSION}`;
	const url = `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${name}.tgz`;
	const tmp = path.join(ROOT, `${name}.tgz`);
	await download(url, tmp);
	untar(tmp, ROOT);
	rmSync(tmp);
	copyFileSync(path.join(ROOT, name, "lib", t.ortLib), libPath);
	execFileSync("chmod", ["0755", libPath]);
	rmSync(path.join(ROOT, name), { recursive: true, force: true });
	log(`ONNX Runtime ${ORT_VERSION} installed`);
}

async function fetchModels() {
	const modelsDir = path.join(ROOT, "models", "kokoro");
	mkdirSync(modelsDir, { recursive: true });
	// hexgrad/Kokoro-82M no longer hosts the ONNX export; kokoro-onnx releases do (ZIP of .npy voices,
	// the exact format the daemon's voice_loader.rs parses).
	const MODEL_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1";
	const files = [
		[`${MODEL_BASE}/kokoro-v1.0.onnx`, "kokoro-v1.0.onnx", 300_000_000],
		[`${MODEL_BASE}/voices-v1.0.bin`, "voices-v1.0.bin", 25_000_000],
	];
	for (const [url, name, minSize] of files) {
		const dest = path.join(modelsDir, name);
		if (existsSync(dest) && statSync(dest).size >= minSize * 0.9) {
			log(`${name} already present`);
			continue;
		}
		log(`downloading ${name} (~${Math.round(minSize / 1e6)} MB) ...`);
		await download(url, dest);
	}
}

/**
 * Fallback when no prebuilt release asset exists yet: clone shallow and build with cargo.
 * ponytail: rebuilds on every reinstall until a release asset is published; fine while releases are rare.
 */
async function buildFromSource() {
	let haveCargo = false;
	try {
		execFileSync("cargo", ["--version"], { stdio: "ignore" });
		haveCargo = true;
	} catch {}
	if (!haveCargo) {
		warn("no cargo toolchain; install Rust (https://rustup.rs) and re-run install, or wait for a release binary");
		return false;
	}
	const src = path.join(ROOT, "src");
	rmSync(src, { recursive: true, force: true });
	log("building daemon from source (first install takes a few minutes) ...");
	execFileSync("git", ["clone", "--depth", "1", `https://github.com/${REPO}.git`, src], { stdio: "inherit" });
	execFileSync("cargo", ["build", "--release"], { cwd: src, stdio: "inherit" });
	const out = path.join(ROOT, "bin", "pi-speak");
	mkdirSync(path.dirname(out), { recursive: true });
	copyFileSync(path.join(src, "target", "release", "pi-speak"), out);
	execFileSync("chmod", ["0755", out]);
	return true;
}

async function main() {
	if (process.env.PI_SPEAK_SKIP_SETUP) {
		log("PI_SPEAK_SKIP_SETUP set — skipping native setup");
		return;
	}
	const t = targets();
	if (!t) {
		warn(`unsupported platform ${platform()}-${arch()}; build manually: https://github.com/${REPO}#manual-install`);
		return;
	}
	mkdirSync(ROOT, { recursive: true });
	await fetchOnnxRuntime(t);
	await fetchModels();
	// Keep any existing daemon binary; re-download only when absent.
	const binPath = path.join(ROOT, "bin", "pi-speak");
	if (existsSync(binPath)) {
		log("daemon binary already installed");
	} else if (!(await fetchDaemonBinary(t)) && !(await buildFromSource())) {
		warn(`daemon binary missing. Install Rust (https://rustup.rs) and re-run install, or
  pi-speak: build it:  git clone https://github.com/${REPO} && cd pi-speak && cargo build --release
  pi-speak: then place it at ${binPath}`);
	}
}

main().catch((e) => {
	// Never break `pi install` over provisioning problems.
	console.warn(`pi-speak: setup incomplete: ${e.message}`);
	console.warn(`pi-speak: run manually later or see https://github.com/${REPO}#install`);
});
