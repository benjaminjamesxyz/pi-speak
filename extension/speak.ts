/**
 * Speak Extension for Pi Coding Agent
 *
 * Provides real-time speech response (TTS) powered by pi-speak daemon (Rust + SIMD).
 * Automatically speaks assistant responses sentence-by-sentence with zero lag and CD quality.
 * Intelligently coordinates across multiple concurrent Pi sessions without voice interleaving.
 *
 * Commands:
 *   /speak [on|off|stop|voice|speed|status|shutdown]
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn, ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";

interface SpeakIPCRequest {
	action: "feed" | "flush" | "say" | "stop" | "stop_all" | "status" | "shutdown" | "set_voice" | "set_speed";
	text?: string;
	voice?: string;
	speed?: number;
	all?: boolean;
	force?: boolean;
}

interface SpeakIPCResponse {
	status: "ok" | "status" | "error";
	message?: string;
	playing?: boolean;
	model?: string;
	sample_rate?: number;
	client_count?: number;
	voice?: string;
	speed?: number;
	error?: string;
}

interface AssistantDeltaEvent {
	type?: string;
	delta?: unknown;
}

interface MessageUpdateEvent {
	assistantMessageEvent?: AssistantDeltaEvent;
}

interface MessageEndEvent {
	message?: {
		role?: string;
		content?: unknown;
	};
}

/**
 * Resolves the location of the pi-speak native daemon binary across common environments:
 * 1. Explicit PI_SPEAK_BIN environment variable
 * 2. Target release / debug directory relative to this extension file
 * 3. User standard binaries: ~/.local/bin/pi-speak, ~/.cargo/bin/pi-speak
 * 4. System PATH directories
 */
function resolveBinaryPath(): string | null {
	if (process.env.PI_SPEAK_BIN && fs.existsSync(process.env.PI_SPEAK_BIN)) {
		return process.env.PI_SPEAK_BIN;
	}

	// Check relative to extension file (e.g. extension/../target/release/pi-speak)
	const extRelease = path.resolve(__dirname, "..", "target", "release", "pi-speak");
	if (fs.existsSync(extRelease)) return extRelease;

	const extDebug = path.resolve(__dirname, "..", "target", "debug", "pi-speak");
	if (fs.existsSync(extDebug)) return extDebug;

	const home = process.env.HOME;
	if (home) {
		const localBin = path.join(home, ".local", "bin", "pi-speak");
		if (fs.existsSync(localBin)) return localBin;

		const cargoBin = path.join(home, ".cargo", "bin", "pi-speak");
		if (fs.existsSync(cargoBin)) return cargoBin;
	}

	// Look up in system PATH
	const pathEnv = process.env.PATH || "";
	const pathDirs = pathEnv.split(path.delimiter);
	for (const dir of pathDirs) {
		if (!dir) continue;
		const candidate = path.join(dir, "pi-speak");
		if (fs.existsSync(candidate)) {
			try {
				fs.accessSync(candidate, fs.constants.X_OK);
				return candidate;
			} catch {}
		}
	}

	return null;
}

let isSpeakingEnabled = true;
let clientSocket: net.Socket | null = null;
let connectingPromise: Promise<net.Socket> | null = null;
let daemonProcess: ChildProcess | null = null;
let sessionContext: ExtensionContext | null = null;

/**
 * Extracts a readable error message from an unknown catch clause value.
 */
function getErrorMessage(error: unknown): string {
	if (error instanceof Error) {
		return error.message;
	}
	return String(error);
}

/**
 * Intelligently resolves the Unix domain socket path across environments:
 * 1. Explicit PI_SPEAK_SOCKET environment variable
 * 2. Active socket in XDG_RUNTIME_DIR or /tmp
 * 3. Default path based on XDG_RUNTIME_DIR or /tmp
 */
function getSocketPath(): string {
	if (process.env.PI_SPEAK_SOCKET) {
		return process.env.PI_SPEAK_SOCKET;
	}

	const tmpSock = "/tmp/pi-speak.sock";
	const xdgRuntime = process.env.XDG_RUNTIME_DIR;
	const xdgSock = xdgRuntime ? path.join(xdgRuntime, "pi-speak.sock") : null;

	// Prefer currently active socket if one exists
	if (xdgSock && fs.existsSync(xdgSock)) return xdgSock;
	if (fs.existsSync(tmpSock)) return tmpSock;

	// Fallback to user runtime dir, then /tmp
	if (xdgSock) return xdgSock;
	return tmpSock;
}

/**
 * Attempts a single non-blocking connection to the Unix domain socket.
 */
function tryConnect(sockPath: string): Promise<net.Socket> {
	return new Promise((resolve, reject) => {
		const sock = net.createConnection(sockPath);
		let resolved = false;

		sock.once("connect", () => {
			resolved = true;
			resolve(sock);
		});

		sock.once("error", (err: Error) => {
			if (!resolved) {
				resolved = true;
				sock.destroy();
				reject(err);
			}
		});
	});
}

/**
 * Attaches lifecycle and drain listeners to the active client socket.
 */
function setupSocket(sock: net.Socket): void {
	clientSocket = sock;

	sock.on("error", () => {
		if (clientSocket === sock) {
			clientSocket = null;
		}
	});

	sock.on("close", () => {
		if (clientSocket === sock) {
			clientSocket = null;
		}
	});

	// Drain incoming ok/ack responses so the buffer never fills
	sock.on("data", () => {});
}

/**
 * Ensures the pi-speak daemon is running and returns an active persistent socket connection.
 * Multi-session intelligent:
 * - Reuses existing daemon across all Pi instances
 * - Only cleans up stale socket if connection is definitively refused (no live process)
 * - Safe against racing starts via file-lock aware retry
 */
async function getSocket(): Promise<net.Socket> {
	if (clientSocket && !clientSocket.destroyed && clientSocket.writable) {
		return clientSocket;
	}

	if (connectingPromise) {
		return connectingPromise;
	}

	connectingPromise = (async (): Promise<net.Socket> => {
		const sockPath = getSocketPath();

		// 1. Try connecting to existing socket
		if (fs.existsSync(sockPath)) {
			try {
				const sock = await tryConnect(sockPath);
				setupSocket(sock);
				return sock;
			} catch (err: unknown) {
				// Only unlink if connection was definitively refused (no listening daemon)
				const nodeErr = err as NodeJS.ErrnoException;
				if (nodeErr?.code === "ECONNREFUSED") {
					try {
						fs.unlinkSync(sockPath);
					} catch {}
				}
			}
		}

		// 2. Spawn daemon if not running
		const binaryPath = resolveBinaryPath();
		if (!binaryPath) {
			throw new Error(
				"pi-speak binary not found. Build it with 'cargo build --release', install to ~/.cargo/bin, or set PI_SPEAK_BIN."
			);
		}

		daemonProcess = spawn(binaryPath, ["daemon"], {
			detached: true,
			stdio: "ignore",
		});
		daemonProcess.unref();

		// 3. Retry connecting for up to 4 seconds
		const startTime = Date.now();
		while (Date.now() - startTime < 4000) {
			await new Promise<void>((r) => setTimeout(r, 100));
			const currentSock = getSocketPath();
			if (fs.existsSync(currentSock)) {
				try {
					const sock = await tryConnect(currentSock);
					setupSocket(sock);
					return sock;
				} catch {
					// Socket still initializing
				}
			}
		}

		throw new Error("Failed to connect to pi-speak daemon after spawning");
	})();

	try {
		return await connectingPromise;
	} finally {
		connectingPromise = null;
	}
}

/**
 * Sends a message to the pi-speak daemon over the persistent connection.
 */
async function sendDaemonMessage(msg: SpeakIPCRequest): Promise<void> {
	try {
		const sock = await getSocket();
		if (!sock.destroyed && sock.writable) {
			sock.write(JSON.stringify(msg) + "\n");
		}
	} catch {
		// Daemon connection errors are non-fatal to the CLI
	}
}

/**
 * Queries the daemon for status over an isolated connection to avoid stream interleaving.
 */
function queryStatus(): Promise<SpeakIPCResponse> {
	return new Promise((resolve, reject) => {
		const sockPath = getSocketPath();
		const sock = net.createConnection(sockPath);
		let buffer = "";

		const timer = setTimeout(() => {
			sock.destroy();
			reject(new Error("Timeout waiting for status response"));
		}, 2000);

		sock.on("connect", () => {
			sock.write(JSON.stringify({ action: "status" }) + "\n");
		});

		sock.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			if (buffer.includes("\n")) {
				clearTimeout(timer);
				try {
					const resp = JSON.parse(buffer.trim()) as SpeakIPCResponse;
					sock.destroy();
					resolve(resp);
				} catch (e: unknown) {
					sock.destroy();
					reject(e instanceof Error ? e : new Error(String(e)));
				}
			}
		});

		sock.on("error", (err: Error) => {
			clearTimeout(timer);
			sock.destroy();
			reject(err);
		});
	});
}

/**
 * Updates Pi TUI status bar widget.
 */
function updateStatus(ctx?: ExtensionContext | ExtensionCommandContext): void {
	const targetCtx = ctx ?? sessionContext;
	if (!targetCtx?.hasUi || !targetCtx?.ui) return;

	if (isSpeakingEnabled) {
		targetCtx.ui.setStatus("speak", "🔊 [TTS: On]");
	} else {
		targetCtx.ui.setStatus("speak", "🔇 [TTS: Muted]");
	}
}

export default function (pi: ExtensionAPI): void {
	// Track session lifecycle
	pi.on("session_start", async (_event: unknown, ctx: ExtensionContext) => {
		sessionContext = ctx;
		// Non-blocking socket warmup
		getSocket().catch(() => {});
		updateStatus(ctx);
	});

	pi.on("session_shutdown", async () => {
		// Stop only this session's speech; other open sessions remain completely unaffected
		await sendDaemonMessage({ action: "stop", all: false });
		if (clientSocket) {
			try {
				clientSocket.end();
			} catch {}
			clientSocket = null;
		}
		sessionContext = null;
	});

	// Barge-in: when user starts a new prompt in this session, immediately silence previous speech in this session
	pi.on("before_agent_start", async () => {
		await sendDaemonMessage({ action: "stop", all: false });
	});

	// Stream tokens to TTS engine as they arrive
	pi.on("message_update", async (event: MessageUpdateEvent) => {
		if (!isSpeakingEnabled) return;

		const evt = event.assistantMessageEvent;
		if (evt && evt.type === "text_delta" && typeof evt.delta === "string") {
			await sendDaemonMessage({
				action: "feed",
				text: evt.delta,
			});
		}
	});

	// Assistant message completed: flush any remaining buffered sentences for this session
	pi.on("message_end", async (event: MessageEndEvent) => {
		if (isSpeakingEnabled && event.message?.role === "assistant") {
			await sendDaemonMessage({ action: "flush" });
		}
	});

	// Turn ended: final flush safety net
	pi.on("turn_end", async () => {
		if (isSpeakingEnabled) {
			await sendDaemonMessage({ action: "flush" });
		}
	});

	// Register /speak command
	pi.registerCommand("speak", {
		description: "Controls voice playback: /speak [on|off|stop|voice|speed|status|shutdown]",
		getArgumentCompletions: (prefix: string) => {
			const subcommands = ["on", "off", "stop", "voice", "speed", "status", "shutdown"];
			const trimmed = prefix.trim();
			if (trimmed.startsWith("voice ")) {
				const voices = [
					"jarvis",
					"bm_george",
					"bm_daniel",
					"bm_fable",
					"bm_lewis",
					"af_heart",
					"af_bella",
					"af_sarah",
					"af_nicole",
					"af_sky",
					"am_adam",
					"am_echo",
					"am_michael",
					"am_onyx",
					"am_fenrir",
					"bf_emma",
					"bf_isabella",
				];
				const vPrefix = trimmed.slice(6).trim();
				return voices
					.filter((v) => v.startsWith(vPrefix))
					.map((v) => ({ value: `voice ${v}`, label: v }));
			}
			const filtered = subcommands.filter((s) => s.startsWith(prefix));
			return filtered.length > 0 ? filtered.map((s) => ({ value: s, label: s })) : null;
		},
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const rawArgs = args ? args.trim() : "";
			const parts = rawArgs.split(/\s+/);
			const sub = (parts[0] || "").toLowerCase();

			switch (sub) {
				case "off":
				case "mute": {
					isSpeakingEnabled = false;
					await sendDaemonMessage({ action: "stop", all: false });
					updateStatus(ctx);
					ctx.ui?.notify("Speech output muted for this session.", "info");
					break;
				}
				case "on":
				case "unmute": {
					isSpeakingEnabled = true;
					updateStatus(ctx);
					ctx.ui?.notify("Speech output enabled (hands-free automatic).", "info");
					break;
				}
				case "voice": {
					const voiceName = parts[1];
					if (!voiceName) {
						try {
							const resp = await queryStatus();
							ctx.ui?.notify(`Current voice: ${resp.voice || "jarvis"} (${resp.model || "Kokoro-82M"})`, "info");
						} catch (e: unknown) {
							ctx.ui?.notify(`Failed to query voice: ${getErrorMessage(e)}`, "error");
						}
					} else {
						await sendDaemonMessage({ action: "set_voice", voice: voiceName });
						ctx.ui?.notify(`Voice set to '${voiceName}'`, "info");
					}
					break;
				}
				case "speed": {
					const speedVal = parseFloat(parts[1] || "");
					if (isNaN(speedVal) || speedVal < 0.2 || speedVal > 4.0) {
						ctx.ui?.notify("Usage: /speak speed <0.5 - 3.0>", "warning");
					} else {
						await sendDaemonMessage({ action: "set_speed", speed: speedVal });
						ctx.ui?.notify(`Speaking speed set to ${speedVal.toFixed(2)}x`, "info");
					}
					break;
				}
				case "stop": {
					const stopAll = parts[1] === "all" || parts[1] === "--all";
					await sendDaemonMessage({ action: "stop", all: stopAll });
					ctx.ui?.notify(
						stopAll ? "Speech stopped across all sessions." : "Speech stopped for this session.",
						"info"
					);
					break;
				}
				case "status": {
					try {
						const resp = await queryStatus();
						const sessionsInfo = resp.client_count !== undefined ? ` | Sessions: ${resp.client_count} active` : "";
						const voiceInfo = resp.voice ? ` | Voice: ${resp.voice}` : "";
						const speedInfo = resp.speed ? ` (${resp.speed.toFixed(2)}x)` : "";
						ctx.ui?.notify(
							`TTS: ${resp.playing ? "Playing" : "Idle"}${sessionsInfo} | ${resp.model ?? "Kokoro-82M"}${voiceInfo}${speedInfo} | Output: ${isSpeakingEnabled ? "Enabled" : "Muted"}`,
							"info"
						);
					} catch (err: unknown) {
						ctx.ui?.notify(`Cannot reach daemon: ${getErrorMessage(err)}`, "error");
					}
					break;
				}
				case "shutdown":
				case "kill": {
					const isForce = parts.includes("force") || parts.includes("--force");
					try {
						const sock = net.createConnection(getSocketPath());
						let buffer = "";
						sock.on("connect", () => {
							sock.write(JSON.stringify({ action: "shutdown", force: isForce }) + "\n");
						});
						sock.on("data", (chunk: Buffer) => {
							buffer += chunk.toString();
							if (buffer.includes("\n")) {
								try {
									const resp = JSON.parse(buffer.trim()) as SpeakIPCResponse;
									if (resp.status === "error") {
										ctx.ui?.notify(resp.error || "Failed to shut down daemon", "warning");
									} else {
										ctx.ui?.notify("pi-speak daemon shut down cleanly.", "info");
										if (clientSocket) {
											clientSocket.end();
											clientSocket = null;
										}
									}
								} catch {}
								sock.destroy();
							}
						});
						sock.on("error", (err: Error) => {
							ctx.ui?.notify(`Error communicating with daemon: ${err.message}`, "error");
							sock.destroy();
						});
					} catch (err: unknown) {
						ctx.ui?.notify(`Error shutting down daemon: ${getErrorMessage(err)}`, "error");
					}
					break;
				}
				case "": {
					// Toggle
					isSpeakingEnabled = !isSpeakingEnabled;
					if (!isSpeakingEnabled) {
						await sendDaemonMessage({ action: "stop", all: false });
					}
					updateStatus(ctx);
					ctx.ui?.notify(
						`Speech output toggled ${isSpeakingEnabled ? "ON" : "OFF"}`,
						"info"
					);
					break;
				}
				default: {
					ctx.ui?.notify("Usage: /speak [on|off|stop [all]|voice [name]|speed [val]|status|shutdown [--force]]", "warning");
					break;
				}
			}
		},
	});
}
