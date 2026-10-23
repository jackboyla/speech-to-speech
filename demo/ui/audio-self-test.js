// @ts-check
import { $ } from "./dom.js";

export function supportsAudioOutputSelection() {
  const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
  return typeof Ctx?.prototype?.setSinkId === "function";
}

/** Audio checks own their resources and never use the conversation's stream. */
export class AudioSelfTest {
  /** @param {{ micConstraints: () => MediaStreamConstraints, onMicReady: () => Promise<void> }} options */
  constructor(options) {
    this.options = options;
    this.micButton = /** @type {HTMLButtonElement} */ ($("#test-microphone"));
    this.speakerButton = /** @type {HTMLButtonElement} */ ($("#test-speakers"));
    this.meter = /** @type {HTMLMeterElement} */ ($("#mic-test-meter"));
    this.status = $("#audio-test-status");
    this.output = /** @type {HTMLSelectElement} */ ($("#audio-output"));
    /** @type {{ kind: string, context: AudioContext | null, stream: MediaStream | null, nodes: AudioNode[], frame: number } | null} */
    this.run = null;
    this.micButton.addEventListener("click", () => void this.start("microphone"));
    this.speakerButton.addEventListener("click", () => void this.start("speakers"));
  }

  /** @param {boolean} available */
  setAvailable(available) {
    this.micButton.disabled = this.speakerButton.disabled = !available;
    if (!available) this.stop("Stop the conversation to test audio.");
    else if (!this.run) this.message("Tests run locally. No server connection needed.");
  }

  /** @param {string} text @param {boolean} [error] */
  message(text, error = false) {
    this.status.textContent = text;
    this.status.classList.toggle("error", error);
  }

  /** @param {string} [message] */
  stop(message = "Tests run locally. No server connection needed.") {
    const run = this.run;
    this.run = null;
    if (run) {
      cancelAnimationFrame(run.frame);
      run.stream?.getTracks().forEach((track) => track.stop());
      run.nodes.forEach((node) => node.disconnect());
      if (run.context) void run.context.close().catch(() => {});
    }
    this.micButton.textContent = "Test microphone";
    this.speakerButton.textContent = "Test speakers";
    this.meter.value = -66;
    this.message(message);
  }

  /** @param {"microphone" | "speakers"} kind */
  async start(kind) {
    if (this.micButton.disabled) return;
    const toggleOff = this.run?.kind === kind;
    this.stop();
    if (toggleOff) return;
    const run = { kind, context: /** @type {AudioContext | null} */ (null), stream: /** @type {MediaStream | null} */ (null), nodes: /** @type {AudioNode[]} */ ([]), frame: 0 };
    this.run = run;
    const button = kind === "microphone" ? this.micButton : this.speakerButton;
    button.textContent = kind === "microphone" ? "Stop microphone test" : "Stop speaker test";
    this.message(kind === "microphone" ? "Allow microphone access to test your input." : "Preparing test sound…");
    try {
      if (kind === "microphone" && !navigator.mediaDevices?.getUserMedia) {
        throw new Error("Microphone access needs HTTPS or localhost and a browser with audio capture support.");
      }
      const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
      if (!Ctx) throw new Error("This browser does not support Web Audio tests.");
      // Resume inside the click gesture, including on iOS.
      const context = /** @type {AudioContext} */ (new Ctx());
      run.context = context;
      await context.resume();
      if (this.run !== run) return;

      if (kind === "microphone") {
        const stream = await navigator.mediaDevices.getUserMedia(this.options.micConstraints());
        // A permission prompt may outlive the dialog or a device change.
        if (this.run !== run) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        run.stream = stream;
        stream.getAudioTracks().forEach((track) => track.addEventListener("ended", () => {
          if (this.run === run) {
            this.stop();
            this.message("Microphone disconnected or unavailable. Reconnect it and test again.", true);
          }
        }, { once: true }));
        const source = context.createMediaStreamSource(stream);
        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        run.nodes.push(source, analyser);
        const samples = new Float32Array(analyser.fftSize);
        const paint = () => {
          if (this.run !== run) return;
          analyser.getFloatTimeDomainData(samples);
          const rms = Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
          this.meter.value = Math.max(-66, 20 * Math.log10(rms || 1e-10));
          run.frame = requestAnimationFrame(paint);
        };
        paint();
        this.message("Speak and watch the input level. If it stays low, check your mic's mute switch.");
        await this.options.onMicReady();
      } else {
        const canPickOutput = supportsAudioOutputSelection();
        if (canPickOutput) await /** @type {any} */ (context).setSinkId(this.output.value);
        if (this.run !== run) return;
        const tone = context.createOscillator();
        const gain = context.createGain();
        tone.frequency.value = 440;
        tone.connect(gain);
        gain.connect(context.destination);
        run.nodes.push(tone, gain);
        const now = context.currentTime;
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.08, now + 0.03);
        gain.gain.setValueAtTime(0.08, now + 0.35);
        gain.gain.linearRampToValueAtTime(0, now + 0.45);
        tone.onended = () => {
          if (this.run === run) this.stop("Test sound finished. If you heard nothing, check your output and volume.");
        };
        tone.start(now);
        tone.stop(now + 0.5);
        this.message(canPickOutput ? "Playing through the selected speakers…" : "Playing through system default speakers; this browser cannot select an output.");
      }
    } catch (error) {
      if (this.run !== run) return;
      this.stop();
      const name = error instanceof Error ? error.name : "";
      let message = error instanceof Error ? error.message : String(error);
      if (name === "NotAllowedError" || name === "SecurityError") {
        message = kind === "microphone"
          ? "Microphone permission denied. Allow microphone access in your browser or system settings, then try again."
          : "Speaker access denied. Check your browser's audio output permissions, then try again.";
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        message = kind === "microphone"
          ? "Selected microphone not found. Connect it or choose another microphone, then try again."
          : "Selected speakers not found. Connect them or choose another output, then try again.";
      } else if (name === "NotReadableError" || name === "AbortError" || name === "InvalidAccessError") {
        message = kind === "microphone"
          ? "Microphone unavailable. Check its connection and whether another app is using it, then try again."
          : "Speakers unavailable. Check their connection and choose an available output.";
      }
      this.message(message, true);
    }
  }
}
