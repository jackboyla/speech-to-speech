// @ts-check
/**
 * Developer mode entry point.
 *
 * Loaded by the DEV button, a saved choice, or `?debug=1`.
 *
 * The lab attaches to a live session through the client's read-only event tap.
 * It cannot send, cancel or reorder anything: the one place it touches the
 * session is the microphone, and only when the scripted mic is switched on.
 */

import { Tracer, classifyResponse, classifyTurn, turnMetrics } from "./tracer.js";
import { LabPanel } from "./panel.js";
import { SyntheticMic, loadFixtures } from "./mic.js";
import { ScenarioRunner, SCENARIOS } from "./scenarios.js";

const SCRIPTED_MIC_KEY = "s2s.lab.scriptedMic";
const BASELINE_KEY = "s2s.lab.baseline";

export { labRequested } from "./flag.js";

class Lab {
  constructor() {
    this.tracer = new Tracer();
    /** @type {AudioContext | null} */
    this.ctx = null;
    /** @type {SyntheticMic | null} */
    this.mic = null;
    /** @type {Map<string, import("./mic.js").Fixture>} */
    this.fixtures = new Map();
    this.fixtureError = "Fixtures load when a session starts.";
    /** @type {any} */
    this.client = null;
    /** @type {ScenarioRunner | null} */
    this.runner = null;
    this.scriptedMic = readFlag(SCRIPTED_MIC_KEY);
    /** True when the *current* session was started with the scripted mic. */
    this.scriptedMicLive = false;

    this.panel = new LabPanel({
      tracer: this.tracer,
      onRun: (id) => void this.runScenario(id),
      onStop: () => this.runner?.stop(),
      onReset: () => {
        this.tracer.reset();
        this.panel.clearLog();
        this.panel.render();
      },
      onExport: () => void this.copyTrace(),
      onSaveBaseline: () => this.saveBaseline(),
      onClearBaseline: () => this.clearBaseline(),
      fixtureError: () => this.fixtureError,
      scriptedMic: () => this.scriptedMic,
      scriptedMicLive: () => this.scriptedMicLive,
      onScriptedMic: (value) => {
        this.scriptedMic = value;
        writeFlag(SCRIPTED_MIC_KEY, value);
        this.panel.render();
      },
    });

    this.tracer.addEventListener("change", () => this._scheduleRender());
    this._renderQueued = false;
    this.panel.setBaseline(readBaseline());
    document.body.append(this.panel.el);
    this.visible = false;
    this.setVisible(true);
    // Developer mode is already an explicit opt-in, so exposing the lab costs
    // nothing and is what makes scripted runs drivable from outside the page
    // (see demo/scripts/run_scenario.mjs).
    Object.defineProperty(globalThis, "__voiceLab", { value: this, configurable: true });
  }

  /** Coalesce renders: a busy turn emits events far faster than a useful repaint. */
  _scheduleRender() {
    if (!this.visible) return;
    if (this._renderQueued) return;
    this._renderQueued = true;
    requestAnimationFrame(() => {
      this._renderQueued = false;
      this.panel.render();
    });
  }

  setVisible(visible) {
    this.visible = visible;
    this.panel.el.hidden = !visible;
    document.body.classList.toggle("lab-open", visible);
    if (visible) this.panel.render();
  }

  /**
   * Wire up a freshly created session.
   * @param {any} client
   * @param {{audioContext: AudioContext | null}} context
   */
  attach(client, context) {
    this.client = client;
    this.ctx = context.audioContext ?? null;
    this.tracer.reset();

    client.addEventListener("protocol-event", (/** @type {any} */ e) => this.tracer.onProtocolEvent(e.detail));
    client.addEventListener("audio-packet", (/** @type {any} */ e) => this.tracer.onAudioPacket(e.detail));
    client.addEventListener("playback-cleared", (/** @type {any} */ e) => this.tracer.onPlaybackCleared(e.detail));
    client.addEventListener("playback-mark", (/** @type {any} */ e) => {
      // Sample the AudioContext clock here so worklet marks can be placed on the
      // audio timeline rather than at message-arrival time.
      const contextNowMs = this.ctx ? this.ctx.currentTime * 1000 : undefined;
      this.tracer.onPlaybackMark(e.detail, contextNowMs);
    });

    if (this.ctx && this.mic?.ctx !== this.ctx) this.mic = new SyntheticMic(this.ctx);
    if (this.mic) {
      this.runner = new ScenarioRunner({
        mic: this.mic,
        fixtures: this.fixtures,
        tracer: this.tracer,
        client,
        log: (line, kind) => this.panel.log(line, kind),
      });
      void this._ensureFixtures();
    }
    this.panel.render();
  }

  detach() {
    this.client = null;
    this.scriptedMicLive = false;
    this.panel.setRunning(false);
    this.panel.render();
  }

  /**
   * Whether the next session will capture from fixtures rather than a mic.
   *
   * Answered from the switch alone, because this is asked before the session
   * exists — the synthetic mic itself is not built until there is an
   * AudioContext to build it in.
   */
  willScriptMic() {
    return this.scriptedMic;
  }

  /**
   * The stream the session should capture from, or null to use the real mic.
   *
   * Returning the synthetic stream means `getUserMedia` is never called for
   * this session: a scripted run should not be able to pick up the room.
   */
  micStreamOverride() {
    if (this.scriptedMic && !this.mic && this.ctx) this.mic = new SyntheticMic(this.ctx);
    if (!this.scriptedMic || !this.mic) {
      this.scriptedMicLive = false;
      return null;
    }
    this.scriptedMicLive = true;
    return this.mic.stream;
  }

  async _ensureFixtures() {
    if (!this.ctx || this.fixtures.size > 0) return;
    const { fixtures, error } = await loadFixtures(this.ctx);
    this.fixtures = fixtures;
    this.fixtureError = error;
    if (this.runner) this.runner.fixtures = fixtures;
    this.panel.render();
  }

  /** @param {string} id @param {{stepOverrides?: Record<string, number>}} [options] */
  async runScenario(id, options = {}) {
    const scenario = SCENARIOS.find((s) => s.id === id);
    if (!scenario || !this.runner) return;
    if (!this.client) {
      this.panel.log("✖ Start a session first — a scenario drives a real conversation.", "error");
      return;
    }
    if (!this.scriptedMicLive) {
      this.panel.log(
        "✖ This session is using the real microphone. Turn on the scripted microphone, then press Restart.",
        "error",
      );
      return;
    }
    await this._ensureFixtures();
    this.panel.setRunning(true);
    try {
      await this.runner.run(scenario, options);
    } finally {
      this.panel.setRunning(false);
      this.panel.show("scenarios");
    }
  }

  /** Save the current superseded-turn outcomes to compare a second build against. */
  saveBaseline() {
    const rows = this.panel.currentSupersededRows();
    if (rows.length === 0) return;
    const baseline = { label: `baseline · ${new Date().toLocaleTimeString()}`, rows };
    try {
      localStorage.setItem(BASELINE_KEY, JSON.stringify(baseline));
    } catch {
      // Storage is not essential; the in-memory baseline still works this session.
    }
    this.panel.setBaseline(baseline);
    this.panel.log(`✔ Saved ${rows.length} superseded response(s) as the baseline.`, "ok");
  }

  clearBaseline() {
    try {
      localStorage.removeItem(BASELINE_KEY);
    } catch {
      // Nothing to clean up.
    }
    this.panel.setBaseline(null);
  }

  /** The whole run, reduced to plain data. */
  buildTrace() {
    return {
      capturedAt: new Date().toISOString(),
      transport: this.tracer.transport,
      outputLatencyMs: Math.round(this.tracer.outputLatencyMs),
      userAgent: navigator.userAgent,
      turns: this.tracer.turns.map((turn) => ({
        number: turn.number,
        transcript: turn.transcript,
        verdict: classifyTurn(turn),
        supersededByTurn: turn.supersededBy?.number ?? null,
        metrics: turnMetrics(turn),
        responses: turn.responses.map((response) => ({
          number: response.number,
          id: response.id,
          verdict: classifyResponse(response),
          status: response.status,
          supersededByTurn: response.supersededBy?.number ?? null,
          audibleWhenSuperseded:
            response.supersededAt === null ? null : response.hadAudioBefore(response.supersededAt),
          textAfterSupersede: response.textAfterSupersede,
          audioAfterSupersede: response.audioAfterSupersede,
          audioPackets: response.audioPackets,
          audioMs: Math.round(response.audioMs),
          text: response.text,
          toolCalls: response.toolCalls.map((call) => ({ name: call.name, durationMs: call.durationMs })),
        })),
      })),
      events: this.tracer.rawEvents.map((entry) => ({
        at: Math.round(entry.at),
        type: entry.type,
        direction: entry.direction,
      })),
    };
  }

  /** Put the whole trace on the clipboard as JSON, for pasting into an issue. */
  async copyTrace() {
    const text = JSON.stringify(this.buildTrace(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      this.panel.log("✔ Trace copied to the clipboard.", "ok");
    } catch {
      console.info("[lab] trace", text);
      this.panel.log("Clipboard refused; the trace was logged to the console instead.", "warn");
    }
  }
}

/** @param {string} key */
function readFlag(key) {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

/** @param {string} key @param {boolean} value */
function writeFlag(key, value) {
  try {
    if (value) localStorage.setItem(key, "1");
    else localStorage.removeItem(key);
  } catch {
    // Non-essential.
  }
}

function readBaseline() {
  try {
    const raw = localStorage.getItem(BASELINE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** Mount developer mode. Call only when `labRequested()` is true. */
export async function createLab() {
  await new Promise((resolve) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./lab.css", import.meta.url).href;
    link.onload = () => resolve(undefined);
    link.onerror = () => resolve(undefined);
    document.head.append(link);
  });
  return new Lab();
}
