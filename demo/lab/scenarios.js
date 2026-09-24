// @ts-check
/**
 * Scripted conversations.
 *
 * Each scenario is a short list of steps the runner performs against a live
 * session. Steps that wait are written against *events*, not stopwatches,
 * wherever the timing matters. A scenario that reproduces a race by sleeping
 * 400 ms reproduces it on one machine on one day; a scenario that waits for
 * `response.created` and then starts talking reproduces it wherever the race
 * exists at all.
 *
 * Every scenario also carries a `check`. Driving a race is not the same as
 * hitting it — the pipeline may answer faster than the script can interrupt —
 * so the runner reports whether the window it was aiming at actually opened.
 * A run that missed says so instead of quietly showing a pass.
 */

import { classifyResponse } from "./tracer.js";

const DEFAULT_TIMEOUT_MS = 15_000;

/**
 * @typedef {Object} Step
 * @property {string} [say]          Fixture name to speak.
 * @property {number} [silence]      Milliseconds of nothing.
 * @property {string} [id]           Name, so a caller can retime this step.
 * @property {string} [awaitEvent]   Protocol event type to wait for.
 * @property {boolean} [awaitAudible] Wait until the assistant is actually heard.
 * @property {boolean} [awaitToolCall] Wait until a tool call is requested.
 * @property {boolean} [awaitDone]   Wait until the open response finishes.
 * @property {number} [timeoutMs]
 * @property {string} [note]         Shown in the run log.
 */

/** @type {Array<{id: string, title: string, purpose: string, aims: string, steps: Step[], check: (t: any) => {met: boolean, note: string}}>} */
export const SCENARIOS = [
  {
    id: "normal-turn",
    title: "Normal turn",
    purpose: "The baseline. One question, one answer, nothing overlapping.",
    aims: "Every stage mark present, so the other scenarios have something to be compared against.",
    steps: [
      { say: "capital_france", note: "ask a short question" },
      { awaitDone: true, timeoutMs: 20_000, note: "let the answer finish" },
    ],
    check: (tracer) => {
      const turn = tracer.turns[tracer.turns.length - 1];
      const response = turn?.responses[0];
      if (!response) return { met: false, note: "No response was created." };
      return {
        met: response.audibleAt !== null || response.firstAudioEventAt !== null,
        note: response.audibleAt !== null ? "Answer was heard." : "No audio reached the output.",
      };
    },
  },
  {
    id: "stop-resume",
    title: "Rapid stop → resume",
    purpose: "A speaker who pauses mid-thought and carries on.",
    aims: "One logical turn, not two. If the pause is read as end-of-turn the pipeline answers half a question.",
    steps: [
      { say: "half_question", note: "start the sentence" },
      { silence: 320, note: "pause, shorter than the reopen window" },
      { say: "half_question_tail", note: "finish the sentence" },
      { awaitDone: true, timeoutMs: 20_000 },
    ],
    check: (tracer) => {
      const turns = tracer.turns.length;
      return {
        met: true,
        note: turns === 1
          ? "Held as one turn — the pause did not end it."
          : `Split into ${turns} turns; the pipeline treated the pause as end-of-turn.`,
      };
    },
  },
  {
    id: "barge-in",
    title: "Barge-in while speaking",
    purpose: "Interrupt the assistant once it is already talking.",
    aims: "Audio stops promptly. The response was committed, so #308 lets the generation finish — the point is the silence, not the drop.",
    steps: [
      { say: "capital_france" },
      { awaitAudible: true, timeoutMs: 20_000, note: "wait until it is actually speaking" },
      { silence: 250, note: "let a little of the answer through" },
      { say: "interrupt_wait", note: "talk over it" },
      { awaitDone: true, timeoutMs: 20_000 },
    ],
    check: (tracer) => {
      const turn = tracer.turns.find((t) => t.bargedIn);
      if (!turn) return { met: false, note: "Nothing was audible to interrupt — the barge-in missed." };
      const ms = turn.bargeInSilencedAt === null ? null : Math.round(turn.bargeInSilencedAt - turn.startedAt);
      return { met: true, note: ms === null ? "Interrupted, but silence was never observed." : `Silent ${ms} ms after the interruption began.` };
    },
  },
  {
    id: "superseded-turn",
    title: "Turn 2 while turn 1 is still generating",
    purpose: "The issue #308 race: a second question arrives while the first answer is being generated and before any of it is audible.",
    aims: "Turn 1's uncommitted work should die. If it survives, the user hears an answer to a question they have already replaced.",
    steps: [
      { say: "capital_france", note: "ask the first question" },
      { awaitEvent: "input_audio_buffer.speech_stopped", timeoutMs: 20_000, note: "turn 1 ends; the pipeline starts thinking" },
      // Long enough that turn 2 is a new turn rather than turn 1 reopening
      // after a pause (the reopen window defaults to 800 ms), and short enough
      // that turn 1's answer is still being generated and still silent.
      { id: "interrupt-at", silence: 1200, note: "past the reopen window, before the answer exists" },
      { say: "planet_count", note: "ask something else while turn 1 is still silent" },
      { awaitDone: true, timeoutMs: 30_000 },
      { silence: 2500, note: "give any stale output time to show itself" },
    ],
    check: (tracer) => {
      const replaced = tracer.turns.filter((turn) => turn.supersededAt !== null);
      if (replaced.length === 0) {
        return {
          met: false,
          note: "No turn was replaced while it was still waiting for an answer — the race window never opened. "
            + "The pipeline may be answering faster than the script can interrupt; widen the window by slowing "
            + "the language model (see demo/scripts/slow_llm.py).",
        };
      }
      const leaked = replaced.filter((turn) =>
        turn.responses.some((response) => classifyResponse(response) === "stale-accepted"));
      const spoke = leaked.flatMap((turn) => turn.responses.map((r) => r.text)).filter(Boolean);
      return {
        met: true,
        note: leaked.length > 0
          ? `Reproduced, and the replaced turn still spoke: ${JSON.stringify(spoke.join(" "))}`
          : `Reproduced: ${replaced.length} turn(s) replaced while unanswered, and none of them produced output.`,
      };
    },
  },
  {
    id: "fragments",
    title: "Repeated short fragments",
    purpose: "Several very short utterances in quick succession.",
    aims: "Each fragment is either folded into one turn or answered once. Fragments that each spawn a response pile answers on top of each other.",
    steps: [
      { say: "yes" },
      { silence: 500 },
      { say: "yes" },
      { silence: 500 },
      { say: "yes" },
      { silence: 3000 },
    ],
    check: (tracer) => {
      const responses = [...tracer.responsesById.values()];
      const stale = responses.filter((r) => classifyResponse(r) === "stale-accepted").length;
      return {
        met: true,
        note: `${tracer.turns.length} turn(s), ${responses.length} response(s), ${stale} stale-accepted.`,
      };
    },
  },
  {
    id: "tool-interrupt",
    title: "Tool call, then interruption",
    purpose: "Interrupt while a tool call is in flight rather than while audio is playing.",
    aims: "The tool result must not resurrect a turn the user has already replaced. Needs the web-search tool enabled.",
    steps: [
      { say: "weather_query", note: "ask something that needs a tool" },
      { awaitToolCall: true, timeoutMs: 20_000, note: "wait for the call to go out" },
      { say: "planet_count", note: "change the subject mid-call" },
      { awaitDone: true, timeoutMs: 25_000 },
      { silence: 1500 },
    ],
    check: (tracer) => {
      const withTools = [...tracer.responsesById.values()].filter((r) => r.toolCalls.length > 0);
      if (withTools.length === 0) {
        return { met: false, note: "No tool call happened — enable the web-search tool and try again." };
      }
      const verdicts = withTools.map(classifyResponse);
      return {
        met: true,
        note: `Tool-bearing response(s) ended as: ${verdicts.join(", ")}.`,
      };
    },
  },
];

/** Thrown when a scenario is stopped by the user. */
export class Aborted extends Error {}

export class ScenarioRunner {
  /**
   * @param {{mic: import("./mic.js").SyntheticMic, fixtures: Map<string, import("./mic.js").Fixture>,
   *          tracer: any, client: EventTarget, log: (line: string, kind?: string) => void}} deps
   */
  constructor(deps) {
    this.mic = deps.mic;
    this.fixtures = deps.fixtures;
    this.tracer = deps.tracer;
    this.client = deps.client;
    this.log = deps.log;
    this.running = false;
    this._abort = null;
  }

  stop() {
    this._abort?.abort();
    this.mic.stop();
  }

  /**
   * @param {typeof SCENARIOS[number]} scenario
   * @param {{stepOverrides?: Record<string, number>}} [options]
   *   Retime a named step. The moment an interruption lands decides which
   *   window it falls in, so being able to sweep it is the difference between
   *   a demonstration and a guess.
   * @returns {Promise<{ok: boolean, met: boolean, note: string}>}
   */
  async run(scenario, options = {}) {
    if (this.running) throw new Error("A scenario is already running");
    this.running = true;
    const controller = new AbortController();
    this._abort = controller;
    const started = performance.now();
    this.log(`▶ ${scenario.title}`, "head");
    try {
      const overrides = options.stepOverrides ?? {};
      for (const raw of scenario.steps) {
        controller.signal.throwIfAborted();
        const override = raw.id !== undefined ? overrides[raw.id] : undefined;
        const step = override === undefined ? raw : { ...raw, silence: override };
        await this._step(step, controller.signal, started);
      }
      const verdict = scenario.check(this.tracer);
      this.log(verdict.met ? `✔ ${verdict.note}` : `… ${verdict.note}`, verdict.met ? "ok" : "warn");
      return { ok: true, ...verdict };
    } catch (err) {
      if (err instanceof Aborted || controller.signal.aborted) {
        this.log("■ stopped", "warn");
        return { ok: false, met: false, note: "Stopped." };
      }
      const message = err instanceof Error ? err.message : String(err);
      this.log(`✖ ${message}`, "error");
      return { ok: false, met: false, note: message };
    } finally {
      this.mic.stop();
      this.running = false;
      this._abort = null;
    }
  }

  /** @param {Step} step @param {AbortSignal} signal @param {number} started */
  async _step(step, signal, started) {
    const at = () => `${Math.round(performance.now() - started)} ms`;
    if (step.say) {
      const fixture = this.fixtures.get(step.say);
      if (!fixture) throw new Error(`Missing fixture "${step.say}" — generate the fixture set first`);
      this.log(`  ${at()}  say “${fixture.text || step.say}”${step.note ? `  — ${step.note}` : ""}`);
      await this.mic.play(fixture.buffer);
      return;
    }
    if (typeof step.silence === "number") {
      this.log(`  ${at()}  silence ${step.silence} ms${step.note ? `  — ${step.note}` : ""}`);
      await sleep(step.silence, signal);
      return;
    }
    if (step.awaitEvent) {
      this.log(`  ${at()}  wait for ${step.awaitEvent}${step.note ? `  — ${step.note}` : ""}`);
      await this._waitForProtocol((type) => type === step.awaitEvent, step, signal);
      return;
    }
    if (step.awaitAudible) {
      this.log(`  ${at()}  wait until audible${step.note ? `  — ${step.note}` : ""}`);
      await this._waitForAudible(step, signal);
      return;
    }
    if (step.awaitToolCall) {
      this.log(`  ${at()}  wait for a tool call${step.note ? `  — ${step.note}` : ""}`);
      await this._waitForProtocol((type) => type === "response.function_call_arguments.done", step, signal);
      return;
    }
    if (step.awaitDone) {
      this.log(`  ${at()}  wait for response.done${step.note ? `  — ${step.note}` : ""}`);
      await this._waitForProtocol((type) => type === "response.done", step, signal);
      return;
    }
    throw new Error(`Unrecognised step: ${JSON.stringify(step)}`);
  }

  /** @param {(type: string) => boolean} predicate @param {Step} step @param {AbortSignal} signal */
  _waitForProtocol(predicate, step, signal) {
    return this._race("protocol-event", (event) => predicate(event.detail?.type), step, signal);
  }

  /** @param {Step} step @param {AbortSignal} signal */
  _waitForAudible(step, signal) {
    return this._race(
      "playback-mark",
      (event) => event.detail?.kind === "playback-started" || event.detail?.kind === "rtc-audible",
      step,
      signal,
    );
  }

  /**
   * Resolve on the first matching client event, reject on timeout or abort.
   * @param {string} name @param {(event: any) => boolean} match @param {Step} step @param {AbortSignal} signal
   */
  _race(name, match, step, signal) {
    const timeoutMs = step.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        this.client.removeEventListener(name, onEvent);
        signal.removeEventListener("abort", onAbort);
        window.clearTimeout(timer);
      };
      const onEvent = (/** @type {any} */ event) => {
        if (!match(event)) return;
        cleanup();
        resolve(undefined);
      };
      const onAbort = () => {
        cleanup();
        reject(new Aborted("stopped"));
      };
      const timer = window.setTimeout(() => {
        cleanup();
        reject(new Error(`Timed out after ${timeoutMs} ms waiting for ${step.awaitEvent ?? name}`));
      }, timeoutMs);
      this.client.addEventListener(name, onEvent);
      signal.addEventListener("abort", onAbort, { once: true });
    });
  }
}

/** @param {number} ms @param {AbortSignal} signal */
function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve(undefined);
    }, ms);
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new Aborted("stopped"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
