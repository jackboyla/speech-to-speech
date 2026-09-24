// @ts-check
/**
 * The developer-mode panel: timeline, latency table, raw events, scenarios.
 *
 * Rendering rules that are not negotiable, because the whole point of the panel
 * is that its numbers can be trusted:
 *
 * - A measurement that does not exist renders as "—". It never renders as 0,
 *   and it is never quietly filled in from a neighbouring mark.
 * - Every latency after end-of-speech is shown with a leading "+", because it
 *   is an offset from the moment the user stopped talking, not a duration of
 *   the stage above it.
 * - Where a number's resolution is worse than a millisecond — the WebRTC level
 *   poll, the device's output latency — the panel says so next to the number.
 */

import { escHtml } from "../ui/dom.js";
import { classifyResponse, turnMetrics, VERDICT_COPY } from "./tracer.js";
import { SCENARIOS } from "./scenarios.js";

const TABS = [
  { id: "timeline", label: "Timeline" },
  { id: "latency", label: "Latency" },
  { id: "events", label: "Events" },
  { id: "scenarios", label: "Scenarios" },
];

/** @param {string} tag @param {Record<string, any>} [props] @param {any[]} [children] */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" || typeof child === "number" ? String(child) : child);
  }
  return node;
}

/** Milliseconds, or an em dash when the mark never happened. */
function ms(value, { sign = false } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const rounded = Math.round(value);
  return `${sign && rounded >= 0 ? "+" : ""}${rounded} ms`;
}

export class LabPanel {
  /**
   * @param {{tracer: any, onRun: (id: string) => void, onStop: () => void,
   *          onReset: () => void, onExport: () => void,
   *          onSaveBaseline: () => void, onClearBaseline: () => void,
   *          fixtureError: () => string}} deps
   */
  constructor(deps) {
    this.deps = deps;
    this.tracer = deps.tracer;
    this.active = "timeline";
    /** @type {Set<number>} Raw events expanded to show their JSON. */
    this.expanded = new Set();
    this.filter = "";
    this.running = false;
    /** @type {{label: string, rows: any[]} | null} */
    this.baseline = null;
    this._build();
  }

  _build() {
    this.logEl = el("div", { class: "lab-log", role: "log", "aria-live": "polite" });
    this.panels = {};
    for (const tab of TABS) this.panels[tab.id] = el("div", { class: "lab-panel", hidden: tab.id !== this.active });

    this.tabButtons = TABS.map((tab) => el("button", {
      class: "lab-tab",
      type: "button",
      role: "tab",
      "aria-selected": tab.id === this.active,
      text: tab.label,
      onclick: () => this.show(tab.id),
    }));

    this.transportEl = el("span", { class: "lab-transport", text: "—" });

    this.el = el("section", { class: "lab", "aria-label": "Developer mode" }, [
      el("header", { class: "lab-head" }, [
        el("span", { class: "lab-title", text: "Voice Lab" }),
        el("div", { class: "lab-tabs", role: "tablist" }, this.tabButtons),
        el("div", { class: "lab-head-right" }, [
          this.transportEl,
          el("button", { class: "lab-btn", type: "button", text: "Copy trace", onclick: () => this.deps.onExport() }),
          el("button", { class: "lab-btn", type: "button", text: "Clear", onclick: () => this.deps.onReset() }),
          el("button", {
            class: "lab-btn",
            type: "button",
            text: "Hide",
            onclick: (/** @type {any} */ e) => {
              const collapsed = this.el.classList.toggle("is-collapsed");
              document.body.classList.toggle("lab-collapsed", collapsed);
              e.target.textContent = collapsed ? "Show" : "Hide";
            },
          }),
        ]),
      ]),
      el("div", { class: "lab-body" }, Object.values(this.panels)),
    ]);
  }

  /** @param {string} id */
  show(id) {
    this.active = id;
    for (const [index, tab] of TABS.entries()) {
      this.panels[tab.id].hidden = tab.id !== id;
      this.tabButtons[index].setAttribute("aria-selected", String(tab.id === id));
    }
    this.render();
  }

  /** @param {string} line @param {string} [kind] */
  log(line, kind = "") {
    this.logEl.append(el("div", { class: kind ? `is-${kind}` : "", text: line }));
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  clearLog() {
    this.logEl.replaceChildren();
  }

  /** @param {boolean} running */
  setRunning(running) {
    this.running = running;
    this.render();
  }

  render() {
    this.transportEl.textContent = `·${this.tracer.transport}`;
    const panel = this.panels[this.active];
    if (!panel) return;
    const scrollTop = panel.scrollTop;
    panel.replaceChildren(...this[`_render_${this.active}`]());
    // Re-rendering on every event would otherwise yank the reader back to the
    // top mid-scroll.
    panel.scrollTop = scrollTop;
  }

  // ── timeline ─────────────────────────────────────────────────────────────

  _render_timeline() {
    if (this.tracer.turns.length === 0) {
      return [el("p", { class: "lab-empty", text: "No turns yet — start a session and say something" })];
    }
    const nodes = [];
    for (const turn of [...this.tracer.turns].reverse()) nodes.push(this._turnCard(turn));
    return nodes;
  }

  /** @param {any} turn */
  _turnCard(turn) {
    const m = turnMetrics(turn);
    const rows = [];
    // The bar spans the whole turn: the speech itself, plus the longest wait
    // after it. Scaling to the offsets alone pushed every late mark off the
    // right-hand edge.
    const offsets = [
      m.sttFirstPartial, m.sttFinal, m.responseCreated,
      m.firstText, m.firstAudio, m.audible, m.total,
    ].filter((value) => value !== null);
    const span = Math.max(1, (m.speechMs ?? 0) + Math.max(0, ...offsets));

    // The first two marks are absolute within the turn; everything after is an
    // offset from end of speech, which is when the user starts waiting.
    rows.push(this._ledgerRow("user speech start", "0 ms", "user", 0, span, turn.speechMs));
    rows.push(this._ledgerRow("user speech end", ms(m.speechMs), "user", m.speechMs ?? 0, span, turn.speechMs));

    const after = [
      ["STT first partial", m.sttFirstPartial, "user"],
      ["STT final", m.sttFinal, "user"],
      ["response created", m.responseCreated, "assistant"],
      ["assistant first text", m.firstText, "assistant"],
      [this.tracer.transport === "webrtc" ? "first audio event" : "first audio packet", m.firstAudio, "assistant"],
      ["audible playback", m.audible, "assistant"],
      ["tool call total", m.toolMs, "tool"],
      ["barge-in → silence", m.bargeInSilence, "tool"],
      ["response done", m.total, "assistant"],
    ];
    for (const [label, value, role] of after) {
      if (value === null && (label === "tool call total" || label === "barge-in → silence")) continue;
      rows.push(this._ledgerRow(label, ms(value, { sign: true }), role, (m.speechMs ?? 0) + (value ?? 0), span, turn.speechMs));
    }

    const responses = turn.responses.map((r) => this._responseRow(r, turn));

    return el("article", { class: "lab-turn" }, [
      el("div", { class: "lab-turn-head" }, [
        el("span", { class: "lab-eyebrow is-user", text: `TURN ${turn.number}` }),
        el("span", { class: "lab-said", text: turn.transcript || turn.partialTranscript || "" }),
      ]),
      el("dl", { class: "lab-ledger" }, rows.flat()),
      ...responses,
      m.audible !== null && this.tracer.outputLatencyMs > 0
        ? el("p", {
            class: "lab-note",
            style: "margin:8px 0 0",
            text: `Audible playback is the first sample leaving the playback node. The output device adds a further `
              + `${Math.round(this.tracer.outputLatencyMs)} ms before it reaches the speaker.`,
          })
        : null,
    ].filter(Boolean));
  }

  /** One ledger line: label, figure, and a tick on the shared bar. */
  _ledgerRow(label, figure, role, absolute, span, speechMs) {
    const missing = figure === null;
    const bar = el("div", { class: "lab-bar" }, [
      speechMs
        ? el("i", { class: `is-span is-${role}`, style: `left:0;width:${Math.max(1, (speechMs / span) * 100)}%` })
        : null,
      missing
        ? null
        : el("i", {
            class: `is-${role}`,
            style: `left:${Math.max(0, Math.min(99, (absolute / span) * 100))}%`,
          }),
    ].filter(Boolean));
    return [
      el("dt", { text: label }),
      el("dd", { class: missing ? "is-missing" : "", text: figure ?? "—" }),
      bar,
    ];
  }

  /** @param {any} response @param {any} turn */
  _responseRow(response, turn) {
    const verdict = classifyResponse(response);
    const copy = VERDICT_COPY[verdict] ?? { label: verdict, detail: "" };
    const evidence = [];
    if (response.supersededAt !== null) {
      evidence.push(`superseded by turn ${response.supersededBy?.number ?? "?"} at `
        + `+${Math.round(response.supersededAt - turn.zero)} ms`);
      evidence.push(response.hadAudioBefore(response.supersededAt)
        ? "audio had already started"
        : "nothing audible yet");
      evidence.push(`${response.textAfterSupersede} text + ${response.audioAfterSupersede} audio events after`);
    }
    if (response.status) evidence.push(`status ${response.status}`);

    return el("div", { class: "lab-response" }, [
      el("div", { class: "lab-turn-head" }, [
        el("span", { class: "lab-eyebrow is-assistant", text: `RESPONSE ${response.number}` }),
        el("span", { class: `lab-verdict is-${verdict}`, text: copy.label }),
        el("span", { class: "lab-verdict-why", text: copy.detail }),
      ]),
      response.text ? el("div", { class: "lab-said", text: response.text }) : null,
      evidence.length ? el("div", { class: "lab-verdict-why", text: evidence.join(" · ") }) : null,
      ...response.toolCalls.map((call) => el("div", { class: "lab-verdict-why" }, [
        el("span", { class: "lab-eyebrow is-tool", text: "TOOL " }),
        `${call.name} — ${call.durationMs === null ? "never returned" : `${Math.round(call.durationMs)} ms`}`,
      ])),
    ].filter(Boolean));
  }

  // ── latency ──────────────────────────────────────────────────────────────

  _render_latency() {
    const turns = this.tracer.turns;
    if (turns.length === 0) return [el("p", { class: "lab-empty", text: "Nothing measured yet" })];

    const columns = [
      ["speech", (m) => m.speechMs],
      ["STT final", (m) => m.sttFinal],
      ["first text", (m) => m.firstText],
      ["first audio", (m) => m.firstAudio],
      ["audible", (m) => m.audible],
      ["pkt→play", (m) => m.packetToPlayback],
      ["barge-in", (m) => m.bargeInSilence],
      ["tool", (m) => m.toolMs],
      ["total", (m) => m.total],
    ];
    const metrics = turns.map(turnMetrics);

    const body = turns.map((turn, index) => el("tr", {}, [
      el("td", { text: `turn ${turn.number}` }),
      ...columns.map(([, pick]) => {
        const value = pick(metrics[index]);
        return el("td", { class: value === null ? "is-missing" : "", text: ms(value) ?? "—" });
      }),
    ]));

    const medians = columns.map(([, pick]) => {
      const values = metrics.map(pick).filter((v) => v !== null && v !== undefined);
      if (values.length === 0) return null;
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    });

    return [
      el("p", { class: "lab-note" }, [
        "Every column except ",
        el("em", { text: "speech" }),
        ", ",
        el("em", { text: "pkt→play" }),
        ", ",
        el("em", { text: "barge-in" }),
        " and ",
        el("em", { text: "tool" }),
        " is measured from the end of the user's speech. A dash means the mark never arrived — "
        + "over WebRTC the page cannot see audio packets or the playback queue at all, so those columns stay empty.",
      ]),
      el("table", { class: "lab-table" }, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "" }), ...columns.map(([label]) => el("th", { text: label }))])]),
        el("tbody", {}, body),
        el("tfoot", {}, [el("tr", {}, [
          el("td", { text: `median of ${turns.length}` }),
          ...medians.map((value) => el("td", { class: value === null ? "is-missing" : "", text: ms(value) ?? "—" })),
        ])]),
      ]),
    ];
  }

  // ── raw events ───────────────────────────────────────────────────────────

  _render_events() {
    const filter = this.filter.trim().toLowerCase();
    const events = this.tracer.rawEvents.filter((e) => !filter || e.type.toLowerCase().includes(filter));
    const first = this.tracer.rawEvents[0]?.at ?? 0;

    const input = el("input", {
      class: "lab-filter",
      type: "search",
      placeholder: "Filter by event type — e.g. response., audio, transcription",
      value: this.filter,
      oninput: (/** @type {any} */ e) => {
        this.filter = e.target.value;
        this.render();
        this.panels.events.querySelector(".lab-filter")?.focus();
      },
    });

    const rows = [...events].reverse().slice(0, 400).map((entry) => {
      const open = this.expanded.has(entry.seq);
      const line = el("div", {
        class: `lab-event${entry.direction === "local" ? " is-local" : ""}`,
        onclick: () => {
          if (open) this.expanded.delete(entry.seq);
          else this.expanded.add(entry.seq);
          this.render();
        },
      }, [
        el("span", { class: "lab-event-at", text: `${Math.round(entry.at - first)}` }),
        el("span", { class: "lab-event-type", text: entry.type }),
      ]);
      if (!open) return line;
      return el("div", {}, [
        line,
        el("pre", { class: "lab-event-json", text: safeJson(entry.event) }),
      ]);
    });

    return [
      input,
      el("p", { class: "lab-note", text:
        "Everything the session saw, in arrival order, including event types this client has no behaviour for. "
        + "Lines marked in amber are the page's own marks (audio packets, playback edges), not server events." }),
      rows.length ? el("div", { class: "lab-events" }, rows) : el("p", { class: "lab-empty", text: "No matching events" }),
    ];
  }

  // ── scenarios ────────────────────────────────────────────────────────────

  _render_scenarios() {
    const fixtureError = this.deps.fixtureError();
    const cards = SCENARIOS.map((scenario) => el("article", { class: "lab-scenario" }, [
      el("h4", { text: scenario.title }),
      el("p", { text: scenario.purpose }),
      el("p", { class: "lab-scenario-aims", text: scenario.aims }),
      el("button", {
        class: "lab-btn is-primary",
        type: "button",
        text: "Run",
        disabled: this.running || Boolean(fixtureError),
        onclick: () => this.deps.onRun(scenario.id),
      }),
    ]));

    return [
      el("div", { class: "lab-cols" }, [
        el("div", { class: "lab-col" }, [
          fixtureError ? el("p", { class: "lab-note", text: fixtureError }) : null,
          el("p", { class: "lab-note", text:
            "Scenarios speak prerecorded audio through a synthetic microphone, so the same utterance arrives the "
            + "same way every run. Start a session first — a scenario drives a live conversation, it does not fake one." }),
          this._scriptedMicSwitch(),
          this.running
            ? el("button", { class: "lab-btn", type: "button", text: "Stop", onclick: () => this.deps.onStop() })
            : null,
          ...cards,
        ].filter(Boolean)),
        el("div", { class: "lab-col" }, [
          this.logEl,
          ...this._compare(),
        ]),
      ]),
    ];
  }

  /**
   * Choose between the real microphone and the scripted one.
   *
   * A session captures from one source for its whole life, so flipping this
   * takes effect on the next Restart rather than immediately. Saying that
   * plainly is better than silently doing nothing until the user reloads.
   */
  _scriptedMicSwitch() {
    const wanted = this.deps.scriptedMic();
    const live = this.deps.scriptedMicLive();
    const id = "lab-scripted-mic";
    return el("div", { class: "lab-scenario" }, [
      el("label", { for: id, style: "display:flex;gap:8px;align-items:center;cursor:pointer" }, [
        el("input", {
          id,
          type: "checkbox",
          checked: wanted,
          onchange: (/** @type {any} */ e) => this.deps.onScriptedMic(e.target.checked),
        }),
        el("span", { text: "Scripted microphone" }),
      ]),
      el("p", {
        class: "lab-note",
        style: "margin:8px 0 0",
        text: wanted === live
          ? (live
              ? "This session is capturing from the fixtures. The real microphone is not open."
              : "This session is capturing from the real microphone. Scenarios need the scripted one.")
          : "Takes effect on the next session — press Restart in Settings.",
      }),
    ]);
  }

  /**
   * What this build did with superseded turns, next to a saved run.
   *
   * There is no way for the page to know which server build it is talking to,
   * so it does not guess. You run a scenario against one build, save it, point
   * the demo at the other build and run it again. Both columns are measured;
   * neither is asserted.
   */
  _compare() {
    const rows = this._supersededRows();
    const header = el("div", { class: "lab-turn-head" }, [
      el("span", { class: "lab-eyebrow is-assistant", text: "SUPERSEDED TURNS" }),
      el("button", {
        class: "lab-btn",
        type: "button",
        text: "Save as baseline",
        disabled: rows.length === 0,
        onclick: () => this.deps.onSaveBaseline(),
      }),
      this.baseline
        ? el("button", { class: "lab-btn", type: "button", text: "Clear baseline", onclick: () => this.deps.onClearBaseline() })
        : null,
    ].filter(Boolean));

    if (rows.length === 0 && !this.baseline) {
      return [
        header,
        el("p", { class: "lab-note", text:
          "Nothing has been superseded yet. Run “Turn 2 while turn 1 is still generating” to open the window "
          + "where a response is replaced before any of it has been heard." }),
      ];
    }

    const verdictCell = (verdict) => el("td", {
      class: verdict === "stale-accepted" ? "is-bad" : verdict === "dropped" ? "is-good" : "is-idle",
      text: verdict ? (VERDICT_COPY[verdict]?.label ?? verdict) : "—",
    });

    const table = el("table", {}, [
      el("thead", {}, [el("tr", {}, [
        el("th", { text: "superseded response" }),
        el("th", { text: "audible when replaced" }),
        el("th", { text: "output after" }),
        el("th", { text: "this run" }),
        this.baseline ? el("th", { text: this.baseline.label }) : null,
      ].filter(Boolean))]),
      el("tbody", {}, rows.map((row, index) => el("tr", {}, [
        el("td", { text: `response ${row.number} → turn ${row.supersededBy}` }),
        el("td", { class: row.hadAudio ? "is-idle" : "", text: row.hadAudio ? "yes" : "no" }),
        el("td", { text: `${row.textAfter} text, ${row.audioAfter} audio` }),
        verdictCell(row.verdict),
        this.baseline ? verdictCell(this.baseline.rows[index]?.verdict ?? "") : null,
      ].filter(Boolean)))),
    ]);

    return [
      header,
      el("p", { class: "lab-note", text:
        "“Audible when replaced” is how this panel infers whether the pipeline had committed the response. The page "
        + "cannot read the server's commit flag; it can only tell whether sound had started. Issue #308 draws its "
        + "line in the same place, so the proxy is a fair one — but it is a proxy." }),
      el("div", { class: "lab-compare" }, [table]),
    ];
  }

  _supersededRows() {
    return [...this.tracer.responsesById.values()]
      .filter((r) => r.supersededAt !== null)
      .sort((a, b) => a.number - b.number)
      .map((r) => ({
        number: r.number,
        supersededBy: r.supersededBy?.number ?? null,
        hadAudio: r.hadAudioBefore(r.supersededAt),
        textAfter: r.textAfterSupersede,
        audioAfter: r.audioAfterSupersede,
        verdict: classifyResponse(r),
      }));
  }

  /** @param {{label: string, rows: any[]} | null} baseline */
  setBaseline(baseline) {
    this.baseline = baseline;
    this.render();
  }

  currentSupersededRows() {
    return this._supersededRows();
  }
}

/** @param {any} value */
function safeJson(value) {
  try {
    return JSON.stringify(value, (_key, inner) => {
      if (typeof inner === "string" && inner.length > 600) return `${inner.slice(0, 600)}… (${inner.length} chars)`;
      return inner;
    }, 2);
  } catch {
    return String(value);
  }
}

export { escHtml };
