// @ts-check
/**
 * Turn model for the realtime voice lab.
 *
 * The tracer is a pure consumer: it takes the client's read-only event tap and
 * builds one record per user turn and one per assistant response. It never
 * calls back into the session, so a traced conversation is the same
 * conversation.
 *
 * Two things make this more than a log viewer:
 *
 * 1. **Every latency is measured from end of speech**, not from the previous
 *    event. "STT took 116 ms" is only meaningful next to the moment the user
 *    stopped talking, because that is when the user starts waiting.
 *
 * 2. **Responses are classified against the turn that owns them.** When a user
 *    starts a new turn while an older response is still open, the older
 *    response is *superseded*. What the pipeline is supposed to do with it
 *    depends on whether it had already committed output — which is the whole
 *    substance of issue #308. See `classifyResponse`.
 */

/**
 * @typedef {"open" | "committed" | "dropped" | "stale-accepted" | "completed" | "failed"} ResponseVerdict
 * @typedef {Object} Mark
 * @property {string} name
 * @property {number} at      Milliseconds on the page clock (`performance.now`).
 * @property {string} [note]
 */

/** Assistant transcript deltas, under both the GA and the older event names. */
const ASSISTANT_TEXT_DELTA = new Set([
  "response.audio_transcript.delta",
  "response.output_audio_transcript.delta",
  "response.text.delta",
  "response.output_text.delta",
]);

/** Wire audio for a response, when the transport carries it as events. */
const ASSISTANT_AUDIO_DELTA = new Set([
  "response.audio.delta",
  "response.output_audio.delta",
]);

const TERMINAL_STATUSES = new Set(["completed", "cancelled", "incomplete", "failed"]);

/** One user turn: everything from speech start to the death of its responses. */
class Turn {
  /** @param {string} itemId @param {number} at @param {number} number */
  constructor(itemId, at, number) {
    this.number = number;
    this.itemId = itemId;
    this.startedAt = at;
    /** @type {number | null} */
    this.endedAt = null;
    /** @type {number | null} */
    this.sttFinalAt = null;
    /** @type {number | null} */
    this.sttFirstPartialAt = null;
    this.transcript = "";
    this.partialTranscript = "";
    /** @type {Response[]} */
    this.responses = [];
    /** @type {Mark[]} */
    this.marks = [];
    /** Set when this turn's start superseded something already in flight. */
    this.supersededOthers = false;
    /** When a newer turn replaced this one while it was still unanswered. */
    /** @type {number | null} */
    this.supersededAt = null;
    /** @type {Turn | null} */
    this.supersededBy = null;
    /** @type {number | null} Playback silence after this turn barged in. */
    this.bargeInSilencedAt = null;
    this.bargedIn = false;
  }

  /** @param {string} name @param {number} at @param {string} [note] */
  mark(name, at, note) {
    this.marks.push({ name, at, ...(note ? { note } : {}) });
  }

  /** The clock everything else is measured against: when the user stopped. */
  get zero() {
    return this.endedAt ?? this.startedAt;
  }

  get speechMs() {
    return this.endedAt === null ? null : this.endedAt - this.startedAt;
  }

  /**
   * Whether this turn is still waiting on the pipeline.
   *
   * A turn the user has finished speaking is owed an answer until one arrives
   * and finishes. Until then a newer turn replaces it, which is the state
   * issue #308 is about.
   */
  get awaitingResponse() {
    if (this.endedAt === null) return false;
    if (this.responses.length === 0) return true;
    return this.responses.some((response) => response.open);
  }
}

/** One assistant response, owned by the turn that was current when it began. */
class Response {
  /** @param {string} id @param {Turn | null} turn @param {number} at @param {number} number */
  constructor(id, turn, at, number) {
    this.number = number;
    this.id = id;
    this.turn = turn;
    this.createdAt = at;
    /** @type {number | null} */
    this.firstTextAt = null;
    /** @type {number | null} */
    this.firstAudioEventAt = null;
    /** @type {number | null} Audio decoded on the main thread (WebSocket). */
    this.firstAudioPacketAt = null;
    /** @type {number | null} First sample actually leaving the playback node. */
    this.audibleAt = null;
    /** @type {number | null} */
    this.doneAt = null;
    /** @type {string} */
    this.status = "";
    this.text = "";
    this.audioPackets = 0;
    this.audioMs = 0;
    /** @type {number | null} When a newer user turn arrived while still open. */
    this.supersededAt = null;
    /** @type {Turn | null} */
    this.supersededBy = null;
    /** Output events seen after `supersededAt` — the evidence for the verdict. */
    this.textAfterSupersede = 0;
    this.audioAfterSupersede = 0;
    /** @type {ToolCall[]} */
    this.toolCalls = [];
    /** @type {Mark[]} */
    this.marks = [];
  }

  /** @param {string} name @param {number} at @param {string} [note] */
  mark(name, at, note) {
    this.marks.push({ name, at, ...(note ? { note } : {}) });
  }

  get open() {
    return this.doneAt === null;
  }

  /**
   * Whether this response had put sound in the user's ears before `at`.
   *
   * This is the browser's best proxy for the pipeline's internal "committed"
   * flag. The page cannot read that flag; what it can say is whether audio for
   * this response had already started. Issue #308 draws its line in the same
   * place — output the user has begun hearing must be allowed to finish — so
   * the proxy is a fair one, but it is a proxy, and the panel says so.
   *
   * @param {number} at
   */
  hadAudioBefore(at) {
    const first = this.firstAudioPacketAt ?? this.firstAudioEventAt ?? this.audibleAt;
    return first !== null && first <= at;
  }
}

/** One tool call inside a response. */
class ToolCall {
  /** @param {string} callId @param {string} name @param {number} at */
  constructor(callId, name, at) {
    this.callId = callId;
    this.name = name;
    this.requestedAt = at;
    /** @type {number | null} */
    this.settledAt = null;
    this.arguments = "";
  }

  get durationMs() {
    return this.settledAt === null ? null : this.settledAt - this.requestedAt;
  }
}

/**
 * Decide what the pipeline did with a response, in the vocabulary of #308.
 *
 * - `completed` — ran to the end without anything newer arriving. The ordinary case.
 * - `committed` — a newer turn arrived, but this response had already started
 *   speaking. #308 says work the user is already hearing finishes; letting it
 *   run is correct.
 * - `dropped` — a newer turn arrived while this response was still silent, and
 *   it produced nothing afterwards. This is the fixed behaviour.
 * - `stale-accepted` — a newer turn arrived while this response was still
 *   silent, and it *kept producing output anyway*. This is the bug: the user
 *   hears an answer to a question they have already moved on from.
 *
 * @param {Response} response
 * @returns {ResponseVerdict}
 */
export function classifyResponse(response) {
  if (response.supersededAt === null) {
    if (response.open) return "open";
    if (response.status === "failed") return "failed";
    return "completed";
  }
  if (response.hadAudioBefore(response.supersededAt)) return "committed";
  const outputAfter = response.textAfterSupersede + response.audioAfterSupersede;
  return outputAfter > 0 ? "stale-accepted" : "dropped";
}

/**
 * What became of a user turn.
 *
 * This is the level issue #308 actually argues about. A response object may
 * never exist for a turn that was replaced before the pipeline produced
 * anything, so asking "what happened to this response" cannot see the case at
 * all; asking "what happened to this turn" can.
 *
 * - `answered` — the turn got its reply. The ordinary case.
 * - `unanswered` — the turn ended and nothing has come back yet.
 * - `replaced-dropped` — a newer turn arrived while this one was still waiting,
 *   and the pipeline produced nothing for it afterwards. Correct.
 * - `replaced-leaked` — a newer turn arrived while this one was still waiting,
 *   and the pipeline answered it anyway. The user hears a reply to a question
 *   they had already moved on from.
 *
 * @param {Turn} turn
 * @returns {"answered" | "unanswered" | "replaced-dropped" | "replaced-leaked"}
 */
export function classifyTurn(turn) {
  if (turn.supersededAt === null) {
    return turn.responses.some((response) => !response.open) ? "answered" : "unanswered";
  }
  const leaked = turn.responses.some((response) => {
    if (response.hadAudioBefore(turn.supersededAt)) return false;
    return response.text.length > 0 || response.audioPackets > 0 || response.firstAudioEventAt !== null;
  });
  return leaked ? "replaced-leaked" : "replaced-dropped";
}

/** Human-readable one-liners for each verdict, shown in the panel. */
export const VERDICT_COPY = {
  "open": { label: "open", detail: "Still generating." },
  "completed": { label: "completed", detail: "Ran to the end; nothing newer arrived." },
  "committed": { label: "committed", detail: "Superseded, but already audible — #308 lets it finish." },
  "dropped": { label: "dropped", detail: "Superseded while silent, and produced nothing after. Correct." },
  "stale-accepted": { label: "stale accepted", detail: "Superseded while silent, then kept emitting. The #308 bug." },
  "failed": { label: "failed", detail: "The server reported an error." },
};

export class Tracer extends EventTarget {
  constructor() {
    super();
    /** @type {Turn[]} */
    this.turns = [];
    /** @type {Map<string, Response>} */
    this.responsesById = new Map();
    /** @type {Array<{seq: number, type: string, at: number, direction: string, event: any}>} */
    this.rawEvents = [];
    /** @type {Turn | null} */
    this.currentTurn = null;
    this.transport = "websocket";
    this.outputLatencyMs = 0;
    this._seq = 0;
    this._turnNumber = 0;
    this._responseNumber = 0;
    this._rawLimit = 2000;
    /** Maps the AudioContext clock to the page clock for worklet marks. */
    this._contextClockOffsetMs = null;
  }

  reset() {
    this.turns = [];
    this.responsesById.clear();
    this.rawEvents = [];
    this.currentTurn = null;
    this._seq = 0;
    this._turnNumber = 0;
    this._responseNumber = 0;
    this._changed();
  }

  _changed() {
    this.dispatchEvent(new Event("change"));
  }

  /** @param {string} type @param {number} at @param {any} event @param {string} [direction] */
  _recordRaw(type, at, event, direction = "server") {
    this._seq += 1;
    this.rawEvents.push({ seq: this._seq, type, at, direction, event });
    if (this.rawEvents.length > this._rawLimit) this.rawEvents.splice(0, this.rawEvents.length - this._rawLimit);
  }

  /**
   * Pick the turn a freshly created response belongs to.
   *
   * The pipeline does not tell the page which user turn a response answers —
   * assistant output carries no `previous_item_id` back to the input item — so
   * this is inference, and the order it tries things in matters:
   *
   * 1. The oldest finished turn that is still owed an answer and was never
   *    replaced. A response is far more likely to be the legitimate answer to
   *    a live question than stale work.
   * 2. Failing that, the oldest finished turn still owed an answer, which by
   *    then must be one that was replaced — so this response is stale work.
   * 3. Failing that, the newest finished turn, which keeps a tool-call
   *    follow-up attached to the turn that caused it.
   *
   * A turn the user has not finished speaking is never a candidate: the
   * pipeline cannot be answering a question that is still being asked.
   *
   * Order 1-before-2 makes the inference deliberately conservative. When two
   * finished turns could both own a response, it hands it to the live one, so
   * the panel under-reports stale output rather than inventing it. A trace
   * that does report stale output is therefore worth believing; a trace that
   * reports none is weaker evidence.
   */
  _claimTurnForResponse() {
    const owed = this.turns.filter((turn) => turn.endedAt !== null && turn.responses.length === 0);
    const live = owed.find((turn) => turn.supersededAt === null);
    if (live) return live;
    if (owed.length > 0) return owed[0];
    for (let i = this.turns.length - 1; i >= 0; i -= 1) {
      if (this.turns[i].endedAt !== null) return this.turns[i];
    }
    return this.currentTurn;
  }

  /** Responses still open that belong to a turn older than `turn`. */
  _openOlderThan(turn) {
    const result = [];
    for (const response of this.responsesById.values()) {
      if (!response.open) continue;
      if (response.turn === null || response.turn === turn) continue;
      if (response.turn.number < turn.number) result.push(response);
    }
    return result;
  }

  /**
   * Feed one tapped protocol event.
   * @param {{event: any, type: string, at: number, transport: string}} detail
   */
  onProtocolEvent(detail) {
    const { event, type, at } = detail;
    this.transport = detail.transport;
    this._recordRaw(type, at, event);

    switch (type) {
      case "input_audio_buffer.speech_started": {
        const itemId = typeof event.item_id === "string" ? event.item_id : `item_${this._seq}`;
        this._turnNumber += 1;
        const turn = new Turn(itemId, at, this._turnNumber);
        turn.mark("user speech start", at);
        // A new turn supersedes anything older still in flight. That has to be
        // tracked on the *turn*, not only on open responses: the pipeline emits
        // response.created when the first output is ready, so a turn can be
        // replaced long before any response object exists for it.
        const superseded = this._openOlderThan(turn);
        for (const response of superseded) {
          response.supersededAt = at;
          response.supersededBy = turn;
          response.mark("superseded by turn " + turn.number, at);
          if (response.hadAudioBefore(at)) turn.bargedIn = true;
        }
        for (const older of this.turns) {
          if (older === turn || !older.awaitingResponse || older.supersededAt !== null) continue;
          older.supersededAt = at;
          older.supersededBy = turn;
          older.mark(`superseded by turn ${turn.number}`, at);
          turn.supersededOthers = true;
        }
        if (superseded.length > 0) turn.supersededOthers = true;
        this.turns.push(turn);
        this.currentTurn = turn;
        break;
      }
      case "input_audio_buffer.speech_stopped": {
        const turn = this.currentTurn;
        if (turn && turn.endedAt === null) {
          turn.endedAt = at;
          turn.mark("user speech end", at);
        }
        break;
      }
      case "conversation.item.input_audio_transcription.delta": {
        const turn = this._turnForItem(event.item_id);
        if (turn && typeof event.delta === "string" && event.delta) {
          if (turn.sttFirstPartialAt === null) {
            turn.sttFirstPartialAt = at;
            turn.mark("STT first partial", at);
          }
          turn.partialTranscript += event.delta;
        }
        break;
      }
      case "conversation.item.input_audio_transcription.completed": {
        const turn = this._turnForItem(event.item_id);
        if (turn && turn.sttFinalAt === null) {
          turn.sttFinalAt = at;
          turn.transcript = typeof event.transcript === "string" ? event.transcript : turn.partialTranscript;
          turn.mark("STT final", at);
        }
        break;
      }
      case "conversation.item.input_audio_transcription.failed": {
        const turn = this._turnForItem(event.item_id);
        if (turn) turn.mark("STT failed", at, event.error?.message ?? "");
        break;
      }
      case "response.created": {
        const id = event.response?.id ?? `resp_${this._seq}`;
        this._responseNumber += 1;
        const owner = this._claimTurnForResponse();
        const response = new Response(id, owner, at, this._responseNumber);
        // A response for a turn that was already replaced inherits the moment
        // it was replaced, so output arriving now counts as arriving after.
        if (owner && owner.supersededAt !== null) {
          response.supersededAt = owner.supersededAt;
          response.supersededBy = owner.supersededBy;
          response.mark(`superseded by turn ${owner.supersededBy?.number ?? "?"}`, owner.supersededAt);
        }
        this.responsesById.set(id, response);
        owner?.responses.push(response);
        response.mark("response created", at);
        break;
      }
      case "response.done": {
        const id = event.response?.id ?? "";
        const response = this.responsesById.get(id) ?? this._latestOpenResponse();
        if (response) {
          response.doneAt = at;
          response.status = event.response?.status ?? "completed";
          const reason = event.response?.status_details?.reason;
          response.mark("response done", at, reason ? `${response.status}: ${reason}` : response.status);
        }
        break;
      }
      case "response.function_call_arguments.done": {
        const response = this.responsesById.get(event.response_id ?? "") ?? this._latestOpenResponse();
        if (response) {
          const call = new ToolCall(event.call_id ?? "", event.name ?? "tool", at);
          call.arguments = typeof event.arguments === "string" ? event.arguments : "";
          response.toolCalls.push(call);
          response.mark(`tool call ${call.name}`, at);
        }
        break;
      }
      case "conversation.item.created": {
        // The tool result going back in closes the call that was waiting on it.
        if (event.item?.type === "function_call_output") {
          const callId = event.item?.call_id ?? "";
          const call = this._findToolCall(callId);
          if (call && call.settledAt === null) call.settledAt = at;
        }
        break;
      }
      case "error": {
        this.currentTurn?.mark("error", at, event.error?.message ?? "");
        break;
      }
      default:
        break;
    }

    if (ASSISTANT_TEXT_DELTA.has(type)) this._onAssistantText(event, at);
    if (ASSISTANT_AUDIO_DELTA.has(type)) this._onAssistantAudioEvent(event, at);
    if (TERMINAL_STATUSES.has(event?.response?.status) && type !== "response.done") {
      // Nothing to do; kept explicit so the reader knows terminal status is
      // only trusted from response.done.
    }
    this._changed();
  }

  /** @param {any} event @param {number} at */
  _onAssistantText(event, at) {
    const response = this.responsesById.get(event.response_id ?? "") ?? this._latestOpenResponse();
    if (!response) return;
    const delta = typeof event.delta === "string" ? event.delta : "";
    if (!delta) return;
    if (response.firstTextAt === null) {
      response.firstTextAt = at;
      response.mark("assistant first text", at);
    }
    response.text += delta;
    if (response.supersededAt !== null && at > response.supersededAt) response.textAfterSupersede += 1;
  }

  /** @param {any} event @param {number} at */
  _onAssistantAudioEvent(event, at) {
    const response = this.responsesById.get(event.response_id ?? "") ?? this._latestOpenResponse();
    if (!response) return;
    if (response.firstAudioEventAt === null) {
      response.firstAudioEventAt = at;
      response.mark("first audio event", at);
    }
    if (response.supersededAt !== null && at > response.supersededAt) response.audioAfterSupersede += 1;
  }

  /**
   * Audio decoded on the main thread and handed to the playback worklet.
   * WebSocket only — over WebRTC the media never passes through the page.
   * @param {{responseId: string, samples: number, durationMs: number, at: number}} detail
   */
  onAudioPacket(detail) {
    const response = this.responsesById.get(detail.responseId) ?? this._latestOpenResponse();
    this._recordRaw("· audio packet", detail.at, detail, "local");
    if (!response) return;
    response.audioPackets += 1;
    response.audioMs += detail.durationMs;
    if (response.firstAudioPacketAt === null) {
      response.firstAudioPacketAt = detail.at;
      response.mark("first audio packet", detail.at);
    }
    if (response.supersededAt !== null && detail.at > response.supersededAt) response.audioAfterSupersede += 1;
    this._changed();
  }

  /**
   * A mark from the playback worklet, or the WebRTC level poll.
   * @param {{kind: string, at: number, contextTime?: number, outputLatency?: number, resolutionMs?: number}} detail
   * @param {number} [contextNowMs] `AudioContext.currentTime * 1000` sampled on arrival.
   */
  onPlaybackMark(detail, contextNowMs) {
    if (typeof detail.outputLatency === "number") this.outputLatencyMs = detail.outputLatency * 1000;
    const at = this._playbackTime(detail, contextNowMs);
    if (detail.kind === "playback-started" || detail.kind === "rtc-audible") {
      const response = this._latestOpenResponse() ?? this._latestResponse();
      this._recordRaw(`· ${detail.kind}`, at, detail, "local");
      if (response && response.audibleAt === null) {
        response.audibleAt = at;
        response.mark("audible playback", at, detail.kind === "rtc-audible" ? "±50 ms (level poll)" : "");
      }
    } else if (detail.kind === "playback-stopped" || detail.kind === "rtc-silent") {
      this._recordRaw(`· ${detail.kind}`, at, detail, "local");
      // If the current turn barged in, this is the moment the room went quiet.
      const turn = this.currentTurn;
      if (turn && turn.bargedIn && turn.bargeInSilencedAt === null) {
        turn.bargeInSilencedAt = at;
        turn.mark("assistant silent", at);
      }
    } else if (detail.kind === "underrun") {
      this._recordRaw("· underrun", at, detail, "local");
    }
    this._changed();
  }

  /** @param {{at: number}} detail */
  onPlaybackCleared(detail) {
    this._recordRaw("· playback cleared", detail.at, detail, "local");
    this.currentTurn?.mark("playback cleared", detail.at);
    this._changed();
  }

  /**
   * Convert a worklet mark's AudioContext timestamp onto the page clock.
   *
   * The worklet stamps the render quantum; the message then queues to the main
   * thread, so arrival time is late by an unknown amount. Using the context
   * clock removes that jitter. If the caller could not sample the context
   * clock, fall back to arrival time and accept the error.
   */
  _playbackTime(detail, contextNowMs) {
    if (typeof detail.contextTime !== "number" || typeof contextNowMs !== "number") return detail.at;
    const contextTimeMs = detail.contextTime * 1000;
    return Math.round((detail.at - (contextNowMs - contextTimeMs)) * 1000) / 1000;
  }

  /** @param {string} itemId */
  _turnForItem(itemId) {
    if (typeof itemId === "string" && itemId) {
      for (let i = this.turns.length - 1; i >= 0; i -= 1) {
        if (this.turns[i].itemId === itemId) return this.turns[i];
      }
    }
    return this.currentTurn;
  }

  _latestOpenResponse() {
    let best = null;
    for (const response of this.responsesById.values()) {
      if (response.open && (best === null || response.createdAt > best.createdAt)) best = response;
    }
    return best;
  }

  _latestResponse() {
    let best = null;
    for (const response of this.responsesById.values()) {
      if (best === null || response.createdAt > best.createdAt) best = response;
    }
    return best;
  }

  /** @param {string} callId */
  _findToolCall(callId) {
    for (const response of this.responsesById.values()) {
      for (const call of response.toolCalls) {
        if (call.callId === callId) return call;
      }
    }
    return null;
  }

  /** Tool executor bookends, reported by the page rather than the protocol. */
  onToolResult(callId, at) {
    const call = this._findToolCall(callId);
    if (call && call.settledAt === null) {
      call.settledAt = at;
      this._changed();
    }
  }
}

/**
 * Reduce one turn to the numbers the panel shows.
 *
 * Every field is milliseconds from end of speech, or `null` when the transport
 * or the run simply did not produce that mark. Null is shown as "—"; it is
 * never rendered as a zero.
 *
 * @param {Turn} turn
 */
export function turnMetrics(turn) {
  const zero = turn.zero;
  const responses = turn.responses;
  const primary = responses.find((r) => classifyResponse(r) !== "dropped") ?? responses[0] ?? null;
  const since = (/** @type {number | null} */ value) => (value === null ? null : value - zero);

  const firstAudio = primary
    ? (primary.firstAudioPacketAt ?? primary.firstAudioEventAt)
    : null;

  return {
    speechMs: turn.speechMs,
    sttFirstPartial: since(turn.sttFirstPartialAt),
    sttFinal: since(turn.sttFinalAt),
    responseCreated: since(primary ? primary.createdAt : null),
    firstText: since(primary ? primary.firstTextAt : null),
    firstAudio: since(firstAudio),
    audible: since(primary ? primary.audibleAt : null),
    packetToPlayback:
      primary && primary.audibleAt !== null && firstAudio !== null
        ? primary.audibleAt - firstAudio
        : null,
    total: since(primary ? primary.doneAt : null),
    bargeInSilence:
      turn.bargedIn && turn.bargeInSilencedAt !== null ? turn.bargeInSilencedAt - turn.startedAt : null,
    toolMs: primary
      ? primary.toolCalls.reduce((sum, call) => sum + (call.durationMs ?? 0), 0) || null
      : null,
  };
}

export { Turn, Response, ToolCall };
