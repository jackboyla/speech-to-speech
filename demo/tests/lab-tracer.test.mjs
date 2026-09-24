// Tracer tests. These drive the tracer with hand-built event streams so the
// turn-ordering verdicts are pinned down without a backend, a microphone or a
// browser. The streams mirror what the two server builds actually emit; the
// fixtures in demo/lab/fixtures drive the same paths against a real pipeline.

import test from "node:test";
import assert from "node:assert/strict";

import { Tracer, classifyResponse, turnMetrics } from "../lab/tracer.js";

/** Feed a protocol event at a given page-clock time. */
function feed(tracer, at, type, event = {}) {
  tracer.onProtocolEvent({ event: { type, ...event }, type, at, transport: "websocket" });
}

function speechStart(tracer, at, itemId) {
  feed(tracer, at, "input_audio_buffer.speech_started", { item_id: itemId });
}

function speechStop(tracer, at, itemId) {
  feed(tracer, at, "input_audio_buffer.speech_stopped", { item_id: itemId });
}

function responseCreated(tracer, at, id) {
  feed(tracer, at, "response.created", { response: { id } });
}

function textDelta(tracer, at, id, delta = "hi") {
  feed(tracer, at, "response.audio_transcript.delta", { response_id: id, delta });
}

function responseDone(tracer, at, id, status = "completed") {
  feed(tracer, at, "response.done", { response: { id, status } });
}

test("a normal turn measures every stage from end of speech", () => {
  const tracer = new Tracer();
  speechStart(tracer, 1000, "item_1");
  speechStop(tracer, 2843, "item_1");
  feed(tracer, 2959, "conversation.item.input_audio_transcription.completed", {
    item_id: "item_1", transcript: "what is the capital of france",
  });
  responseCreated(tracer, 3000, "resp_1");
  textDelta(tracer, 3114, "resp_1", "Paris");
  tracer.onAudioPacket({ responseId: "resp_1", samples: 480, durationMs: 20, at: 3261 });
  tracer.onPlaybackMark({ kind: "playback-started", at: 3294, contextTime: 3.294 }, 3294);
  responseDone(tracer, 4500, "resp_1");

  const turn = tracer.turns[0];
  const m = turnMetrics(turn);
  assert.equal(m.speechMs, 1843);
  assert.equal(m.sttFinal, 116);
  assert.equal(m.firstText, 271);
  assert.equal(m.firstAudio, 418);
  assert.equal(m.audible, 451);
  assert.equal(m.packetToPlayback, 33);
  assert.equal(m.total, 1657);
  assert.equal(classifyResponse(turn.responses[0]), "completed");
});

test("a metric with no mark stays null rather than becoming zero", () => {
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 500, "item_1");
  responseCreated(tracer, 600, "resp_1");
  const m = turnMetrics(tracer.turns[0]);
  assert.equal(m.sttFinal, null);
  assert.equal(m.firstAudio, null);
  assert.equal(m.audible, null);
  assert.equal(m.total, null);
});

test("a response still silent when the next turn starts, then emitting, is stale-accepted", () => {
  // This is issue #308 on an unfixed server: turn 1's generation is still
  // running with nothing audible when turn 2 begins, and it goes on to speak.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  responseCreated(tracer, 900, "resp_1");
  speechStart(tracer, 1200, "item_2");           // turn 2 supersedes resp_1
  textDelta(tracer, 1500, "resp_1", "stale answer");
  tracer.onAudioPacket({ responseId: "resp_1", samples: 480, durationMs: 20, at: 1600 });
  responseDone(tracer, 2000, "resp_1");

  const stale = tracer.responsesById.get("resp_1");
  assert.equal(stale.supersededAt, 1200);
  assert.equal(stale.supersededBy.number, 2);
  assert.equal(stale.textAfterSupersede, 1);
  assert.equal(stale.audioAfterSupersede, 1);
  assert.equal(classifyResponse(stale), "stale-accepted");
});

test("a response silenced by the next turn and producing nothing after is dropped", () => {
  // The same stream against a server that enforces global turn order.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  responseCreated(tracer, 900, "resp_1");
  speechStart(tracer, 1200, "item_2");
  responseDone(tracer, 1250, "resp_1", "cancelled");

  const dropped = tracer.responsesById.get("resp_1");
  assert.equal(dropped.textAfterSupersede, 0);
  assert.equal(dropped.audioAfterSupersede, 0);
  assert.equal(classifyResponse(dropped), "dropped");
});

test("a response already audible when the next turn starts is committed, not a bug", () => {
  // #308 explicitly allows committed work to finish, so continuing to emit
  // here must not be reported as the bug.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  responseCreated(tracer, 900, "resp_1");
  tracer.onAudioPacket({ responseId: "resp_1", samples: 480, durationMs: 20, at: 1000 });
  speechStart(tracer, 1200, "item_2");           // barge-in, audio already flowing
  textDelta(tracer, 1300, "resp_1", "tail");
  responseDone(tracer, 1400, "resp_1");

  const committed = tracer.responsesById.get("resp_1");
  assert.equal(classifyResponse(committed), "committed");
  assert.equal(tracer.turns[1].bargedIn, true);
});

test("barge-in latency runs from the interrupting speech to real silence", () => {
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  responseCreated(tracer, 900, "resp_1");
  tracer.onAudioPacket({ responseId: "resp_1", samples: 480, durationMs: 20, at: 1000 });
  speechStart(tracer, 2000, "item_2");
  tracer.onPlaybackCleared({ at: 2005 });
  tracer.onPlaybackMark({ kind: "playback-stopped", at: 2042, contextTime: 2.042 }, 2042);

  const m = turnMetrics(tracer.turns[1]);
  assert.equal(m.bargeInSilence, 42);
});

test("tool-call duration spans the request and the result going back", () => {
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 500, "item_1");
  responseCreated(tracer, 600, "resp_1");
  feed(tracer, 700, "response.function_call_arguments.done", {
    response_id: "resp_1", call_id: "call_1", name: "web_search", arguments: "{}",
  });
  feed(tracer, 1450, "conversation.item.created", {
    item: { type: "function_call_output", call_id: "call_1" },
  });
  const call = tracer.responsesById.get("resp_1").toolCalls[0];
  assert.equal(call.name, "web_search");
  assert.equal(call.durationMs, 750);
});

test("repeated short fragments each open their own turn", () => {
  const tracer = new Tracer();
  for (let i = 0; i < 3; i += 1) {
    speechStart(tracer, i * 1000, `item_${i}`);
    speechStop(tracer, i * 1000 + 300, `item_${i}`);
  }
  assert.equal(tracer.turns.length, 3);
  assert.deepEqual(tracer.turns.map((t) => t.number), [1, 2, 3]);
});

test("worklet marks are placed by the audio clock, not by message arrival", () => {
  // The worklet stamps the render quantum; the message reaches the main thread
  // later. Using arrival time would inflate audible latency by that queueing.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 100, "item_1");
  responseCreated(tracer, 200, "resp_1");
  // Started at context time 5.000 s, but the message only arrived 18 ms late.
  tracer.onPlaybackMark({ kind: "playback-started", at: 5018, contextTime: 5.0 }, 5018);
  assert.equal(tracer.responsesById.get("resp_1").audibleAt, 5000);
});

test("transcription is attributed by item id, not by arrival order", () => {
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 400, "item_1");
  speechStart(tracer, 1000, "item_2");
  speechStop(tracer, 1400, "item_2");
  // Turn 1's transcript lands after turn 2 has already started.
  feed(tracer, 1500, "conversation.item.input_audio_transcription.completed", {
    item_id: "item_1", transcript: "first",
  });
  assert.equal(tracer.turns[0].transcript, "first");
  assert.equal(tracer.turns[1].transcript, "");
});

test("a response created after its turn was replaced is attributed to that turn", () => {
  // The pipeline emits response.created when the first output is ready, not
  // when generation starts. A turn can therefore be replaced before any
  // response object for it exists, and the response that eventually arrives
  // still belongs to the older turn — not to whichever turn is current.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  speechStart(tracer, 2000, "item_2");          // turn 1 replaced while unanswered
  responseCreated(tracer, 3000, "resp_1");      // turn 1's answer, while turn 2 is still being spoken
  textDelta(tracer, 3010, "resp_1", "Paris");
  responseDone(tracer, 3200, "resp_1");

  const stale = tracer.responsesById.get("resp_1");
  assert.equal(stale.turn.number, 1, "response belongs to turn 1, not the current turn");
  assert.equal(stale.supersededAt, 2000);
  assert.equal(classifyResponse(stale), "stale-accepted");
  assert.equal(tracer.turns[0].supersededBy.number, 2);
});

test("a turn replaced while unanswered and never answered counts as dropped", () => {
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 800, "item_1");
  speechStart(tracer, 2000, "item_2");
  speechStop(tracer, 2800, "item_2");
  responseCreated(tracer, 3000, "resp_2");      // only turn 2 is answered
  textDelta(tracer, 3010, "resp_2", "Eight");
  responseDone(tracer, 3200, "resp_2");

  // Turn 1 was superseded and never produced anything.
  assert.equal(tracer.turns[0].supersededBy.number, 2);
  assert.equal(tracer.turns[0].responses.length, 0);
  // The only response goes to the turn still legitimately owed one, so a
  // server that correctly drops turn 1 is never reported as leaking output.
  assert.equal(tracer.responsesById.get("resp_2").turn.number, 2);
  assert.equal(classifyResponse(tracer.responsesById.get("resp_2")), "completed");
});

test("a second response for the same turn attaches to that turn", () => {
  // Tool-call follow-ups produce another response for a turn that already has
  // one; it must not be handed to some other turn.
  const tracer = new Tracer();
  speechStart(tracer, 0, "item_1");
  speechStop(tracer, 500, "item_1");
  responseCreated(tracer, 600, "resp_1");
  responseDone(tracer, 900, "resp_1");
  responseCreated(tracer, 1000, "resp_2");
  assert.equal(tracer.responsesById.get("resp_2").turn.number, 1);
});
