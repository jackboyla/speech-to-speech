import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

test("settings tests audio before connecting and releases its resources", { timeout: 60_000 }, async (t) => {
  const requests = [];
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    requests.push(name);
    if (name === "/api/config") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ lb: true, allowDirect: false }));
      return;
    }
    const file = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(await readFile(file));
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const browser = await chromium.launch({ headless: true, args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("https://**/*", (route) => route.abort());
  await page.addInitScript(() => {
    localStorage.setItem("s2s.ws.tools", JSON.stringify({ camera_snapshot: false, web_search: false }));
    const audio = window.audioChecks = { contexts: [], streams: [], constraints: [], sinks: [], failure: null, defer: false };
    const OriginalContext = window.AudioContext;
    window.AudioContext = class extends OriginalContext {
      constructor(...args) { super(...args); audio.contexts.push(this); }
      async setSinkId(id) { audio.sinks.push(id); return super.setSinkId(id); }
    };
    // Fake permission exposes labels early; model the pre-permission list explicitly.
    const enumerateDevices = navigator.mediaDevices.enumerateDevices.bind(navigator.mediaDevices);
    navigator.mediaDevices.enumerateDevices = async () => (await enumerateDevices()).map((device) =>
      audio.streams.length ? device : { kind: device.kind, deviceId: device.deviceId, label: "" });
    const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      audio.constraints.push(constraints);
      if (audio.failure) throw new DOMException("injected device failure", audio.failure);
      const stream = await getUserMedia(constraints);
      audio.streams.push(stream);
      if (audio.defer) return new Promise((resolve) => { audio.release = () => resolve(stream); });
      return stream;
    };
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  const mic = page.getByRole("button", { name: "Test microphone", exact: true });
  const stopMic = page.getByRole("button", { name: "Stop microphone test", exact: true });
  const speakers = page.getByRole("button", { name: "Test speakers", exact: true });
  const status = page.locator("#audio-test-status");
  const released = () => page.waitForFunction(() =>
    audioChecks.contexts.every((context) => context.state === "closed") &&
    audioChecks.streams.every((stream) => stream.getTracks().every((track) => track.readyState === "ended")));
  const startMic = async () => {
    await mic.click({ timeout: 2000 });
    await page.waitForFunction(() => document.querySelector("#mic-test-meter").value > -60);
  };

  await page.waitForFunction(() => document.querySelector("#audio-input").options.length > 1);
  assert.match(await page.locator("#audio-input option").nth(1).textContent(), /^Microphone \d/);
  await startMic();
  await page.waitForFunction(() => !/^Microphone \d/.test(document.querySelector("#audio-input").options[1].text));
  if (process.env.S2S_UI_SCREENSHOTS) {
    for (const width of [390, 1280]) {
      await page.setViewportSize({ width, height: 844 });
      await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/audio-settings-${width}.png` });
    }
    await page.setViewportSize({ width: 390, height: 844 });
  }
  const inputId = await page.locator("#audio-input option").nth(1).getAttribute("value");
  assert.ok(inputId);
  assert.doesNotMatch(await page.locator("#audio-input option").nth(1).textContent(), /^Microphone \d/);
  await stopMic.click();
  await released();
  assert.equal(await page.locator("#mic-test-meter").evaluate((meter) => meter.value), -66);

  // Use the editable selection before Save, including when permission refreshes labels.
  await page.selectOption("#audio-input", inputId);
  await startMic();
  assert.equal(await page.evaluate(() => audioChecks.constraints.at(-1).audio.deviceId.exact), inputId);
  assert.equal(await page.locator("#audio-input").inputValue(), inputId);
  await page.selectOption("#audio-input", "");
  await released();

  const outputId = await page.locator("#audio-output option").nth(1).getAttribute("value");
  await page.selectOption("#audio-output", outputId);
  await startMic();
  // Starting the speaker test stops the microphone test first.
  await speakers.click();
  await speakers.waitFor();
  await released();
  assert.equal(await page.evaluate(() => audioChecks.sinks.at(-1)), outputId);
  assert.match(await status.textContent(), /Test sound finished/);

  // Device lists must preserve unsaved choices, including an explicit system default.
  await page.evaluate(() => navigator.mediaDevices.dispatchEvent(new Event("devicechange")));
  assert.equal(await page.locator("#audio-input").inputValue(), "");
  assert.equal(await page.locator("#audio-output").inputValue(), outputId);
  for (const [failure, message] of [
    ["NotAllowedError", /permission denied/],
    ["NotFoundError", /microphone not found/],
    ["NotReadableError", /Microphone unavailable/],
  ]) {
    await page.evaluate((failure) => { audioChecks.failure = failure; }, failure);
    await mic.click();
    await page.waitForFunction(() => document.querySelector("#audio-test-status").classList.contains("error"));
    assert.match(await status.textContent(), message);
    await released();
  }
  await page.evaluate(() => { audioChecks.failure = null; });
  await page.evaluate(() => {
    const option = new Option("Unplugged microphone", "missing-mic");
    document.querySelector("#audio-input").add(option);
  });
  await page.selectOption("#audio-input", "missing-mic");
  await mic.click();
  await page.waitForFunction(() => document.querySelector("#audio-test-status").classList.contains("error"));
  assert.match(await status.textContent(), /microphone not found/);
  await released();
  await page.selectOption("#audio-input", "");

  // Cancel an unresolved permission request; its late stream must be stopped.
  await page.evaluate(() => { audioChecks.defer = true; });
  await mic.click();
  await page.waitForFunction(() => Boolean(audioChecks.release));
  await page.keyboard.press("Escape");
  await page.evaluate(() => { audioChecks.release(); audioChecks.defer = false; });
  await released();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await startMic();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await released();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await startMic();
  await page.evaluate(() => navigator.mediaDevices.dispatchEvent(new Event("devicechange")));
  await released();
  assert.match(await status.textContent(), /devices changed/);

  // Output failure must not silently play through the default device.
  await page.evaluate(() => {
    document.querySelector("#audio-output").add(new Option("Unplugged speakers", "missing-output"));
  });
  await page.selectOption("#audio-output", "missing-output");
  await speakers.click();
  await page.waitForFunction(() => document.querySelector("#audio-test-status").classList.contains("error"));
  assert.match(await status.textContent(), /speakers not found/);
  await released();
  await page.selectOption("#audio-output", outputId);

  // Browsers without Web Audio output selection still test the system default.
  await page.evaluate(() => { AudioContext.prototype.setSinkId = undefined; });
  await page.evaluate(() => navigator.mediaDevices.dispatchEvent(new Event("devicechange")));
  await page.waitForFunction(() => document.querySelector("#audio-output").disabled);
  assert.match(await page.locator("#audio-output-hint").textContent(), /system default/);
  const sinkCount = await page.evaluate(() => audioChecks.sinks.length);
  await speakers.click();
  await page.waitForFunction(() => document.querySelector("#audio-test-status").textContent.includes("system default"));
  await page.keyboard.press("Escape");
  await released();
  assert.equal(await page.evaluate(() => audioChecks.sinks.length), sinkCount);
  assert.equal(requests.some((url) => /api\/(session|calls)/.test(url)), false);

  // Restart releases the test before normal capture and disables self-tests while connecting.
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await startMic();
  const testResources = await page.evaluate(() => ({
    context: audioChecks.contexts.length - 1, stream: audioChecks.streams.length - 1,
  }));
  let sessionRequest;
  await page.route("**/api/session", (route) => { sessionRequest = route; });
  await Promise.all([
    page.waitForRequest("**/api/session"),
    page.getByRole("button", { name: "Restart conversation with these settings" }).click(),
  ]);
  await page.waitForFunction(() => document.querySelector("#test-microphone").disabled);
  await page.waitForFunction(({ context, stream }) => audioChecks.contexts[context].state === "closed" &&
    audioChecks.streams[stream].getTracks().every((track) => track.readyState === "ended"), testResources);
  assert.equal(await page.locator("#test-microphone").isDisabled(), true);
  await page.waitForFunction(() => audioChecks.constraints.length >= 2);
  assert.deepEqual(await page.evaluate(() => audioChecks.constraints.at(-1)), { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  assert.deepEqual(errors, []);
  assert.ok(sessionRequest, "normal startup should reach session creation after releasing test capture");
  await sessionRequest.fulfill({ status: 503, body: "test complete" });
  await released();
});
