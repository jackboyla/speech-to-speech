import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const timing = {
  version: 2, turn_id: "turn_1", turn_revision: 0, response_key: "key-1", status: "completed",
  stt_s: 0.18, llm_s: 1.24, tts_ttfa_s: 0.12, e2e_s: 1.61,
  vad_decision_s: 0.32,
  hold_s: 0, smart_status: "incomplete",
};

test("history timings and scroll following work on desktop and phone", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    const file = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      let content = await readFile(file, "utf8");
      // Render the actual demo shell without network/auth/audio startup.
      if (file.endsWith(".html")) content = content.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "");
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(content);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  for (const width of [1280, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 850 }, reducedMotion: "reduce" });
    await page.route("https://**/*", (route) => route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async (timing) => {
      const { ChatView } = await import("/ui/chat.js");
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js");
      document.body.classList.remove("booting");
      window.chat = new ChatView();
      window.client = new S2sRealtimeClient({ transport: "webrtc" });
      client.addEventListener("response-finished", (event) => chat.onResponseFinished(event.detail));
      window.finish = (id, timing, status = "completed") => client._onTransportEvent({
        type: "response.done", response: { id, status, metadata: timing ? {
          "speech_to_speech.turn_latency": JSON.stringify(timing),
        } : {}, output: [] },
      });
      chat.onTranscript({ role: "assistant", text: "The sky looks blue because air scatters blue light more strongly.", partial: false, responseId: "r1" });
      finish("r1", timing);
      chat._openPanel();
    }, timing);
    await page.evaluate(() => document.querySelector(".chat-panel-inner").getAnimations().forEach((animation) => animation.finish()));
    assert.equal(await page.locator(".hist-msg.assistant").count(), 1);
    const summary = page.locator(".hist-timings summary").first();
    assert.match(await summary.textContent(), /E2E 1.61 s/);
    await summary.focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator(".hist-timings").first().getAttribute("open"), "");
    assert.match(await page.locator(".hist-timings dl").innerText(), /0.00 s/);
    assert.deepEqual(await page.locator(".hist-timings dt").allTextContents(), [
      "E2E time", "VAD end decision", "Smart Turn decision", "Transcription",
      "Response generation", "Voice synthesis to first audio", "Hold time before response",
    ]);
    assert.match(await page.locator(".hist-timings dl").innerText(), /Hold time before response/);
    assert.match(await page.locator(".hist-timings").innerText(), /excluding browser playback/);
    await page.waitForFunction(() => {
      const box = document.querySelector(".hist-timings summary").getBoundingClientRect();
      return box.right <= innerWidth;
    });
    const box = await summary.boundingBox();
    assert.ok(box.height >= 44 && box.x >= 0 && box.x + box.width <= width);
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/timings-${width}.png` });
    await page.evaluate((timing) => {
      // A tool-only terminal still gets a record, distinct from its follow-up.
      finish("tool", { ...timing, response_key: "tool", stt_s: null, e2e_s: null, tts_ttfa_s: null });
      chat.onTranscript({ role: "assistant", text: "Checking that for you…", partial: true, responseId: "cancelled" });
      finish("cancelled", { ...timing, response_key: "cancelled", status: "cancelled" }, "cancelled");
      chat.onTranscript({ role: "assistant", text: "An older server reply.", partial: false, responseId: "legacy" });
      finish("legacy", null);
    }, timing);
    assert.equal(await page.locator(".hist-timings").count(), 3);
    const tool = page.locator(".hist-timings").nth(1);
    await tool.locator("summary").click();
    assert.match(await tool.innerText(), /Unavailable/);
    assert.equal(await page.locator(".hist-note").textContent(), "Interrupted");
    assert.equal(await page.locator(".hist-msg.assistant").last().locator(".hist-timings").count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);

    await page.evaluate(() => {
      for (let i = 0; i < 25; i++) {
        chat.onTranscript({ role: "user", itemId: `history-${i}`, text: `Earlier message ${i}. ` + "Some conversation text. ".repeat(5), partial: false });
      }
      chat.onTranscript({ role: "user", itemId: "stream", text: "Incoming text.", partial: true });
    });
    const atBottom = () => {
      const history = document.querySelector("#chat-history");
      return history.scrollHeight - history.clientHeight - history.scrollTop <= 1;
    };
    await page.waitForFunction(atBottom);
    // A growing existing row must keep following even when one delta adds
    // more than the near-bottom threshold.
    await page.evaluate(() => chat.onTranscript({ role: "user", itemId: "stream", text: "Incoming text. ".repeat(60), partial: true }));
    await page.waitForFunction(atBottom);
    await page.evaluate(() => {
      document.querySelector("#chat-history").scrollTop = 150;
      chat.onTranscript({ role: "assistant", responseId: "new-reply", text: "New reply.", partial: true });
      chat.onTranscript({ role: "user", itemId: "stream", text: "Incoming text. ".repeat(100), partial: false });
    });
    await page.waitForTimeout(400); // Catch delayed smooth scrolls as well as frame-queued writes.
    assert.equal(await page.locator("#chat-history").evaluate((history) => history.scrollTop), 150);
    const newMessages = page.getByRole("button", { name: "New messages ↓", exact: true });
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/history-paused-${width}.png` });
    await newMessages.focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(atBottom);
    assert.equal(await newMessages.isVisible(), false);

    // Follow when near the end, even if text arrives before the scroll event.
    await page.evaluate(() => {
      const history = document.querySelector("#chat-history");
      history.scrollTop = history.scrollHeight - history.clientHeight - 32;
      chat.onTranscript({ role: "assistant", responseId: "new-reply", text: "Growing reply. ".repeat(100), partial: false });
    });
    await page.waitForFunction(atBottom);
    assert.equal(await newMessages.isVisible(), false);

    await page.locator("#chat-history").evaluate((history) => { history.scrollTop = 150; });
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.evaluate(() => {
      chat._closePanel();
      chat.onTranscript({ role: "user", itemId: "while-closed", text: "Received while closed.", partial: false });
      chat._openPanel();
    });
    await page.waitForFunction(atBottom);
    assert.equal(await newMessages.isVisible(), false);

    await page.locator("#chat-history").evaluate((history) => { history.scrollTop = 150; });
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.evaluate(() => chat.onTranscript({ role: "user", itemId: "before-reset", text: "Unread before reset.", partial: false }));
    assert.equal(await newMessages.isVisible(), true);
    await page.evaluate(() => { chat.clear(); chat.reset(); });
    assert.equal(await newMessages.isVisible(), false);
    assert.equal(await page.locator(".hist-timings").count(), 0);
    await page.evaluate(() => chat.onTranscript({ role: "user", itemId: "fresh-session", text: "Fresh session. ".repeat(200), partial: true }));
    await page.waitForFunction(atBottom);
    await page.close();
  }
});
