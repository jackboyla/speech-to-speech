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

test("history shows timings and copies current message text on desktop and phone", async (t) => {
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
    const page = await browser.newPage({ viewport: { width, height: 850 }, reducedMotion: "reduce", permissions: ["clipboard-read", "clipboard-write"] });
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
      chat.renderEmptyState();
      chat._openPanel();
    }, timing);
    await page.evaluate(() => document.querySelector(".chat-panel-inner").getAnimations().forEach((animation) => animation.finish()));
    const copy = page.getByRole("button", { name: "Copy conversation", exact: true });
    const copyStatus = page.locator("#chat-copy-status");
    await page.evaluate(() => navigator.clipboard.writeText("keep existing clipboard"));
    await copy.click({ timeout: 2000 });
    assert.equal(await copyStatus.textContent(), "No conversation text to copy yet.");
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), "keep existing clipboard");
    await page.evaluate((timing) => {
      chat.onTranscript({ role: "user", text: "Why is the sky", partial: true, itemId: "u1" });
      chat.onTranscript({ role: "assistant", text: "The sky looks blue because air scatters blue light more strongly.", partial: false, responseId: "r1" });
      finish("r1", timing);
    }, timing);
    await copy.click();
    await page.waitForFunction(() => document.querySelector("#chat-copy-status").textContent === "Conversation text copied.");
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()),
      "You:\nWhy is the sky\n\nAssistant:\nThe sky looks blue because air scatters blue light more strongly.");
    await page.evaluate(() => {
      chat.onTranscript({ role: "user", text: "Why is the sky blue?\n<café> & clouds ☁", partial: false, itemId: "u1" });
      chat.onTranscript({ role: "assistant", text: "Checking", partial: true, responseId: "stream" });
    });
    await copy.click();
    assert.match(await page.evaluate(() => navigator.clipboard.readText()), /Assistant:\nChecking$/);
    await page.evaluate(() => chat.onTranscript({ role: "assistant", text: "Checking the latest answer…", partial: true, responseId: "stream" }));
    await copy.click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()),
      "You:\nWhy is the sky blue?\n<café> & clouds ☁\n\nAssistant:\nThe sky looks blue because air scatters blue light more strongly.\n\nAssistant:\nChecking the latest answer…");
    assert.equal(await copy.getAttribute("aria-describedby"), "chat-copy-hint");
    assert.equal(await page.locator(".hist-msg.assistant").count(), 2);
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
      chat.onToolResult("web_search", '{"query":"sky"}', "tool output", "data:image/png;base64,");
      chat.onUserAudio({ itemId: "audio-only", audio: new Blob([], { type: "audio/wav" }) });
    });
    await copy.click();
    const copied = await page.evaluate(() => navigator.clipboard.readText());
    assert.match(copied, /Assistant \(Interrupted\):\nChecking that for you…/);
    assert.doesNotMatch(copied, /Server timings|Audio sent|web_search|tool output|data:image/);
    assert.equal(copied.split("You:").length - 1, 1);
    const copyBox = await copy.boundingBox();
    assert.ok(copyBox.x >= 0 && copyBox.x + copyBox.width <= width && copyBox.height >= 44);
    await page.evaluate(() => {
      navigator.clipboard.writeText = async () => { throw new DOMException("Blocked", "NotAllowedError"); };
    });
    await copy.click();
    await page.waitForFunction(() => document.querySelector("#chat-copy-status").textContent.includes("Could not copy"));
    assert.match(await copyStatus.textContent(), /Select the message text and copy it manually/);
    await page.evaluate(() => Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined }));
    await copy.click();
    assert.match(await copyStatus.textContent(), /Could not copy/);
    await page.evaluate(() => { chat.clear(); chat.reset(); });
    assert.equal(await copyStatus.textContent(), "");
    await copy.click();
    assert.equal(await copyStatus.textContent(), "No conversation text to copy yet.");
    assert.equal(await page.locator(".hist-timings").count(), 0);
    await page.close();
  }
});
