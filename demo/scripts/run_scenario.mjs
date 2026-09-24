#!/usr/bin/env node
/**
 * Drive one lab scenario in a real browser and print the trace.
 *
 * This is the same code path a person exercises by clicking "Run" in developer
 * mode — same page, same client, same audio graph. It is scripted so a race
 * can be run thirty times in a row, and so the two server builds can be
 * compared without a human trying to interrupt at the same millisecond twice.
 *
 * Chromium is launched with a fake media device, but the scripted microphone
 * means no capture device is ever opened: the session captures from the
 * fixture WAVs through a MediaStreamDestination.
 *
 * Usage:
 *   node scripts/run_scenario.mjs --scenario superseded-turn --repeat 5
 *   node scripts/run_scenario.mjs --list
 */

import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const args = parseArgs(process.argv.slice(2));
const url = args.url ?? "http://127.0.0.1:7870/";
const scenario = args.scenario ?? "superseded-turn";
const repeat = Number(args.repeat ?? 1);
const timeoutMs = Number(args.timeout ?? 120_000);
const outDir = args.out ?? null;

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) out[key] = true;
    else {
      out[key] = next;
      i += 1;
    }
  }
  return out;
}

const browser = await chromium.launch({
  args: [
    // No real capture device is opened — the scripted mic feeds the graph — but
    // Chromium still wants a device to exist before it will build an audio
    // graph at all.
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});

const context = await browser.newContext({ permissions: ["microphone"] });
await context.addInitScript(() => {
  // Developer mode, scripted microphone, and no greeting racing the first turn.
  localStorage.setItem("s2s.debug", "1");
  localStorage.setItem("s2s.lab.scriptedMic", "1");
});

const page = await context.newPage();
page.on("console", (message) => {
  if (message.type() === "error") console.error("  [page error]", message.text());
});

await page.goto(`${url}?debug=1`, { waitUntil: "load" });
await page.waitForFunction(() => Boolean(globalThis.__voiceLab), null, { timeout: 15_000 });

if (args.list) {
  const names = await page.evaluate(async () => {
    const module = await import("./lab/scenarios.js");
    return module.SCENARIOS.map((s) => `${s.id.padEnd(20)} ${s.title}`);
  });
  console.log(names.join("\n"));
  await browser.close();
  process.exit(0);
}

console.log(`scenario ${scenario} · ${repeat} run(s) · ${url}`);
const traces = [];

for (let run = 1; run <= repeat; run += 1) {
  process.stdout.write(`\nrun ${run}/${repeat}  `);
  await startSession(page);
  const result = await page.evaluate(
    async ([id, interruptMs]) => {
      const lab = globalThis.__voiceLab;
      lab.tracer.reset();
      const stepOverrides = interruptMs === null ? {} : { "interrupt-at": interruptMs };
      await lab.runScenario(id, { stepOverrides });
      return lab.buildTrace();
    },
    [scenario, args["interrupt-ms"] === undefined ? null : Number(args["interrupt-ms"])],
  );
  traces.push(result);
  summarise(result);
  await stopSession(page);
}

report(traces);

if (outDir) {
  await mkdir(outDir, { recursive: true });
  const file = path.join(outDir, `${scenario}-${Date.now()}.json`);
  await writeFile(file, JSON.stringify({ url, scenario, traces }, null, 2));
  console.log(`\nwrote ${file}`);
}

await browser.close();

/** Click the orb and wait for the session to go live. */
async function startSession(page) {
  await page.click("#main-circle");
  await page.waitForFunction(
    () => {
      const state = document.body.getAttribute("data-state") ?? "";
      const caption = document.querySelector("#circle-caption")?.textContent ?? "";
      return /listening|connected|ready/i.test(`${state} ${caption}`);
    },
    null,
    { timeout: timeoutMs },
  ).catch(() => {});
  // The client reports its own status; wait for it rather than a fixed sleep.
  await page.waitForFunction(
    () => ["connected", "listening", "user-speaking", "processing", "ai-speaking"]
      .includes(globalThis.__voiceLab?.client?.status),
    null,
    { timeout: timeoutMs },
  );
}

async function stopSession(page) {
  await page.evaluate(() => document.querySelector("#stop-btn")?.click());
  await page.waitForTimeout(1200);
}

/** @param {any} trace */
function summarise(trace) {
  for (const turn of trace.turns) {
    const said = turn.responses.map((r) => r.text).filter(Boolean).join(" ");
    console.log(
      `turn ${turn.number} ${JSON.stringify(turn.transcript)} → ${turn.verdict.toUpperCase()}`
      + (turn.supersededByTurn ? ` (replaced by turn ${turn.supersededByTurn})` : "")
      + (said ? ` · spoke ${JSON.stringify(said)}` : ""),
    );
  }
  const replaced = trace.turns.filter((t) => t.supersededByTurn !== null);
  if (replaced.length === 0) console.log("  no turn was replaced while waiting — the race window did not open");
}

/** @param {any[]} traces */
function report(traces) {
  const turns = traces.flatMap((t) => t.turns);
  const replaced = turns.filter((t) => t.supersededByTurn !== null);
  const leaked = replaced.filter((t) => t.verdict === "replaced-leaked");
  const dropped = replaced.filter((t) => t.verdict === "replaced-dropped");

  console.log("\n─────────────────────────────────────────────");
  console.log(`runs                              ${traces.length}`);
  console.log(`turns replaced while waiting      ${replaced.length}   ← the #308 window`);
  console.log(`   answered anyway (leaked)       ${leaked.length}   ${leaked.length ? "⚠ STALE OUTPUT" : ""}`);
  console.log(`   dropped                        ${dropped.length}   ${dropped.length ? "✓" : ""}`);
  console.log("─────────────────────────────────────────────");
  if (replaced.length === 0) {
    console.log("The race window never opened, so this run proves nothing either way.");
    console.log("Sweep the interruption point with --interrupt-ms to find it.");
  }
}
