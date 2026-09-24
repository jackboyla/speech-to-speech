// @ts-check
/**
 * A microphone the lab can script.
 *
 * Scenarios need the same utterance to arrive the same way every run, which a
 * person at a laptop cannot provide. This replaces the capture *source* with a
 * `MediaStreamAudioDestinationNode` and leaves everything downstream alone —
 * the capture worklet, the noise gate, the resampler and the 40 ms send cadence
 * all still run. Swapping the source rather than injecting PCM further down
 * matters: the pipeline's turn detection reacts to how audio is paced, so a
 * shortcut that posted a whole utterance at once would measure a path no user
 * ever takes.
 *
 * Fixture audio is played through the graph in real time for the same reason.
 * There is no way to make this faster without making it wrong.
 */

/** @typedef {{name: string, text: string, buffer: AudioBuffer}} Fixture */

export class SyntheticMic {
  /** @param {AudioContext} ctx */
  constructor(ctx) {
    this.ctx = ctx;
    this.destination = ctx.createMediaStreamDestination();
    /** @type {AudioBufferSourceNode | null} */
    this._playing = null;
  }

  /** The stream to hand the client in place of `getUserMedia`. */
  get stream() {
    return this.destination.stream;
  }

  /**
   * Play one fixture and resolve when the last sample has been rendered.
   * @param {AudioBuffer} buffer
   * @param {{gain?: number}} [options]
   */
  play(buffer, options = {}) {
    return new Promise((resolve) => {
      const source = this.ctx.createBufferSource();
      source.buffer = buffer;
      const gain = this.ctx.createGain();
      gain.gain.value = options.gain ?? 1;
      source.connect(gain);
      gain.connect(this.destination);
      source.onended = () => {
        this._playing = null;
        resolve(undefined);
      };
      this._playing = source;
      source.start();
    });
  }

  /** Cut playback short, for a scenario that is being abandoned. */
  stop() {
    try {
      this._playing?.stop();
    } catch {
      // Already stopped; nothing to do.
    }
    this._playing = null;
  }
}

/**
 * Fetch and decode the fixture set described by `fixtures/manifest.json`.
 *
 * A missing fixture directory is reported rather than thrown: the lab is still
 * useful for tracing a live conversation without any fixtures present, and the
 * panel explains how to generate them.
 *
 * @param {AudioContext} ctx
 * @param {string} [base]
 * @returns {Promise<{fixtures: Map<string, Fixture>, error: string}>}
 */
export async function loadFixtures(ctx, base = "./lab/fixtures/") {
  /** @type {Map<string, Fixture>} */
  const fixtures = new Map();
  let manifest;
  try {
    const response = await fetch(`${base}manifest.json`, { cache: "no-store" });
    if (!response.ok) throw new Error(`manifest.json returned ${response.status}`);
    manifest = await response.json();
  } catch (err) {
    return {
      fixtures,
      error: `No fixtures found (${err instanceof Error ? err.message : String(err)}). `
        + "Generate them with demo/scripts/make_lab_fixtures.py.",
    };
  }

  const entries = Array.isArray(manifest?.fixtures) ? manifest.fixtures : [];
  const failures = [];
  await Promise.all(entries.map(async (entry) => {
    try {
      const response = await fetch(`${base}${entry.file}`, { cache: "force-cache" });
      if (!response.ok) throw new Error(`${entry.file} returned ${response.status}`);
      const buffer = await ctx.decodeAudioData(await response.arrayBuffer());
      fixtures.set(entry.name, { name: entry.name, text: entry.text ?? "", buffer });
    } catch (err) {
      failures.push(`${entry.name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }));

  return { fixtures, error: failures.length ? `Some fixtures failed to load — ${failures.join("; ")}` : "" };
}
