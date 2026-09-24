// @ts-check
/**
 * Whether developer mode was asked for.
 *
 * Deliberately its own module with no imports: `main.js` needs the answer on
 * every page load, and importing it must not drag the rest of the lab into the
 * initial bundle. Everything else in `lab/` is behind a dynamic import that
 * only runs when this returns true.
 */

/** @returns {boolean} */
export function labRequested() {
  try {
    const flag = new URLSearchParams(location.search).get("debug");
    if (flag === "1") {
      // Persist, so a reload that drops the query string stays in developer mode.
      localStorage.setItem("s2s.debug", "1");
      return true;
    }
    if (flag === "0") {
      localStorage.removeItem("s2s.debug");
      return false;
    }
    return localStorage.getItem("s2s.debug") === "1";
  } catch {
    return false;
  }
}
