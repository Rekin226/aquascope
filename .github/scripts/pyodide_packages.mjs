import { setTimeout } from "node:timers/promises";

export async function loadPackagesWithRetry(
  pyodide, packages, { sleep = setTimeout, warn = console.warn } = {},
) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    const pending = packages.filter((name) => !pyodide.loadedPackages[name]);
    if (!pending.length) return;
    try {
      await pyodide.loadPackage(pending);
      // loadPackage can log download errors without rejecting its promise.
      const missing = packages.filter((name) => !pyodide.loadedPackages[name]);
      if (missing.length) {
        throw new Error(`Packages not loaded: ${missing.join(", ")}`);
      }
      return;
    } catch (cause) {
      if (attempt === 3) {
        throw new Error("Pyodide package loading failed after 3 attempts", { cause });
      }
      const delay = 1000 * 2 ** (attempt - 1);
      warn(`Pyodide package load attempt ${attempt}/3 failed; retrying in ${delay}ms: ${cause.message}`);
      await sleep(delay);
    }
  }
}
