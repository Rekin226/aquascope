import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const code = await readFile(new URL("../src/gr4j-area.js", import.meta.url), "utf8");
const { confirmLargeArea } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

test("the area threshold requires an explicit choice only above 10,000 km²", async () => {
  const warning = { hidden: true }, message = { textContent: "" }, button = { onclick: null };
  assert.equal(await confirmLargeArea(10000, warning, message, button).result, true);
  assert.equal(warning.hidden, true);
  const gate = confirmLargeArea(10001, warning, message, button);
  let resolved = false;
  gate.result.then(() => { resolved = true; });
  await Promise.resolve();
  assert.equal(resolved, false);
  assert.equal(warning.hidden, false);
  assert.match(message.textContent, /not averaged over the catchment/);
  button.onclick();
  assert.equal(await gate.result, true);
  assert.equal(warning.hidden, true);
  assert.equal(button.onclick, null);
});

test("stop or a station reset releases the pending confirmation", async () => {
  const warning = { hidden: true }, message = {}, button = { onclick: null };
  const gate = confirmLargeArea(20000, warning, message, button);
  gate.cancel();
  assert.equal(await gate.result, false);
  assert.equal(warning.hidden, true);
  assert.equal(button.onclick, null);
});
