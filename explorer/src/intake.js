// Solve's intake from a sentence, on the reader's device. Pure functions, no
// DOM: the prompt a small model gets, the JSON schema that constrains its
// reply, and the reader of that reply. The values themselves are made safe
// by aquascope.playbooks.coerce_intake in the worker, so the page and the CLI
// apply one set of rules; this module only decides whether there is a reply
// worth sending there.

// "return_period (int, default 100, at least 2)": one line per field.
function fieldLine(f) {
  const bits = [f.type];
  if (f.type === "choice" && (f.options || []).length) bits.push(`one of: ${f.options.join(", ")}`);
  if (f.default !== null && f.default !== undefined) bits.push(`default ${f.default}`);
  if (f.min !== null && f.min !== undefined) bits.push(`at least ${f.min}`);
  if (f.max !== null && f.max !== undefined) bits.push(`at most ${f.max}`);
  return `    ${f.name}: ${f.label || f.name} (${bits.join(", ")})`;
}

/** The system prompt: the playbooks with their intake fields, and the one-object reply rule. */
export function intakePrompt(playbooks) {
  const lines = (playbooks || []).map((p) =>
    `- ${p.id}: ${p.title}${p.description ? `. ${p.description}` : ""}\n${(p.intake || []).map(fieldLine).join("\n")}`);
  return [
    "You read one sentence stating a water problem and fill a short form.",
    "Playbooks and their fields:",
    ...lines,
    "",
    'Reply with ONE JSON object and nothing else: {"playbook": "<id>", "intake": {<field>: <value>}}.',
    'Pick the playbook whose problem the sentence describes; use "none" when no playbook fits.',
    "Fill only the fields the sentence states, with values of the given type; leave the others out.",
  ].join("\n");
}

/** A JSON schema for the reply, so a constrained decoder cannot wander. */
export function intakeSchema(playbooks) {
  const fields = {};
  for (const p of playbooks || []) {
    for (const f of p.intake || []) {
      if (fields[f.name]) continue;
      if (f.type === "int") fields[f.name] = { type: "integer" };
      else if (f.type === "float") fields[f.name] = { type: "number" };
      else if (f.type === "bool") fields[f.name] = { type: "boolean" };
      else if (f.type === "choice") fields[f.name] = { type: "string", enum: (f.options || []).map(String) };
      else fields[f.name] = { type: "string" };
    }
  }
  return {
    type: "object",
    properties: {
      playbook: { type: "string", enum: [...(playbooks || []).map((p) => p.id), "none"] },
      intake: { type: "object", properties: fields },
    },
    required: ["playbook"],
  };
}

/**
 * The model's text as { playbook, intake }, or null when it named no known
 * playbook: the caller then falls back to the keyword rules. Field values are
 * passed through untouched; coerce_intake in the worker settles them.
 */
export function parseIntakeReply(text, playbooks) {
  const raw = String(text || "").trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  let obj;
  try { obj = JSON.parse(raw.slice(start, end + 1)); } catch { return null; }
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
  const id = typeof obj.playbook === "string" ? obj.playbook.trim() : "";
  if (!id || !(playbooks || []).some((p) => p.id === id)) return null;
  const intake = obj.intake && typeof obj.intake === "object" && !Array.isArray(obj.intake) ? obj.intake : {};
  return { playbook: id, intake };
}
