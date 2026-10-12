// "Export for..." on the station panel: presentation only. Which tools take which record, and every
// byte of every file, come from aquascope.io.engineering in the worker; this module turns its answer
// into menu options and its base64 zip into bytes. No imports, so node can test it.

export const EXPORT_PROMPT = "Export for…";

// The worker's menu ({kind, tools: [{id, label, what}]}) as <option> data: a disabled prompt first,
// then one option per tool. An empty list when nothing takes the record.
export function exportOptions(menu) {
  const tools = (menu && Array.isArray(menu.tools)) ? menu.tools : [];
  if (!tools.length) return [];
  return [{ value: "", label: EXPORT_PROMPT, title: "", prompt: true },
    ...tools.filter((t) => t && t.id).map((t) => ({ value: String(t.id), label: String(t.label || t.id),
                                                   title: String(t.what || ""), prompt: false }))];
}

// A base64 string to bytes (atob where the page has it, Buffer under node).
export function base64ToBytes(b64) {
  const text = String(b64 || "");
  if (typeof atob === "function") {
    const bin = atob(text);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(text, "base64"));
}

// One line for the status bar after a download: the file and the first note worth reading.
export function exportSummary(result) {
  if (!result || !result.filename) return "";
  const n = Array.isArray(result.files) ? result.files.length : 0;
  const note = (result.notes || []).find((x) => /placeholder|filled|interpolat|averaged|time zone/i.test(x));
  return `${result.label}: ${result.filename} (${n} file${n === 1 ? "" : "s"}).${note ? " " + note : ""}`;
}
