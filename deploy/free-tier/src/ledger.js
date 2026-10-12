// The free tier's ledger: dollars and runs per day, per month, per visitor and per network.
//
// It runs inside one Durable Object, so every check and every charge is serialised: two calls arriving together
// cannot both slip under a cap. It takes any store with `get`, `put`, `delete` and `list({prefix})` (a Durable
// Object's storage, or a Map-backed fake in the tests). Keys hold numbers and counts only, never content.

import { RUN_KINDS, overBudget, utcDay, utcMonth } from "./core.js";

const VISITOR = { usd: 0, asks: 0, studies: 0, calls: 0 };
const DAY = { usd: 0, calls: 0, refused: 0, runs: { ask: 0, map: 0, study: 0 }, visitors: 0 };

export class Ledger {
  constructor(storage, limits, now = () => new Date()) {
    this.storage = storage;
    this.limits = limits;
    this.now = now;
  }

  async _get(key, empty) {
    const v = await this.storage.get(key);
    return v === undefined ? structuredClone(empty) : v;
  }

  async spent(visitor, network) {
    const day = utcDay(this.now());
    const [d, m, v, n] = await Promise.all([
      this._get(`day:${day}`, DAY), this._get(`month:${utcMonth(this.now())}`, 0),
      this._get(`v:${day}:${visitor}`, VISITOR), this._get(`n:${day}:${network}`, 0),
    ]);
    return { day: d.usd, month: m, visitor: v.usd, network: n, visitorRow: v, dayRow: d };
  }

  /** Before a model call: the budget it would break, or null. A refusal is counted (it is how demand shows). */
  async check(visitor, network) {
    const s = await this.spent(visitor, network);
    const over = overBudget(s, this.limits);
    if (over) {
      s.dayRow.refused += 1;
      await this.storage.put(`day:${utcDay(this.now())}`, s.dayRow);
    }
    return over;
  }

  /** After a model call: charge what it cost to the day, the month, the visitor and the network. */
  async charge(visitor, network, usd) {
    const day = utcDay(this.now());
    const month = utcMonth(this.now());
    const s = await this.spent(visitor, network);
    const d = s.dayRow;
    d.usd += usd;
    d.calls += 1;
    const v = s.visitorRow;
    if (v.calls === 0) d.visitors += 1;
    v.usd += usd;
    v.calls += 1;
    await Promise.all([
      this.storage.put(`day:${day}`, d),
      this.storage.put(`month:${month}`, s.month + usd),
      this.storage.put(`v:${day}:${visitor}`, v),
      this.storage.put(`n:${day}:${network}`, s.network + usd),
    ]);
    return { day: d.usd, visitor: v.usd };
  }

  /** The start of an Ask, a map command or a study: refused past the visitor's run cap or any budget. */
  async startRun(visitor, network, kind) {
    const cap = RUN_KINDS[kind];
    if (!cap) return { ok: false, reason: "unknown", message: `unknown run kind ${kind}` };
    const s = await this.spent(visitor, network);
    const over = overBudget(s, this.limits);
    if (over) return { ok: false, reason: over };
    const v = s.visitorRow;
    const field = cap === "VISITOR_STUDIES" ? "studies" : "asks";
    if (v[field] >= this.limits[cap]) return { ok: false, reason: field, left: this.left(v) };
    v[field] += 1;
    const d = s.dayRow;
    d.runs[kind] = (d.runs[kind] || 0) + 1;
    const day = utcDay(this.now());
    await Promise.all([this.storage.put(`v:${day}:${visitor}`, v), this.storage.put(`day:${day}`, d)]);
    return { ok: true, left: this.left(v) };
  }

  left(v) {
    return {
      asks: Math.max(0, this.limits.VISITOR_ASKS - v.asks),
      studies: Math.max(0, this.limits.VISITOR_STUDIES - v.studies),
    };
  }

  async allowance(visitor) {
    const v = await this._get(`v:${utcDay(this.now())}:${visitor}`, VISITOR);
    return this.left(v);
  }

  /** Days of totals, newest first: what the stop-rule review reads. */
  async stats(days = 14) {
    const rows = [];
    const listed = await this.storage.list({ prefix: "day:" });
    for (const [key, value] of listed) rows.push({ day: key.slice(4), ...value, usd: Math.round(value.usd * 1e4) / 1e4 });
    rows.sort((a, b) => (a.day < b.day ? 1 : -1));
    const month = await this._get(`month:${utcMonth(this.now())}`, 0);
    return { month_usd: Math.round(month * 1e4) / 1e4, days: rows.slice(0, days), limits: this.limits };
  }

  /** Drop the per-visitor and per-network rows of days before yesterday; the daily totals stay. */
  async prune() {
    const cutoff = utcDay(new Date(this.now().getTime() - 86_400_000));
    let removed = 0;
    for (const prefix of ["v:", "n:"]) {
      const listed = await this.storage.list({ prefix });
      const old = [...listed.keys()].filter((k) => k.split(":")[1] < cutoff);
      for (const k of old) await this.storage.delete(k);
      removed += old.length;
    }
    return removed;
  }
}
