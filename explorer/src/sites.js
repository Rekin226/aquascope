// Local counterpart of archive/catalog.py; shared fixtures pin the ranking rule.
export function groupStationSites(rows, allRows = rows, today = new Date().toISOString().slice(0, 10)) {
  const day = (value) => {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return NaN;
    const stamp = Date.parse(value);
    return Number.isFinite(stamp) && new Date(stamp).toISOString().slice(0, 10) === value ? stamp : NaN;
  };
  const span = (r) => {
    const days = (day(r.period_end || today) - day(r.period_start)) / 86400000;
    return Number.isFinite(days) ? Math.max(-1, days) : -1;
  };
  const compare = (a, b) => span(b) - span(a) || (a.station_id < b.station_id ? -1 : a.station_id > b.station_id ? 1 : 0);
  const groups = new Map();
  for (const row of rows) {
    const key = siteKey(row);
    if (!groups.has(key)) groups.set(key, { representative: row, records: [] });
    const group = groups.get(key);
    if (compare(row, group.representative) < 0) group.representative = row;
  }
  for (const row of allRows) groups.get(siteKey(row))?.records.push(row);
  return [...groups.values()].map(({ representative, records }) => ({
    ...representative, site_id: representative.site_id || representative.station_id,
    record_count: records.length, records: [...records].sort(compare),
  }));
}

export function siteKey(row) {
  return JSON.stringify([row.source, row.site_id || row.station_id]);
}

export function colocatedOffsets(rows) {
  const groups = new Map();
  for (const row of rows) {
    const coordinate = JSON.stringify([row.lon, row.lat]);
    if (!groups.has(coordinate)) groups.set(coordinate, []);
    groups.get(coordinate).push(row);
  }
  const offsets = new Map();
  for (const members of groups.values()) {
    members.sort((a, b) => `${a.source}/${a.station_id}`.localeCompare(`${b.source}/${b.station_id}`));
    const n = members.length;
    // At least 24 icon units between centres, including larger groups.
    const radius = n > 1 ? 12 / Math.sin(Math.PI / n) : 0;
    members.forEach((row, i) => {
      const angle = 2 * Math.PI * i / n;
      offsets.set(`${row.source}/${row.station_id}`, [radius * Math.cos(angle), radius * Math.sin(angle)]);
    });
  }
  return offsets;
}
