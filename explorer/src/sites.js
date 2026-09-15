// Site identity and screen offsets only; record ranking lives in Python.
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
