// The 10,000 km² threshold is a screening judgment for point forcing.
export const AREA_WARN_KM2 = 10000;

export function confirmLargeArea(area, warning, message, proceed) {
  if (area <= AREA_WARN_KM2) return { result: Promise.resolve(true), cancel: () => {} };
  message.textContent = `This catchment is ${area.toLocaleString()} km². ` +
    "Forcing is taken at the gauge, not averaged over the catchment, so the fit may be unreliable " +
    "above the 10,000 km² screening threshold (a judgment, not a fixed physical limit). Run it anyway? ";
  warning.hidden = false;
  let finish;
  const result = new Promise((resolve) => {
    finish = (answer) => { warning.hidden = true; proceed.onclick = null; resolve(answer); };
    proceed.onclick = () => finish(true);
  });
  return { result, cancel: () => finish(false) };
}
