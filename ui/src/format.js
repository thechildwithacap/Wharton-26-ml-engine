export const pct = (x, d = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(d)}%`
export const num = (x, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d)

// 0-100 score -> red→amber→green gradient color.
export function scoreColor(s) {
  if (s === null || s === undefined || Number.isNaN(s)) return '#28323f'
  const t = Math.max(0, Math.min(100, s)) / 100
  const stops = [[229, 97, 95], [224, 163, 78], [63, 178, 127]]
  const seg = t < 0.5 ? 0 : 1
  const f = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5
  const a = stops[seg], b = stops[seg + 1]
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}
