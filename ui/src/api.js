// Thin fetch helpers against the stdlib engine API.
const base = ''  // same origin (served by the Python server) or Vite proxy

export async function getIndices() {
  const r = await fetch(`${base}/api/indices`)
  if (!r.ok) throw new Error(`indices ${r.status}`)
  return r.json()
}

export async function getReport({ index, objective, risk, drift }) {
  const p = new URLSearchParams({ index, objective, risk: String(risk) })
  if (drift !== undefined && drift !== null && drift !== '') p.set('drift', String(drift))
  const r = await fetch(`${base}/api/report?${p.toString()}`)
  const data = await r.json()
  if (!r.ok || data.error) throw new Error(data.error || `report ${r.status}`)
  return data
}
