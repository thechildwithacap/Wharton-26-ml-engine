import React from 'react'
import { pct, num } from '../format.js'

export default function RiskPanel({ risk, ml, confidence }) {
  const sectors = Object.entries(risk.sector_exposure || {}).sort((a, b) => b[1] - a[1])
  const maxSec = Math.max(...sectors.map(s => s[1]), 0.0001)
  return (
    <div>
      <div className="tiles" style={{ marginBottom: 14 }}>
        <Tile k="Volatility" v={pct(risk.volatility)} />
        <Tile k="Beta" v={num(risk.beta, 2)} />
        <Tile k="Top-5 weight" v={pct(risk.top5_weight, 0)} />
        <Tile k="Effective N" v={num(risk.effective_n, 1)} />
      </div>
      {ml && (
        <div className="small mut" style={{ marginBottom: 12 }}>
          ML alpha (out-of-sample): IC <b style={{ color: 'var(--ink)' }}>{num(ml.mean_ic, 3)}</b>
          {' '}(t={num(ml.ic_t_stat, 2)}, hit-rate {pct(ml.hit_rate, 0)}) · signal confidence {num(confidence, 2)}
        </div>
      )}
      <div className="small mut" style={{ marginBottom: 6 }}>Sector exposure</div>
      {sectors.map(([s, w]) => (
        <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '4px 0' }}>
          <div style={{ width: 130 }} className="small">{s}</div>
          <div className="wbar" style={{ flex: 1 }}><span style={{ width: `${(w / maxSec) * 100}%` }} /></div>
          <div className="small num" style={{ width: 44 }}>{pct(w, 0)}</div>
        </div>
      ))}
    </div>
  )
}
function Tile({ k, v }) {
  return <div className="tile"><div className="k">{k}</div><div className="v">{v}</div></div>
}
