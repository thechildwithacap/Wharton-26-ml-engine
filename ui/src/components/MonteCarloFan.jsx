import React from 'react'
import { pct } from '../format.js'

// Horizontal distribution "fan": p05..p95 range, p25..p75 interquartile band,
// median marker, and a 0% reference line — plus stat tiles.
export default function MonteCarloFan({ mc }) {
  if (!mc) return <div className="mut small">Not enough price history to simulate.</div>
  const p = mc.percentiles
  const lo = Math.min(p.p05, 0), hi = Math.max(p.p95, 0)
  const span = (hi - lo) || 1
  const W = 100
  const x = (v) => ((v - lo) / span) * W
  const zeroX = x(0)

  return (
    <div>
      <div className="tiles" style={{ marginBottom: 14 }}>
        <Tile k="Expected (median)" v={pct(mc.median_return)} cls={mc.median_return >= 0 ? 'good' : 'bad'} />
        <Tile k="Prob. of loss" v={pct(mc.prob_loss, 0)} cls={mc.prob_loss > 0.4 ? 'bad' : mc.prob_loss > 0.2 ? 'warn' : 'good'} />
        <Tile k="VaR 95%" v={pct(mc.var_95)} cls="bad" />
        <Tile k="CVaR 95%" v={pct(mc.cvar_95)} cls="bad" />
      </div>
      <div className="scroll">
        <svg viewBox="0 0 100 26" preserveAspectRatio="none" style={{ width: '100%', height: 90 }}>
          {/* p05-p95 range */}
          <rect x={x(p.p05)} y="8" width={x(p.p95) - x(p.p05)} height="10" rx="1.2"
            fill="rgba(79,157,255,.18)" stroke="rgba(79,157,255,.5)" strokeWidth="0.2" />
          {/* interquartile p25-p75 */}
          <rect x={x(p.p25)} y="8" width={x(p.p75) - x(p.p25)} height="10" rx="1.2" fill="rgba(79,157,255,.42)" />
          {/* median */}
          <line x1={x(p.p50)} x2={x(p.p50)} y1="6.5" y2="19.5" stroke="#e6edf6" strokeWidth="0.5" />
          {/* zero reference */}
          <line x1={zeroX} x2={zeroX} y1="5" y2="21" stroke="#8b9bb4" strokeWidth="0.4" strokeDasharray="1,0.8" />
        </svg>
      </div>
      <div className="small mut" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
        <span>{pct(p.p05)} <span style={{ opacity: .6 }}>(5th)</span></span>
        <span style={{ opacity: .8 }}>0%</span>
        <span>{pct(p.p95)} <span style={{ opacity: .6 }}>(95th)</span></span>
      </div>
      <div className="small mut" style={{ marginTop: 10 }}>
        {mc.n_sims.toLocaleString()} paths · {mc.horizon_days}-day horizon · {mc.method.replace('_', ' ')} ·
        median max-drawdown <b style={{ color: 'var(--bad)' }}>{pct(mc.max_drawdown_median)}</b>
        {mc.meta && mc.meta.annual_drift_override != null &&
          <> · drift override {pct(mc.meta.annual_drift_override)}/yr</>}
      </div>
    </div>
  )
}
function Tile({ k, v, cls }) {
  return <div className="tile"><div className="k">{k}</div><div className={`v ${cls || ''}`}>{v}</div></div>
}
