import React from 'react'
import { num, scoreColor } from '../format.js'

const LABEL = {
  value: 'Value', quality: 'Quality', growth: 'Growth', garp: 'GARP',
  discipline: 'Discip.', momentum: 'Mom.', low_vol: 'LowVol', size: 'Size',
  factor: 'Factor', macro_tilt: 'MacTilt', macro_fit: 'MacFit', analyst: 'Analyst',
}

// Holdings × factor-score heatmap (0-100, red→amber→green).
export default function FactorHeatmap({ holdings, columns, limit = 15 }) {
  const rows = holdings.slice(0, limit)
  return (
    <div>
      <div className="scroll">
        <table className="heat">
          <thead>
            <tr>
              <th>Ticker</th>
              {columns.map(c => <th key={c} className="num" style={{ textAlign: 'center' }}>{LABEL[c] || c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map(h => (
              <tr key={h.ticker}>
                <td><b>{h.ticker}</b></td>
                {columns.map(c => {
                  const s = h.factors ? h.factors[c] : null
                  return <td key={c} className="cell" style={{ background: scoreColor(s) }}>{num(s, 0)}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        <span>weak</span>
        <span className="sw" style={{ background: scoreColor(10) }} />
        <span className="sw" style={{ background: scoreColor(50) }} />
        <span className="sw" style={{ background: scoreColor(90) }} />
        <span>strong (0–100 cross-sectional percentile)</span>
      </div>
    </div>
  )
}
