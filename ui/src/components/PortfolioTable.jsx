import React from 'react'
import { pct, num } from '../format.js'

export default function PortfolioTable({ holdings }) {
  const maxW = Math.max(...holdings.map(h => h.weight || 0), 0.0001)
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Ticker</th><th>Name</th><th>Sector</th>
            <th className="num">Weight</th><th style={{ width: 120 }}>Weight</th>
            <th className="num">Score</th><th className="num">Fit</th><th className="num">Risk</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h, i) => (
            <tr key={h.ticker}>
              <td className="mut">{i + 1}</td>
              <td><b>{h.ticker}</b></td>
              <td className="mut" style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis' }}>{h.name}</td>
              <td className="mut small">{h.sector}</td>
              <td className="num">{pct(h.weight, 1)}</td>
              <td><div className="wbar"><span style={{ width: `${(h.weight / maxW) * 100}%` }} /></div></td>
              <td className="num">{num(h.integrated_score, 0)}</td>
              <td className="num">{num(h.client_fit, 0)}</td>
              <td className="num">{num(h.risk_score, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
