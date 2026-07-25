import React, { useEffect, useState, useCallback } from 'react'
import { getIndices, getReport } from './api.js'
import { pct } from './format.js'
import MonteCarloFan from './components/MonteCarloFan.jsx'
import PortfolioTable from './components/PortfolioTable.jsx'
import FactorHeatmap from './components/FactorHeatmap.jsx'
import RiskPanel from './components/RiskPanel.jsx'

const OBJECTIVES = ['balanced', 'growth', 'income', 'capital_preservation']
const RISK_LABELS = { 1: 'very low', 2: 'low', 3: 'moderate', 4: 'high', 5: 'very high' }

export default function App() {
  const [meta, setMeta] = useState(null)
  const [index, setIndex] = useState('SPX')
  const [objective, setObjective] = useState('growth')
  const [risk, setRisk] = useState(4)
  const [drift, setDrift] = useState('')
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getIndices().then(setMeta).catch(e => setError(e.message))
  }, [])

  const load = useCallback(() => {
    setLoading(true); setError(null)
    getReport({ index, objective, risk, drift })
      .then(r => { setReport(r); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [index, objective, risk, drift])

  useEffect(() => { if (meta) load() }, [meta, load])

  const idxInfo = meta && meta.indices.find(i => i.id === index)

  return (
    <div className="wrap">
      <div className="masthead">
        <div>
          <h1>Wharton ML Investment Engine</h1>
          <div className="sub">
            Multi-factor decision support{meta ? ` · data as of ${meta.as_of} · ${meta.dataset}` : ''}
          </div>
        </div>
        {report && (
          <div style={{ textAlign: 'right' }}>
            <span className={`badge ${report.decision.action}`}>{report.decision.action}</span>
            <div className="sub" style={{ marginTop: 4 }}>
              regime: <b>{report.regime.label}</b>
            </div>
          </div>
        )}
      </div>

      <div className="controls">
        <div className="field">
          <label>Index / universe</label>
          <select value={index} onChange={e => setIndex(e.target.value)}>
            {meta && meta.indices.map(i =>
              <option key={i.id} value={i.id}>{i.name} ({i.n_available})</option>)}
          </select>
        </div>
        <div className="field">
          <label>Objective</label>
          <select value={objective} onChange={e => setObjective(e.target.value)}>
            {OBJECTIVES.map(o => <option key={o} value={o}>{o.replace('_', ' ')}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Risk tolerance</label>
          <select value={risk} onChange={e => setRisk(Number(e.target.value))}>
            {[1, 2, 3, 4, 5].map(r => <option key={r} value={r}>{r} — {RISK_LABELS[r]}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Expected return / yr (optional)</label>
          <input type="number" step="0.01" placeholder="e.g. 0.08" value={drift}
            onChange={e => setDrift(e.target.value)} />
        </div>
        <button className="chip" style={{ cursor: 'pointer', padding: '9px 16px' }} onClick={load}>
          {loading ? 'Running…' : 'Run engine'}
        </button>
      </div>

      {error && <div className="err">Error: {error}</div>}
      {!report && !error && <div className="loading">Loading engine…</div>}

      {report && (
        <div className="grid">
          <div className="card col-8">
            <h2>Decision</h2>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
              <span className={`badge ${report.decision.action}`}>{report.decision.action}</span>
              <span className="chip">{report.n_holdings} holdings / {report.n_universe} in {index}</span>
              <span className="chip">client: {report.client.objective} · {report.client.risk} risk</span>
              <span className="chip">net score {report.decision.net_score_improvement >= 0 ? '+' : ''}{report.decision.net_score_improvement?.toFixed(2)}</span>
            </div>
            <ul className="reasons">
              {report.decision.reasons.slice(0, 5).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
            <div className="small mut" style={{ marginTop: 8 }}>
              Style tilt:{' '}
              {Object.entries(report.style_weights).sort((a, b) => b[1] - a[1])
                .filter(([, w]) => w > 0.01)
                .map(([k, w]) => `${k} ${pct(w, 0)}`).join(' · ')}
            </div>
          </div>

          <div className="card col-4">
            <h2>Portfolio risk</h2>
            <RiskPanel risk={report.risk} ml={report.ml_metrics} confidence={report.signal_confidence} />
          </div>

          <div className="card col-6">
            <h2>Monte Carlo — {report.monte_carlo ? `${report.monte_carlo.horizon_days}-day outcome range` : 'outcome range'}</h2>
            <MonteCarloFan mc={report.monte_carlo} />
          </div>

          <div className="card col-6">
            <h2>Factor scorecard (top holdings)</h2>
            <FactorHeatmap holdings={report.holdings} columns={report.factor_columns} />
          </div>

          <div className="card col-12">
            <h2>Recommended portfolio</h2>
            <PortfolioTable holdings={report.holdings} />
          </div>

          {report.concentration_alerts.length > 0 && (
            <div className="card col-12">
              <h2>Alerts</h2>
              <ul className="reasons">
                {report.concentration_alerts.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="sub" style={{ marginTop: 28, textAlign: 'center' }}>
        Research / decision-support tool. Not investment advice.
      </div>
    </div>
  )
}
