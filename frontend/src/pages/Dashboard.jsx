import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../api/client'

function StatCard({ icon, label, value, valueClass, bg }) {
  return (
    <div className="card" style={{ background: bg }}>
      <div className="card-header">
        <span className="card-title">{label}</span>
        <div className="card-icon" style={{ background: 'rgba(255,255,255,0.06)' }}>{icon}</div>
      </div>
      <div className={`card-value ${valueClass || ''}`}>{value ?? '—'}</div>
    </div>
  )
}

function fmt(n, decimals = 2) {
  if (n == null) return '—'
  return Number(n).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function timeSince(iso) {
  if (!iso) return '—'
  const diff = Math.floor((Date.now() - new Date(iso)) / 1000)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

export default function Dashboard({ onAgentStatusChange }) {
  const [status, setStatus] = useState(null)
  const [diary, setDiary] = useState([])
  const [decisions, setDecisions] = useState([])
  const [logs, setLogs] = useState('')
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('decisions')
  const logViewerRef = useRef(null)

  // Auto-scroll log viewer to bottom when new logs arrive or when switching to the logs tab
  useEffect(() => {
    if (logViewerRef.current) {
      logViewerRef.current.scrollTop = logViewerRef.current.scrollHeight
    }
  }, [logs, activeTab])

  const fetchAll = useCallback(async () => {
    try {
      const [s, d, dec, l] = await Promise.all([
        api.getStatus(),
        api.getDiary(50),
        api.getDecisions(30),
        api.getLogs(150),
      ])
      setStatus(s)
      setDiary(d.entries || [])
      setDecisions(dec.entries || [])
      setLogs(l.content || '')
      onAgentStatusChange?.(s.running)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [onAgentStatusChange])

  useEffect(() => {
    fetchAll()
    const id = setInterval(fetchAll, 10000)
    return () => clearInterval(id)
  }, [fetchAll])

  const handleStartStop = async () => {
    setActionLoading(true)
    setError('')
    try {
      if (status?.running) { await api.stopAgent() } else { await api.startAgent() }
      await fetchAll()
    } catch (e) {
      setError(e.message)
    } finally {
      setActionLoading(false)
    }
  }

  const handleClearLogs = async () => {
    setActionLoading(true)
    setError('')
    try {
      await api.clearLogs()
      setLogs('')
    } catch (e) {
      setError(e.message)
    } finally {
      setActionLoading(false)
    }
  }

  const handleClearDecisions = async () => {
    setActionLoading(true)
    setError('')
    try {
      await api.clearDecisions()
      setDecisions([])
    } catch (e) {
      setError(e.message)
    } finally {
      setActionLoading(false)
    }
  }

  const handleClearDiary = async () => {
    setActionLoading(true)
    setError('')
    try {
      await api.clearDiary()
      setDiary([])
    } catch (e) {
      setError(e.message)
    } finally {
      setActionLoading(false)
    }
  }

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
      <div className="spinner" style={{ width: 32, height: 32 }} />
    </div>
  )

  const uptimeStr = status?.uptime_seconds
    ? `${Math.floor(status.uptime_seconds / 60)}m ${Math.floor(status.uptime_seconds % 60)}s`
    : '—'

  const latestDecision = decisions[decisions.length - 1]

  return (
    <div className="page">
      <div className="section-header">
        <div>
          <h1 className="section-title">Dashboard</h1>
          <p className="section-subtitle">Live trading agent overview — refreshes every 10s</p>
        </div>
        <button
          id="agent-toggle-btn"
          className={`btn ${status?.running ? 'btn-danger' : 'btn-success'} btn-lg`}
          onClick={handleStartStop}
          disabled={actionLoading}
        >
          {actionLoading
            ? <><span className="spinner" /> Working…</>
            : status?.running ? '⏹ Stop Agent' : '▶ Start Agent'
          }
        </button>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}

      {/* Agent status bar */}
      <div className="agent-status-bar">
        <div className={`status-dot ${status?.running ? 'running' : 'stopped'}`} />
        <div style={{ flex: 1 }}>
          <span style={{ fontWeight: 600, color: status?.running ? 'var(--success)' : 'var(--text-muted)' }}>
            {status?.running ? 'Agent Running' : 'Agent Stopped'}
          </span>
          {status?.pid && <span style={{ color: 'var(--text-muted)', fontSize: 12, marginLeft: 8 }}>PID {status.pid}</span>}
        </div>
        <span className={`env-pill ${status?.environment}`}>{status?.environment}</span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {status?.llm_provider} · {status?.llm_model}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Uptime: {uptimeStr}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Last cycle: {timeSince(status?.last_cycle)}
        </span>
      </div>

      {/* Stats */}
      <div className="stat-grid">
        <StatCard icon="💰" label="Environment" value={status?.environment?.toUpperCase()} />
        <StatCard icon="🤖" label="LLM Provider" value={status?.llm_provider} />
        <StatCard icon="📐" label="LLM Model" value={status?.llm_model} />
        <StatCard icon="⏱" label="Uptime" value={uptimeStr} />
      </div>

      {/* Data tabs */}
      <div className="tabs">
        {['decisions', 'diary', 'logs'].map(t => (
          <button key={t} className={`tab ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t === 'decisions' ? '🎯 Decisions' : t === 'diary' ? '📓 Trade Diary' : '🖥 Logs'}
          </button>
        ))}
      </div>

      {activeTab === 'decisions' && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent LLM Decisions ({decisions.length})</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={fetchAll}>↻ Refresh</button>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleClearDecisions}
                disabled={actionLoading}
              >
                {actionLoading ? <><span className="spinner" /> Working…</> : 'Clear Decisions'}
              </button>
            </div>
          </div>
          {decisions.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No decisions yet. Start the agent to begin trading.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {[...decisions].reverse().map((d, i) => (
                <div key={i} style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 16px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {d.timestamp ? new Date(d.timestamp).toLocaleString() : '—'}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Cycle #{d.cycle}</span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      Balance: <strong style={{ color: 'var(--text-primary)' }}>${fmt(d.account_value)}</strong>
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {(d.decisions || []).map((dec, j) => (
                      <span key={j} style={{
                        display: 'inline-flex', alignItems: 'center', gap: 4,
                        background: 'var(--bg-card)', border: '1px solid var(--border)',
                        borderRadius: 6, padding: '3px 10px', fontSize: 12
                      }}>
                        <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{dec.asset}</span>
                        <span className={`badge badge-${dec.action}`}>{dec.action}</span>
                        {dec.allocation_usd > 0 && (
                          <span style={{ color: 'var(--text-muted)' }}>${fmt(dec.allocation_usd, 0)}</span>
                        )}
                      </span>
                    ))}
                  </div>
                  {d.reasoning && (
                    <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.5 }}>
                      {d.reasoning.slice(0, 300)}{d.reasoning.length > 300 ? '…' : ''}
                    </p>
                  )}
                  {d.thinking && (
                    <div style={{
                      marginTop: 10,
                      padding: '8px 12px',
                      background: 'rgba(0,0,0,0.2)',
                      borderLeft: '2px solid var(--primary)',
                      fontSize: 11,
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-mono)',
                      maxHeight: 120,
                      overflowY: 'auto'
                    }}>
                      <div style={{ textTransform: 'uppercase', fontSize: 9, fontWeight: 700, marginBottom: 4, opacity: 0.5 }}>Internal Thinking</div>
                      {d.thinking}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'diary' && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Trade Diary ({diary.length})</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={fetchAll}>↻ Refresh</button>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleClearDiary}
                disabled={actionLoading}
              >
                {actionLoading ? <><span className="spinner" /> Working…</> : 'Clear Diary'}
              </button>
            </div>
          </div>
          {diary.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No diary entries yet.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Asset</th>
                    <th>Action</th>
                    <th>Allocation</th>
                    <th>Entry Price</th>
                    <th>TP</th>
                    <th>SL</th>
                    <th>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {[...diary].reverse().map((e, i) => (
                    <tr key={i}>
                      <td className="font-mono text-xs text-muted">
                        {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—'}
                      </td>
                      <td style={{ fontWeight: 700 }}>{e.asset}</td>
                      <td><span className={`badge badge-${e.action}`}>{e.action}</span></td>
                      <td>{e.allocation_usd ? `$${fmt(e.allocation_usd, 0)}` : '—'}</td>
                      <td>{e.entry_price ? `$${fmt(e.entry_price)}` : '—'}</td>
                      <td style={{ color: 'var(--success)' }}>{e.tp_price ? `$${fmt(e.tp_price)}` : '—'}</td>
                      <td style={{ color: 'var(--danger)' }}>{e.sl_price ? `$${fmt(e.sl_price)}` : '—'}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 12, maxWidth: 200 }}>
                        {(e.rationale || '').slice(0, 80)}{(e.rationale || '').length > 80 ? '…' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Agent Process Logs</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={fetchAll}>↻ Refresh</button>
              <button
                className="btn btn-danger btn-sm"
                onClick={handleClearLogs}
                disabled={actionLoading}
              >
                {actionLoading ? <><span className="spinner" /> Working…</> : 'Clear Logs'}
              </button>
            </div>
          </div>
          <div className="log-viewer" ref={logViewerRef}>{logs || 'No logs yet. Start the agent to see output here.'}</div>
        </div>
      )}
    </div>
  )
}
