import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'

const TABS = ['exchange', 'llm', 'trading', 'risk', 'advanced']
const TAB_ICONS = { exchange: '🔗', llm: '🤖', trading: '📈', risk: '🛡', advanced: '⚙️' }
const TAB_LABELS = { exchange: 'Exchange', llm: 'LLM Model', trading: 'Trading', risk: 'Risk Mgmt', advanced: 'Advanced' }

const TESTNET_URL = 'https://cdn-ind.testnet.deltaex.org'
const MAINNET_URL = 'https://api.india.delta.exchange'

const LLM_PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic Claude', models: ['claude-opus-4-5-20251101', 'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'] },
  { value: 'openai', label: 'OpenAI', models: ['gpt-4o', 'gpt-4o-mini', 'o1', 'o3-mini'] },
  { value: 'deepseek', label: 'DeepSeek', models: ['deepseek-chat', 'deepseek-reasoner', 'deepseek-v4-pro'] },
  { value: 'openrouter', label: 'OpenRouter', models: ['meta-llama/llama-3.3-70b-instruct', 'anthropic/claude-opus-4-5', 'google/gemini-2.0-flash'] },
  { value: 'openai_compat', label: 'OpenAI Compatible (Ollama, LM Studio…)', models: [] },
]

function ToggleInput({ id, value, onChange }) {
  const checked = value === 'true' || value === true || value === '1'
  return (
    <label className="toggle" htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked ? 'true' : 'false')}
      />
      <div className="toggle-track" />
      <div className="toggle-thumb" />
    </label>
  )
}

function SettingField({ setting, value, onChange }) {
  const isBool = setting.key.startsWith('ENABLE_') || setting.key.startsWith('THINKING_ENABLED')
  const isSecret = setting.is_secret

  if (isBool) return (
    <div className="toggle-wrap">
      <ToggleInput id={`s-${setting.key}`} value={value} onChange={onChange} />
      <span className="toggle-label">{value === 'true' ? 'Enabled' : 'Disabled'}</span>
    </div>
  )

  return (
    <input
      id={`s-${setting.key}`}
      type={isSecret ? 'password' : 'text'}
      className={`form-input${isSecret ? ' secret' : ''}`}
      value={value ?? ''}
      onChange={e => onChange(e.target.value)}
      placeholder={`Enter ${setting.label}`}
      autoComplete="off"
    />
  )
}

export default function Settings() {
  const [tab, setTab] = useState('exchange')
  const [settings, setSettings] = useState([])
  const [values, setValues] = useState({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [success, setSuccess] = useState('')
  const [error, setError] = useState('')
  const [currentEnv, setCurrentEnv] = useState('testnet')
  const [envSwitching, setEnvSwitching] = useState(false)

  const fetchSettings = useCallback(async () => {
    setLoading(true)
    try {
      const allData = await api.getSettings()
      // Filter out DB credentials from the UI
      const data = allData.filter(s => !s.key.startsWith('DB_'))

      setSettings(data)
      const map = {}
      data.forEach(s => { map[s.key] = s.value ?? '' })
      setValues(map)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchEnv = useCallback(async () => {
    try {
      const e = await api.getCurrentEnv()
      setCurrentEnv(e.environment)
    } catch { }
  }, [])

  useEffect(() => {
    fetchSettings()
    fetchEnv()
  }, [fetchSettings, fetchEnv])

  const handleSave = async () => {
    setSaving(true); setSuccess(''); setError('')
    try {
      const tabSettings = settings.filter(s => s.category === tab)
      const payload = tabSettings.map(s => ({ key: s.key, value: values[s.key] ?? '' }))
      await api.bulkUpdateSettings(payload)
      setSuccess('Settings saved and .env file updated!')
      setTimeout(() => setSuccess(''), 4000)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  const handleEnvSwitch = async () => {
    const target = currentEnv === 'testnet' ? 'mainnet' : 'testnet'
    setEnvSwitching(true); setError('')
    try {
      await api.switchEnv(target)
      setCurrentEnv(target)
      // Update local value too
      setValues(v => ({
        ...v,
        DELTA_BASE_URL: target === 'testnet' ? TESTNET_URL : MAINNET_URL,
      }))
      setSuccess(`Switched to ${target.toUpperCase()}!`)
      setTimeout(() => setSuccess(''), 3000)
    } catch (e) {
      setError(e.message)
    } finally {
      setEnvSwitching(false)
    }
  }

  const tabSettings = settings.filter(s => s.category === tab)

  // LLM provider-aware model suggestions
  const selectedProvider = values['LLM_PROVIDER'] || 'anthropic'
  const providerInfo = LLM_PROVIDERS.find(p => p.value === selectedProvider)

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
      <div className="spinner" style={{ width: 32, height: 32 }} />
    </div>
  )

  return (
    <div className="page">
      <div className="section-header">
        <div>
          <h1 className="section-title">Settings</h1>
          <p className="section-subtitle">Configure your trading agent — saved to MySQL and synced to .env</p>
        </div>
        <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? <><span className="spinner" /> Saving…</> : '💾 Save Changes'}
        </button>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}
      {success && <div className="alert alert-success">✅ {success}</div>}

      {/* Environment quick-switch */}
      {tab === 'exchange' && (
        <div className="env-switch" style={{ marginBottom: 24 }}>
          <div className="env-switch-label">
            <h3>Network Environment</h3>
            <p>Current: {currentEnv === 'mainnet' ? MAINNET_URL : TESTNET_URL}</p>
          </div>
          <span className={`env-pill ${currentEnv}`}>{currentEnv}</span>
          <label className="big-toggle" htmlFor="env-big-toggle" title={`Switch to ${currentEnv === 'testnet' ? 'Mainnet' : 'Testnet'}`}>
            <input
              id="env-big-toggle"
              type="checkbox"
              checked={currentEnv === 'mainnet'}
              onChange={handleEnvSwitch}
              disabled={envSwitching}
            />
            <div className="big-toggle-track" />
            <div className="big-toggle-thumb">{envSwitching ? '⏳' : currentEnv === 'mainnet' ? '🟢' : '🟡'}</div>
          </label>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'right' }}>
            <div style={{ color: 'var(--warning)', fontWeight: 600 }}>TESTNET</div>
            <div>↕ tap to switch</div>
            <div style={{ color: 'var(--success)', fontWeight: 600 }}>MAINNET</div>
          </div>
        </div>
      )}

      {/* LLM provider selector */}
      {tab === 'llm' && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
            🤖 LLM Provider Selection
          </h3>
          <div className="settings-grid">
            <div className="form-group">
              <label className="form-label" htmlFor="llm-provider-select">Provider</label>
              <select
                id="llm-provider-select"
                className="form-select"
                value={values['LLM_PROVIDER'] || ''}
                onChange={e => setValues(v => ({ ...v, LLM_PROVIDER: e.target.value }))}
              >
                <option value="">Select provider…</option>
                {LLM_PROVIDERS.map(p => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="llm-model-select">Model</label>
              {providerInfo?.models?.length > 0 ? (
                <select
                  id="llm-model-select"
                  className="form-select"
                  value={values['LLM_MODEL'] || ''}
                  onChange={e => setValues(v => ({ ...v, LLM_MODEL: e.target.value }))}
                >
                  <option value="">Select model…</option>
                  {providerInfo.models.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              ) : (
                <input
                  id="llm-model-select"
                  type="text"
                  className="form-input"
                  value={values['LLM_MODEL'] || ''}
                  onChange={e => setValues(v => ({ ...v, LLM_MODEL: e.target.value }))}
                  placeholder="Enter model name (e.g. llama3.2)"
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab bar */}
      <div className="tabs">
        {TABS.map(t => (
          <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {TAB_ICONS[t]} {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      {/* Settings fields */}
      <div className="card">
        <div className="settings-grid">
          {tabSettings.map(s => (
            <div key={s.key} className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" htmlFor={`s-${s.key}`}>
                {s.label}
                {s.is_secret && <span style={{ color: 'var(--warning)', marginLeft: 6, fontSize: 10 }}>🔒 SECRET</span>}
              </label>
              {s.description && <p className="form-description">{s.description}</p>}
              <SettingField
                setting={s}
                value={values[s.key] ?? ''}
                onChange={v => setValues(prev => ({ ...prev, [s.key]: v }))}
              />
            </div>
          ))}
        </div>

        <div style={{ marginTop: 24, paddingTop: 20, borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
          <button className="btn btn-secondary" onClick={fetchSettings}>↺ Reset</button>
          <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? <><span className="spinner" /> Saving…</> : '💾 Save Changes'}
          </button>
        </div>
      </div>
    </div>
  )
}
