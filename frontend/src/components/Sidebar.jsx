import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const NAV = [
  { label: 'Dashboard',  icon: '📊', to: '/' },
  { label: 'Settings',   icon: '⚙️', to: '/settings' },
]

export default function Sidebar({ agentRunning }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">⚡</div>
        <h2>Trading Agent</h2>
        <p>Delta Exchange · LLM</p>
      </div>

      <nav className="sidebar-nav">
        <p className="nav-section-label">Navigation</p>
        {NAV.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span className="nav-item-icon">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}

        <p className="nav-section-label" style={{ marginTop: 24 }}>Status</p>
        <div style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className={`status-dot ${agentRunning ? 'running' : 'stopped'}`} />
          <span style={{ fontSize: 13, color: agentRunning ? 'var(--success)' : 'var(--text-muted)' }}>
            {agentRunning ? 'Agent Running' : 'Agent Stopped'}
          </span>
        </div>
      </nav>

      <div className="sidebar-bottom">
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, padding: '0 4px' }}>
          Signed in as <strong style={{ color: 'var(--text-secondary)' }}>{user?.username}</strong>
        </div>
        <button className="btn btn-secondary btn-sm w-full" onClick={handleLogout}>
          🚪 Sign Out
        </button>
      </div>
    </aside>
  )
}
