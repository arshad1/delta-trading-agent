/**
 * API client — thin wrapper around fetch with JWT injection.
 */

const BASE = '/api'

function getToken() {
  return localStorage.getItem('ta_token')
}

function authHeaders() {
  const token = getToken()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function request(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: authHeaders(),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (res.status === 401) {
    localStorage.removeItem('ta_token')
    window.location.href = '/login'
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    let err = 'Request failed'
    try { const d = await res.json(); err = d.detail || JSON.stringify(d) } catch {}
    throw new Error(err)
  }
  const ct = res.headers.get('content-type') || ''
  return ct.includes('application/json') ? res.json() : res.text()
}

export const api = {
  // Auth
  login: (username, password) =>
    request('POST', '/auth/login', { username, password }),
  me: () => request('GET', '/auth/me'),
  changePassword: (current_password, new_password) =>
    request('POST', '/auth/change-password', { current_password, new_password }),

  // Settings
  getSettings: (category) =>
    request('GET', `/settings${category ? `?category=${category}` : ''}`),
  updateSetting: (key, value) =>
    request('PUT', `/settings/${key}`, { key, value }),
  bulkUpdateSettings: (settings) =>
    request('PUT', '/settings', { settings }),
  switchEnv: (environment) =>
    request('POST', '/settings/actions/switch-env', { environment }),
  getCurrentEnv: () =>
    request('GET', '/settings/actions/current-env'),

  // Agent
  getStatus: () => request('GET', '/agent/status'),
  startAgent: () => request('POST', '/agent/start'),
  stopAgent: () => request('POST', '/agent/stop'),
  getDiary: (limit = 100) => request('GET', `/agent/diary?limit=${limit}`),
  getDecisions: (limit = 50) => request('GET', `/agent/decisions?limit=${limit}`),
  getLogs: (lines = 200) => request('GET', `/agent/logs?lines=${lines}`),
}
