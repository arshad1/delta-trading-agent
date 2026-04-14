import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchMe = useCallback(async () => {
    const token = localStorage.getItem('ta_token')
    if (!token) { setLoading(false); return }
    try {
      const me = await api.me()
      setUser(me)
    } catch {
      localStorage.removeItem('ta_token')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchMe() }, [fetchMe])

  const login = async (username, password) => {
    const data = await api.login(username, password)
    localStorage.setItem('ta_token', data.access_token)
    const me = await api.me()
    setUser(me)
    return me
  }

  const logout = () => {
    localStorage.removeItem('ta_token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
