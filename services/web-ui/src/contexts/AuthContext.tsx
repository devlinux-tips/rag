import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
}

interface Tenant {
  id: string;
  slug: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  tenant: Tenant | null;
  language: string;
  features: string[];
  permissions: string[];
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_URL = import.meta.env.VITE_API_URL || '/api/v1';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [language, setLanguage] = useState<string>('hr');
  const [features, setFeatures] = useState<string[]>([]);
  const [permissions, setPermissions] = useState<string[]>([]);
  const [accessToken, setAccessToken] = useState<string | null>(
    () => localStorage.getItem('accessToken')
  );
  const [refreshToken, setRefreshToken] = useState<string | null>(
    () => localStorage.getItem('refreshToken')
  );
  const [isLoading, setIsLoading] = useState(true);

  // Fetch user info on mount if tokens exist
  useEffect(() => {
    if (accessToken) {
      fetchUserInfo();
    } else {
      setIsLoading(false);
    }
  }, [accessToken]);

  const fetchUserInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/user`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setTenant(data.tenant);
        setLanguage(data.language);
        setFeatures(data.features);
        setPermissions(data.permissions);
      } else {
        // Token invalid, clear auth
        clearAuth();
      }
    } catch (error) {
      console.error('Failed to fetch user info:', error);
      clearAuth();
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || 'Login failed');
    }

    const data = await response.json();
    const { user: userData, tenant: tenantData, tokens } = data.data;

    // Store tokens
    localStorage.setItem('accessToken', tokens.accessToken);
    localStorage.setItem('refreshToken', tokens.refreshToken);
    setAccessToken(tokens.accessToken);
    setRefreshToken(tokens.refreshToken);

    // Set user data (features and permissions will be fetched via /user endpoint)
    setUser(userData);
    setTenant(tenantData);
    setLanguage(userData.language);
  };

  const register = async (email: string, password: string, name: string) => {
    const response = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, name }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || 'Registration failed');
    }

    const data = await response.json();
    const { user: userData, tenant: tenantData, tokens } = data.data;

    // Store tokens
    localStorage.setItem('accessToken', tokens.accessToken);
    localStorage.setItem('refreshToken', tokens.refreshToken);
    setAccessToken(tokens.accessToken);
    setRefreshToken(tokens.refreshToken);

    // Set user data
    setUser(userData);
    setTenant(tenantData);
    setLanguage('hr'); // Default language
  };

  const logout = async () => {
    try {
      if (refreshToken) {
        await fetch(`${API_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ refreshToken }),
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      clearAuth();
    }
  };

  const clearAuth = () => {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    setAccessToken(null);
    setRefreshToken(null);
    setUser(null);
    setTenant(null);
    setLanguage('hr');
    setFeatures([]);
    setPermissions([]);
  };

  const value: AuthContextType = {
    user,
    tenant,
    language,
    features,
    permissions,
    accessToken,
    refreshToken,
    isAuthenticated: !!user && !!accessToken,
    isLoading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}