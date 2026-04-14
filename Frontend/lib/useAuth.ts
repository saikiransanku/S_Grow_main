"use client";

import { useState, useEffect } from "react";

export interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  token: string | null;
}

export function useAuth(): AuthState {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    isLoading: true,
    token: null,
  });

  useEffect(() => {
    // Check for token in localStorage
    const token = localStorage.getItem("token");

    setAuthState({
      isAuthenticated: !!token,
      isLoading: false,
      token: token,
    });
  }, []);

  return authState;
}

export function logout() {
  localStorage.removeItem("token");
  window.location.href = "/login";
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}
