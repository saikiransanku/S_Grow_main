import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2000/api";

export const apiClient = axios.create({
  baseURL: API_URL,
});

// Add token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  },
);

export const api = {
  auth: {
    register: (data: any) => apiClient.post("/auth/register", data),
    login: (data: any) => apiClient.post("/auth/login", data),
  },
  users: {
    getAll: () => apiClient.get("/users"),
    getMe: () => apiClient.get("/users/me"),
    getById: (id: string) => apiClient.get(`/users/${id}`),
    updateMe: (data: any) => apiClient.put("/users/me", data),
    update: (id: string, data: any) => apiClient.put(`/users/${id}`, data),
  },
  laws: {
    getAll: () => apiClient.get("/laws"),
    getById: (id: string) => apiClient.get(`/laws/${id}`),
    getByCategory: (category: string) =>
      apiClient.get(`/laws/category/${category}`),
    create: (data: any) => apiClient.post("/laws", data),
  },
  history: {
    getUserHistory: (userId: string) =>
      apiClient.get(`/history/user/${userId}`),
    logAction: (data: any) => apiClient.post("/history", data),
  },
};

export default apiClient;
