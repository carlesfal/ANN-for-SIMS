import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor
api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const msg = error.response?.data?.error || error.message || 'Request failed';
    return Promise.reject(new Error(msg));
  }
);

// Training API
export const trainingApi = {
  startTraining: (formData) =>
    api.post('/training/start', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    }),

  getStatus: (jobId) => api.get(`/training/${jobId}/status`),

  cancelJob: (jobId) => api.delete(`/training/${jobId}`),

  listJobs: () => api.get('/training'),
};

// Predictions API
export const predictionsApi = {
  batchPredict: (formData) =>
    api.post('/predictions/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    }),

  singlePredict: (data) => api.post('/predictions/single', data),
};

// Results API
export const resultsApi = {
  getResults: (jobId) => api.get(`/results/${jobId}`),

  getPlots: (jobId) => api.get(`/results/${jobId}/plots`),

  getDownloadUrl: (jobId, format = 'json') =>
    `${BASE_URL}/api/results/${jobId}/download?format=${format}`,
};

// Template API
export const templateApi = {
  downloadUrl: ({ labels = 1, inputs = 3, outputs = 1 } = {}) =>
    `${BASE_URL}/api/template/download?labels=${labels}&inputs=${inputs}&outputs=${outputs}`,
};

// Health check
export const healthApi = {
  check: () => axios.get(`${BASE_URL}/health`).then((r) => r.data),
};

// WebSocket helper
export function createWebSocket(jobId, onMessage, onError) {
  const wsBase = BASE_URL
    ? BASE_URL.replace(/^http/, 'ws')
    : `ws://${window.location.host}`;

  const ws = new WebSocket(`${wsBase}/ws`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'subscribe', jobId }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      // Ignore parse errors
    }
  };

  ws.onerror = (err) => {
    if (onError) onError(err);
  };

  return ws;
}

export default api;
