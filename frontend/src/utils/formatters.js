/**
 * Formatting utilities for the ANN-SIMS frontend.
 */

/**
 * Format a number to a fixed number of decimal places.
 * @param {number} value
 * @param {number} [decimals=4]
 * @returns {string}
 */
export function formatNumber(value, decimals = 4) {
  if (value === null || value === undefined || isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

/**
 * Format a metric value with auto-detected precision.
 */
export function formatMetric(value, metricKey = '') {
  if (value === null || value === undefined || isNaN(value)) return '—';
  const lk = metricKey.toLowerCase();
  if (lk.includes('mape')) return `${Number(value).toFixed(2)}%`;
  if (lk.includes('r2')) return Number(value).toFixed(4);
  return Number(value).toFixed(4);
}

/**
 * Format a percentage (0-1 range).
 */
export function formatPercent(value, decimals = 1) {
  if (value === null || value === undefined || isNaN(value)) return '—';
  return `${(Number(value) * 100).toFixed(decimals)}%`;
}

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * Format a date/timestamp.
 */
export function formatDate(dateString) {
  if (!dateString) return '—';
  return new Date(dateString).toLocaleString();
}

/**
 * Format a date showing only the date portion (locale-independent).
 */
export function formatDateShort(dateString) {
  if (!dateString) return '—';
  return new Date(dateString).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Format elapsed time in human-readable form.
 */
export function formatDuration(startIso, endIso) {
  if (!startIso) return '—';
  const start = new Date(startIso).getTime();
  const end = endIso ? new Date(endIso).getTime() : Date.now();
  const ms = end - start;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
}

/**
 * Format a job status to a display string.
 */
export function formatStatus(status) {
  const map = {
    queued: 'Queued',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
  };
  return map[status] || status || '—';
}

/**
 * Truncate a string to a maximum length.
 */
export function truncate(str, maxLength = 40) {
  if (!str) return '';
  return str.length > maxLength ? `${str.slice(0, maxLength)}…` : str;
}

/**
 * Convert an array of objects to a CSV string.
 */
export function arrayToCSV(data) {
  if (!data || data.length === 0) return '';
  const headers = Object.keys(data[0]);
  const rows = data.map((row) =>
    headers.map((h) => {
      const val = row[h];
      if (val === null || val === undefined) return '';
      if (typeof val === 'string' && val.includes(',')) return `"${val}"`;
      return val;
    }).join(',')
  );
  return [headers.join(','), ...rows].join('\n');
}

/**
 * Trigger a browser download of a string as a file.
 */
export function downloadString(content, filename, mimeType = 'text/plain') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
