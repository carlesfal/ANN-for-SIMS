import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { trainingApi, createWebSocket } from '../services/api.js';
import { formatDuration, formatMetric } from '../utils/formatters.js';

export default function TrainingMonitor({ jobId, onReset }) {
  const [jobStatus, setJobStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  const pollRef = useRef(null);

  const updateFromData = useCallback((data) => {
    setJobStatus((prev) => ({ ...prev, ...data }));
    if (data.logs?.length) {
      setLogs((prev) => {
        const combined = [...prev, ...data.logs];
        return combined.slice(-200);
      });
    }
  }, []);

  // WebSocket connection
  useEffect(() => {
    if (!jobId) return;

    // Initial poll
    trainingApi.getStatus(jobId).then((data) => {
      setJobStatus(data);
      setLogs(data.logs || []);
    }).catch(console.error);

    // Try WebSocket
    try {
      const ws = createWebSocket(
        jobId,
        (msg) => {
          if (msg.type === 'job_updated' && msg.jobId === jobId) {
            updateFromData(msg.data);
          }
        },
        () => setWsConnected(false)
      );
      wsRef.current = ws;
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }

    // Fallback polling
    pollRef.current = setInterval(async () => {
      try {
        const data = await trainingApi.getStatus(jobId);
        setJobStatus(data);
        if (data.logs?.length) setLogs(data.logs);
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
          clearInterval(pollRef.current);
        }
      } catch (e) {
        // ignore
      }
    }, 3000);

    return () => {
      wsRef.current?.close();
      clearInterval(pollRef.current);
    };
  }, [jobId, updateFromData]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  if (!jobStatus) {
    return (
      <div className="card flex-center" style={{ padding: '3rem' }}>
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <span style={{ marginLeft: '1rem', color: 'var(--text-secondary)' }}>Connecting...</span>
      </div>
    );
  }

  const { status, progress = 0, currentFold, currentEpoch, metrics, statusMessage, startedAt, completedAt, error: jobError } = jobStatus;
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(status);

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <span className="card-title">📡 Training Monitor</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '0.75rem', color: wsConnected ? 'var(--secondary)' : 'var(--text-muted)' }}>
              {wsConnected ? '● Live' : '○ Polling'}
            </span>
            <span className={`badge badge-${status}`}>{status}</span>
          </div>
        </div>

        <div className="grid grid-2" style={{ marginBottom: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Job ID</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {jobId}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Elapsed</div>
            <div style={{ fontWeight: 600 }}>{formatDuration(startedAt, completedAt)}</div>
          </div>
          {currentFold && (
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Current Fold</div>
              <div style={{ fontWeight: 600 }}>Fold {currentFold} / {jobStatus.config?.folds || '?'}</div>
            </div>
          )}
          {statusMessage && (
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Status</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{statusMessage}</div>
            </div>
          )}
        </div>

        {/* Progress bar */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Progress</span>
            <span style={{ fontWeight: 700, color: 'var(--primary-light)' }}>{Math.round(progress)}%</span>
          </div>
          <div className="progress-bar-wrapper">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>

        {/* Live metrics */}
        {metrics && (
          <div className="grid grid-4" style={{ marginBottom: '1.5rem' }}>
            {['rmse', 'mae', 'r2', 'mape'].map((key) =>
              metrics[key] !== undefined ? (
                <div className="metric-card" key={key}>
                  <div className="metric-value" style={{ fontSize: '1.4rem' }}>
                    {formatMetric(metrics[key], key)}
                  </div>
                  <div className="metric-label">{key.toUpperCase()}</div>
                </div>
              ) : null
            )}
          </div>
        )}

        {/* Error */}
        {jobError && (
          <div className="alert alert-error" style={{ marginBottom: '1rem' }}>
            <span>⚠️</span>
            <span>{jobError}</span>
          </div>
        )}

        {/* Actions */}
        {isTerminal && (
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
            {status === 'completed' && (
              <Link to={`/results/${jobId}`} className="btn btn-success">
                📊 View Results
              </Link>
            )}
            <button className="btn btn-secondary" onClick={onReset}>
              🔄 New Training Job
            </button>
          </div>
        )}
      </div>

      {/* Log panel */}
      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="card-header">
          <span className="card-title">🖥️ Logs</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{logs.length} entries</span>
        </div>
        <div className="log-panel">
          {logs.length === 0 && (
            <div style={{ color: 'var(--text-muted)' }}>Waiting for output...</div>
          )}
          {logs.map((log, i) => (
            <div key={i} className={`log-entry${log.type === 'error' ? ' log-error' : ''}`}>
              <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
              <span className="log-msg">{log.message}</span>
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
}
