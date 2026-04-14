import React, { useState } from 'react';
import { resultsApi } from '../services/api.js';
import { arrayToCSV, downloadString } from '../utils/formatters.js';

export default function DownloadManager({ jobId, results }) {
  const [downloading, setDownloading] = useState('');

  const downloadJSON = () => {
    if (!results) return;
    downloadString(JSON.stringify(results, null, 2), `results_${jobId}.json`, 'application/json');
  };

  const downloadPredictionsCSV = () => {
    if (!results?.predictions) return;
    const rows = results.predictions.yTrue.map((y, i) => ({
      actual: y,
      predicted: results.predictions.yPred[i],
      lower_95: results.predictions.yLower[i],
      upper_95: results.predictions.yUpper[i],
    }));
    downloadString(arrayToCSV(rows), `predictions_${jobId}.csv`, 'text/csv');
  };

  const downloadMetricsCSV = () => {
    if (!results?.foldMetrics) return;
    const rows = results.foldMetrics.map((m, i) => ({ fold: i + 1, ...m }));
    downloadString(arrayToCSV(rows), `metrics_${jobId}.csv`, 'text/csv');
  };

  const downloadFromServer = async (format) => {
    setDownloading(format);
    try {
      const url = resultsApi.getDownloadUrl(jobId, format);
      window.open(url, '_blank');
    } finally {
      setDownloading('');
    }
  };

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <span className="card-title">📥 Download Results</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '1.25rem',
          }}>
            <div style={{ fontWeight: 600, marginBottom: '0.4rem' }}>📊 Full Results (JSON)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Complete training results including hyperparameters, metrics, and all cross-validation data.
            </div>
            <button className="btn btn-primary btn-sm" onClick={downloadJSON}>
              ⬇ Download results.json
            </button>
          </div>

          <div style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '1.25rem',
          }}>
            <div style={{ fontWeight: 600, marginBottom: '0.4rem' }}>🎯 Predictions (CSV)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Cross-validation predictions with actual values and 95% prediction intervals.
            </div>
            <button className="btn btn-secondary btn-sm" onClick={downloadPredictionsCSV}>
              ⬇ Download predictions.csv
            </button>
          </div>

          <div style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '1.25rem',
          }}>
            <div style={{ fontWeight: 600, marginBottom: '0.4rem' }}>📋 Fold Metrics (CSV)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Per-fold RMSE, MAE, R², and MAPE values for all cross-validation folds.
            </div>
            <button className="btn btn-secondary btn-sm" onClick={downloadMetricsCSV}>
              ⬇ Download metrics.csv
            </button>
          </div>

          <div style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '1.25rem',
          }}>
            <div style={{ fontWeight: 600, marginBottom: '0.4rem' }}>🔗 Server-side Download</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Download results directly from the server storage.
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => downloadFromServer('json')}
                disabled={downloading === 'json'}
              >
                {downloading === 'json' ? <span className="spinner" /> : '⬇'} JSON
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => downloadFromServer('csv')}
                disabled={downloading === 'csv'}
              >
                {downloading === 'csv' ? <span className="spinner" /> : '⬇'} CSV
              </button>
            </div>
          </div>
        </div>

        {results && (
          <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
            <span>ℹ️</span>
            <div style={{ fontSize: '0.85rem' }}>
              <strong>Job ID:</strong>{' '}
              <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary-light)' }}>{jobId}</code>
              <br />
              Use this ID to load the model for predictions on the Predictions page.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
