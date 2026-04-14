import React, { useState } from 'react';
import DataUploader from './DataUploader.jsx';
import { predictionsApi, trainingApi } from '../services/api.js';
import { formatNumber } from '../utils/formatters.js';

export default function PredictionInterface({ mode }) {
  const [file, setFile] = useState(null);
  const [modelJobId, setModelJobId] = useState('');
  const [targetColumn, setTargetColumn] = useState('target');
  const [featureColumns, setFeatureColumns] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // For single prediction
  const [singleFeatures, setSingleFeatures] = useState('');
  const [singleResult, setSingleResult] = useState(null);

  const handleBatchPredict = async (e) => {
    e.preventDefault();
    if (!file) { setError('Please upload a data file'); return; }
    if (!modelJobId.trim()) { setError('Please enter a Model Job ID'); return; }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('dataFile', file);
      formData.append('modelJobId', modelJobId.trim());
      formData.append('targetColumn', targetColumn);
      if (featureColumns) formData.append('featureColumns', featureColumns);

      const data = await predictionsApi.batchPredict(formData);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSinglePredict = async (e) => {
    e.preventDefault();
    if (!modelJobId.trim()) { setError('Please enter a Model Job ID'); return; }

    let features;
    try {
      features = JSON.parse(singleFeatures);
    } catch (err) {
      setError('Invalid JSON for features. Example: {"feature1": 1.5, "feature2": 3.2}');
      return;
    }

    setLoading(true);
    setError('');
    setSingleResult(null);

    try {
      const data = await predictionsApi.singlePredict({ modelJobId: modelJobId.trim(), features });
      setSingleResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {error && (
        <div className="alert alert-error" style={{ marginBottom: '1rem' }}>
          <span>⚠️</span><span>{error}</span>
        </div>
      )}

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header"><span className="card-title">🔑 Model Selection</span></div>
        <div className="form-group">
          <label className="form-label">Model Job ID</label>
          <input
            type="text"
            className="form-control"
            value={modelJobId}
            placeholder="Paste the Job ID from a completed training run"
            onChange={(e) => setModelJobId(e.target.value)}
          />
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Find your Job ID in the Results page or from the training job output.
          </div>
        </div>
        <div className="grid grid-2">
          <div className="form-group">
            <label className="form-label">Target Column</label>
            <input type="text" className="form-control" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">Feature Columns (optional)</label>
            <input
              type="text"
              className="form-control"
              value={featureColumns}
              placeholder="col1, col2 (blank = all)"
              onChange={(e) => setFeatureColumns(e.target.value)}
            />
          </div>
        </div>
      </div>

      {mode === 'batch' && (
        <form onSubmit={handleBatchPredict}>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="card-header"><span className="card-title">📦 Upload Prediction Data</span></div>
            <DataUploader onFileSelected={setFile} />
          </div>

          <button type="submit" className="btn btn-primary btn-lg" disabled={loading || !file || !modelJobId}>
            {loading ? <><span className="spinner" /> &nbsp;Running...</> : '🔍 Run Batch Prediction'}
          </button>

          {results && (
            <div className="card" style={{ marginTop: '1.5rem' }}>
              <div className="card-header">
                <span className="card-title">📊 Prediction Results</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{results.results?.n_samples} samples</span>
              </div>
              {results.results?.metrics && (
                <div className="grid grid-4" style={{ marginBottom: '1rem' }}>
                  {Object.entries(results.results.metrics).map(([k, v]) => (
                    <div className="metric-card" key={k}>
                      <div className="metric-value" style={{ fontSize: '1.3rem' }}>{formatNumber(v)}</div>
                      <div className="metric-label">{k.toUpperCase()}</div>
                    </div>
                  ))}
                </div>
              )}
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      {results.results?.feature_columns?.map((c) => <th key={c}>{c}</th>)}
                      <th>Prediction</th>
                      <th>Lower 95%</th>
                      <th>Upper 95%</th>
                      {results.results?.predictions?.[0]?.actual !== undefined && <th>Actual</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {results.results?.predictions?.slice(0, 100).map((row, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                        {results.results.feature_columns?.map((c) => (
                          <td key={c}>{formatNumber(row[c], 3)}</td>
                        ))}
                        <td style={{ fontWeight: 600, color: 'var(--primary-light)' }}>{formatNumber(row.prediction)}</td>
                        <td style={{ color: 'var(--text-secondary)' }}>{formatNumber(row.lower_95)}</td>
                        <td style={{ color: 'var(--text-secondary)' }}>{formatNumber(row.upper_95)}</td>
                        {row.actual !== undefined && <td>{formatNumber(row.actual)}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {results.results?.predictions?.length > 100 && (
                  <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                    Showing first 100 rows of {results.results.predictions.length}
                  </div>
                )}
              </div>
            </div>
          )}
        </form>
      )}

      {mode === 'single' && (
        <form onSubmit={handleSinglePredict}>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="card-header"><span className="card-title">🔬 Single Sample Features</span></div>
            <div className="form-group">
              <label className="form-label">Feature Values (JSON)</label>
              <textarea
                className="form-control"
                rows={5}
                value={singleFeatures}
                placeholder={'{\n  "feature1": 1.5,\n  "feature2": 3.2,\n  "feature3": 0.8\n}'}
                onChange={(e) => setSingleFeatures(e.target.value)}
                style={{ fontFamily: 'var(--font-mono)', resize: 'vertical' }}
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading || !modelJobId || !singleFeatures}>
              {loading ? <><span className="spinner" /> &nbsp;Predicting...</> : '🔍 Predict'}
            </button>
          </div>

          {singleResult && (
            <div className="card">
              <div className="card-header"><span className="card-title">🎯 Prediction Result</span></div>
              <div className="grid grid-3" style={{ marginBottom: '1rem' }}>
                <div className="metric-card">
                  <div className="metric-value">{formatNumber(singleResult.prediction)}</div>
                  <div className="metric-label">Prediction</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: '1.3rem' }}>{formatNumber(singleResult.lower_95)}</div>
                  <div className="metric-label">Lower 95% PI</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value" style={{ fontSize: '1.3rem' }}>{formatNumber(singleResult.upper_95)}</div>
                  <div className="metric-label">Upper 95% PI</div>
                </div>
              </div>
              <div style={{ padding: '0.75rem', background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                <pre style={{ margin: 0, color: 'var(--text-secondary)' }}>{JSON.stringify(singleResult, null, 2)}</pre>
              </div>
            </div>
          )}
        </form>
      )}
    </div>
  );
}
