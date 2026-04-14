import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { trainingApi, resultsApi } from '../services/api.js';
import MetricsDisplay from '../components/MetricsDisplay.jsx';
import Plot3DViewer from '../components/Plot3DViewer.jsx';
import DownloadManager from '../components/DownloadManager.jsx';
import { formatDate, formatDuration, formatStatus } from '../utils/formatters.js';

export default function Results() {
  const { jobId: paramJobId } = useParams();
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(paramJobId || null);
  const [results, setResults] = useState(null);
  const [plots, setPlots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('metrics');

  useEffect(() => {
    loadJobs();
  }, []);

  useEffect(() => {
    if (selectedJobId) {
      loadResults(selectedJobId);
    }
  }, [selectedJobId]);

  const loadJobs = async () => {
    try {
      const data = await trainingApi.listJobs();
      setJobs(data.jobs || []);
      if (!selectedJobId && data.jobs?.length > 0) {
        const completed = data.jobs.find((j) => j.status === 'completed');
        if (completed) setSelectedJobId(completed.jobId);
      }
    } catch (err) {
      console.error('Failed to load jobs:', err.message);
    }
  };

  const loadResults = async (jobId) => {
    setLoading(true);
    setError('');
    setResults(null);
    setPlots([]);
    try {
      const [resultsData, plotsData] = await Promise.all([
        resultsApi.getResults(jobId),
        resultsApi.getPlots(jobId),
      ]);
      setResults(resultsData);
      setPlots(plotsData.plots || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header flex-between flex-wrap gap-1">
        <div>
          <h1 className="page-title">Results</h1>
          <p className="page-subtitle">View training results, metrics, and plots.</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={loadJobs}>🔄 Refresh</button>
      </div>

      <div className="grid grid-2" style={{ gridTemplateColumns: '280px 1fr', gap: '1.5rem' }}>
        {/* Job List Sidebar */}
        <div className="card" style={{ alignSelf: 'start' }}>
          <div className="card-header">
            <span className="card-title">📋 Training Jobs</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{jobs.length} jobs</span>
          </div>
          {jobs.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem 0' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📭</div>
              <div>No training jobs yet.</div>
              <Link to="/training" className="btn btn-primary btn-sm mt-2" style={{ marginTop: '1rem' }}>
                Start Training
              </Link>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {jobs.map((job) => (
                <button
                  key={job.jobId}
                  onClick={() => setSelectedJobId(job.jobId)}
                  style={{
                    background: selectedJobId === job.jobId ? 'rgba(99,102,241,0.15)' : 'transparent',
                    border: `1px solid ${selectedJobId === job.jobId ? 'var(--primary)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.75rem',
                    textAlign: 'left',
                    cursor: 'pointer',
                    color: 'var(--text-primary)',
                    transition: 'all 0.2s',
                  }}
                >
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                    {job.originalName || 'Training Job'}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'space-between' }}>
                    <span className={`badge badge-${job.status}`}>{formatStatus(job.status)}</span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {formatDate(job.createdAt).split(',')[0]}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div>
          {!selectedJobId && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
              <div style={{ color: 'var(--text-secondary)' }}>Select a training job to view its results.</div>
            </div>
          )}

          {loading && (
            <div className="card flex-center" style={{ padding: '3rem' }}>
              <div className="spinner" style={{ width: 32, height: 32 }} />
            </div>
          )}

          {error && (
            <div className="alert alert-error">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {results && !loading && (
            <div>
              <div className="tabs">
                {[
                  { key: 'metrics', label: '📊 Metrics' },
                  { key: 'plots', label: '🗺️ Plots' },
                  { key: 'download', label: '📥 Download' },
                ].map(({ key, label }) => (
                  <button
                    key={key}
                    className={`tab-btn ${activeTab === key ? 'active' : ''}`}
                    onClick={() => setActiveTab(key)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {activeTab === 'metrics' && (
                <MetricsDisplay results={results} />
              )}

              {activeTab === 'plots' && (
                <Plot3DViewer plots={plots} jobId={selectedJobId} />
              )}

              {activeTab === 'download' && (
                <DownloadManager jobId={selectedJobId} results={results} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
