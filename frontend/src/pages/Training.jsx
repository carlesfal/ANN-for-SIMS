import React, { useState } from 'react';
import DataUploader from '../components/DataUploader.jsx';
import TrainingConfig from '../components/TrainingConfig.jsx';
import TrainingMonitor from '../components/TrainingMonitor.jsx';
import { trainingApi } from '../services/api.js';

export default function Training() {
  const [step, setStep] = useState('upload'); // 'upload' | 'config' | 'monitor'
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleFileSelected = (selectedFile) => {
    setFile(selectedFile);
    setError('');
    setStep('config');
  };

  const handleConfigSubmit = async (config) => {
    if (!file) {
      setError('Please upload a data file first');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('dataFile', file);
      Object.entries(config).forEach(([key, value]) => {
        formData.append(key, value);
      });

      const result = await trainingApi.startTraining(formData);
      setJobId(result.jobId);
      setStep('monitor');
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    setStep('upload');
    setFile(null);
    setJobId(null);
    setError('');
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Training</h1>
        <p className="page-subtitle">
          Upload your SIMS dataset and train an ANN with automated hyperparameter search.
        </p>
      </div>

      {/* Step indicator */}
      <div className="flex gap-2 mb-2" style={{ marginBottom: '1.5rem' }}>
        {[
          { key: 'upload', label: '1. Upload Data' },
          { key: 'config', label: '2. Configure' },
          { key: 'monitor', label: '3. Monitor' },
        ].map(({ key, label }) => (
          <div
            key={key}
            style={{
              padding: '0.4rem 1rem',
              borderRadius: '999px',
              fontSize: '0.8rem',
              fontWeight: 600,
              background: step === key
                ? 'rgba(99,102,241,0.2)'
                : 'rgba(255,255,255,0.04)',
              color: step === key ? 'var(--primary-light)' : 'var(--text-muted)',
              border: `1px solid ${step === key ? 'var(--primary)' : 'var(--border)'}`,
            }}
          >
            {label}
          </div>
        ))}
      </div>

      {error && (
        <div className="alert alert-error">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {step === 'upload' && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">📂 Upload Dataset</span>
          </div>
          <DataUploader onFileSelected={handleFileSelected} />
        </div>
      )}

      {step === 'config' && (
        <div>
          <div className="alert alert-success" style={{ marginBottom: '1rem' }}>
            <span>✅</span>
            <span>File selected: <strong>{file?.name}</strong> ({(file?.size / 1024).toFixed(1)} KB)</span>
            <button
              className="btn btn-sm btn-secondary"
              style={{ marginLeft: 'auto' }}
              onClick={() => setStep('upload')}
            >Change File</button>
          </div>
          <div className="card">
            <div className="card-header">
              <span className="card-title">⚙️ Training Configuration</span>
            </div>
            <TrainingConfig onSubmit={handleConfigSubmit} loading={submitting} />
          </div>
        </div>
      )}

      {step === 'monitor' && jobId && (
        <div>
          <TrainingMonitor jobId={jobId} onReset={handleReset} />
        </div>
      )}
    </div>
  );
}
