import React, { useState } from 'react';

const DEFAULT_CONFIG = {
  targetColumn: 'target',
  featureColumns: '',
  epochs: 100,
  folds: 10,
  learningRate: 0.001,
  units: 64,
  layers: 3,
  dropout: 0.2,
  batchSize: 32,
  patience: 20,
  usePretraining: true,
};

export default function TrainingConfig({ onSubmit, loading }) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);

  const handleChange = (key, value) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      ...config,
      usePretraining: config.usePretraining ? 'true' : 'false',
    });
  };

  const Field = ({ label, name, type = 'number', min, max, step = 1, helpText }) => (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <input
        type={type}
        className="form-control"
        value={config[name]}
        min={min}
        max={max}
        step={step}
        onChange={(e) => handleChange(name, type === 'number' ? Number(e.target.value) : e.target.value)}
      />
      {helpText && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>{helpText}</div>}
    </div>
  );

  return (
    <form onSubmit={handleSubmit}>
      <div className="section">
        <div className="section-title">📋 Data Configuration</div>
        <div className="grid grid-2">
          <Field
            label="Target Column Name"
            name="targetColumn"
            type="text"
            helpText="Name of the output/dependent variable column"
          />
          <div className="form-group">
            <label className="form-label">Feature Columns (optional)</label>
            <input
              type="text"
              className="form-control"
              value={config.featureColumns}
              placeholder="col1, col2, col3 (leave blank for all)"
              onChange={(e) => handleChange('featureColumns', e.target.value)}
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Comma-separated. Leave blank to use all non-target columns.
            </div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-title">🔧 Hyperparameters</div>
        <div className="grid grid-3">
          <Field label="Training Epochs" name="epochs" min={10} max={1000} helpText="Max epochs per fold" />
          <Field label="CV Folds" name="folds" min={2} max={20} helpText="K for k-fold cross-validation" />
          <Field label="Batch Size" name="batchSize" min={8} max={512} step={8} helpText="Samples per training batch" />
          <Field label="Units per Layer" name="units" min={8} max={512} step={8} helpText="Used as starting point for tuner" />
          <Field label="Number of Layers" name="layers" min={1} max={8} helpText="Dense layers (search space)" />
          <Field label="Dropout Rate" name="dropout" min={0} max={0.8} step={0.05} helpText="Fraction of units to drop" />
          <Field label="Learning Rate" name="learningRate" min={0.0001} max={0.1} step={0.0001} helpText="Adam optimizer LR" />
          <Field label="Early Stopping Patience" name="patience" min={5} max={100} helpText="Epochs without improvement before stopping" />
        </div>
      </div>

      <div className="section">
        <div className="section-title">🔬 Advanced Options</div>
        <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <input
            type="checkbox"
            id="usePretraining"
            checked={config.usePretraining}
            onChange={(e) => handleChange('usePretraining', e.target.checked)}
            style={{ width: 18, height: 18, cursor: 'pointer', accentColor: 'var(--primary)' }}
          />
          <label htmlFor="usePretraining" style={{ cursor: 'pointer' }}>
            <span style={{ fontWeight: 600 }}>Enable Hill Function Pre-training</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
              — Warm-start weights from synthetic Hill-function data for faster convergence
            </span>
          </label>
        </div>
      </div>

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.25rem', display: 'flex', gap: '1rem' }}>
        <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
          {loading ? (
            <><span className="spinner" />&nbsp; Submitting...</>
          ) : (
            '🚀 Start Training'
          )}
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setConfig(DEFAULT_CONFIG)}
        >
          Reset Defaults
        </button>
      </div>
    </form>
  );
}
