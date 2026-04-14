import React from 'react';
import { Link } from 'react-router-dom';

const features = [
  {
    icon: '🧠',
    title: 'Neural Network Training',
    desc: 'Train deep ANNs with automatic hyperparameter tuning via Keras Tuner.',
  },
  {
    icon: '🔬',
    title: 'Hill Function Pre-training',
    desc: 'Warm-start weights using synthetic Hill-function data for faster convergence.',
  },
  {
    icon: '📊',
    title: '10-Fold Cross Validation',
    desc: 'Robust model evaluation with k-fold CV, per-fold metrics, and averages.',
  },
  {
    icon: '📈',
    title: 'Comprehensive Metrics',
    desc: 'RMSE, MAE, R², and MAPE with standard deviations across folds.',
  },
  {
    icon: '🎯',
    title: 'Prediction Intervals',
    desc: '95% prediction intervals via MC Dropout for uncertainty quantification.',
  },
  {
    icon: '🗺️',
    title: '3D Surface Plots',
    desc: 'Interactive 3D visualization of the learned response surface.',
  },
  {
    icon: '⚡',
    title: 'Real-time Monitoring',
    desc: 'Live training progress via WebSocket with log streaming.',
  },
  {
    icon: '📥',
    title: 'Export Results',
    desc: 'Download predictions and metrics as JSON or CSV.',
  },
];

export default function Home() {
  return (
    <div>
      <section className="hero">
        <h1 className="hero-title">ANN-SIMS<br />Prediction System</h1>
        <p className="hero-subtitle">
          A production-ready Artificial Neural Network pipeline for SIMS data analysis
          — with hyperparameter optimization, cross-validation, and uncertainty quantification.
        </p>
        <div className="hero-actions">
          <Link to="/training" className="btn btn-primary btn-lg">
            🚀 Start Training
          </Link>
          <Link to="/predictions" className="btn btn-secondary btn-lg">
            🔍 Run Predictions
          </Link>
        </div>
      </section>

      <div className="feature-grid">
        {features.map((f, i) => (
          <div key={i} className="feature-card">
            <div className="feature-icon">{f.icon}</div>
            <div className="feature-title">{f.title}</div>
            <div className="feature-desc">{f.desc}</div>
          </div>
        ))}
      </div>

      <div className="card mt-3" style={{ marginTop: '3rem' }}>
        <div className="card-header">
          <span className="card-title">📖 Quick Start</span>
        </div>
        <div className="grid grid-3" style={{ gap: '1rem' }}>
          {[
            { step: '1', title: 'Upload Data', desc: 'Upload a CSV or Excel file with your SIMS measurements. Designate a target column and feature columns.' },
            { step: '2', title: 'Configure & Train', desc: 'Set training hyperparameters or use defaults. Enable Hill pre-training for faster convergence. Start the training job.' },
            { step: '3', title: 'Analyze Results', desc: 'Review cross-validation metrics, view 3D surface plots, and download predictions with uncertainty bounds.' },
          ].map((s) => (
            <div key={s.step} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div style={{
                width: 36, height: 36, borderRadius: '50%',
                background: 'var(--primary)', color: '#fff',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: 700, fontSize: '1rem', flexShrink: 0,
              }}>{s.step}</div>
              <div>
                <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{s.title}</div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
