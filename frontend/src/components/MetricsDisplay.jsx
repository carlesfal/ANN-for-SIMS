import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Scatter, Bar } from 'react-chartjs-2';
import { formatMetric, formatNumber } from '../utils/formatters.js';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
);

const METRIC_LABELS = { rmse: 'RMSE', mae: 'MAE', r2: 'R²', mape: 'MAPE (%)' };

function MetricCard({ label, value, std }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{formatMetric(value, label)}</div>
      <div className="metric-label">{METRIC_LABELS[label] || label}</div>
      {std !== undefined && (
        <div className="metric-sub">± {formatNumber(std, 4)}</div>
      )}
    </div>
  );
}

function FoldMetricsTable({ foldMetrics }) {
  if (!foldMetrics?.length) return null;
  const bestFold = foldMetrics.reduce(
    (best, m, i) => (m.r2 > (foldMetrics[best]?.r2 || -Infinity) ? i : best),
    0
  );

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Fold</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>R²</th>
            <th>MAPE (%)</th>
          </tr>
        </thead>
        <tbody>
          {foldMetrics.map((m, i) => (
            <tr key={i} className={i === bestFold ? 'best-fold' : ''}>
              <td>
                {i + 1}
                {i === bestFold && (
                  <span style={{ marginLeft: '0.4rem', fontSize: '0.7rem', color: 'var(--secondary)' }}>★ best</span>
                )}
              </td>
              <td>{formatNumber(m.rmse)}</td>
              <td>{formatNumber(m.mae)}</td>
              <td>{formatNumber(m.r2)}</td>
              <td>{formatNumber(m.mape, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PredVsActualChart({ predictions }) {
  if (!predictions?.yTrue?.length) return null;

  const maxPoints = 500;
  const step = Math.ceil(predictions.yTrue.length / maxPoints);
  const pts = predictions.yTrue
    .filter((_, i) => i % step === 0)
    .map((y, i) => ({ x: y, y: predictions.yPred[i * step] }));

  const allVals = [...predictions.yTrue, ...predictions.yPred];
  const mn = Math.min(...allVals);
  const mx = Math.max(...allVals);

  const data = {
    datasets: [
      {
        label: 'Predicted vs Actual',
        data: pts,
        backgroundColor: 'rgba(99,102,241,0.5)',
        pointRadius: 3,
      },
      {
        label: 'Perfect Fit',
        data: [{ x: mn, y: mn }, { x: mx, y: mx }],
        type: 'line',
        borderColor: '#ef4444',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        showLine: true,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      title: { display: true, text: 'Predicted vs. Actual (CV)', color: '#f1f5f9', font: { size: 14 } },
    },
    scales: {
      x: { title: { display: true, text: 'Actual', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { title: { display: true, text: 'Predicted', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
  };

  return <Scatter data={data} options={options} />;
}

function FoldMetricsChart({ foldMetrics }) {
  if (!foldMetrics?.length) return null;
  const labels = foldMetrics.map((_, i) => `Fold ${i + 1}`);

  const data = {
    labels,
    datasets: [
      {
        label: 'RMSE',
        data: foldMetrics.map((m) => m.rmse),
        backgroundColor: 'rgba(99,102,241,0.7)',
        borderColor: 'rgba(99,102,241,1)',
        borderWidth: 1,
      },
      {
        label: 'MAE',
        data: foldMetrics.map((m) => m.mae),
        backgroundColor: 'rgba(16,185,129,0.7)',
        borderColor: 'rgba(16,185,129,1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      title: { display: true, text: 'Per-Fold RMSE & MAE', color: '#f1f5f9', font: { size: 14 } },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
  };

  return <Bar data={data} options={options} />;
}

export default function MetricsDisplay({ results }) {
  const [activeTab, setActiveTab] = useState('summary');

  if (!results) return null;

  const { averageMetrics, foldMetrics, predictions, bestHyperparameters, featureColumns, targetColumn, nSamples } = results;

  return (
    <div>
      {/* Summary Stats */}
      <div className="grid grid-4" style={{ marginBottom: '1.5rem' }}>
        <MetricCard label="rmse" value={averageMetrics?.rmse} std={averageMetrics?.rmse_std} />
        <MetricCard label="mae" value={averageMetrics?.mae} std={averageMetrics?.mae_std} />
        <MetricCard label="r2" value={averageMetrics?.r2} std={averageMetrics?.r2_std} />
        <MetricCard label="mape" value={averageMetrics?.mape} std={averageMetrics?.mape_std} />
      </div>

      {/* Sub-tabs */}
      <div className="tabs">
        {['summary', 'folds', 'charts'].map((t) => (
          <button key={t} className={`tab-btn ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t === 'summary' ? '📋 Summary' : t === 'folds' ? '📊 Per-Fold' : '📈 Charts'}
          </button>
        ))}
      </div>

      {activeTab === 'summary' && (
        <div className="grid grid-2">
          <div className="card">
            <div className="card-header"><span className="card-title">🔧 Best Hyperparameters</span></div>
            <table className="data-table">
              <tbody>
                {bestHyperparameters && Object.entries(bestHyperparameters).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ color: 'var(--text-muted)' }}>{k}</td>
                    <td style={{ fontWeight: 600 }}>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <div className="card-header"><span className="card-title">📂 Dataset Info</span></div>
            <table className="data-table">
              <tbody>
                <tr><td style={{ color: 'var(--text-muted)' }}>Samples</td><td style={{ fontWeight: 600 }}>{nSamples}</td></tr>
                <tr><td style={{ color: 'var(--text-muted)' }}>Target</td><td style={{ fontWeight: 600 }}>{targetColumn}</td></tr>
                <tr><td style={{ color: 'var(--text-muted)' }}>Features</td><td style={{ fontWeight: 600 }}>{featureColumns?.length}</td></tr>
                <tr><td style={{ color: 'var(--text-muted)' }}>Feature Names</td><td style={{ fontSize: '0.8rem' }}>{featureColumns?.join(', ')}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'folds' && (
        <div className="card">
          <div className="card-header"><span className="card-title">📊 Per-Fold Metrics</span></div>
          <FoldMetricsTable foldMetrics={foldMetrics} />
        </div>
      )}

      {activeTab === 'charts' && (
        <div className="grid grid-2">
          <div className="card">
            <PredVsActualChart predictions={predictions} />
          </div>
          <div className="card">
            <FoldMetricsChart foldMetrics={foldMetrics} />
          </div>
        </div>
      )}
    </div>
  );
}
