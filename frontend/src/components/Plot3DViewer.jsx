import React, { useState } from 'react';

export default function Plot3DViewer({ plots, jobId }) {
  const [selected, setSelected] = useState(null);

  if (!plots?.length) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🗺️</div>
        <div style={{ color: 'var(--text-secondary)' }}>
          No plots available for this job yet.
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Plots are generated during training after the final model is saved.
        </div>
      </div>
    );
  }

  const plotUrl = (plot) => {
    const base = import.meta.env.VITE_API_BASE_URL || '';
    return `${base}/results/${jobId}/plots/${plot.name}`;
  };

  return (
    <div>
      {selected && (
        <div
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, cursor: 'pointer',
          }}
          onClick={() => setSelected(null)}
        >
          <div onClick={(e) => e.stopPropagation()} style={{ maxWidth: '90vw', maxHeight: '90vh' }}>
            <img
              src={plotUrl(selected)}
              alt={selected.name}
              style={{ maxWidth: '100%', maxHeight: '85vh', borderRadius: 12, boxShadow: '0 20px 60px rgba(0,0,0,0.8)' }}
            />
            <div style={{ textAlign: 'center', color: '#94a3b8', marginTop: '0.75rem', fontSize: '0.85rem' }}>
              {selected.name} &nbsp;·&nbsp; Click anywhere to close
            </div>
          </div>
        </div>
      )}

      <div className="plot-grid">
        {plots.map((plot) => (
          <div
            key={plot.name}
            className="plot-card"
            style={{ cursor: 'pointer', transition: 'transform 0.2s' }}
            onClick={() => setSelected(plot)}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = ''; }}
          >
            <img
              src={plotUrl(plot)}
              alt={plot.name}
              loading="lazy"
              style={{ width: '100%', display: 'block' }}
            />
            <div className="plot-card-label">
              {plot.name.replace(/_/g, ' ').replace('.png', '')}
              <span style={{ marginLeft: '0.5rem', color: 'var(--primary-light)', fontSize: '0.7rem' }}>
                🔍 Click to enlarge
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
