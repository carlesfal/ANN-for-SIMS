import React, { useState } from 'react';
import { templateApi } from '../services/api.js';

const MIN_INPUTS  = 1;
const MAX_INPUTS  = 50;
const MIN_OUTPUTS = 1;
const MAX_OUTPUTS = 20;
const MIN_LABELS  = 0;
const MAX_LABELS  = 20;

function NumberInput({ label, value, min, max, onChange, hint }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', flex: 1 }}>
      <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
        {label}
      </label>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const n = parseInt(e.target.value, 10);
          if (!Number.isFinite(n)) return;
          onChange(Math.max(min, Math.min(max, n)));
        }}
        style={{
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          color: 'var(--text-primary)',
          padding: '0.45rem 0.65rem',
          fontSize: '0.95rem',
          width: '100%',
          outline: 'none',
        }}
      />
      {hint && (
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{hint}</span>
      )}
    </div>
  );
}

export default function TemplateDownloader() {
  const [open, setOpen]       = useState(false);
  const [labels,  setLabels]  = useState(1);
  const [inputs,  setInputs]  = useState(3);
  const [outputs, setOutputs] = useState(1);

  const handleDownload = () => {
    const url = templateApi.downloadUrl({ labels, inputs, outputs });
    const a = document.createElement('a');
    a.href = url;
    a.download = `ann_sims_template_L${labels}_I${inputs}_O${outputs}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setOpen(false);
  };

  const totalCols = labels + inputs + outputs;

  return (
    <>
      {/* Trigger button */}
      <button
        className="btn btn-secondary"
        style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
        onClick={() => setOpen(true)}
        title="Download a blank Excel template pre-formatted for ANN-SIMS"
      >
        📥 Download Template
      </button>

      {/* Modal backdrop */}
      {open && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(0,0,0,0.6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            padding: '1rem',
          }}
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}
        >
          {/* Modal card */}
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.75rem',
              width: '100%',
              maxWidth: 440,
              boxShadow: 'var(--shadow)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                📥 Download Excel Template
              </h3>
              <button
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', color: 'var(--text-muted)' }}
                onClick={() => setOpen(false)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: 1.6 }}>
              Choose how many columns of each type to include. The generated&nbsp;
              <strong>.xlsx</strong> file will have colour-coded headers so you can
              fill in your data and upload it directly.
            </p>

            {/* Column legend chips */}
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
              {[
                { label: 'Label', color: '#93C5FD', desc: 'Identifier / sample name' },
                { label: 'Input', color: '#86EFAC', desc: 'Model feature (numeric)' },
                { label: 'Output', color: '#FCD34D', desc: 'Target variable' },
              ].map(({ label, color, desc }) => (
                <div
                  key={label}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.35rem',
                           fontSize: '0.75rem', color: 'var(--text-secondary)' }}
                  title={desc}
                >
                  <span style={{ width: 12, height: 12, borderRadius: 3, background: color, display: 'inline-block' }} />
                  {label}
                </div>
              ))}
            </div>

            {/* Number inputs */}
            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <NumberInput
                label={`🔵 Label columns (${MIN_LABELS}–${MAX_LABELS})`}
                value={labels}
                min={MIN_LABELS}
                max={MAX_LABELS}
                onChange={setLabels}
                hint="e.g. sample IDs"
              />
              <NumberInput
                label={`🟢 Input columns (${MIN_INPUTS}–${MAX_INPUTS})`}
                value={inputs}
                min={MIN_INPUTS}
                max={MAX_INPUTS}
                onChange={setInputs}
                hint="numeric features"
              />
              <NumberInput
                label={`🟡 Output columns (${MIN_OUTPUTS}–${MAX_OUTPUTS})`}
                value={outputs}
                min={MIN_OUTPUTS}
                max={MAX_OUTPUTS}
                onChange={setOutputs}
                hint="target variable(s)"
              />
            </div>

            {/* Summary */}
            <div
              style={{
                background: 'rgba(99,102,241,0.08)',
                border: '1px solid rgba(99,102,241,0.2)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.65rem 1rem',
                fontSize: '0.82rem',
                color: 'var(--text-secondary)',
                marginBottom: '1.25rem',
                lineHeight: 1.6,
              }}
            >
              Template will have&nbsp;
              <strong style={{ color: 'var(--text-primary)' }}>{totalCols}</strong>&nbsp;column{totalCols !== 1 ? 's' : ''}:&nbsp;
              {labels > 0 && <><span style={{ color: '#93C5FD' }}>{labels} label</span>, </>}
              <span style={{ color: '#86EFAC' }}>{inputs} input</span>,&nbsp;
              <span style={{ color: '#FCD34D' }}>{outputs} output</span>
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              <button className="btn btn-secondary" onClick={() => setOpen(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleDownload}>
                ⬇ Download .xlsx
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
