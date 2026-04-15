import React, { useState, useRef } from 'react';
import TemplateDownloader from './TemplateDownloader.jsx';

export default function DataUploader({ onFileSelected }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const ALLOWED_TYPES = ['.csv', '.xlsx', '.xls'];
  const MAX_SIZE_MB = 50;

  const validateFile = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(ext)) {
      return `Invalid file type. Allowed: ${ALLOWED_TYPES.join(', ')}`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum size: ${MAX_SIZE_MB}MB`;
    }
    return null;
  };

  const handleFile = (file) => {
    if (!file) return;
    const err = validateFile(file);
    if (err) {
      setError(err);
      setSelectedFile(null);
      return;
    }
    setError('');
    setSelectedFile(file);
    onFileSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleChange = (e) => {
    handleFile(e.target.files[0]);
  };

  return (
    <div>
      <div
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="upload-icon">
          {selectedFile ? '✅' : '📂'}
        </div>
        <div className="upload-text">
          {selectedFile
            ? <span className="upload-filename">{selectedFile.name}</span>
            : <span>Drag & drop your dataset here, or <strong style={{ color: 'var(--primary-light)' }}>click to browse</strong></span>
          }
        </div>
        <div className="upload-hint">
          Supported formats: CSV, XLSX, XLS &nbsp;·&nbsp; Max size: 50MB
        </div>
        {selectedFile && (
          <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            {(selectedFile.size / 1024).toFixed(1)} KB
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={handleChange}
        />
      </div>

      {error && (
        <div className="alert alert-error mt-1">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {selectedFile && (
        <div style={{ marginTop: '1rem', display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-primary"
            onClick={() => onFileSelected(selectedFile)}
          >
            Continue with this file →
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => { setSelectedFile(null); if (inputRef.current) inputRef.current.value = ''; }}
          >
            Clear
          </button>
        </div>
      )}

      <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
        <span>ℹ️</span>
        <div>
          <strong>Data format tips:</strong>
          <ul style={{ marginTop: '0.25rem', paddingLeft: '1.25rem', fontSize: '0.85rem', lineHeight: 1.8 }}>
            <li>First row should contain column headers</li>
            <li>One column must be your target/output variable</li>
            <li>All other numeric columns will be used as features</li>
            <li>Missing values (NaN) will be automatically removed</li>
          </ul>
        </div>
      </div>

      <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Need a starting point?
        </span>
        <TemplateDownloader />
      </div>
    </div>
  );
}
