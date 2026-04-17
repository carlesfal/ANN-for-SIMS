import React, { useState } from 'react';
import PredictionInterface from '../components/PredictionInterface.jsx';

export default function Predictions() {
  const [activeTab, setActiveTab] = useState('batch');

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Predictions</h1>
        <p className="page-subtitle">
          Run batch or single-sample predictions using a trained model.
        </p>
      </div>

      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          📦 Batch Prediction
        </button>
        <button
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          🔬 Single Sample
        </button>
      </div>

      <PredictionInterface mode={activeTab} />
    </div>
  );
}
