# ANN-SIMS Prediction System

A production-ready full-stack web application for training and deploying Artificial Neural Networks on SIMS (Secondary Ion Mass Spectrometry) data.

## Features

- 🧠 **ANN Training** — Dense neural networks with TensorFlow/Keras
- 🔬 **Hill Function Pre-training** — Warm-start from synthetic Hill-function data
- 🔧 **Hyperparameter Tuning** — Automated search via Keras Tuner
- 📊 **10-Fold Cross-Validation** — Robust model evaluation
- 📈 **Comprehensive Metrics** — RMSE, MAE, R², MAPE with standard deviations
- 🎯 **Prediction Intervals** — 95% PI via MC Dropout uncertainty quantification
- 🗺️ **3D Surface Plots** — Interactive visualization of the learned response surface
- ⚡ **Real-time Monitoring** — Live training progress via WebSocket
- 📥 **Export** — Download predictions and metrics as JSON/CSV

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Chart.js, react-router-dom v6 |
| Backend | Node.js, Express, WebSocket (ws) |
| ML Pipeline | Python, TensorFlow/Keras, Keras Tuner, scikit-learn |
| Data | pandas, numpy, joblib, matplotlib |

## Quick Start

### Prerequisites
- Node.js >= 18
- Python >= 3.9
- npm >= 9

### Installation

```bash
# 1. Install backend dependencies
cd backend && npm install

# 2. Install Python dependencies
cd python && pip install -r requirements.txt

# 3. Install frontend dependencies
cd ../../frontend && npm install
```

### Development

```bash
# Terminal 1 — Backend
cd backend && npm run dev

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Docker

```bash
docker-compose up --build
```

## Documentation

- [Setup Guide](docs/SETUP.md) — Detailed installation instructions
- [API Reference](docs/API.md) — All REST endpoints and WebSocket protocol
- [Architecture](docs/ARCHITECTURE.md) — System design and data flow
- [User Guide](docs/USER_GUIDE.md) — How to use the application

## Project Structure

```
├── backend/
│   ├── server.js              # Express + WebSocket server
│   ├── config.js              # Configuration
│   ├── routes/                # API routes
│   ├── controllers/           # Business logic
│   ├── middleware/            # Multer upload
│   ├── utils/                 # Job queue
│   ├── python/                # ML pipeline
│   │   ├── train_model.py     # Main training pipeline
│   │   ├── pretraining.py     # Hill function pre-training
│   │   ├── predict.py         # Inference with PI
│   │   ├── plot_3d.py         # 3D surface plot generation
│   │   ├── utils.py           # Shared utilities
│   │   └── config.py          # ML hyperparameter defaults
│   ├── uploads/               # Uploaded files
│   ├── models/                # Saved models & scalers
│   └── results/               # Training results & plots
├── frontend/
│   ├── src/
│   │   ├── pages/             # Home, Training, Predictions, Results
│   │   ├── components/        # DataUploader, TrainingMonitor, MetricsDisplay, etc.
│   │   ├── services/api.js    # Axios client + WebSocket helper
│   │   └── utils/formatters.js
│   └── vite.config.js
└── docs/
    ├── SETUP.md
    ├── API.md
    ├── ARCHITECTURE.md
    └── USER_GUIDE.md
```

## License

MIT
