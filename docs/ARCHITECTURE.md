# System Architecture

## Overview

ANN-SIMS is a full-stack web application for training and deploying Artificial Neural Networks on SIMS (Secondary Ion Mass Spectrometry) data with the following layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Browser / Client                      │
│              React + Vite + Chart.js                     │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / WebSocket
┌────────────────────────▼────────────────────────────────┐
│                  Node.js Express Backend                 │
│   REST API + WebSocket Server (port 5000)                │
│   ┌─────────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │  Training   │  │Prediction│  │    Results       │  │
│   │   Routes    │  │  Routes  │  │    Routes        │  │
│   └──────┬──────┘  └────┬─────┘  └───────┬──────────┘  │
│          │               │                │             │
│   ┌──────▼───────────────▼────────────────▼──────────┐  │
│   │            Controllers + Job Queue                │  │
│   │   (spawn Python child processes, track status)    │  │
│   └──────────────────────┬────────────────────────────┘  │
└──────────────────────────┼─────────────────────────────┘
                           │ spawn child_process
┌──────────────────────────▼─────────────────────────────┐
│              Python ML Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ pretraining  │  │ train_model  │  │   predict    │  │
│  │  (Hill fn)   │  │  (main pipe) │  │  (inference) │  │
│  └──────────────┘  └──────┬───────┘  └──────────────┘  │
│                           │                             │
│  ┌──────────────┐  ┌──────▼───────┐                    │
│  │   plot_3d    │  │  Keras Tuner │                    │
│  │  (surfaces)  │  │ + 10-fold CV │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                           │ read/write
┌──────────────────────────▼─────────────────────────────┐
│                 File System                             │
│  uploads/  models/  results/                           │
└─────────────────────────────────────────────────────────┘
```

## Components

### Frontend (React + Vite)
- **Pages:** Home, Training, Predictions, Results
- **Components:** DataUploader, TrainingConfig, TrainingMonitor, MetricsDisplay, Plot3DViewer, PredictionInterface, DownloadManager
- **Services:** Axios HTTP client, WebSocket client
- **State:** Local component state (no external state library needed)

### Backend (Node.js + Express)
- **REST API:** Training, Predictions, Results endpoints
- **WebSocket:** Real-time job progress broadcast
- **Job Queue:** In-memory job store with status tracking and WebSocket broadcasting
- **File Handling:** Multer for multipart upload, UUID-named files

### Python ML Pipeline
- **pretraining.py:** Generates synthetic Hill-function data, trains warm-start model
- **train_model.py:** Full pipeline: load data → hyperparameter search (Keras Tuner) → 10-fold CV → final model
- **predict.py:** Load model + scalers, batch/single inference, MC Dropout uncertainty
- **plot_3d.py:** 3D surface plots, scatter plots, residual analysis
- **utils.py:** Data loading, scaling, metrics computation, logging helpers

## Data Flow

1. User uploads CSV/Excel → stored in `backend/uploads/`
2. Training job spawned → Python process runs `train_model.py`
3. Python outputs structured `PROGRESS:{}` lines → parsed by Node.js controller
4. Job status + progress broadcast via WebSocket to subscribed clients
5. On completion: model saved to `backend/models/<jobId>/`, results to `backend/results/<jobId>/`
6. Frontend polls or receives WS updates → displays live metrics and logs
7. On completion: user views results, plots, downloads CSV/JSON

## Real-time Communication

- WebSocket server at `ws://localhost:5000/ws`
- Client subscribes to a specific `jobId`
- Server broadcasts `job_updated` events whenever job state changes
- Fallback polling at 3-second intervals when WS unavailable

## Security Considerations

- File type and size validation on upload
- UUIDs for all file/job names (no path traversal)
- CORS restricted to configured frontend origin
- No authentication (add as needed for production)
