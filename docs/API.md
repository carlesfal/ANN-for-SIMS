# API Documentation

Base URL: `http://localhost:5000`

---

## Health

### GET /health
Returns server health status.

**Response:**
```json
{ "status": "ok", "timestamp": "...", "version": "1.0.0" }
```

---

## Training

### POST /api/training/start
Start a new training job.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dataFile` | File | ✅ | — | CSV or Excel file |
| `targetColumn` | string | ✅ | `target` | Target variable column name |
| `featureColumns` | string | ❌ | all | Comma-separated feature columns |
| `epochs` | number | ❌ | 100 | Maximum training epochs |
| `folds` | number | ❌ | 10 | Cross-validation folds |
| `learningRate` | number | ❌ | 0.001 | Adam learning rate |
| `units` | number | ❌ | 64 | Units per dense layer |
| `layers` | number | ❌ | 3 | Number of dense layers |
| `dropout` | number | ❌ | 0.2 | Dropout rate (0-1) |
| `batchSize` | number | ❌ | 32 | Training batch size |
| `patience` | number | ❌ | 20 | Early stopping patience |
| `usePretraining` | string | ❌ | `true` | Enable Hill function pre-training |

**Response (202):**
```json
{ "jobId": "uuid", "status": "queued", "message": "Training job queued" }
```

---

### GET /api/training/:jobId/status
Get the status of a training job.

**Response:**
```json
{
  "jobId": "...",
  "status": "running",
  "progress": 45,
  "currentFold": 4,
  "metrics": { "rmse": 0.1, "mae": 0.08, "r2": 0.95, "mape": 5.2 },
  "logs": [...],
  "createdAt": "...",
  "startedAt": "..."
}
```

**Status values:** `queued`, `running`, `completed`, `failed`, `cancelled`

---

### DELETE /api/training/:jobId
Cancel a running training job.

---

### GET /api/training
List all training jobs.

---

## Predictions

### POST /api/predictions/batch
Run batch predictions from a file.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataFile` | File | ✅ | CSV or Excel file |
| `modelJobId` | string | ✅ | Job ID of the trained model |
| `targetColumn` | string | ❌ | Optional actual target column |
| `featureColumns` | string | ❌ | Comma-separated feature columns |

**Response:**
```json
{
  "predictionId": "...",
  "results": {
    "predictions": [
      { "feature1": 1.2, "prediction": 3.4, "lower_95": 2.9, "upper_95": 3.9 }
    ],
    "metrics": { "rmse": 0.1, ... },
    "summary": { "mean_prediction": 3.0, ... }
  }
}
```

---

### POST /api/predictions/single
Predict a single sample.

**Body:**
```json
{
  "modelJobId": "...",
  "features": { "feature1": 1.5, "feature2": 3.2 }
}
```

**Response:**
```json
{
  "prediction": 3.4,
  "lower_95": 2.9,
  "upper_95": 3.9,
  "features": { ... }
}
```

---

## Results

### GET /api/results/:jobId
Get full results for a completed job.

### GET /api/results/:jobId/download?format=json|csv
Download results file.

### GET /api/results/:jobId/plots
List available plot files.

**Response:**
```json
{ "plots": [{ "name": "surface_x1_vs_x2.png", "url": "/results/.../plots/..." }] }
```

---

## WebSocket

Connect to `ws://localhost:5000/ws`

**Subscribe to a job:**
```json
{ "type": "subscribe", "jobId": "..." }
```

**Receive updates:**
```json
{
  "type": "job_updated",
  "jobId": "...",
  "data": {
    "status": "running",
    "progress": 45,
    "metrics": { ... },
    "logs": [...]
  }
}
```
