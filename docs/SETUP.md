# ANN-SIMS Setup Guide

## Prerequisites

- Node.js >= 18.0.0
- Python >= 3.9
- npm >= 9.0.0

## Local Development Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd ANN-for-SIMS
```

### 2. Backend Setup

```bash
cd backend
npm install
```

Create a `.env` file in `backend/`:
```env
PORT=5000
FRONTEND_URL=http://localhost:3000
NODE_ENV=development
PYTHON_EXECUTABLE=python3
```

### 3. Python Dependencies

```bash
cd backend/python
pip install -r requirements.txt
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Start Development Servers

**Backend** (in `backend/`):
```bash
npm run dev
```

**Frontend** (in `frontend/`):
```bash
npm run dev
```

The frontend is available at `http://localhost:3000` and proxies API calls to `http://localhost:5000`.

---

## Docker Setup

```bash
docker-compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5000`
- Full stack via Nginx: `http://localhost:80`

## Podman Setup

Install [podman-compose](https://github.com/containers/podman-compose) if you
haven't already:

```bash
pip install podman-compose        # or: dnf install podman-compose
```

Then run:

```bash
podman-compose -f podman-compose.yml up --build
```

To build and run individual containers with plain Podman:

```bash
# Backend
podman build -t ann-sims-backend -f backend/Containerfile backend/
podman run -d -p 5000:5000 \
  -v ./backend/uploads:/app/uploads:Z \
  -v ./backend/models:/app/models:Z \
  -v ./backend/results:/app/results:Z \
  --name ann-sims-backend ann-sims-backend

# Frontend
podman build -t ann-sims-frontend -f frontend/Containerfile frontend/
podman run -d -p 3000:80 --name ann-sims-frontend ann-sims-frontend
```

> **Note:** The `Containerfile` is the Podman-native equivalent of a
> `Dockerfile`.  Both `docker build` and `podman build` can consume either
> filename, so the same images work with both runtimes.

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5000`
- Full stack via Nginx: `http://localhost:80`

---

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Backend server port |
| `FRONTEND_URL` | `http://localhost:3000` | CORS allowed origin |
| `NODE_ENV` | `development` | Environment mode |
| `PYTHON_EXECUTABLE` | `python3` | Path to Python interpreter |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `` (empty) | Backend base URL (empty = proxy) |

---

## Troubleshooting

### Python not found
Ensure `python3` is in your PATH, or set `PYTHON_EXECUTABLE` in your `.env` file to the full path.

### TensorFlow installation issues
Try installing with CPU-only version:
```bash
pip install tensorflow-cpu
```

### Port conflicts
Change the PORT environment variable in `backend/.env` and update `vite.config.js` proxy accordingly.
