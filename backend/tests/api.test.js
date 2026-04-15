'use strict';

/**
 * Integration tests for the Express API routes.
 * Uses supertest to make requests without starting a real server.
 */

const request = require('supertest');

// Re-require a fresh app instance for each test suite
let app;

beforeAll(() => {
  jest.resetModules();
  ({ app } = require('../server'));
});

afterAll(() => {
  // server.js exports the http.Server; close it to release the port
  const { server } = require('../server');
  server.close();
});

// ─── Health check ─────────────────────────────────────────────────────────────

describe('GET /health', () => {
  test('returns 200 with status ok', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(typeof res.body.timestamp).toBe('string');
  });
});

// ─── 404 handler ──────────────────────────────────────────────────────────────

describe('Unknown route', () => {
  test('returns 404 for an unregistered path', async () => {
    const res = await request(app).get('/api/does-not-exist');
    expect(res.status).toBe(404);
    expect(res.body.error).toBe('Route not found');
  });
});

// ─── Training routes ──────────────────────────────────────────────────────────

describe('GET /api/training', () => {
  test('returns 200 with a jobs array', async () => {
    const res = await request(app).get('/api/training');
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.jobs)).toBe(true);
  });
});

describe('GET /api/training/:jobId/status', () => {
  test('returns 400 for an invalid UUID', async () => {
    const res = await request(app).get('/api/training/not-a-uuid/status');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Invalid job ID format');
  });

  test('returns 400 for a path-traversal attempt', async () => {
    const res = await request(app).get('/api/training/..%2F..%2Fetc%2Fpasswd/status');
    expect(res.status).toBe(400);
  });

  test('returns 404 for an unknown (but valid) UUID', async () => {
    const res = await request(app).get(
      '/api/training/00000000-0000-4000-8000-000000000000/status',
    );
    expect(res.status).toBe(404);
    expect(res.body.error).toBeDefined();
  });
});

describe('POST /api/training/start', () => {
  test('returns 400 when no file is uploaded', async () => {
    const res = await request(app)
      .post('/api/training/start')
      .field('targetColumn', 'target');
    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });
});

describe('DELETE /api/training/:jobId', () => {
  test('returns 400 for an invalid job ID', async () => {
    const res = await request(app).delete('/api/training/bad-id');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Invalid job ID format');
  });

  test('returns 404 for a valid but unknown UUID', async () => {
    const res = await request(app).delete(
      '/api/training/00000000-0000-4000-8000-000000000001',
    );
    expect(res.status).toBe(404);
  });
});

// ─── Results routes ───────────────────────────────────────────────────────────

describe('GET /api/results/:jobId', () => {
  test('returns 400 for a non-UUID job ID', async () => {
    const res = await request(app).get('/api/results/invalid-id');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Invalid job ID format');
  });

  test('returns 404 for an unknown valid UUID', async () => {
    const res = await request(app).get(
      '/api/results/00000000-0000-4000-8000-000000000000',
    );
    expect(res.status).toBe(404);
    expect(res.body.error).toBeDefined();
  });
});

describe('GET /api/results/:jobId/plots', () => {
  test('returns 400 for an invalid job ID', async () => {
    const res = await request(app).get('/api/results/bad-id/plots');
    expect(res.status).toBe(400);
  });

  test('returns 200 with empty plots array for an unknown valid UUID', async () => {
    const res = await request(app).get(
      '/api/results/00000000-0000-4000-8000-000000000000/plots',
    );
    expect(res.status).toBe(200);
    expect(res.body.plots).toEqual([]);
  });
});

describe('GET /api/results/:jobId/download', () => {
  test('returns 400 for an invalid job ID', async () => {
    const res = await request(app).get('/api/results/bad-id/download');
    expect(res.status).toBe(400);
  });

  test('returns 404 for an unknown valid UUID', async () => {
    const res = await request(app).get(
      '/api/results/00000000-0000-4000-8000-000000000000/download',
    );
    expect(res.status).toBe(404);
  });
});

// ─── Prediction routes ────────────────────────────────────────────────────────

describe('POST /api/predictions/single', () => {
  test('returns 400 when required fields are missing', async () => {
    const res = await request(app)
      .post('/api/predictions/single')
      .send({});
    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  test('returns 400 for an invalid (non-UUID) modelJobId', async () => {
    const res = await request(app)
      .post('/api/predictions/single')
      .send({ modelJobId: 'not-a-uuid', features: [1, 2, 3] });
    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  test('returns 404 when modelJobId is a valid UUID but unknown', async () => {
    const res = await request(app)
      .post('/api/predictions/single')
      .send({
        modelJobId: '00000000-0000-4000-8000-000000000000',
        features: [1, 2, 3],
      });
    expect(res.status).toBe(404);
  });
});

describe('POST /api/predictions/batch', () => {
  test('returns 400 when no file is uploaded', async () => {
    const res = await request(app)
      .post('/api/predictions/batch')
      .field('modelJobId', '00000000-0000-4000-8000-000000000000');
    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });
});
