'use strict';

/**
 * Unit tests for middleware/validation.js
 */

const { validateJobId, sanitizeJobId, UUID_REGEX } = require('../middleware/validation');

// ─── UUID_REGEX ──────────────────────────────────────────────────────────────

describe('UUID_REGEX', () => {
  const validUuids = [
    '550e8400-e29b-41d4-a716-446655440000',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    '00000000-0000-4000-8000-000000000000',
  ];

  const invalidUuids = [
    '',
    'not-a-uuid',
    '550e8400-e29b-41d4-a716',           // too short
    '550e8400-e29b-41d4-a716-44665544000g', // invalid hex char
    '../etc/passwd',
    '550e8400-e29b-31d4-a716-446655440000', // version 3, not 4
  ];

  test.each(validUuids)('accepts valid UUID v4: %s', (uuid) => {
    expect(UUID_REGEX.test(uuid)).toBe(true);
  });

  test.each(invalidUuids)('rejects invalid value: %s', (value) => {
    expect(UUID_REGEX.test(value)).toBe(false);
  });
});

// ─── validateJobId middleware ────────────────────────────────────────────────

function makeMockReqRes(jobId) {
  const req = { params: { jobId } };
  const res = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
  };
  const next = jest.fn();
  return { req, res, next };
}

describe('validateJobId', () => {
  test('calls next() for a valid UUID v4', () => {
    const { req, res, next } = makeMockReqRes('550e8400-e29b-41d4-a716-446655440000');
    validateJobId(req, res, next);
    expect(next).toHaveBeenCalledTimes(1);
    expect(res.status).not.toHaveBeenCalled();
  });

  test('returns 400 for an empty jobId', () => {
    const { req, res, next } = makeMockReqRes('');
    validateJobId(req, res, next);
    expect(res.status).toHaveBeenCalledWith(400);
    expect(res.json).toHaveBeenCalledWith({ error: 'Invalid job ID format' });
    expect(next).not.toHaveBeenCalled();
  });

  test('returns 400 for a path-traversal string', () => {
    const { req, res, next } = makeMockReqRes('../../../etc/passwd');
    validateJobId(req, res, next);
    expect(res.status).toHaveBeenCalledWith(400);
    expect(next).not.toHaveBeenCalled();
  });

  test('returns 400 for a UUID v1 (not v4)', () => {
    const { req, res, next } = makeMockReqRes('550e8400-e29b-11d4-a716-446655440000');
    validateJobId(req, res, next);
    expect(res.status).toHaveBeenCalledWith(400);
    expect(next).not.toHaveBeenCalled();
  });

  test('returns 400 when jobId is undefined', () => {
    const req = { params: {} };
    const res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    const next = jest.fn();
    validateJobId(req, res, next);
    expect(res.status).toHaveBeenCalledWith(400);
    expect(next).not.toHaveBeenCalled();
  });
});

// ─── sanitizeJobId ────────────────────────────────────────────────────────────

describe('sanitizeJobId', () => {
  test('returns the UUID in lowercase for a valid input', () => {
    const input = '550E8400-E29B-41D4-A716-446655440000';
    const result = sanitizeJobId(input);
    expect(result).toBe('550e8400-e29b-41d4-a716-446655440000');
  });

  test('returns unchanged lowercase UUID', () => {
    const input = '550e8400-e29b-41d4-a716-446655440000';
    expect(sanitizeJobId(input)).toBe(input);
  });

  test('throws for an invalid ID', () => {
    expect(() => sanitizeJobId('invalid')).toThrow('Invalid job ID');
  });

  test('throws for a path-traversal attempt', () => {
    expect(() => sanitizeJobId('../etc/passwd')).toThrow('Invalid job ID');
  });

  test('throws for null', () => {
    expect(() => sanitizeJobId(null)).toThrow('Invalid job ID');
  });

  test('throws for empty string', () => {
    expect(() => sanitizeJobId('')).toThrow('Invalid job ID');
  });
});
