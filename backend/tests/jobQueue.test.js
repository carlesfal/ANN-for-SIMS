'use strict';

/**
 * Unit tests for utils/jobQueue.js
 */

// Require fresh module state for each test suite run
let jobQueue;

beforeEach(() => {
  // Re-require to get a fresh in-memory store each time
  jest.resetModules();
  jobQueue = require('../utils/jobQueue');
});

describe('createJob', () => {
  test('creates a job with queued status and default fields', () => {
    const job = jobQueue.createJob('job-1', { type: 'training' });

    expect(job).toMatchObject({
      type: 'training',
      status: 'queued',
      progress: 0,
    });
    expect(Array.isArray(job.logs)).toBe(true);
    expect(typeof job.createdAt).toBe('string');
  });

  test('stores the job so it can be retrieved', () => {
    jobQueue.createJob('job-2', { type: 'prediction' });
    const retrieved = jobQueue.getJob('job-2');
    expect(retrieved).not.toBeNull();
    expect(retrieved.type).toBe('prediction');
  });

  test('allows custom fields in the job data', () => {
    jobQueue.createJob('job-3', { type: 'training', config: { folds: 5 } });
    const job = jobQueue.getJob('job-3');
    expect(job.config.folds).toBe(5);
  });
});

describe('updateJob', () => {
  test('updates existing job fields', () => {
    jobQueue.createJob('job-upd', { type: 'training' });
    const updated = jobQueue.updateJob('job-upd', { status: 'running', progress: 50 });

    expect(updated.status).toBe('running');
    expect(updated.progress).toBe(50);
  });

  test('merges updates without overwriting unrelated fields', () => {
    jobQueue.createJob('job-merge', { type: 'training', config: { folds: 10 } });
    jobQueue.updateJob('job-merge', { status: 'running' });

    const job = jobQueue.getJob('job-merge');
    expect(job.config.folds).toBe(10); // original field preserved
    expect(job.status).toBe('running');
  });

  test('returns null for a non-existent job', () => {
    const result = jobQueue.updateJob('does-not-exist', { status: 'running' });
    expect(result).toBeNull();
  });
});

describe('getJob', () => {
  test('returns null for an unknown job ID', () => {
    expect(jobQueue.getJob('unknown-id')).toBeNull();
  });

  test('returns the correct job by ID', () => {
    jobQueue.createJob('job-get', { type: 'training' });
    const job = jobQueue.getJob('job-get');
    expect(job).not.toBeNull();
    expect(job.type).toBe('training');
  });
});

describe('getAllJobs', () => {
  test('returns an array of [id, job] entries', () => {
    jobQueue.createJob('job-all-1', { type: 'training' });
    jobQueue.createJob('job-all-2', { type: 'prediction' });

    const all = jobQueue.getAllJobs();
    expect(Array.isArray(all)).toBe(true);
    const ids = all.map(([id]) => id);
    expect(ids).toContain('job-all-1');
    expect(ids).toContain('job-all-2');
  });

  test('returns an empty array when no jobs exist', () => {
    // Fresh module — no jobs
    const all = jobQueue.getAllJobs();
    expect(all).toHaveLength(0);
  });
});

describe('deleteJob', () => {
  test('removes a job from the store', () => {
    jobQueue.createJob('job-del', { type: 'training' });
    jobQueue.deleteJob('job-del');
    expect(jobQueue.getJob('job-del')).toBeNull();
  });

  test('returns false when deleting a non-existent job', () => {
    const result = jobQueue.deleteJob('no-such-job');
    expect(result).toBe(false);
  });
});
