const WebSocket = require('ws');

// In-memory job store
const jobs = new Map();
let wssRef = null;

function attachWebSocketServer(wss) {
  wssRef = wss;
}

function createJob(jobId, data) {
  jobs.set(jobId, {
    ...data,
    status: 'queued',
    progress: 0,
    logs: [],
    createdAt: new Date().toISOString(),
  });
  _broadcast(jobId, 'job_created', jobs.get(jobId));
  return jobs.get(jobId);
}

function updateJob(jobId, updates) {
  const job = jobs.get(jobId);
  if (!job) return null;

  const updated = { ...job, ...updates };
  jobs.set(jobId, updated);
  _broadcast(jobId, 'job_updated', updated);
  return updated;
}

function getJob(jobId) {
  return jobs.get(jobId) || null;
}

function getAllJobs() {
  return Array.from(jobs.entries());
}

function deleteJob(jobId) {
  return jobs.delete(jobId);
}

function _broadcast(jobId, eventType, jobData) {
  if (!wssRef) return;
  const message = JSON.stringify({
    type: eventType,
    jobId,
    data: {
      status: jobData.status,
      progress: jobData.progress,
      currentFold: jobData.currentFold,
      currentEpoch: jobData.currentEpoch,
      metrics: jobData.metrics,
      statusMessage: jobData.statusMessage,
      logs: (jobData.logs || []).slice(-10),
      completedAt: jobData.completedAt,
      error: jobData.error,
      results: jobData.status === 'completed' ? jobData.results : undefined,
    },
    timestamp: new Date().toISOString(),
  });

  wssRef.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN && client.subscribedJobId === jobId) {
      client.send(message);
    }
  });
}

// Cleanup old completed jobs after 24 hours
setInterval(() => {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  jobs.forEach((job, id) => {
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      const ts = new Date(job.completedAt || job.createdAt).getTime();
      if (ts < cutoff) {
        jobs.delete(id);
      }
    }
  });
}, 60 * 60 * 1000);

module.exports = { attachWebSocketServer, createJob, updateJob, getJob, getAllJobs, deleteJob };
