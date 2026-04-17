const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const config = require('../config');
const { getJob } = require('../utils/jobQueue');
const { validateJobId, apiRateLimiter, sanitizeJobId } = require('../middleware/validation');

// Apply rate limiting to all results routes
router.use(apiRateLimiter);

// GET /api/results/:jobId - Get results for a completed job
router.get('/:jobId', validateJobId, (req, res) => {
  const jobId = sanitizeJobId(req.params.jobId);
  const job = getJob(jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  if (job.status !== 'completed') {
    return res.status(400).json({ error: 'Job not yet completed', status: job.status });
  }

  const resultsPath = path.join(config.resultsDir, jobId, 'results.json');
  if (!fs.existsSync(resultsPath)) {
    return res.status(404).json({ error: 'Results file not found' });
  }

  try {
    const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
    res.json(results);
  } catch (err) {
    res.status(500).json({ error: 'Failed to read results', details: err.message });
  }
});

// GET /api/results/:jobId/download - Download results as CSV/JSON
router.get('/:jobId/download', validateJobId, (req, res) => {
  const jobId = sanitizeJobId(req.params.jobId);
  const { format = 'json' } = req.query;
  const job = getJob(jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  // Only allow json or csv formats — prevents directory traversal via format param
  const ext = format === 'csv' ? 'csv' : 'json';
  const filename = `results_${jobId}.${ext}`;
  const filePath = path.join(config.resultsDir, jobId, filename);

  if (!fs.existsSync(filePath)) {
    const jsonPath = path.join(config.resultsDir, jobId, 'results.json');
    if (!fs.existsSync(jsonPath)) {
      return res.status(404).json({ error: 'Results file not found' });
    }
    if (format === 'json') {
      return res.download(jsonPath, `results_${jobId}.json`);
    }
    return res.status(404).json({ error: 'CSV format not available for this job' });
  }

  res.download(filePath, filename);
});

// GET /api/results/:jobId/plots - List available plot images
router.get('/:jobId/plots', validateJobId, (req, res) => {
  const jobId = sanitizeJobId(req.params.jobId);
  const plotsDir = path.join(config.resultsDir, jobId, 'plots');

  if (!fs.existsSync(plotsDir)) {
    return res.json({ plots: [] });
  }

  const plots = fs.readdirSync(plotsDir)
    .filter(f => /\.(png|jpg)$/i.test(f))
    .map(f => ({
      name: f,
      url: `/results/${jobId}/plots/${f}`,
    }));

  res.json({ plots });
});

module.exports = router;
