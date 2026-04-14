const express = require('express');
const router = express.Router();
const upload = require('../middleware/upload');
const trainingController = require('../controllers/trainingController');
const { validateJobId, heavyRateLimiter, apiRateLimiter } = require('../middleware/validation');

// POST /api/training/start - Start a new training job
router.post('/start', heavyRateLimiter, upload.single('dataFile'), trainingController.startTraining);

// GET /api/training/:jobId/status - Get job status and progress
router.get('/:jobId/status', apiRateLimiter, validateJobId, trainingController.getStatus);

// DELETE /api/training/:jobId - Cancel a running job
router.delete('/:jobId', apiRateLimiter, validateJobId, trainingController.cancelJob);

// GET /api/training/jobs - List all jobs
router.get('/', apiRateLimiter, trainingController.listJobs);

module.exports = router;
