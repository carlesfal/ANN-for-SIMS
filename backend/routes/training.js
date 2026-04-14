const express = require('express');
const router = express.Router();
const upload = require('../middleware/upload');
const trainingController = require('../controllers/trainingController');

// POST /api/training/start - Start a new training job
router.post('/start', upload.single('dataFile'), trainingController.startTraining);

// GET /api/training/:jobId/status - Get job status and progress
router.get('/:jobId/status', trainingController.getStatus);

// DELETE /api/training/:jobId - Cancel a running job
router.delete('/:jobId', trainingController.cancelJob);

// GET /api/training/jobs - List all jobs
router.get('/', trainingController.listJobs);

module.exports = router;
