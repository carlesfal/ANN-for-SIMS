const express = require('express');
const router = express.Router();
const upload = require('../middleware/upload');
const predictionController = require('../controllers/predictionController');
const { heavyRateLimiter } = require('../middleware/validation');

// POST /api/predictions/batch - Batch predictions from uploaded file
router.post('/batch', heavyRateLimiter, upload.single('dataFile'), predictionController.batchPredict);

// POST /api/predictions/single - Single row prediction from JSON body
router.post('/single', heavyRateLimiter, predictionController.singlePredict);

module.exports = router;
