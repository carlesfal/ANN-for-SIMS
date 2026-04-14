/**
 * Middleware utilities for validation and rate limiting.
 */
const rateLimit = require('express-rate-limit');

// UUID v4 pattern
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Validate that req.params.jobId is a valid UUID v4.
 * Returns 400 if invalid, preventing path injection attacks.
 */
function validateJobId(req, res, next) {
  const { jobId } = req.params;
  if (!jobId || !UUID_REGEX.test(jobId)) {
    return res.status(400).json({ error: 'Invalid job ID format' });
  }
  next();
}

/**
 * Rate limiter for API routes — 100 requests per 15 minutes per IP.
 */
const apiRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' },
});

/**
 * Stricter rate limiter for resource-intensive routes (training/prediction).
 * 10 requests per 15 minutes per IP.
 */
const heavyRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many training/prediction requests, please try again later.' },
});

module.exports = { validateJobId, apiRateLimiter, heavyRateLimiter };
