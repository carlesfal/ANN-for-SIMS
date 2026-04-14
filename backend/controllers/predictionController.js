const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const config = require('../config');
const { sanitizeJobId } = require('../middleware/validation');

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function validateModelJobId(modelJobId) {
  return modelJobId && UUID_REGEX.test(modelJobId);
}

exports.batchPredict = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No data file provided' });
    }

    const { modelJobId, targetColumn = 'target', featureColumns = '' } = req.body;

    if (!modelJobId) {
      return res.status(400).json({ error: 'modelJobId is required' });
    }

    if (!validateModelJobId(modelJobId)) {
      return res.status(400).json({ error: 'Invalid modelJobId format' });
    }

    // Sanitize at point of use for defense-in-depth
    const safeModelJobId = sanitizeJobId(modelJobId);
    const modelDir = path.join(config.modelsDir, safeModelJobId);
    if (!fs.existsSync(modelDir)) {
      return res.status(404).json({ error: 'Model not found for the specified job' });
    }

    const predictionId = uuidv4();
    const outputDir = path.join(config.resultsDir, predictionId);
    fs.mkdirSync(outputDir, { recursive: true });

    const scriptPath = path.join(config.pythonDir, 'predict.py');
    const args = [
      scriptPath,
      '--input', req.file.path,
      '--model-dir', modelDir,
      '--output-dir', outputDir,
      '--target-column', targetColumn,
    ];

    if (featureColumns) {
      args.push('--feature-columns', featureColumns);
    }

    await new Promise((resolve, reject) => {
      const proc = spawn(config.pythonExecutable, args, {
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', (d) => { stdout += d.toString(); });
      proc.stderr.on('data', (d) => { stderr += d.toString(); });

      proc.on('close', (code) => {
        if (code === 0) resolve(stdout);
        else reject(new Error(`Prediction failed: ${stderr}`));
      });
      proc.on('error', reject);
    });

    const resultsPath = path.join(outputDir, 'predictions.json');
    if (!fs.existsSync(resultsPath)) {
      return res.status(500).json({ error: 'Prediction output not found' });
    }

    const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
    res.json({ predictionId, results });
  } catch (err) {
    console.error('[PREDICT] Batch predict error:', err);
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
};

exports.singlePredict = async (req, res) => {
  try {
    const { modelJobId, features } = req.body;

    if (!modelJobId || !features) {
      return res.status(400).json({ error: 'modelJobId and features are required' });
    }

    if (!validateModelJobId(modelJobId)) {
      return res.status(400).json({ error: 'Invalid modelJobId format' });
    }

    // Sanitize at point of use for defense-in-depth
    const safeModelJobId = sanitizeJobId(modelJobId);
    const modelDir = path.join(config.modelsDir, safeModelJobId);
    if (!fs.existsSync(modelDir)) {
      return res.status(404).json({ error: 'Model not found for the specified job' });
    }

    const inputData = JSON.stringify(features);
    const scriptPath = path.join(config.pythonDir, 'predict.py');
    const args = [
      scriptPath,
      '--model-dir', modelDir,
      '--single',
      '--input-json', inputData,
    ];

    let stdout = '';
    let stderr = '';

    await new Promise((resolve, reject) => {
      const proc = spawn(config.pythonExecutable, args, {
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });
      proc.stdout.on('data', (d) => { stdout += d.toString(); });
      proc.stderr.on('data', (d) => { stderr += d.toString(); });
      proc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`Single prediction failed: ${stderr}`));
      });
      proc.on('error', reject);
    });

    const result = JSON.parse(stdout);
    res.json(result);
  } catch (err) {
    console.error('[PREDICT] Single predict error:', err);
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
};
