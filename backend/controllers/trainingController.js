const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const config = require('../config');
const { createJob, updateJob, getJob, getAllJobs } = require('../utils/jobQueue');

exports.startTraining = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No data file provided' });
    }

    const jobId = uuidv4();
    const {
      targetColumn = 'target',
      featureColumns = '',
      epochs = '100',
      folds = '10',
      learningRate = '0.001',
      units = '64',
      layers = '3',
      dropout = '0.2',
      batchSize = '32',
      usePretraining = 'true',
      patience = '20',
    } = req.body;

    // Create output directory for this job
    const outputDir = path.join(config.resultsDir, jobId);
    fs.mkdirSync(outputDir, { recursive: true });
    fs.mkdirSync(path.join(outputDir, 'plots'), { recursive: true });

    // Create job record
    createJob(jobId, {
      type: 'training',
      filePath: req.file.path,
      originalName: req.file.originalname,
      config: {
        targetColumn,
        featureColumns,
        epochs: parseInt(epochs),
        folds: parseInt(folds),
        learningRate: parseFloat(learningRate),
        units: parseInt(units),
        layers: parseInt(layers),
        dropout: parseFloat(dropout),
        batchSize: parseInt(batchSize),
        usePretraining: usePretraining === 'true',
        patience: parseInt(patience),
      },
    });

    res.status(202).json({
      jobId,
      status: 'queued',
      message: 'Training job queued successfully',
    });

    // Spawn Python training process asynchronously
    _spawnTrainingProcess(jobId, req.file.path, outputDir, {
      targetColumn,
      featureColumns,
      epochs,
      folds,
      learningRate,
      units,
      layers,
      dropout,
      batchSize,
      usePretraining,
      patience,
    });
  } catch (err) {
    console.error('[TRAINING] Error starting training:', err);
    res.status(500).json({ error: 'Failed to start training job', details: err.message });
  }
};

function _spawnTrainingProcess(jobId, filePath, outputDir, params) {
  const scriptPath = path.join(config.pythonDir, 'train_model.py');

  const args = [
    scriptPath,
    '--input', filePath,
    '--output-dir', outputDir,
    '--job-id', jobId,
    '--target-column', params.targetColumn,
    '--epochs', params.epochs,
    '--folds', params.folds,
    '--learning-rate', params.learningRate,
    '--units', params.units,
    '--layers', params.layers,
    '--dropout', params.dropout,
    '--batch-size', params.batchSize,
    '--patience', params.patience,
  ];

  if (params.featureColumns) {
    args.push('--feature-columns', params.featureColumns);
  }
  if (params.usePretraining === 'true') {
    args.push('--use-pretraining');
  }

  updateJob(jobId, { status: 'running', startedAt: new Date().toISOString(), progress: 0, logs: [] });

  const pythonProcess = spawn(config.pythonExecutable, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  updateJob(jobId, { pid: pythonProcess.pid });

  let stdoutBuffer = '';

  pythonProcess.stdout.on('data', (data) => {
    const text = data.toString();
    stdoutBuffer += text;
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop();

    lines.forEach((line) => {
      if (!line.trim()) return;
      _handleProgressLine(jobId, line.trim());
    });
  });

  pythonProcess.stderr.on('data', (data) => {
    const text = data.toString().trim();
    if (text) {
      console.error(`[TRAINING][${jobId}] STDERR:`, text);
      const job = getJob(jobId);
      if (job) {
        const logs = [...(job.logs || []), { type: 'error', message: text, timestamp: new Date().toISOString() }];
        updateJob(jobId, { logs });
      }
    }
  });

  pythonProcess.on('close', (code) => {
    const job = getJob(jobId);
    if (code === 0) {
      // Load results from output directory
      const resultsPath = path.join(outputDir, 'results.json');
      let results = null;
      if (fs.existsSync(resultsPath)) {
        try { results = JSON.parse(fs.readFileSync(resultsPath, 'utf8')); } catch (e) { /* ignore */ }
      }
      updateJob(jobId, {
        status: 'completed',
        progress: 100,
        completedAt: new Date().toISOString(),
        results,
      });
      console.log(`[TRAINING][${jobId}] Completed successfully`);

      // Regenerate TF Serving model config so the new model is discoverable
      _regenerateTfServingConfig();
    } else {
      updateJob(jobId, {
        status: 'failed',
        completedAt: new Date().toISOString(),
        error: `Python process exited with code ${code}`,
      });
      console.error(`[TRAINING][${jobId}] Failed with exit code ${code}`);
    }
  });

  pythonProcess.on('error', (err) => {
    updateJob(jobId, {
      status: 'failed',
      completedAt: new Date().toISOString(),
      error: err.message,
    });
    console.error(`[TRAINING][${jobId}] Process error:`, err.message);
  });
}

function _handleProgressLine(jobId, line) {
  const job = getJob(jobId);
  if (!job) return;

  const logs = [...(job.logs || []), { type: 'info', message: line, timestamp: new Date().toISOString() }];
  let updates = { logs };

  // Parse structured progress messages
  try {
    if (line.startsWith('PROGRESS:')) {
      const data = JSON.parse(line.slice(9));
      updates.progress = data.progress || job.progress;
      updates.currentFold = data.fold;
      updates.currentEpoch = data.epoch;
      updates.metrics = data.metrics;
    } else if (line.startsWith('STATUS:')) {
      updates.statusMessage = line.slice(7).trim();
    }
  } catch (e) {
    // Not structured, just log it
  }

  updateJob(jobId, updates);
}

/**
 * Regenerate the TF Serving model_config_list file by invoking the
 * export_savedmodel.py helper.  Runs fire-and-forget — a failure here
 * does not affect the training result.
 */
function _regenerateTfServingConfig() {
  const scriptPath = path.join(config.pythonDir, 'export_savedmodel.py');
  const proc = spawn(config.pythonExecutable, [
    scriptPath,
    '--models-root', config.modelsDir,
    '--generate-config',
  ]);
  proc.on('close', (code) => {
    if (code === 0) {
      console.log('[TF-SERVING] Model config regenerated successfully');
    } else {
      console.warn(`[TF-SERVING] Config regeneration exited with code ${code}`);
    }
  });
  proc.on('error', (err) => {
    console.warn('[TF-SERVING] Config regeneration failed:', err.message);
  });
}

exports.getStatus = (req, res) => {
  const { jobId } = req.params;
  const job = getJob(jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  res.json({
    jobId,
    status: job.status,
    progress: job.progress || 0,
    currentFold: job.currentFold,
    currentEpoch: job.currentEpoch,
    metrics: job.metrics,
    statusMessage: job.statusMessage,
    logs: (job.logs || []).slice(-50), // Return last 50 log entries
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt,
    error: job.error,
    config: job.config,
    results: job.status === 'completed' ? job.results : undefined,
  });
};

exports.cancelJob = (req, res) => {
  const { jobId } = req.params;
  const job = getJob(jobId);

  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  if (job.pid) {
    try {
      process.kill(job.pid, 'SIGTERM');
    } catch (e) {
      // Process may have already exited
    }
  }

  updateJob(jobId, { status: 'cancelled', completedAt: new Date().toISOString() });
  res.json({ message: 'Job cancelled', jobId });
};

exports.listJobs = (req, res) => {
  const jobs = getAllJobs().map(([id, job]) => ({
    jobId: id,
    status: job.status,
    progress: job.progress || 0,
    createdAt: job.createdAt,
    completedAt: job.completedAt,
    originalName: job.originalName,
    type: job.type,
  }));
  res.json({ jobs });
};
