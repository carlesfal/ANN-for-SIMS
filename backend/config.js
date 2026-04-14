const path = require('path');

module.exports = {
  port: process.env.PORT || 5000,
  frontendUrl: process.env.FRONTEND_URL || 'http://localhost:3000',
  uploadsDir: path.join(__dirname, 'uploads'),
  modelsDir: path.join(__dirname, 'models'),
  resultsDir: path.join(__dirname, 'results'),
  pythonDir: path.join(__dirname, 'python'),
  pythonExecutable: process.env.PYTHON_EXECUTABLE || 'python3',
  maxFileSize: 50 * 1024 * 1024, // 50MB
  allowedFileTypes: ['.csv', '.xlsx', '.xls'],
  jobTimeoutMs: 30 * 60 * 1000, // 30 minutes
  wsHeartbeatInterval: 30000,
};
