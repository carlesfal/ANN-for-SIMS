require('dotenv').config();
const express = require('express');
const cors = require('cors');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const config = require('./config');

const trainingRoutes = require('./routes/training');
const predictionRoutes = require('./routes/predictions');
const resultsRoutes = require('./routes/results');
const { attachWebSocketServer } = require('./utils/jobQueue');

const app = express();
const server = http.createServer(app);

// WebSocket server
const wss = new WebSocket.Server({ server, path: '/ws' });
attachWebSocketServer(wss);

// CORS
app.use(cors({
  origin: [config.frontendUrl, 'http://localhost:3000', 'http://127.0.0.1:3000'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Static results
app.use('/results', express.static(config.resultsDir));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString(), version: '1.0.0' });
});

// API routes
app.use('/api/training', trainingRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/results', resultsRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('[ERROR]', err.stack);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});

// WebSocket heartbeat
const heartbeat = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) return ws.terminate();
    ws.isAlive = false;
    ws.ping();
  });
}, config.wsHeartbeatInterval);

wss.on('connection', (ws) => {
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });
  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data);
      if (msg.type === 'subscribe' && msg.jobId) {
        ws.subscribedJobId = msg.jobId;
      }
    } catch (e) {
      // Ignore malformed messages
    }
  });
});

wss.on('close', () => clearInterval(heartbeat));

server.listen(config.port, () => {
  console.log(`[SERVER] ANN-SIMS backend running on port ${config.port}`);
  console.log(`[SERVER] WebSocket server active on ws://localhost:${config.port}/ws`);
  console.log(`[SERVER] Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = { app, server, wss };
