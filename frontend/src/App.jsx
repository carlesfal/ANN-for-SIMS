import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Home from './pages/Home.jsx';
import Training from './pages/Training.jsx';
import Predictions from './pages/Predictions.jsx';
import Results from './pages/Results.jsx';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="brand-icon">⚡</span>
        <span className="brand-text">ANN-SIMS</span>
      </div>
      <div className="navbar-links">
        <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Home
        </NavLink>
        <NavLink to="/training" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Training
        </NavLink>
        <NavLink to="/predictions" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Predictions
        </NavLink>
        <NavLink to="/results" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Results
        </NavLink>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/training" element={<Training />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/results" element={<Results />} />
            <Route path="/results/:jobId" element={<Results />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <p>ANN-SIMS Prediction System &mdash; Built with TensorFlow &amp; React</p>
        </footer>
      </div>
    </BrowserRouter>
  );
}
