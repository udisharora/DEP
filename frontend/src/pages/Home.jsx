import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, ShieldCheck, Zap, Layers } from 'lucide-react';
import './Home.css';

const Home = () => {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content animate-fade-in">
          <div className="pill-badge">v2.0 Architecture Live</div>
          <h1 className="hero-title">
            Auto-Adaptive <br/>
            <span className="text-gradient">License Plate Recognition</span>
          </h1>
          <p className="hero-subtitle">
            A detection-first ALPR pipeline powered by deep learning. Built to always deblur before detection, featuring DarkIR, DeHaze and DeRain fallback nodes, culminating in an upscaled TrOCR extraction phase.
          </p>
          <div className="hero-actions">
            <Link to="/scanner" className="btn-primary">
              Launch Scanner <ArrowRight size={18} />
            </Link>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="btn-secondary">
              View Documentation
            </a>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="features-section container">
        <h2 className="section-header animate-fade-in stagger-1">Pipeline Highlights</h2>
        <div className="features-grid">
          <div className="feature-card glass-panel animate-fade-in stagger-2">
            <div className="feature-icon"><Layers size={24} color="var(--primary)" /></div>
            <h3>Test-Time Augmentation</h3>
            <p>Employs dynamic image padding scales combined with majority voting to drastically minimize false negatives and guarantee robust character extraction.</p>
          </div>
          <div className="feature-card glass-panel animate-fade-in stagger-3">
            <div className="feature-icon"><Zap size={24} color="var(--accent)" /></div>
            <h3>Adaptive Fallbacks</h3>
            <p>If initial detection fails, the image routes through specialized state-of-the-art restorative models: DarkIR, DeHaze, and DeRain consecutively.</p>
          </div>
          <div className="feature-card glass-panel animate-fade-in stagger-4">
            <div className="feature-icon"><ShieldCheck size={24} color="#10b981" /></div>
            <h3>RegCheck Metadata Verification</h3>
            <p>Decoded license strings are dynamically cross-referenced against live API endpoints, confirming Registration States, Vahan owners, engine CC, and validity.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
