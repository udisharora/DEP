import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ScanSearch, Home, Activity } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar glass-panel">
      <div className="container nav-container">
        <Link to="/" className="nav-brand">
          <Activity className="nav-logo" />
          <span className="nav-title">ALPR<span className="text-gradient">Vision</span></span>
        </Link>
        
        <div className="nav-links">
          <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
            <Home size={18} />
            <span>Home</span>
          </Link>
          <Link to="/scanner" className={`nav-link ${location.pathname === '/scanner' ? 'active' : ''}`}>
            <ScanSearch size={18} />
            <span>Scanner</span>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
