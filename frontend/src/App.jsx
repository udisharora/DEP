import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Scanner from './pages/Scanner';

function App() {
  return (
    <Router>
      <Navbar />
      <main style={{ flex: 1, padding: '40px 0', marginTop: '60px' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/scanner" element={<Scanner />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
