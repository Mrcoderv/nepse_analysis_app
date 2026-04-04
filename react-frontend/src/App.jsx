import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';

import Market from './components/Market';
import StockDetail from './components/StockDetail';

import Portfolio from './components/Portfolio';

// Mock/Placeholder Components until implemented
const About = () => (
  <div className="max-w-2xl mx-auto space-y-6 text-neutral-400">
    <h2 className="text-2xl font-bold text-neutral-100">About NEPSE AI</h2>
    <p>This is a professional-grade stock analysis platform for the Nepal Stock Exchange (NEPSE).</p>
    <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-2xl">
      <h3 className="font-bold text-neutral-200 mb-2">Features</h3>
      <ul className="list-disc list-inside space-y-2 text-sm">
        <li>Real-time market data synchronization</li>
        <li>Advanced technical indicators (RSI, SMA, MACD)</li>
        <li>Automated structural trend detection</li>
        <li>Gemini 1.5 Flash AI powered stock insights</li>
      </ul>
    </div>
  </div>
);

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/market" element={<Market />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/stock/:symbol" element={<StockDetail />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<div className="text-center py-20 text-neutral-500 text-xl font-medium">Page Not Found</div>} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
