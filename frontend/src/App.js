import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ResultPage from './components/ResultPage';
import AboutPage from './components/AboutPage';

const App = () => (
  <Router>
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/upload" element={<UploadPage />} />
      <Route path="/results" element={<ResultPage />} />
      <Route path="/about" element={<AboutPage />} />
    </Routes>
  </Router>
);

export default App;
