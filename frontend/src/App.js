import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ResultPage from './components/ResultPage';
import AboutPage from './components/AboutPage';

const App = () => (
  <Router>
    <Switch>
      <Route path="/" exact component={HomePage} />
      <Route path="/upload" component={UploadPage} />
      <Route path="/results" component={ResultPage} />
      <Route path="/about" component={AboutPage} />
    </Switch>
  </Router>
);

export default App;
