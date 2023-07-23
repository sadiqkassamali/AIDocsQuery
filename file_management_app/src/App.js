import React from 'react';
import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Tab, Tabs } from '@mui/material';
import FileUpload from './components/FileUpload';
import GetAllFiles from './components/GetAllFiles';
import ApproveReject from './components/ApproveReject';
import QueryImage from './components/QueryImage';

function App() {
  return (
    <Router>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">File Management App</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ marginTop: '20px' }}>
        <Tabs>
          <Tab label="Upload" component={Link} to="/upload" />
          <Tab label="Get All Files" component={Link} to="/get-all-files" />
          <Tab label="Approve/Reject" component={Link} to="/approve-reject" />
          <Tab label="Query Image" component={Link} to="/query-image" />
        </Tabs>
        <Switch>
          <Route path="/upload" component={FileUpload} />
          <Route path="/get-all-files" component={GetAllFiles} />
          <Route path="/approve-reject" component={ApproveReject} />
          <Route path="/query-image" component={QueryImage} />
        </Switch>
      </Container>
    </Router>
  );
}

export default App;
