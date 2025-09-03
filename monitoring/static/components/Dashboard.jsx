
import React, { useState, useEffect } from 'react';
import {
    Container, Grid, Paper, Typography, Box,
    AppBar, Toolbar, IconButton, Menu, MenuItem
} from '@mui/material';
import {
    Menu as MenuIcon,
    Refresh as RefreshIcon,
    Settings as SettingsIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

import MetricsPanel from './components/MetricsPanel';
import GameViewer from './components/GameViewer';
import ControlPanel from './components/ControlPanel';
import EventLog from './components/EventLog';
import StatusBar from './components/StatusBar';

const Dashboard = () => {
    const [refreshInterval, setRefreshInterval] = useState(1000);
    const [metrics, setMetrics] = useState({});
    const [gameState, setGameState] = useState({});
    const [events, setEvents] = useState([]);
    
    useEffect(() => {
        const ws = new WebSocket('ws://' + window.location.host + '/ws');
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'metrics':
                    setMetrics(data.data);
                    break;
                case 'game_state':
                    setGameState(data.data);
                    break;
                case 'event':
                    setEvents(prev => [...prev, data.data]);
                    break;
            }
        };
        
        return () => ws.close();
    }, []);
    
    return (
        <Box sx={{ flexGrow: 1 }}>
            <AppBar position="static">
                <Toolbar>
                    <IconButton edge="start" color="inherit" aria-label="menu">
                        <MenuIcon />
                    </IconButton>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Pokemon Crystal RL Monitor
                    </Typography>
                    <IconButton color="inherit">
                        <RefreshIcon />
                    </IconButton>
                    <IconButton color="inherit">
                        <SettingsIcon />
                    </IconButton>
                </Toolbar>
            </AppBar>
            
            <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                <Grid container spacing={3}>
                    {/* Metrics Panel */}
                    <Grid item xs={12} md={8}>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                            <MetricsPanel metrics={metrics} />
                        </Paper>
                    </Grid>
                    
                    {/* Control Panel */}
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                            <ControlPanel />
                        </Paper>
                    </Grid>
                    
                    {/* Game Viewer */}
                    <Grid item xs={12} md={8}>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                            <GameViewer gameState={gameState} />
                        </Paper>
                    </Grid>
                    
                    {/* Event Log */}
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                            <EventLog events={events} />
                        </Paper>
                    </Grid>
                </Grid>
            </Container>
            
            <StatusBar />
        </Box>
    );
};

export default Dashboard;
