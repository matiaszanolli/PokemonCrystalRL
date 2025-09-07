"""
Web interface module for Pokemon Crystal RL monitoring.

This module defines the React components and layouts for the monitoring interface,
including:
- Dashboard layouts and grids
- Training metrics visualization
- Game state display
- Control panels
- Real-time updates
- Interactive charts
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import pkg_resources

# UI Component Definitions
UI_COMPONENTS = {
    "Dashboard": {
        "type": "component",
        "template": """
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
"""
    },
    
    "MetricsPanel": {
        "type": "component",
        "template": """
import React from 'react';
import {
    Box, Typography, Grid,
    Card, CardContent, CardHeader
} from '@mui/material';
import {
    LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend,
    ResponsiveContainer
} from 'recharts';

const MetricsPanel = ({ metrics }) => {
    const formatMetric = (value) => {
        if (typeof value === 'number') {
            return value.toFixed(2);
        }
        return value;
    };
    
    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Training Metrics
            </Typography>
            
            <Grid container spacing={2}>
                {/* Episode Rewards */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardHeader title="Episode Rewards" />
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={metrics.episode_rewards || []}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="episode" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="reward"
                                        stroke="#8884d8"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>
                
                {/* Loss Values */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardHeader title="Loss Values" />
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={metrics.losses || []}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="step" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="policy_loss"
                                        stroke="#82ca9d"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="value_loss"
                                        stroke="#ffc658"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>
                
                {/* Training Stats */}
                <Grid item xs={12}>
                    <Card>
                        <CardHeader title="Training Statistics" />
                        <CardContent>
                            <Grid container spacing={2}>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="subtitle2">
                                        Episodes
                                    </Typography>
                                    <Typography variant="h6">
                                        {metrics.total_episodes || 0}
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="subtitle2">
                                        Total Steps
                                    </Typography>
                                    <Typography variant="h6">
                                        {metrics.total_steps || 0}
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="subtitle2">
                                        Average Reward
                                    </Typography>
                                    <Typography variant="h6">
                                        {formatMetric(metrics.avg_reward)}
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="subtitle2">
                                        Learning Rate
                                    </Typography>
                                    <Typography variant="h6">
                                        {formatMetric(metrics.learning_rate)}
                                    </Typography>
                                </Grid>
                            </Grid>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default MetricsPanel;
"""
    },
    
    "GameViewer": {
        "type": "component",
        "template": """
import React, { useRef, useEffect } from 'react';
import {
    Box, Typography, Card, CardContent,
    CardHeader, Grid, IconButton
} from '@mui/material';
import {
    PlayArrow, Pause, SkipNext,
    Screenshot, Fullscreen
} from '@mui/icons-material';

const GameViewer = ({ gameState }) => {
    const canvasRef = useRef(null);
    
    useEffect(() => {
        if (!gameState.screen || !canvasRef.current) return;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        
        img.src = `data:image/png;base64,${gameState.screen}`;
    }, [gameState.screen]);
    
    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Game View
            </Typography>
            
            <Card>
                <CardHeader
                    action={
                        <Box>
                            <IconButton>
                                <Screenshot />
                            </IconButton>
                            <IconButton>
                                <Fullscreen />
                            </IconButton>
                        </Box>
                    }
                />
                <CardContent>
                    <Box
                        sx={{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            flexDirection: 'column'
                        }}
                    >
                        <canvas
                            ref={canvasRef}
                            width={480}
                            height={432}
                            style={{
                                border: '1px solid #ccc',
                                imageRendering: 'pixelated'
                            }}
                        />
                        
                        <Box sx={{ mt: 2 }}>
                            <IconButton>
                                <PlayArrow />
                            </IconButton>
                            <IconButton>
                                <Pause />
                            </IconButton>
                            <IconButton>
                                <SkipNext />
                            </IconButton>
                        </Box>
                    </Box>
                    
                    <Grid container spacing={2} sx={{ mt: 2 }}>
                        <Grid item xs={6}>
                            <Typography variant="subtitle2">
                                Game State
                            </Typography>
                            <Typography>
                                {gameState.state || 'Unknown'}
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <Typography variant="subtitle2">
                                Last Action
                            </Typography>
                            <Typography>
                                {gameState.last_action || 'None'}
                            </Typography>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>
        </Box>
    );
};

export default GameViewer;
"""
    },
    
    "ControlPanel": {
        "type": "component",
        "template": """
import React, { useState } from 'react';
import {
    Box, Typography, Button, TextField,
    FormControl, InputLabel, Select,
    MenuItem, Switch, FormControlLabel,
    Grid, Divider
} from '@mui/material';
import {
    PlayArrow, Stop, Save, Refresh,
    Settings
} from '@mui/icons-material';

const ControlPanel = () => {
    const [trainingEnabled, setTrainingEnabled] = useState(true);
    const [learningRate, setLearningRate] = useState(0.001);
    const [episodeLimit, setEpisodeLimit] = useState(1000);
    
    const handleStartTraining = () => {
        fetch('/api/training/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                command: 'start',
                parameters: {
                    learning_rate: learningRate,
                    episode_limit: episodeLimit
                }
            })
        });
    };
    
    const handleStopTraining = () => {
        fetch('/api/training/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                command: 'stop'
            })
        });
    };
    
    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Control Panel
            </Typography>
            
            <Grid container spacing={2}>
                <Grid item xs={12}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={trainingEnabled}
                                onChange={(e) => setTrainingEnabled(e.target.checked)}
                            />
                        }
                        label="Training Enabled"
                    />
                </Grid>
                
                <Grid item xs={12}>
                    <TextField
                        fullWidth
                        label="Learning Rate"
                        type="number"
                        value={learningRate}
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        inputProps={{
                            step: 0.0001,
                            min: 0
                        }}
                    />
                </Grid>
                
                <Grid item xs={12}>
                    <TextField
                        fullWidth
                        label="Episode Limit"
                        type="number"
                        value={episodeLimit}
                        onChange={(e) => setEpisodeLimit(parseInt(e.target.value))}
                        inputProps={{
                            step: 100,
                            min: 0
                        }}
                    />
                </Grid>
                
                <Grid item xs={12}>
                    <Divider sx={{ my: 2 }} />
                </Grid>
                
                <Grid item xs={6}>
                    <Button
                        fullWidth
                        variant="contained"
                        color="primary"
                        startIcon={<PlayArrow />}
                        onClick={handleStartTraining}
                    >
                        Start
                    </Button>
                </Grid>
                
                <Grid item xs={6}>
                    <Button
                        fullWidth
                        variant="contained"
                        color="secondary"
                        startIcon={<Stop />}
                        onClick={handleStopTraining}
                    >
                        Stop
                    </Button>
                </Grid>
                
                <Grid item xs={12}>
                    <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<Save />}
                    >
                        Save Model
                    </Button>
                </Grid>
                
                <Grid item xs={12}>
                    <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<Settings />}
                    >
                        Advanced Settings
                    </Button>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ControlPanel;
"""
    },
    
    "EventLog": {
        "type": "component",
        "template": """
import React from 'react';
import {
    Box, Typography, List, ListItem,
    ListItemText, ListItemIcon, IconButton,
    Paper, Divider
} from '@mui/material';
import {
    Error as ErrorIcon,
    Warning as WarningIcon,
    Info as InfoIcon,
    Clear as ClearIcon
} from '@mui/icons-material';

const EventLog = ({ events }) => {
    const getEventIcon = (type) => {
        switch (type.toLowerCase()) {
            case 'error':
                return <ErrorIcon color="error" />;
            case 'warning':
                return <WarningIcon color="warning" />;
            default:
                return <InfoIcon color="info" />;
        }
    };
    
    const formatTimestamp = (timestamp) => {
        return new Date(timestamp).toLocaleTimeString();
    };
    
    return (
        <Box>
            <Box sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 2
            }}>
                <Typography variant="h6">
                    Event Log
                </Typography>
                <IconButton size="small">
                    <ClearIcon />
                </IconButton>
            </Box>
            
            <Paper
                sx={{
                    height: 400,
                    overflow: 'auto'
                }}
            >
                <List dense>
                    {events.map((event, index) => (
                        <React.Fragment key={index}>
                            <ListItem>
                                <ListItemIcon>
                                    {getEventIcon(event.type)}
                                </ListItemIcon>
                                <ListItemText
                                    primary={event.message}
                                    secondary={formatTimestamp(event.timestamp)}
                                />
                            </ListItem>
                            <Divider component="li" />
                        </React.Fragment>
                    ))}
                </List>
            </Paper>
        </Box>
    );
};

export default EventLog;
"""
    },
    
    "StatusBar": {
        "type": "component",
        "template": """
import React from 'react';
import {
    AppBar, Toolbar, Typography,
    Box, Chip
} from '@mui/material';
import {
    Memory as MemoryIcon,
    Speed as SpeedIcon,
    Storage as StorageIcon
} from '@mui/icons-material';

const StatusBar = () => {
    return (
        <AppBar
            position="fixed"
            color="primary"
            sx={{
                top: 'auto',
                bottom: 0,
                backgroundColor: 'background.paper'
            }}
        >
            <Toolbar variant="dense">
                <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
                    <Chip
                        icon={<MemoryIcon />}
                        label="CPU: 25%"
                        size="small"
                        color="primary"
                    />
                    <Chip
                        icon={<StorageIcon />}
                        label="Memory: 1.2GB"
                        size="small"
                        color="primary"
                    />
                    <Chip
                        icon={<SpeedIcon />}
                        label="FPS: 60"
                        size="small"
                        color="primary"
                    />
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                    Last Update: {new Date().toLocaleTimeString()}
                </Typography>
            </Toolbar>
        </AppBar>
    );
};

export default StatusBar;
"""
    }
}

# Component build configuration
@dataclass
class ComponentConfig:
    """Configuration for a UI component."""
    name: str
    type: str
    template: str
    dependencies: List[str] = None
    styles: Dict[str, Any] = None


class WebInterface:
    """Manager for web interface components and assets."""
    
    def __init__(self, static_dir: str = "static"):
        self.static_dir = Path(static_dir)
        self.components: Dict[str, ComponentConfig] = {}
        self.logger = logging.getLogger("web_interface")
        
        # Ensure static directory exists
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Load component definitions
        self._load_components()
        
        self.logger.info("🎨 Web interface components loaded")
    
    def _load_components(self) -> None:
        """Load component definitions from UI_COMPONENTS."""
        for name, config in UI_COMPONENTS.items():
            self.components[name] = ComponentConfig(
                name=name,
                type=config["type"],
                template=config["template"],
                dependencies=config.get("dependencies", []),
                styles=config.get("styles", {})
            )
    
    def build_components(self) -> None:
        """Build React components to static files."""
        try:
            components_dir = self.static_dir / "components"
            components_dir.mkdir(exist_ok=True)
            
            for name, config in self.components.items():
                component_file = components_dir / f"{name}.jsx"
                with open(component_file, "w") as f:
                    f.write(config.template)
            
            self.logger.info("📦 Components built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build components: {e}")
            raise
    
    def get_component_config(self, name: str) -> Optional[ComponentConfig]:
        """Get configuration for a specific component."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all available components."""
        return list(self.components.keys())
    
    def validate_component(self, name: str) -> bool:
        """Validate a component's configuration."""
        config = self.get_component_config(name)
        if not config:
            return False
        
        # Check required fields
        if not config.name or not config.type or not config.template:
            return False
        
        # Validate dependencies
        if config.dependencies:
            try:
                for dep in config.dependencies:
                    pkg_resources.require(dep)
            except pkg_resources.DistributionNotFound:
                return False
        
        return True
    
    def save_component_metadata(self) -> None:
        """Save component metadata to JSON file."""
        try:
            metadata = {
                name: {
                    "type": config.type,
                    "dependencies": config.dependencies,
                    "styles": config.styles
                }
                for name, config in self.components.items()
            }
            
            metadata_file = self.static_dir / "components.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("📝 Component metadata saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save component metadata: {e}")
            raise
    
    def load_component_metadata(self) -> Dict[str, Any]:
        """Load component metadata from JSON file."""
        try:
            metadata_file = self.static_dir / "components.json"
            if not metadata_file.exists():
                return {}
            
            with open(metadata_file, "r") as f:
                return json.load(f)
            
        except Exception as e:
            self.logger.error(f"Failed to load component metadata: {e}")
            return {}
