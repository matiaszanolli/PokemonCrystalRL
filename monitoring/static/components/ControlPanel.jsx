
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
