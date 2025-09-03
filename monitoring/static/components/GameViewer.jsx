
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
