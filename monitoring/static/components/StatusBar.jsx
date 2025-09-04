
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
