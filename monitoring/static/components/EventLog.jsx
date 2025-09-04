
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
