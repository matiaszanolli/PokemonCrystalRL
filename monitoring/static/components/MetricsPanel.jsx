
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
