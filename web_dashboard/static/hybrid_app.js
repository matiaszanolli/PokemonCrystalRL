/**
 * Hybrid LLM-RL Training Dashboard
 * Real-time monitoring with efficient data visualization
 */

class HybridDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.isConnected = false;
        this.isPaused = false;

        // Data buffers for efficient rendering
        this.rewardHistory = [];
        this.decisionHistory = [];
        this.performanceHistory = [];
        this.logBuffer = [];

        // Chart instances
        this.charts = {
            distribution: null,
            performance: null
        };

        // Performance tracking
        this.lastUpdateTime = Date.now();
        this.updateCounter = 0;
        this.frameRate = 0;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.startPerformanceMonitoring();

        // Initial UI state
        this.updateConnectionStatus('CONNECTING');
        this.showNoData();
    }

    setupEventListeners() {
        // Fullscreen controls
        document.getElementById('fullscreen-btn').addEventListener('click', () => {
            this.toggleFullscreen();
        });

        document.getElementById('close-fullscreen').addEventListener('click', () => {
            this.closeFullscreen();
        });

        // Log controls
        document.getElementById('clear-logs').addEventListener('click', () => {
            this.clearLogs();
        });

        document.getElementById('pause-logs').addEventListener('click', () => {
            this.toggleLogPause();
        });

        // Time selector for distribution chart
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updateDistributionChart(e.target.dataset.period);
            });
        });

        // Chart controls
        document.querySelectorAll('.chart-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updatePerformanceChart(e.target.dataset.metric);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeFullscreen();
            if (e.key === 'f' && e.ctrlKey) {
                e.preventDefault();
                this.toggleFullscreen();
            }
            if (e.key === 'p' && e.ctrlKey) {
                e.preventDefault();
                this.toggleLogPause();
            }
        });
    }

    initializeCharts() {
        // Distribution donut chart
        const distributionCanvas = document.getElementById('distribution-chart');
        if (distributionCanvas) {
            this.charts.distribution = this.createDistributionChart(distributionCanvas);
        }

        // Performance line chart
        const performanceCanvas = document.getElementById('performance-chart');
        if (performanceCanvas) {
            this.charts.performance = this.createPerformanceChart(performanceCanvas);
        }
    }

    createDistributionChart(canvas) {
        const ctx = canvas.getContext('2d');
        return {
            canvas: ctx,
            data: { llm: 0, rl: 0, hybrid: 0, explore: 0 },
            draw: function() {
                const total = this.data.llm + this.data.rl + this.data.hybrid + this.data.explore;
                if (total === 0) return;

                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const radius = Math.min(centerX, centerY) - 10;
                const innerRadius = radius * 0.6;

                this.canvas.clearRect(0, 0, canvas.width, canvas.height);

                const colors = ['#7c3aed', '#059669', '#dc2626', '#ea580c'];
                const values = [this.data.llm, this.data.rl, this.data.hybrid, this.data.explore];
                const labels = ['LLM', 'RL', 'Hybrid', 'Explore'];

                let currentAngle = -Math.PI / 2;

                values.forEach((value, index) => {
                    if (value > 0) {
                        const sliceAngle = (value / total) * 2 * Math.PI;

                        // Draw slice
                        this.canvas.beginPath();
                        this.canvas.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
                        this.canvas.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
                        this.canvas.closePath();
                        this.canvas.fillStyle = colors[index];
                        this.canvas.fill();

                        // Draw percentage in center of slice
                        const midAngle = currentAngle + sliceAngle / 2;
                        const textRadius = (radius + innerRadius) / 2;
                        const textX = centerX + Math.cos(midAngle) * textRadius;
                        const textY = centerY + Math.sin(midAngle) * textRadius;

                        const percentage = Math.round((value / total) * 100);
                        if (percentage > 5) {
                            this.canvas.fillStyle = 'white';
                            this.canvas.font = 'bold 10px monospace';
                            this.canvas.textAlign = 'center';
                            this.canvas.fillText(percentage + '%', textX, textY);
                        }

                        currentAngle += sliceAngle;
                    }
                });
            }
        };
    }

    createPerformanceChart(canvas) {
        const ctx = canvas.getContext('2d');
        return {
            canvas: ctx,
            data: [],
            maxPoints: 100,
            metric: 'reward',
            draw: function() {
                if (this.data.length === 0) return;

                this.canvas.clearRect(0, 0, canvas.width, canvas.height);

                const padding = 20;
                const chartWidth = canvas.width - padding * 2;
                const chartHeight = canvas.height - padding * 2;

                // Find min/max values
                const values = this.data.map(d => d[this.metric] || 0);
                const minValue = Math.min(...values);
                const maxValue = Math.max(...values);
                const range = maxValue - minValue || 1;

                // Draw grid lines
                this.canvas.strokeStyle = '#333';
                this.canvas.lineWidth = 1;
                for (let i = 0; i <= 4; i++) {
                    const y = padding + (i / 4) * chartHeight;
                    this.canvas.beginPath();
                    this.canvas.moveTo(padding, y);
                    this.canvas.lineTo(padding + chartWidth, y);
                    this.canvas.stroke();
                }

                // Draw line
                if (this.data.length > 1) {
                    this.canvas.strokeStyle = '#3b82f6';
                    this.canvas.lineWidth = 2;
                    this.canvas.beginPath();

                    this.data.forEach((point, index) => {
                        const x = padding + (index / (this.data.length - 1)) * chartWidth;
                        const y = padding + chartHeight - ((point[this.metric] - minValue) / range) * chartHeight;

                        if (index === 0) {
                            this.canvas.moveTo(x, y);
                        } else {
                            this.canvas.lineTo(x, y);
                        }
                    });

                    this.canvas.stroke();
                }

                // Draw points
                this.canvas.fillStyle = '#3b82f6';
                this.data.forEach((point, index) => {
                    const x = padding + (index / Math.max(1, this.data.length - 1)) * chartWidth;
                    const y = padding + chartHeight - ((point[this.metric] - minValue) / range) * chartHeight;

                    this.canvas.beginPath();
                    this.canvas.arc(x, y, 2, 0, 2 * Math.PI);
                    this.canvas.fill();
                });

                // Draw labels
                this.canvas.fillStyle = '#666';
                this.canvas.font = '10px monospace';
                this.canvas.textAlign = 'right';
                this.canvas.fillText(maxValue.toFixed(1), padding - 5, padding + 3);
                this.canvas.fillText(minValue.toFixed(1), padding - 5, padding + chartHeight + 3);
            }
        };
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.hostname}:8081/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('🔗 WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('ONLINE');
                this.addLogEntry('success', 'WebSocket connection established');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('OFFLINE');
                this.addLogEntry('warning', 'WebSocket connection lost');
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.addLogEntry('error', 'WebSocket connection error');
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus('ERROR');
            this.scheduleReconnect();
        }
    }

    handleWebSocketMessage(data) {
        this.updateCounter++;

        // Debug logging for WebSocket messages
        console.log('📥 WebSocket message received:', {
            type: data.type,
            dataKeys: Object.keys(data.data || {}),
            updateCounter: this.updateCounter
        });

        switch (data.type) {
            case 'training_update':
                console.log('🎯 Processing training_update:', data.data);
                this.updateTrainingData(data.data);
                break;
            case 'decision_made':
                console.log('🧠 Processing decision_made:', data.data);
                this.addDecision(data.data);
                break;
            case 'game_screen':
                this.updateGameScreen(data.data);
                break;
            case 'system_status':
                this.updateSystemStatus(data.data);
                break;
            case 'log_entry':
                this.addLogEntry(data.data.level, data.data.message);
                break;
            case 'connection_established':
                console.log('✅ WebSocket connection established');
                this.addLogEntry('info', 'WebSocket connection established');
                break;
            default:
                console.log('❓ Unknown message type:', data.type, data);
        }
    }

    updateTrainingData(data) {
        console.log('🔄 updateTrainingData called with:', {
            episode: data.episode,
            isTraining: data.is_training,
            hasMetrics: !!data.metrics,
            hasTemporalMemory: !!data.temporal_memory,
            hasCurriculum: !!data.curriculum,
            dataStructure: Object.keys(data)
        });

        // Update status bar
        this.updateElement('current-episode', data.episode || 0);
        this.updateElement('max-episodes', data.max_episodes || 100);
        this.updateElement('current-action', data.action || 0);
        this.updateElement('actions-per-sec', (data.actions_per_second || 0).toFixed(1));

        // Update training status
        const status = data.is_training ? 'RUNNING' : 'IDLE';
        this.updateStatusBadge(status);
        console.log('📊 Status updated:', status);

        // Update decision mode
        this.updateDecisionMode(data.current_mode || 'hybrid');

        // Update key metrics
        if (data.metrics) {
            console.log('📈 Updating metrics:', data.metrics);
            this.updateMetrics(data.metrics);
        }

        // Update temporal memory stats
        if (data.temporal_memory) {
            console.log('🧠 Updating temporal memory:', data.temporal_memory);
            this.updateTemporalMemory(data.temporal_memory);
        }

        // Update curriculum progress
        if (data.curriculum) {
            console.log('🎓 Updating curriculum:', data.curriculum);
            this.updateCurriculum(data.curriculum);
        }

        // Update performance history
        this.addPerformanceData(data);
        console.log('✅ updateTrainingData completed');
    }

    updateMetrics(metrics) {
        // Total reward
        this.updateElement('total-reward', (metrics.total_reward || 0).toFixed(1));
        this.updateMetricChange('reward-change', metrics.reward_change || 0);

        // LLM performance
        this.updateElement('llm-success-rate', Math.round((metrics.llm_success_rate || 0) * 100) + '%');
        this.updateElement('llm-decisions-ratio', `${metrics.llm_decisions || 0}/${metrics.total_decisions || 0}`);

        // RL performance
        this.updateElement('rl-success-rate', Math.round((metrics.rl_success_rate || 0) * 100) + '%');
        this.updateElement('rl-decisions-ratio', `${metrics.rl_decisions || 0}/${metrics.total_decisions || 0}`);

        // Exploration
        this.updateElement('exploration-rate', Math.round((metrics.exploration_rate || 0) * 100) + '%');
        this.updateElement('exploration-ratio', `${metrics.exploration_decisions || 0}/${metrics.total_decisions || 0}`);

        // Update distribution chart
        if (this.charts.distribution) {
            this.charts.distribution.data = {
                llm: metrics.llm_decisions || 0,
                rl: metrics.rl_decisions || 0,
                hybrid: metrics.hybrid_decisions || 0,
                explore: metrics.exploration_decisions || 0
            };
            this.charts.distribution.draw();
        }

        // Update weights
        this.updateElement('current-llm-weight', (metrics.llm_weight || 0.7).toFixed(2));
        this.updateElement('current-rl-weight', (metrics.rl_weight || 0.3).toFixed(2));
        this.updateElement('mode-switches', metrics.mode_switches || 0);
    }

    updateTemporalMemory(memory) {
        this.updateElement('buffer-size', this.formatNumber(memory.buffer_size || 0));
        this.updateElement('episodes-stored', memory.episodes_stored || 0);
        this.updateElement('avg-novelty', (memory.avg_novelty || 0.5).toFixed(2));

        // Update memory usage bar
        const usage = Math.min(100, (memory.buffer_size || 0) / (memory.max_buffer_size || 100000) * 100);
        document.getElementById('memory-usage').style.width = usage + '%';
    }

    updateCurriculum(curriculum) {
        // Update current stage
        this.updateElement('curriculum-stage', (curriculum.current_stage || 'TUTORIAL').toUpperCase());

        // Update stage indicators
        document.querySelectorAll('.stage').forEach(stage => {
            stage.classList.remove('active');
        });
        const currentStage = document.querySelector(`[data-stage="${curriculum.current_stage}"]`);
        if (currentStage) {
            currentStage.classList.add('active');
        }

        // Update progress
        const progress = (curriculum.progress || 0) * 100;
        document.getElementById('curriculum-progress').style.width = progress + '%';

        // Update stats
        this.updateElement('curriculum-episodes', `${curriculum.episodes || 0}/${curriculum.max_episodes || 10}`);
        this.updateElement('curriculum-success', Math.round((curriculum.success_rate || 0) * 100) + '%');
    }

    addDecision(decision) {
        const stream = document.getElementById('decision-stream');

        // Create decision element
        const element = document.createElement('div');
        element.className = `decision-item ${decision.mode || 'hybrid'} fade-in`;

        const iconMap = {
            llm: '🧠',
            rl: '⚡',
            hybrid: '🔄',
            explore: '🎯'
        };

        element.innerHTML = `
            <div class="decision-icon">${iconMap[decision.mode] || '🔄'}</div>
            <div class="decision-details">
                <div class="decision-action">${this.formatAction(decision.action)}</div>
                <div class="decision-meta">${new Date().toLocaleTimeString()} • ${decision.reasoning || ''}</div>
            </div>
            <div class="decision-confidence">${Math.round((decision.confidence || 0) * 100)}%</div>
        `;

        // Add to stream
        stream.insertBefore(element, stream.firstChild);

        // Limit to last 50 decisions
        while (stream.children.length > 50) {
            stream.removeChild(stream.lastChild);
        }

        // Remove no-data message
        const noData = stream.querySelector('.no-data');
        if (noData) {
            noData.remove();
        }

        // Store in history
        this.decisionHistory.unshift(decision);
        if (this.decisionHistory.length > 1000) {
            this.decisionHistory.pop();
        }
    }

    updateGameScreen(screenData) {
        const img = document.getElementById('game-screen');
        if (screenData.image) {
            img.src = `data:image/png;base64,${screenData.image}`;
        }

        // Update game state overlay
        if (screenData.game_state) {
            const state = screenData.game_state;
            this.updateElement('player-position', `${state.player_x || 0},${state.player_y || 0}`);
            this.updateElement('current-map', state.map_id || 0);
            this.updateElement('player-hp', `${state.player_hp || 0}/${state.player_max_hp || 100}`);
            this.updateElement('player-badges', state.badges || 0);
        }
    }

    updateSystemStatus(status) {
        // Update diagnostics
        this.updateElement('llm-response-time', (status.llm_response_time || 0) + 'ms');
        this.updateElement('rl-inference-time', (status.rl_inference_time || 0) + 'ms');
        this.updateElement('memory-usage-value', this.formatBytes(status.memory_usage || 0));
        this.updateElement('gpu-usage', Math.round(status.gpu_usage || 0) + '%');

        // Update status indicators
        this.updateDiagnosticStatus('llm-status', status.llm_response_time, 500);
        this.updateDiagnosticStatus('rl-status', status.rl_inference_time, 100);
        this.updateDiagnosticStatus('memory-status-indicator', status.memory_usage, 1024 * 1024 * 1024); // 1GB
        this.updateDiagnosticStatus('gpu-status', status.gpu_usage, 80);
    }

    addPerformanceData(data) {
        const point = {
            timestamp: Date.now(),
            reward: data.metrics?.total_reward || 0,
            success: data.metrics?.avg_success_rate || 0,
            confidence: data.metrics?.avg_confidence || 0
        };

        this.performanceHistory.push(point);
        if (this.performanceHistory.length > 200) {
            this.performanceHistory.shift();
        }

        // Update performance chart
        if (this.charts.performance) {
            this.charts.performance.data = this.performanceHistory;
            this.charts.performance.draw();
        }
    }

    addLogEntry(level, message) {
        if (this.isPaused) return;

        const stream = document.getElementById('log-stream');
        const element = document.createElement('div');
        element.className = `log-entry ${level} fade-in`;

        const timestamp = new Date().toLocaleTimeString();
        element.innerHTML = `
            <span class="log-timestamp">${timestamp}</span>
            <span class="log-message">${message}</span>
        `;

        stream.insertBefore(element, stream.firstChild);

        // Limit to last 100 entries
        while (stream.children.length > 100) {
            stream.removeChild(stream.lastChild);
        }

        // Store in buffer
        this.logBuffer.unshift({ timestamp, level, message });
        if (this.logBuffer.length > 500) {
            this.logBuffer.pop();
        }
    }

    // Utility functions
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateStatusBadge(status) {
        const badge = document.querySelector('.status-badge');
        if (badge) {
            badge.textContent = `${status === 'RUNNING' ? '▶' : '⏸'} ${status}`;
            badge.className = `status-badge ${status.toLowerCase()}`;
        }
    }

    updateDecisionMode(mode) {
        const indicator = document.getElementById('decision-mode');
        if (indicator) {
            const iconMap = { llm: '🧠', rl: '⚡', hybrid: '🔄', explore: '🎯' };
            indicator.textContent = `${iconMap[mode] || '🔄'} ${mode.toUpperCase()}`;
            indicator.className = `mode-indicator ${mode}`;
        }
    }

    updateConnectionStatus(status) {
        this.updateElement('connection-status', status);
        const element = document.getElementById('connection-status');
        if (element) {
            element.className = `status-value ${status.toLowerCase()}`;
        }
    }

    updateMetricChange(id, change) {
        const element = document.getElementById(id);
        if (element) {
            const sign = change >= 0 ? '+' : '';
            element.textContent = `${sign}${change.toFixed(1)}`;
            element.className = `metric-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    updateDiagnosticStatus(id, value, threshold) {
        const element = document.getElementById(id);
        if (element) {
            if (value > threshold) {
                element.textContent = 'WARN';
                element.className = 'diagnostic-status warning';
            } else if (value > threshold * 1.5) {
                element.textContent = 'ERR';
                element.className = 'diagnostic-status error';
            } else {
                element.textContent = 'OK';
                element.className = 'diagnostic-status ok';
            }
        }
    }

    formatAction(action) {
        const actionMap = {
            0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT',
            4: 'A', 5: 'B', 6: 'START', 7: 'SELECT'
        };
        return actionMap[action] || `Action ${action}`;
    }

    formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    formatBytes(bytes) {
        if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + 'GB';
        if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + 'MB';
        if (bytes >= 1024) return (bytes / 1024).toFixed(1) + 'KB';
        return bytes + 'B';
    }

    // Event handlers
    toggleFullscreen() {
        const modal = document.getElementById('fullscreen-modal');
        const img = document.getElementById('fullscreen-image');
        const gameScreen = document.getElementById('game-screen');

        modal.classList.remove('hidden');
        img.src = gameScreen.src;
    }

    closeFullscreen() {
        const modal = document.getElementById('fullscreen-modal');
        modal.classList.add('hidden');
    }

    clearLogs() {
        const stream = document.getElementById('log-stream');
        stream.innerHTML = '<div class="log-entry info"><span class="log-timestamp">' +
                          new Date().toLocaleTimeString() + '</span><span class="log-message">Logs cleared</span></div>';
        this.logBuffer = [];
    }

    toggleLogPause() {
        this.isPaused = !this.isPaused;
        const btn = document.getElementById('pause-logs');
        btn.textContent = this.isPaused ? '▶' : '⏸';

        if (!this.isPaused) {
            this.addLogEntry('info', 'Log streaming resumed');
        }
    }

    updateDistributionChart(period) {
        // Placeholder for time-based filtering
        // Would filter decisionHistory based on period
    }

    updatePerformanceChart(metric) {
        if (this.charts.performance) {
            this.charts.performance.metric = metric;
            this.charts.performance.draw();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);

            this.addLogEntry('info', `Reconnecting in ${delay/1000}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connectWebSocket();
            }, delay);
        } else {
            this.addLogEntry('error', 'Max reconnection attempts reached');
        }
    }

    startPerformanceMonitoring() {
        setInterval(() => {
            const now = Date.now();
            const elapsed = now - this.lastUpdateTime;
            this.frameRate = Math.round(1000 / elapsed);
            this.lastUpdateTime = now;

            // Update performance indicators
            this.updateElement('update-rate', this.frameRate);
        }, 1000);
    }

    showNoData() {
        // Initial state for various panels
        const panels = ['decision-stream'];
        panels.forEach(id => {
            const element = document.getElementById(id);
            if (element && element.children.length === 0) {
                element.innerHTML = '<div class="no-data">Waiting for training data...</div>';
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new HybridDashboard();
});