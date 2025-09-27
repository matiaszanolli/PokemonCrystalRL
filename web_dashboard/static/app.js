/**
 * Unified Dashboard JavaScript for Pokemon Crystal RL
 *
 * This script handles all frontend functionality including:
 * - API communication
 * - WebSocket real-time updates
 * - UI updates and state management
 * - Error handling and recovery
 */

class PokemonDashboard {
    constructor() {
        // Configuration
        this.config = {
            apiBaseUrl: '',  // Same origin
            wsUrl: `ws://${window.location.hostname}:8081`,
            updateIntervals: {
                api: 2000,      // API polling interval (ms)
                screen: 100,    // Screen update interval (ms)
                stats: 1000     // Stats update interval (ms)
            },
            maxRetries: 5,
            retryDelay: 1000
        };

        // State
        this.state = {
            connected: false,
            wsConnected: false,
            retryCount: 0,
            lastUpdate: null,
            updateCount: 0,
            errorCount: 0
        };

        // WebSocket connection
        this.ws = null;
        this.wsReconnectTimer = null;

        // Performance tracking
        this.performance = {
            updateTimes: [],
            lastUpdateTime: 0,
            updateRate: 0
        };

        // Initialize dashboard
        this.init();
    }

    /**
     * Initialize the dashboard
     */
    init() {
        console.log('🎮 Initializing Pokemon Crystal RL Dashboard');

        // Setup event listeners
        this.setupEventListeners();

        // Start API polling
        this.startApiPolling();

        // Connect WebSocket
        this.connectWebSocket();

        // Start performance monitoring
        this.startPerformanceMonitoring();

        // Start periodic screen updates (fallback for WebSocket)
        this.startScreenUpdates();

        // Initialize visualizations
        this.initializeVisualizations();

        // Initialize A/B testing
        this.initializeABTesting();

        // Setup WebSocket handlers for A/B testing
        this.setupABTestingWebSocket();

        console.log('✅ Dashboard initialized successfully');
    }

    /**
     * Setup event listeners for UI interactions
     */
    setupEventListeners() {
        // Error banner dismiss
        const dismissError = document.getElementById('dismiss-error');
        if (dismissError) {
            dismissError.addEventListener('click', () => {
                this.hideError();
            });
        }

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.handlePageHidden();
            } else {
                this.handlePageVisible();
            }
        });

        // Handle window beforeunload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    /**
     * Start API polling for dashboard data
     */
    startApiPolling() {
        const pollApi = async () => {
            try {
                await this.updateDashboardData();
                this.state.retryCount = 0; // Reset retry count on success
            } catch (error) {
                console.error('API polling error:', error);
                this.handleApiError(error);
            }
        };

        // Initial load
        pollApi();

        // Set up interval
        this.apiInterval = setInterval(pollApi, this.config.updateIntervals.api);
    }

    /**
     * Update all dashboard data from API
     */
    async updateDashboardData() {
        const startTime = performance.now();

        try {
            // Fetch dashboard data
            const response = await fetch('/api/dashboard');

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.updateUI(data.data);
                this.updateConnectionStatus(true);
                this.state.lastUpdate = new Date();
                this.state.updateCount++;
            } else {
                throw new Error(data.error || 'API returned error');
            }

            // Track performance
            const updateTime = performance.now() - startTime;
            this.trackPerformance(updateTime);

        } catch (error) {
            console.error('Dashboard update error:', error);
            this.updateConnectionStatus(false);
            throw error;
        }
    }

    /**
     * Update UI with dashboard data
     */
    updateUI(data) {
        // Update training statistics
        if (data.training_stats) {
            this.updateTrainingStats(data.training_stats);
        }

        // Update game state
        if (data.game_state) {
            this.updateGameState(data.game_state);
        }

        // Update memory debug
        if (data.memory_debug) {
            this.updateMemoryDebug(data.memory_debug);
        }

        // Update LLM decisions
        if (data.recent_llm_decisions) {
            this.updateLLMDecisions(data.recent_llm_decisions);
        }

        // Update system status
        if (data.system_status) {
            this.updateSystemStatus(data.system_status);
        }

        // Update last update time
        this.updateElement('last-update', this.formatTime(new Date()));
    }

    /**
     * Update training statistics section
     */
    updateTrainingStats(stats) {
        this.updateElement('total-actions', this.formatNumber(stats.total_actions || 0));
        this.updateElement('actions-per-second', this.formatNumber(stats.actions_per_second || 0, 1));
        this.updateElement('llm-decisions', this.formatNumber(stats.llm_decisions || 0));
        this.updateElement('total-reward', this.formatNumber(stats.total_reward || 0, 2));
    }

    /**
     * Update game state section
     */
    updateGameState(gameState) {
        this.updateElement('current-map', gameState.current_map || '-');

        const position = gameState.player_position || { x: 0, y: 0 };
        this.updateElement('player-position', `${position.x},${position.y}`);

        this.updateElement('player-money', `¥${this.formatNumber(gameState.money || 0)}`);
        this.updateElement('player-badges', `${gameState.badges_earned || 0}/16`);
    }

    /**
     * Update memory debug section
     */
    updateMemoryDebug(memoryData) {
        const container = document.getElementById('memory-debug-container');
        if (!container) return;

        if (memoryData.memory_read_success && memoryData.memory_addresses) {
            const addresses = memoryData.memory_addresses;
            let html = '';

            // Convert memory data to debug items
            Object.entries(addresses).forEach(([key, value]) => {
                // Skip timestamp and other metadata
                if (key === 'timestamp' || key === 'debug_info') return;

                // Format the key for display
                const displayKey = key.replace(/_/g, ' ').toUpperCase();
                const displayValue = this.formatMemoryValue(key, value);

                html += `
                    <div class="debug-item">
                        <span class="debug-label">${displayKey}:</span>
                        <span class="debug-value">${displayValue}</span>
                    </div>
                `;
            });

            container.innerHTML = html || '<div class="no-data">No memory data available</div>';
        } else {
            container.innerHTML = '<div class="no-data">Memory reading failed</div>';
        }
    }

    /**
     * Update LLM decisions section
     */
    updateLLMDecisions(decisions) {
        const container = document.getElementById('llm-decisions-container');
        if (!container) return;

        if (decisions && decisions.length > 0) {
            let html = '';

            decisions.slice(-5).reverse().forEach(decision => {
                const timestamp = new Date(decision.timestamp * 1000);
                const confidence = decision.confidence ? (decision.confidence * 100).toFixed(1) : 0;

                html += `
                    <div class="decision-item fade-in">
                        <div class="decision-action">Action: ${decision.action_name || decision.action}</div>
                        <div class="decision-reasoning">${decision.reasoning || 'No reasoning provided'}</div>
                        <div class="decision-meta">
                            <span>${this.formatTime(timestamp)}</span>
                            <span>Confidence: ${confidence}%</span>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        } else {
            container.innerHTML = '<div class="no-data">No LLM decisions yet...</div>';
        }
    }

    /**
     * Update system status section
     */
    updateSystemStatus(status) {
        this.updateElement('training-active', status.training_active ? '✅ Yes' : '❌ No');
        this.updateElement('websocket-connections', status.websocket_connections || 0);
        this.updateElement('api-status', '✅ Connected');
    }

    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        try {
            console.log(`🔌 Connecting to WebSocket: ${this.config.wsUrl}`);

            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.state.wsConnected = true;
                this.clearReconnectTimer();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('WebSocket message parse error:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('📡 WebSocket disconnected');
                this.state.wsConnected = false;
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.state.wsConnected = false;
            };

        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'screen_update':
                this.updateGameScreen(data.data);
                break;
            case 'stats_update':
                this.updateUI({ training_stats: data.data });
                break;
            case 'connection_established':
                console.log('🔗 WebSocket connection established');
                break;
            case 'pong':
                // Handle ping/pong for connection testing
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    /**
     * Update game screen image
     */
    updateGameScreen(imageData) {
        const gameScreen = document.getElementById('game-screen');
        const screenStatus = document.getElementById('screen-status');

        if (gameScreen && imageData) {
            gameScreen.src = imageData;
            gameScreen.classList.add('updating');
            setTimeout(() => gameScreen.classList.remove('updating'), 200);

            if (screenStatus) {
                screenStatus.textContent = `Updated: ${this.formatTime(new Date())}`;
            }
        }
    }

    /**
     * Schedule WebSocket reconnection
     */
    scheduleReconnect() {
        if (this.wsReconnectTimer) return;

        const delay = Math.min(this.config.retryDelay * Math.pow(2, this.state.retryCount), 30000);

        this.wsReconnectTimer = setTimeout(() => {
            this.state.retryCount++;
            this.connectWebSocket();
            this.wsReconnectTimer = null;
        }, delay);

        console.log(`🔄 WebSocket reconnecting in ${delay}ms (attempt ${this.state.retryCount + 1})`);
    }

    /**
     * Clear WebSocket reconnection timer
     */
    clearReconnectTimer() {
        if (this.wsReconnectTimer) {
            clearTimeout(this.wsReconnectTimer);
            this.wsReconnectTimer = null;
        }
        this.state.retryCount = 0;
    }

    /**
     * Handle API errors
     */
    handleApiError(error) {
        this.state.errorCount++;

        if (this.state.retryCount < this.config.maxRetries) {
            this.state.retryCount++;
            console.log(`🔄 Retrying API call (${this.state.retryCount}/${this.config.maxRetries})`);
        } else {
            this.showError(`Connection lost: ${error.message}`);
        }
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        if (!indicator) return;

        this.state.connected = connected;

        if (connected) {
            indicator.textContent = '🟢 Connected';
            indicator.className = 'status-indicator connected';
        } else {
            indicator.textContent = '🔴 Disconnected';
            indicator.className = 'status-indicator disconnected';
        }
    }

    /**
     * Show error banner
     */
    showError(message) {
        const banner = document.getElementById('error-banner');
        const messageEl = document.getElementById('error-message');

        if (banner && messageEl) {
            messageEl.textContent = message;
            banner.classList.remove('hidden');
        }
    }

    /**
     * Hide error banner
     */
    hideError() {
        const banner = document.getElementById('error-banner');
        if (banner) {
            banner.classList.add('hidden');
        }
    }

    /**
     * Start performance monitoring
     */
    startPerformanceMonitoring() {
        setInterval(() => {
            this.updatePerformanceDisplay();
        }, 1000);
    }

    /**
     * Start periodic screen updates (fallback for WebSocket)
     */
    startScreenUpdates() {
        const updateScreen = async () => {
            try {
                // Only update if WebSocket isn't providing screen updates
                const response = await fetch('/api/screen');
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    const dataUrl = await this.blobToDataUrl(blob);
                    this.updateGameScreen(dataUrl);
                }
            } catch (error) {
                console.debug('Screen update error:', error);
            }
        };

        // Update screen every 33ms (30fps) for smooth local streaming
        setInterval(updateScreen, 33);

        // Initial update
        updateScreen();
    }

    /**
     * Convert blob to data URL
     */
    async blobToDataUrl(blob) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    }

    /**
     * Track performance metrics
     */
    trackPerformance(updateTime) {
        this.performance.updateTimes.push(updateTime);
        if (this.performance.updateTimes.length > 10) {
            this.performance.updateTimes.shift();
        }

        const now = Date.now();
        if (this.performance.lastUpdateTime) {
            const timeSinceLastUpdate = now - this.performance.lastUpdateTime;
            this.performance.updateRate = 1000 / timeSinceLastUpdate;
        }
        this.performance.lastUpdateTime = now;
    }

    /**
     * Update performance display
     */
    updatePerformanceDisplay() {
        const updateRate = document.getElementById('update-rate');
        const latency = document.getElementById('latency');

        if (updateRate) {
            updateRate.textContent = this.performance.updateRate.toFixed(1);
        }

        if (latency && this.performance.updateTimes.length > 0) {
            const avgLatency = this.performance.updateTimes.reduce((a, b) => a + b, 0) / this.performance.updateTimes.length;
            latency.textContent = `${avgLatency.toFixed(0)}ms`;
        }
    }

    /**
     * Handle page becoming hidden
     */
    handlePageHidden() {
        // Reduce update frequency when page is hidden
        if (this.apiInterval) {
            clearInterval(this.apiInterval);
            this.apiInterval = setInterval(() => this.updateDashboardData(), this.config.updateIntervals.api * 2);
        }
    }

    /**
     * Handle page becoming visible
     */
    handlePageVisible() {
        // Restore normal update frequency
        if (this.apiInterval) {
            clearInterval(this.apiInterval);
            this.apiInterval = setInterval(() => this.updateDashboardData(), this.config.updateIntervals.api);
        }

        // Force immediate update
        this.updateDashboardData();
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.apiInterval) {
            clearInterval(this.apiInterval);
        }

        if (this.ws) {
            this.ws.close();
        }

        this.clearReconnectTimer();
    }

    // Utility methods

    /**
     * Update element text content safely
     */
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * Format numbers for display
     */
    formatNumber(num, decimals = 0) {
        if (typeof num !== 'number') return num;
        return num.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    /**
     * Initialize visualization components
     */
    initializeVisualizations() {
        console.log('📊 Initializing visualizations');

        // Initialize visualization state
        this.visualizations = {
            charts: {},
            activeTab: 'rewards',
            rewardHistory: [],
            actionPerformance: {},
            decisionPatterns: [],
            performanceMetrics: []
        };

        // Setup tab switching
        this.setupVisualizationTabs();

        // Initialize canvas contexts
        this.initializeCanvases();

        // Start visualization data polling
        this.startVisualizationUpdates();
    }

    /**
     * Setup visualization tab switching
     */
    setupVisualizationTabs() {
        const tabs = document.querySelectorAll('.viz-tab');
        const panels = document.querySelectorAll('.viz-panel');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');

                // Update active tab
                tabs.forEach(t => t.classList.remove('active'));
                panels.forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                document.getElementById(`${tabName}-viz`).classList.add('active');

                this.visualizations.activeTab = tabName;
                this.updateVisualization(tabName);
            });
        });
    }

    /**
     * Initialize canvas contexts for charts
     */
    initializeCanvases() {
        const canvases = ['rewards-chart', 'actions-heatmap', 'exploration-heatmap'];
        canvases.forEach(canvasId => {
            const canvas = document.getElementById(canvasId);
            if (canvas) {
                this.visualizations.charts[canvasId] = canvas.getContext('2d');
                // Set canvas resolution for crisp rendering
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width * devicePixelRatio;
                canvas.height = rect.height * devicePixelRatio;
                canvas.style.width = rect.width + 'px';
                canvas.style.height = rect.height + 'px';
                this.visualizations.charts[canvasId].scale(devicePixelRatio, devicePixelRatio);
            }
        });
    }

    /**
     * Start visualization data updates
     */
    startVisualizationUpdates() {
        setInterval(() => {
            this.fetchVisualizationData();
        }, this.config.updateIntervals.stats);

        // Initial fetch
        this.fetchVisualizationData();
    }

    /**
     * Fetch visualization data from API
     */
    async fetchVisualizationData() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/api/visualization_data`);
            const data = await response.json();

            if (data.success) {
                this.updateVisualizationData(data.data);
            }
        } catch (error) {
            console.warn('Visualization data fetch failed:', error);
        }
    }

    /**
     * Update visualization data and refresh active charts
     */
    updateVisualizationData(data) {
        this.visualizations.rewardHistory = data.reward_history || [];
        this.visualizations.actionPerformance = data.action_performance || {};
        this.visualizations.decisionPatterns = data.decision_patterns || [];
        this.visualizations.performanceMetrics = data.performance_metrics || [];
        this.visualizations.explorationData = data.exploration_data || {};

        // Update the active visualization
        this.updateVisualization(this.visualizations.activeTab);
    }

    /**
     * Update specific visualization based on tab
     */
    updateVisualization(tabName) {
        switch (tabName) {
            case 'rewards':
                this.drawRewardChart();
                break;
            case 'actions':
                this.drawActionHeatmap();
                break;
            case 'decisions':
                this.updateDecisionFlow();
                break;
            case 'exploration':
                this.drawExplorationHeatmap();
                break;
        }
    }

    /**
     * Draw reward progress chart
     */
    drawRewardChart() {
        const canvas = document.getElementById('rewards-chart');
        const ctx = this.visualizations.charts['rewards-chart'];
        if (!ctx || !canvas) return;

        const data = this.visualizations.rewardHistory;
        if (data.length === 0) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width / devicePixelRatio, canvas.height / devicePixelRatio);

        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;
        const padding = 40;

        // Find data bounds
        const rewards = data.map(d => d.reward);
        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);
        const range = maxReward - minReward || 1;

        // Draw axes
        ctx.strokeStyle = '#787c99';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Draw reward line
        if (data.length > 1) {
            ctx.strokeStyle = '#2196F3';
            ctx.lineWidth = 2;
            ctx.beginPath();

            data.forEach((point, i) => {
                const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
                const y = height - padding - ((point.reward - minReward) / range) * (height - 2 * padding);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();
        }

        // Draw points
        ctx.fillStyle = '#2196F3';
        data.forEach((point, i) => {
            const x = padding + (i / Math.max(data.length - 1, 1)) * (width - 2 * padding);
            const y = height - padding - ((point.reward - minReward) / range) * (height - 2 * padding);

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    /**
     * Draw action performance heatmap
     */
    drawActionHeatmap() {
        const ctx = this.visualizations.charts['actions-heatmap'];
        const canvas = document.getElementById('actions-heatmap');
        if (!ctx || !canvas) return;

        const data = this.visualizations.actionPerformance;
        const actions = Object.keys(data);
        if (actions.length === 0) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width / devicePixelRatio, canvas.height / devicePixelRatio);

        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;
        const cellWidth = width / actions.length;
        const cellHeight = height / 2;

        actions.forEach((action, i) => {
            const actionData = data[action];
            const frequency = actionData.frequency || 0;
            const successRate = actionData.success_rate || 0;

            // Draw frequency bar
            ctx.fillStyle = `hsl(200, 60%, ${50 + frequency * 30}%)`;
            ctx.fillRect(i * cellWidth, 0, cellWidth - 2, cellHeight);

            // Draw success rate bar
            ctx.fillStyle = `hsl(120, 60%, ${30 + successRate * 40}%)`;
            ctx.fillRect(i * cellWidth, cellHeight, cellWidth - 2, cellHeight);

            // Draw labels
            ctx.fillStyle = '#a9b1d6';
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(action, i * cellWidth + cellWidth / 2, cellHeight / 2);
        });

        // Update action stats
        if (actions.length > 0) {
            const mostUsed = actions.reduce((a, b) => data[a].count > data[b].count ? a : b);
            const avgSuccess = actions.reduce((sum, action) => sum + data[action].success_rate, 0) / actions.length;

            document.getElementById('most-used-action').textContent = mostUsed;
            document.getElementById('avg-success-rate').textContent = (avgSuccess * 100).toFixed(1) + '%';
        }
    }

    /**
     * Update decision flow visualization
     */
    updateDecisionFlow() {
        const container = document.getElementById('decision-flow');
        if (!container) return;

        const decisions = this.visualizations.decisionPatterns;
        if (decisions.length === 0) {
            container.innerHTML = '<div class="flow-item">No decision data available</div>';
            return;
        }

        container.innerHTML = decisions.map(decision => `
            <div class="flow-item">
                <strong>Action:</strong> ${decision.action_name}
                <div class="decision-meta">
                    Confidence: ${(decision.confidence * 100).toFixed(1)}% |
                    Reasoning: ${decision.reasoning_length} chars
                </div>
            </div>
        `).join('');
    }

    /**
     * Draw exploration heatmap
     */
    drawExplorationHeatmap() {
        const ctx = this.visualizations.charts['exploration-heatmap'];
        const canvas = document.getElementById('exploration-heatmap');
        if (!ctx || !canvas) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width / devicePixelRatio, canvas.height / devicePixelRatio);

        const data = this.visualizations.explorationData;
        const coverage = data.exploration_coverage || 0;
        const currentMap = data.current_map || 0;

        // Simple visualization showing exploration coverage
        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;

        // Draw coverage bar
        ctx.fillStyle = '#4CAF50';
        ctx.fillRect(10, height / 2 - 10, (width - 20) * coverage, 20);

        ctx.fillStyle = '#787c99';
        ctx.font = '16px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`Exploration: ${(coverage * 100).toFixed(1)}%`, width / 2, height / 2 + 40);

        // Update exploration stats
        document.getElementById('exploration-coverage').textContent = (coverage * 100).toFixed(1) + '%';
        document.getElementById('current-map').textContent = currentMap;
    }

    /**
     * Format memory values for display
     */
    formatMemoryValue(key, value) {
        if (typeof value === 'boolean') {
            return value ? '1' : '0';
        }
        if (typeof value === 'number') {
            if (key.includes('ADDRESS') || key.includes('POINTER')) {
                return `0x${value.toString(16).toUpperCase().padStart(4, '0')}`;
            }
            return value.toString();
        }
        if (Array.isArray(value)) {
            return value.join(',');
        }
        return String(value);
    }

    /**
     * Format time for display
     */
    formatTime(date) {
        return date.toLocaleTimeString();
    }

    /**
     * Initialize A/B Testing functionality
     */
    initializeABTesting() {
        this.abTesting = {
            experiments: [],
            templates: [],
            activeExperiment: null,
            updateInterval: null
        };

        this.setupABTestingTabs();
        this.setupABTestingEventListeners();
        this.loadABTestingData();
    }

    /**
     * Setup A/B testing tab switching
     */
    setupABTestingTabs() {
        const abTabs = document.querySelectorAll('.ab-tab');
        const abPanels = document.querySelectorAll('.ab-panel');

        abTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');

                // Update active tab
                abTabs.forEach(t => t.classList.remove('active'));
                abPanels.forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');

                // Load data for specific tabs
                if (tabName === 'templates') {
                    this.loadTemplates();
                } else if (tabName === 'analytics') {
                    this.loadAnalytics();
                }
            });
        });
    }

    /**
     * Setup A/B testing event listeners
     */
    setupABTestingEventListeners() {
        // Create experiment button
        const createBtn = document.getElementById('create-experiment');
        if (createBtn) {
            createBtn.addEventListener('click', () => this.createExperiment());
        }

        // Reset form button
        const resetBtn = document.getElementById('reset-form');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetExperimentForm());
        }

        // Template selection
        document.addEventListener('click', (e) => {
            if (e.target.closest('.template-card')) {
                const templateId = e.target.closest('.template-card').dataset.template;
                this.selectTemplate(templateId);
            }
        });

        // Modal close
        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeExperimentModal());
        }

        // Modal control buttons
        const modalStartBtn = document.getElementById('modal-start-btn');
        const modalStopBtn = document.getElementById('modal-stop-btn');
        const modalAnalyzeBtn = document.getElementById('modal-analyze-btn');

        if (modalStartBtn) {
            modalStartBtn.addEventListener('click', () => this.controlExperiment('start'));
        }
        if (modalStopBtn) {
            modalStopBtn.addEventListener('click', () => this.controlExperiment('stop'));
        }
        if (modalAnalyzeBtn) {
            modalAnalyzeBtn.addEventListener('click', () => this.analyzeExperiment());
        }
    }

    /**
     * Load A/B testing data
     */
    async loadABTestingData() {
        try {
            // Load experiments list
            const response = await fetch('/api/v1/experiments');
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.abTesting.experiments = data.data.experiments;
                    this.updateExperimentsList(data.data);
                    this.updateExperimentStats(data.data);
                }
            }
        } catch (error) {
            console.error('Failed to load A/B testing data:', error);
        }
    }

    /**
     * Load experiment templates
     */
    async loadTemplates() {
        try {
            const response = await fetch('/api/v1/experiments/templates');
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.abTesting.templates = data.data.templates;
                    this.updateTemplatesList(data.data.templates);
                }
            }
        } catch (error) {
            console.error('Failed to load templates:', error);
        }
    }

    /**
     * Load analytics data
     */
    async loadAnalytics() {
        try {
            const response = await fetch('/api/v1/experiments/stats');
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.updateAnalytics(data.data);
                }
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    /**
     * Update experiments list UI
     */
    updateExperimentsList(experimentsData) {
        const container = document.getElementById('experiments-list');
        if (!container) return;

        if (experimentsData.experiments.length === 0) {
            container.innerHTML = `
                <div class="experiment-item placeholder">
                    <div class="experiment-header">
                        <h5>No experiments yet</h5>
                        <span class="experiment-status pending">Create your first A/B test</span>
                    </div>
                    <div class="experiment-description">
                        Use the "Create Test" tab to start comparing different configurations
                    </div>
                </div>
            `;
            return;
        }

        const experimentsHtml = experimentsData.experiments.map(exp => `
            <div class="experiment-item" data-experiment-id="${exp.experiment_id}">
                <div class="experiment-header">
                    <h5>${exp.name}</h5>
                    <span class="experiment-status ${exp.status}">${exp.status.toUpperCase()}</span>
                </div>
                <div class="experiment-description">
                    ${exp.experiment_type} • ${exp.variant_count} variants • ${exp.total_samples} samples
                </div>
                <div class="experiment-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${exp.progress_percentage}%"></div>
                    </div>
                    <span class="progress-text">${exp.progress_percentage.toFixed(1)}%</span>
                </div>
            </div>
        `).join('');

        container.innerHTML = experimentsHtml;

        // Add click listeners for experiment details
        container.querySelectorAll('.experiment-item').forEach(item => {
            if (!item.classList.contains('placeholder')) {
                item.addEventListener('click', () => {
                    const experimentId = item.dataset.experimentId;
                    this.showExperimentDetails(experimentId);
                });
            }
        });
    }

    /**
     * Update experiment statistics
     */
    updateExperimentStats(experimentsData) {
        const totalEl = document.getElementById('total-experiments');
        const runningEl = document.getElementById('running-experiments');
        const completedEl = document.getElementById('completed-experiments');

        if (totalEl) totalEl.textContent = `${experimentsData.total_count} Total`;
        if (runningEl) runningEl.textContent = `${experimentsData.active_count} Running`;
        if (completedEl) completedEl.textContent = `${experimentsData.completed_count} Completed`;
    }

    /**
     * Update templates list
     */
    updateTemplatesList(templates) {
        const container = document.getElementById('templates-grid');
        if (!container) return;

        const templatesHtml = templates.map(template => `
            <div class="template-card" data-template="${template.template_id}">
                <div class="template-header">
                    <h5>${template.name}</h5>
                    <span class="template-badge ${template.difficulty_level}">${template.difficulty_level}</span>
                </div>
                <div class="template-description">
                    ${template.description}
                </div>
            </div>
        `).join('');

        container.innerHTML = templatesHtml;
    }

    /**
     * Update analytics display
     */
    updateAnalytics(stats) {
        const successRateEl = document.getElementById('success-rate');
        const avgImprovementEl = document.getElementById('avg-improvement');
        const testsThisWeekEl = document.getElementById('tests-this-week');

        if (successRateEl) successRateEl.textContent = `${(stats.success_rate * 100).toFixed(1)}%`;
        if (avgImprovementEl) avgImprovementEl.textContent = `${(stats.avg_improvement * 100).toFixed(1)}%`;
        if (testsThisWeekEl) testsThisWeekEl.textContent = `${stats.tests_this_week}`;
    }

    /**
     * Create new experiment
     */
    async createExperiment() {
        const form = {
            name: document.getElementById('experiment-name')?.value,
            experiment_type: document.getElementById('experiment-type')?.value,
            sample_size_per_variant: parseInt(document.getElementById('sample-size')?.value) || 30,
            max_runtime_seconds: (parseInt(document.getElementById('max-runtime')?.value) || 60) * 60,
            primary_metrics: []
        };

        // Get selected metrics
        const metricsCheckboxes = document.querySelectorAll('#primary-metrics input[type="checkbox"]:checked');
        form.primary_metrics = Array.from(metricsCheckboxes).map(cb => cb.value);

        if (!form.name || !form.experiment_type) {
            this.showError('Please fill in all required fields');
            return;
        }

        try {
            const response = await fetch('/api/v1/experiments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form)
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.showSuccess('Experiment created successfully!');
                    this.resetExperimentForm();
                    this.loadABTestingData();

                    // Switch to experiments tab
                    document.querySelector('.ab-tab[data-tab="experiments"]').click();
                } else {
                    this.showError(data.error || 'Failed to create experiment');
                }
            } else {
                this.showError('Failed to create experiment');
            }
        } catch (error) {
            console.error('Create experiment error:', error);
            this.showError('Failed to create experiment');
        }
    }

    /**
     * Reset experiment creation form
     */
    resetExperimentForm() {
        const form = document.querySelector('.create-form');
        if (form) {
            const inputs = form.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (input.type === 'checkbox') {
                    input.checked = input.value === 'total_reward';
                } else if (input.type === 'number') {
                    input.value = input.id === 'sample-size' ? '30' : '60';
                } else {
                    input.value = '';
                }
            });
        }
    }

    /**
     * Select and apply a template
     */
    async selectTemplate(templateId) {
        try {
            const response = await fetch(`/api/v1/experiments/templates/${templateId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: `Template: ${templateId}`,
                    sample_size_per_variant: 30,
                    max_runtime_seconds: 3600
                })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.showSuccess('Experiment created from template!');
                    this.loadABTestingData();

                    // Switch to experiments tab
                    document.querySelector('.ab-tab[data-tab="experiments"]').click();
                } else {
                    this.showError(data.error || 'Failed to create experiment from template');
                }
            }
        } catch (error) {
            console.error('Template selection error:', error);
            this.showError('Failed to create experiment from template');
        }
    }

    /**
     * Show experiment details modal
     */
    async showExperimentDetails(experimentId) {
        try {
            const response = await fetch(`/api/v1/experiments/${experimentId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.populateExperimentModal(data.data);
                    document.getElementById('experiment-modal').style.display = 'flex';

                    // Subscribe to real-time updates for this experiment
                    this.subscribeToExperiment(experimentId);
                }
            }
        } catch (error) {
            console.error('Failed to load experiment details:', error);
        }
    }

    /**
     * Populate experiment modal with data
     */
    populateExperimentModal(experiment) {
        document.getElementById('modal-experiment-name').textContent = experiment.name;
        document.getElementById('modal-status').textContent = experiment.status;
        document.getElementById('modal-progress').textContent = `${experiment.progress_percentage.toFixed(1)}%`;
        document.getElementById('modal-progress-bar').style.width = `${experiment.progress_percentage}%`;

        if (experiment.duration_seconds) {
            const minutes = Math.floor(experiment.duration_seconds / 60);
            document.getElementById('modal-runtime').textContent = `${minutes} min`;
        } else {
            document.getElementById('modal-runtime').textContent = '-';
        }

        this.abTesting.activeExperiment = experiment.experiment_id;
    }

    /**
     * Close experiment modal
     */
    closeExperimentModal() {
        // Unsubscribe from updates if we have an active experiment
        if (this.abTesting.activeExperiment) {
            this.unsubscribeFromExperiment(this.abTesting.activeExperiment);
        }

        document.getElementById('experiment-modal').style.display = 'none';
        this.abTesting.activeExperiment = null;
    }

    /**
     * Control experiment (start/stop)
     */
    async controlExperiment(action) {
        if (!this.abTesting.activeExperiment) return;

        try {
            const response = await fetch(`/api/v1/experiments/${this.abTesting.activeExperiment}/control`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.showSuccess(`Experiment ${action}ed successfully!`);
                    this.loadABTestingData();
                    this.closeExperimentModal();
                } else {
                    this.showError(data.error || `Failed to ${action} experiment`);
                }
            }
        } catch (error) {
            console.error(`Control experiment error:`, error);
            this.showError(`Failed to ${action} experiment`);
        }
    }

    /**
     * Analyze experiment results
     */
    async analyzeExperiment() {
        if (!this.abTesting.activeExperiment) return;

        try {
            const response = await fetch(`/api/v1/experiments/${this.abTesting.activeExperiment}/analysis`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    this.showAnalysisResults(data.data);
                } else {
                    this.showError(data.error || 'Failed to analyze experiment');
                }
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Failed to analyze experiment');
        }
    }

    /**
     * Show analysis results
     */
    showAnalysisResults(analysis) {
        // Simple display of analysis results - could be enhanced with charts
        const resultsHtml = `
            <div class="analysis-results">
                <h4>Analysis Results</h4>
                <p><strong>Has Significant Results:</strong> ${analysis.has_significant_results ? 'Yes' : 'No'}</p>
                ${analysis.winning_variant ? `<p><strong>Winning Variant:</strong> ${analysis.winning_variant}</p>` : ''}
                <p><strong>Confidence:</strong> ${(analysis.confidence_score * 100).toFixed(1)}%</p>
                <p><strong>Summary:</strong> ${analysis.summary}</p>
            </div>
        `;

        document.getElementById('modal-variants').innerHTML = resultsHtml;
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        // Use existing error banner but with success styling
        const banner = document.getElementById('error-banner');
        const messageEl = document.getElementById('error-message');

        if (banner && messageEl) {
            messageEl.textContent = message;
            banner.className = 'error-banner success';
            banner.style.display = 'block';

            setTimeout(() => {
                banner.style.display = 'none';
            }, 3000);
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const banner = document.getElementById('error-banner');
        const messageEl = document.getElementById('error-message');

        if (banner && messageEl) {
            messageEl.textContent = message;
            banner.className = 'error-banner';
            banner.style.display = 'block';
        }
    }

    /**
     * Setup WebSocket handlers for A/B testing real-time updates
     */
    setupABTestingWebSocket() {
        // Add A/B testing message handlers to existing WebSocket
        if (this.ws) {
            // Store original onmessage handler
            const originalOnMessage = this.ws.onmessage;

            this.ws.onmessage = (event) => {
                // Call original handler first
                if (originalOnMessage) {
                    originalOnMessage.call(this.ws, event);
                }

                // Handle A/B testing messages
                try {
                    const message = JSON.parse(event.data);
                    this.handleABTestingWebSocketMessage(message);
                } catch (error) {
                    // Ignore parsing errors for non-JSON messages
                }
            };
        }

        // Request initial A/B testing data via WebSocket
        setTimeout(() => {
            this.requestABTestingUpdates();
        }, 1000);
    }

    /**
     * Handle WebSocket messages for A/B testing
     */
    handleABTestingWebSocketMessage(message) {
        switch (message.type) {
            case 'experiments_update':
                this.handleExperimentsUpdate(message.data);
                break;
            case 'experiment_progress':
                this.handleExperimentProgress(message.data);
                break;
            default:
                // Ignore other message types
                break;
        }
    }

    /**
     * Handle real-time experiments list update
     */
    handleExperimentsUpdate(experimentsData) {
        // Update experiments list if visible
        const experimentsTab = document.getElementById('experiments-tab');
        if (experimentsTab && experimentsTab.classList.contains('active')) {
            this.updateExperimentsList(experimentsData);
            this.updateExperimentStats(experimentsData);
        }

        // Update global A/B testing state
        this.abTesting.experiments = experimentsData.experiments;
    }

    /**
     * Handle real-time experiment progress update
     */
    handleExperimentProgress(progressData) {
        const experimentId = progressData.experiment_id;

        // Update experiment in list if visible
        const experimentElement = document.querySelector(`[data-experiment-id="${experimentId}"]`);
        if (experimentElement) {
            // Update progress bar
            const progressBar = experimentElement.querySelector('.progress-fill');
            const progressText = experimentElement.querySelector('.progress-text');
            const statusElement = experimentElement.querySelector('.experiment-status');

            if (progressBar) {
                progressBar.style.width = `${progressData.progress_percentage}%`;
            }
            if (progressText) {
                progressText.textContent = `${progressData.progress_percentage.toFixed(1)}%`;
            }
            if (statusElement) {
                statusElement.textContent = progressData.status.toUpperCase();
                statusElement.className = `experiment-status ${progressData.status}`;
            }
        }

        // Update modal if open for this experiment
        const modal = document.getElementById('experiment-modal');
        if (modal.style.display === 'flex' && this.abTesting.activeExperiment === experimentId) {
            document.getElementById('modal-status').textContent = progressData.status;
            document.getElementById('modal-progress').textContent = `${progressData.progress_percentage.toFixed(1)}%`;
            document.getElementById('modal-progress-bar').style.width = `${progressData.progress_percentage}%`;

            if (progressData.elapsed_seconds) {
                const minutes = Math.floor(progressData.elapsed_seconds / 60);
                document.getElementById('modal-runtime').textContent = `${minutes} min`;
            }

            // Update live metrics if available
            if (progressData.live_metrics && Object.keys(progressData.live_metrics).length > 0) {
                this.updateLiveMetrics(progressData.live_metrics);
            }
        }
    }

    /**
     * Update live metrics display in experiment modal
     */
    updateLiveMetrics(liveMetrics) {
        const variantsContainer = document.getElementById('modal-variants');
        if (!variantsContainer) return;

        let metricsHtml = '<h4>Live Metrics</h4>';
        metricsHtml += '<div class="live-metrics-grid">';

        for (const [variantName, metrics] of Object.entries(liveMetrics)) {
            metricsHtml += `
                <div class="variant-metrics">
                    <h5>${variantName}</h5>
                    <div class="metrics-row">
                        <span class="metric-label">Samples:</span>
                        <span class="metric-value">${metrics.sample_count}</span>
                    </div>
                    <div class="metrics-row">
                        <span class="metric-label">Avg Reward:</span>
                        <span class="metric-value">${metrics.average_reward?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metrics-row">
                        <span class="metric-label">Latest Reward:</span>
                        <span class="metric-value">${metrics.latest_reward?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metrics-row">
                        <span class="metric-label">Actions/sec:</span>
                        <span class="metric-value">${metrics.latest_actions_per_second?.toFixed(1) || 'N/A'}</span>
                    </div>
                </div>
            `;
        }

        metricsHtml += '</div>';
        variantsContainer.innerHTML = metricsHtml;
    }

    /**
     * Request A/B testing updates via WebSocket
     */
    requestABTestingUpdates() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Request experiments list
            this.ws.send(JSON.stringify({
                type: 'request_experiments',
                timestamp: Date.now()
            }));
        }
    }

    /**
     * Subscribe to real-time updates for a specific experiment
     */
    subscribeToExperiment(experimentId) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe_experiment',
                experiment_id: experimentId,
                timestamp: Date.now()
            }));
        }
    }

    /**
     * Unsubscribe from experiment updates
     */
    unsubscribeFromExperiment(experimentId) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'unsubscribe_experiment',
                experiment_id: experimentId,
                timestamp: Date.now()
            }));
        }
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new PokemonDashboard();
});

// Export for potential external use
window.PokemonDashboard = PokemonDashboard;