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
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new PokemonDashboard();
});

// Export for potential external use
window.PokemonDashboard = PokemonDashboard;