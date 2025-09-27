/**
 * Advanced Analytics & Debugging Dashboard JavaScript
 *
 * Provides interactive analytics, real-time debugging, and visualization
 * capabilities for the Pokemon Crystal RL training platform.
 */

class AdvancedDashboard {
    constructor() {
        // Configuration
        this.config = {
            apiBaseUrl: '',
            wsUrl: `ws://${window.location.hostname}:8081`,
            updateIntervals: {
                analytics: 5000,    // Analytics update interval (5s)
                debug: 2000,        // Debug console update interval (2s)
                profiler: 3000,     // Profiler update interval (3s)
                health: 1000        // Health monitoring interval (1s)
            },
            chartOptions: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    }
                }
            }
        };

        // State management
        this.state = {
            currentTab: 'overview',
            monitoringActive: false,
            wsConnected: false,
            analytics: {},
            debugData: {},
            profilerData: {},
            alerts: [],
            charts: {}
        };

        // WebSocket connection
        this.ws = null;
        this.wsReconnectTimer = null;

        // Update timers
        this.updateTimers = {};

        // Chart instances
        this.charts = {};

        // Initialize dashboard
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Advanced Dashboard...');

        try {
            // Setup event listeners
            this.setupEventListeners();

            // Initialize tabs
            this.initializeTabs();

            // Setup WebSocket connection
            this.setupWebSocket();

            // Start initial data loading
            await this.loadInitialData();

            // Start monitoring if enabled
            if (this.state.monitoringActive) {
                this.startMonitoring();
            }

            this.showToast('Dashboard initialized successfully', 'success');
            console.log('✅ Advanced Dashboard initialized');

        } catch (error) {
            console.error('❌ Failed to initialize dashboard:', error);
            this.showToast('Failed to initialize dashboard', 'error');
        }
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tabId = e.target.dataset.tab;
                this.switchTab(tabId);
            });
        });

        // Control buttons
        document.getElementById('start-monitoring')?.addEventListener('click', () => {
            this.startMonitoring();
        });

        document.getElementById('stop-monitoring')?.addEventListener('click', () => {
            this.stopMonitoring();
        });

        document.getElementById('export-data')?.addEventListener('click', () => {
            this.exportData();
        });

        // Quick actions
        document.getElementById('clear-cache')?.addEventListener('click', () => {
            this.clearCache();
        });

        document.getElementById('restart-components')?.addEventListener('click', () => {
            this.restartComponents();
        });

        document.getElementById('run-diagnostics')?.addEventListener('click', () => {
            this.runDiagnostics();
        });

        document.getElementById('optimize-performance')?.addEventListener('click', () => {
            this.optimizePerformance();
        });

        // Debug controls
        document.getElementById('clear-console')?.addEventListener('click', () => {
            this.clearDebugConsole();
        });

        document.getElementById('add-breakpoint')?.addEventListener('click', () => {
            this.addBreakpoint();
        });

        document.getElementById('add-watch')?.addEventListener('click', () => {
            this.addWatchVariable();
        });

        // Profiler controls
        document.getElementById('start-profiling')?.addEventListener('click', () => {
            this.startProfiling();
        });

        document.getElementById('stop-profiling')?.addEventListener('click', () => {
            this.stopProfiling();
        });

        document.getElementById('reset-profiler')?.addEventListener('click', () => {
            this.resetProfiler();
        });

        // Memory controls
        document.getElementById('gc-collect')?.addEventListener('click', () => {
            this.forceGarbageCollection();
        });

        document.getElementById('memory-snapshot')?.addEventListener('click', () => {
            this.takeMemorySnapshot();
        });

        // Chart controls
        document.getElementById('metrics-selector')?.addEventListener('change', () => {
            this.updateTrendsChart();
        });

        document.getElementById('time-range')?.addEventListener('change', () => {
            this.updateTrendsChart();
        });

        document.getElementById('create-chart')?.addEventListener('click', () => {
            this.createCustomChart();
        });

        // Console filters
        document.getElementById('log-level')?.addEventListener('change', () => {
            this.filterDebugConsole();
        });

        document.getElementById('console-filter')?.addEventListener('input', () => {
            this.filterDebugConsole();
        });
    }

    initializeTabs() {
        // Set default active tab
        this.switchTab('overview');
    }

    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabId}-tab`)?.classList.add('active');

        this.state.currentTab = tabId;

        // Load tab-specific data
        this.loadTabData(tabId);
    }

    async loadTabData(tabId) {
        try {
            switch (tabId) {
                case 'overview':
                    await this.loadOverviewData();
                    break;
                case 'analytics':
                    await this.loadAnalyticsData();
                    break;
                case 'debugger':
                    await this.loadDebuggerData();
                    break;
                case 'profiler':
                    await this.loadProfilerData();
                    break;
                case 'visualizations':
                    await this.loadVisualizationData();
                    break;
            }
        } catch (error) {
            console.error(`Failed to load ${tabId} data:`, error);
            this.showToast(`Failed to load ${tabId} data`, 'error');
        }
    }

    async loadInitialData() {
        console.log('📊 Loading initial dashboard data...');

        try {
            // Load overview data first
            await this.loadOverviewData();

            console.log('✅ Initial data loaded');
        } catch (error) {
            console.error('❌ Failed to load initial data:', error);
            throw error;
        }
    }

    async loadOverviewData() {
        try {
            // Load system health
            const healthData = await this.apiCall('/api/v1/analytics/health');
            this.updateSystemHealth(healthData);

            // Load key metrics
            const metricsData = await this.apiCall('/api/v1/analytics/metrics');
            this.updateKeyMetrics(metricsData);

            // Load alerts
            const alertsData = await this.apiCall('/api/v1/analytics/alerts');
            this.updateAlerts(alertsData);

        } catch (error) {
            console.error('Failed to load overview data:', error);
        }
    }

    async loadAnalyticsData() {
        try {
            // Load analytics summary
            const analyticsData = await this.apiCall('/api/v1/analytics/summary');
            this.state.analytics = analyticsData;

            // Update trends chart
            this.updateTrendsChart();

            // Update correlation matrix
            this.updateCorrelationMatrix(analyticsData.correlations);

            // Update insights
            this.updateInsights(analyticsData.insights);

        } catch (error) {
            console.error('Failed to load analytics data:', error);
        }
    }

    async loadDebuggerData() {
        try {
            // Load debug dashboard
            const debugData = await this.apiCall('/api/v1/debug/dashboard');
            this.state.debugData = debugData;

            // Update debug console
            this.updateDebugConsole(debugData.recent_events);

            // Update component health
            this.updateComponentHealth(debugData.component_status);

            // Update memory analysis
            this.updateMemoryAnalysis(debugData.memory_analysis);

        } catch (error) {
            console.error('Failed to load debugger data:', error);
        }
    }

    async loadProfilerData() {
        try {
            // Load profiler data
            const profilerData = await this.apiCall('/api/v1/debug/profiler');
            this.state.profilerData = profilerData;

            // Update profiler table
            this.updateProfilerTable(profilerData.performance_summary);

            // Update hotspots
            this.updateHotspots(profilerData.hotspots);

            // Update recommendations
            this.updateRecommendations(profilerData.recommendations);

        } catch (error) {
            console.error('Failed to load profiler data:', error);
        }
    }

    async loadVisualizationData() {
        try {
            // Load visualization data
            const vizData = await this.apiCall('/api/v1/analytics/visualization');

            // Update performance heatmap
            this.updatePerformanceHeatmap(vizData.heatmap_data);

            // Update dependency graph
            this.updateDependencyGraph(vizData.dependencies);

        } catch (error) {
            console.error('Failed to load visualization data:', error);
        }
    }

    updateSystemHealth(healthData) {
        if (!healthData) return;

        // Update health score
        document.getElementById('health-score').textContent = Math.round(healthData.score || 0);
        document.getElementById('health-label').textContent = healthData.status || 'Unknown';

        // Update health metrics
        document.getElementById('memory-usage').textContent =
            `${(healthData.memory_usage_mb || 0).toFixed(1)}GB`;
        document.getElementById('thread-count').textContent = healthData.thread_count || 0;
        document.getElementById('error-rate').textContent =
            `${(healthData.error_rate || 0).toFixed(2)}%`;

        // Update health score color
        const healthScore = document.getElementById('health-score');
        const score = healthData.score || 0;
        if (score >= 80) {
            healthScore.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
        } else if (score >= 60) {
            healthScore.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
        } else {
            healthScore.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
        }
    }

    updateKeyMetrics(metricsData) {
        if (!metricsData || !metricsData.metrics) return;

        const metrics = metricsData.metrics;

        // Update actions per second
        if (metrics.actions_per_second) {
            document.getElementById('actions-per-sec').textContent =
                metrics.actions_per_second.latest?.toFixed(1) || '0.0';
            this.updateTrend('actions-trend', metrics.actions_per_second.change_rate);
        }

        // Update total reward
        if (metrics.total_reward) {
            document.getElementById('total-reward').textContent =
                Math.round(metrics.total_reward.latest || 0).toLocaleString();
            this.updateTrend('reward-trend', metrics.total_reward.change_rate);
        }

        // Update LLM response time
        if (metrics.llm_response_time) {
            document.getElementById('llm-response').textContent =
                `${Math.round(metrics.llm_response_time.latest * 1000 || 0)}ms`;
            this.updateTrend('llm-trend', -metrics.llm_response_time.change_rate); // Negative because lower is better
        }

        // Calculate success rate (example metric)
        const successRate = 85 + Math.random() * 10; // Placeholder
        document.getElementById('success-rate').textContent = `${Math.round(successRate)}%`;
        this.updateTrend('success-trend', 0);
    }

    updateTrend(elementId, changeRate) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const absChange = Math.abs(changeRate || 0);
        const sign = changeRate > 0 ? '+' : changeRate < 0 ? '-' : '';

        element.textContent = `${changeRate > 0 ? '↗' : changeRate < 0 ? '↘' : '→'} ${sign}${absChange.toFixed(1)}%`;

        // Update class
        element.className = 'metric-trend';
        if (changeRate > 2) {
            element.classList.add('up');
        } else if (changeRate < -2) {
            element.classList.add('down');
        } else {
            element.classList.add('stable');
        }
    }

    updateAlerts(alertsData) {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer || !alertsData) return;

        alertsContainer.innerHTML = '';

        const alerts = alertsData.alerts || [];
        if (alerts.length === 0) {
            alertsContainer.innerHTML = '<div class="text-center text-muted">No active alerts</div>';
            return;
        }

        alerts.slice(0, 5).forEach(alert => {
            const alertElement = document.createElement('div');
            alertElement.className = `alert ${alert.severity || 'info'}`;
            alertElement.innerHTML = `
                <div class="alert-title">${alert.alert_type || 'Alert'}</div>
                <div class="alert-message">${alert.message || 'No message'}</div>
                <div class="alert-time">${this.formatTime(alert.timestamp)}</div>
            `;
            alertsContainer.appendChild(alertElement);
        });
    }

    updateTrendsChart() {
        const canvas = document.getElementById('trends-chart');
        if (!canvas) return;

        // Get selected metrics and time range
        const selector = document.getElementById('metrics-selector');
        const timeRange = document.getElementById('time-range');

        const selectedMetrics = Array.from(selector.selectedOptions).map(option => option.value);
        const range = timeRange?.value || 'medium';

        // Destroy existing chart
        if (this.charts.trends) {
            this.charts.trends.destroy();
        }

        // Create new chart
        const ctx = canvas.getContext('2d');

        // Generate sample data (replace with real data)
        const datasets = selectedMetrics.map((metric, index) => {
            const color = this.getChartColor(index);
            return {
                label: this.formatMetricName(metric),
                data: this.generateSampleTimeSeriesData(range),
                borderColor: color,
                backgroundColor: color + '20',
                fill: false,
                tension: 0.2
            };
        });

        this.charts.trends = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                ...this.config.chartOptions,
                plugins: {
                    ...this.config.chartOptions.plugins,
                    title: {
                        display: true,
                        text: 'Performance Trends',
                        color: '#f8fafc'
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'HH:mm'
                            }
                        },
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }

    updateCorrelationMatrix(correlations) {
        const container = document.getElementById('correlation-matrix');
        if (!container || !correlations) return;

        // Create correlation matrix visualization
        container.innerHTML = '';

        const metrics = ['actions_per_second', 'total_reward', 'llm_response_time', 'memory_usage'];

        // Create grid
        container.style.gridTemplateColumns = `repeat(${metrics.length + 1}, 1fr)`;

        // Header row
        container.appendChild(this.createCorrelationCell('', true));
        metrics.forEach(metric => {
            container.appendChild(this.createCorrelationCell(this.formatMetricName(metric), true));
        });

        // Data rows
        metrics.forEach(metric1 => {
            container.appendChild(this.createCorrelationCell(this.formatMetricName(metric1), true));
            metrics.forEach(metric2 => {
                const corrKey = `${metric1}_vs_${metric2}`;
                const correlation = correlations[corrKey] || { correlation: metric1 === metric2 ? 1 : 0 };
                container.appendChild(this.createCorrelationCell(
                    correlation.correlation.toFixed(2),
                    false,
                    correlation.correlation
                ));
            });
        });
    }

    createCorrelationCell(text, isHeader, value = 0) {
        const cell = document.createElement('div');
        cell.className = 'correlation-cell';
        cell.textContent = text;

        if (isHeader) {
            cell.classList.add('header');
        } else {
            if (Math.abs(value) > 0.7) {
                cell.classList.add(value > 0 ? 'strong-positive' : 'strong-negative');
            } else if (Math.abs(value) > 0.3) {
                cell.classList.add('moderate');
            }
        }

        return cell;
    }

    updateInsights(insights) {
        const container = document.getElementById('insights-container');
        if (!container || !insights) return;

        container.innerHTML = '';

        const allInsights = [
            ...(insights.recommendations || []),
            ...(insights.optimization_opportunities || []),
            ...(insights.performance_bottlenecks || [])
        ];

        if (allInsights.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No insights available</div>';
            return;
        }

        allInsights.slice(0, 5).forEach(insight => {
            const insightElement = document.createElement('div');
            insightElement.className = `recommendation ${insight.priority || 'low'}`;
            insightElement.innerHTML = `
                <div class="recommendation-title">${insight.title || insight.type}</div>
                <div class="recommendation-description">${insight.description || insight.message}</div>
            `;
            container.appendChild(insightElement);
        });
    }

    updateDebugConsole(events) {
        const console = document.getElementById('debug-console');
        if (!console || !events) return;

        // Filter events based on current filters
        const filteredEvents = this.filterEvents(events);

        console.innerHTML = '';
        filteredEvents.forEach(event => {
            const messageElement = document.createElement('div');
            messageElement.className = `console-message ${event.event_type}`;
            messageElement.innerHTML = `
                <span class="console-timestamp">[${this.formatTime(event.timestamp)}]</span>
                <span class="console-component">[${event.component}]</span>
                <span class="console-text">${event.message}</span>
            `;
            console.appendChild(messageElement);
        });

        // Auto-scroll to bottom
        console.scrollTop = console.scrollHeight;
    }

    updateComponentHealth(componentStatus) {
        const container = document.getElementById('component-health');
        if (!container || !componentStatus) return;

        container.innerHTML = '';

        Object.entries(componentStatus).forEach(([component, status]) => {
            const statusElement = document.createElement('div');
            statusElement.className = `component-status ${status}`;
            statusElement.innerHTML = `
                <span class="component-name">${component}</span>
                <span class="component-indicator ${status}"></span>
            `;
            container.appendChild(statusElement);
        });
    }

    updateMemoryAnalysis(memoryData) {
        const container = document.getElementById('memory-analysis');
        if (!container || !memoryData) return;

        container.innerHTML = '';

        const stats = [
            { label: 'Current Usage', value: `${memoryData.current_mb?.toFixed(1) || 0} MB` },
            { label: 'Average Usage', value: `${memoryData.average_mb?.toFixed(1) || 0} MB` },
            { label: 'Peak Usage', value: `${memoryData.peak_mb?.toFixed(1) || 0} MB` },
            { label: 'Trend', value: memoryData.trend || 'Unknown' }
        ];

        stats.forEach(stat => {
            const statElement = document.createElement('div');
            statElement.className = 'memory-stat';
            statElement.innerHTML = `
                <span>${stat.label}</span>
                <span>${stat.value}</span>
            `;
            container.appendChild(statElement);
        });
    }

    updateProfilerTable(profilerData) {
        const container = document.getElementById('profiler-table');
        if (!container || !profilerData) return;

        const functions = Object.entries(profilerData).slice(0, 20); // Top 20 functions

        container.innerHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Function</th>
                        <th>Calls</th>
                        <th>Total Time</th>
                        <th>Avg Time</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    ${functions.map(([funcName, data]) => `
                        <tr>
                            <td title="${funcName}">${this.truncateText(funcName, 30)}</td>
                            <td>${data.call_count || 0}</td>
                            <td>${(data.total_time || 0).toFixed(3)}s</td>
                            <td>${(data.average_time || 0).toFixed(3)}s</td>
                            <td>${data.performance_trend || 'unknown'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    updateHotspots(hotspotsData) {
        const container = document.getElementById('hotspots-list');
        if (!container) return;

        container.innerHTML = '';

        const hotspots = hotspotsData || [];
        if (hotspots.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hotspots detected</div>';
            return;
        }

        hotspots.slice(0, 10).forEach(hotspot => {
            const hotspotElement = document.createElement('div');
            hotspotElement.className = `hotspot-item ${hotspot.severity}`;
            hotspotElement.innerHTML = `
                <div class="hotspot-function">${hotspot.function}</div>
                <div class="hotspot-stats">
                    <div class="hotspot-stat">Calls: ${hotspot.call_count}</div>
                    <div class="hotspot-stat">Total: ${hotspot.total_time.toFixed(3)}s</div>
                    <div class="hotspot-stat">Avg: ${hotspot.average_time.toFixed(3)}s</div>
                </div>
            `;
            container.appendChild(hotspotElement);
        });
    }

    updateRecommendations(recommendations) {
        const container = document.getElementById('recommendations-list');
        if (!container) return;

        container.innerHTML = '';

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No recommendations available</div>';
            return;
        }

        recommendations.forEach(rec => {
            const recElement = document.createElement('div');
            recElement.className = `recommendation ${rec.priority}`;
            recElement.innerHTML = `
                <div class="recommendation-title">${rec.title}</div>
                <div class="recommendation-description">${rec.description}</div>
            `;
            container.appendChild(recElement);
        });
    }

    // WebSocket functionality
    setupWebSocket() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                console.log('🔌 WebSocket connected');
                this.state.wsConnected = true;
                this.updateConnectionStatus();
            };

            this.ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.state.wsConnected = false;
                this.updateConnectionStatus();
                this.scheduleReconnect();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'analytics_update':
                this.handleAnalyticsUpdate(data.payload);
                break;
            case 'debug_event':
                this.handleDebugEvent(data.payload);
                break;
            case 'alert':
                this.handleAlert(data.payload);
                break;
            case 'health_update':
                this.handleHealthUpdate(data.payload);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    handleAnalyticsUpdate(payload) {
        // Update analytics data in real-time
        if (this.state.currentTab === 'analytics') {
            this.updateKeyMetrics(payload);
        }
    }

    handleDebugEvent(payload) {
        // Add new debug event to console
        if (this.state.currentTab === 'debugger') {
            this.addDebugMessage(payload);
        }
    }

    handleAlert(payload) {
        // Show new alert
        this.showToast(payload.message, payload.severity);
        this.addAlert(payload);
    }

    handleHealthUpdate(payload) {
        // Update system health indicators
        this.updateSystemHealth(payload);
    }

    // Monitoring controls
    async startMonitoring() {
        try {
            await this.apiCall('/api/v1/analytics/start', 'POST');
            await this.apiCall('/api/v1/debug/start', 'POST');

            this.state.monitoringActive = true;
            this.updateMonitoringStatus();

            // Start update timers
            this.startUpdateTimers();

            this.showToast('Monitoring started', 'success');
        } catch (error) {
            console.error('Failed to start monitoring:', error);
            this.showToast('Failed to start monitoring', 'error');
        }
    }

    async stopMonitoring() {
        try {
            await this.apiCall('/api/v1/analytics/stop', 'POST');
            await this.apiCall('/api/v1/debug/stop', 'POST');

            this.state.monitoringActive = false;
            this.updateMonitoringStatus();

            // Stop update timers
            this.stopUpdateTimers();

            this.showToast('Monitoring stopped', 'success');
        } catch (error) {
            console.error('Failed to stop monitoring:', error);
            this.showToast('Failed to stop monitoring', 'error');
        }
    }

    startUpdateTimers() {
        // Clear existing timers
        this.stopUpdateTimers();

        // Analytics updates
        this.updateTimers.analytics = setInterval(() => {
            if (this.state.currentTab === 'analytics') {
                this.loadAnalyticsData();
            }
        }, this.config.updateIntervals.analytics);

        // Debug updates
        this.updateTimers.debug = setInterval(() => {
            if (this.state.currentTab === 'debugger') {
                this.loadDebuggerData();
            }
        }, this.config.updateIntervals.debug);

        // Profiler updates
        this.updateTimers.profiler = setInterval(() => {
            if (this.state.currentTab === 'profiler') {
                this.loadProfilerData();
            }
        }, this.config.updateIntervals.profiler);

        // Health updates
        this.updateTimers.health = setInterval(() => {
            this.loadOverviewData();
        }, this.config.updateIntervals.health);
    }

    stopUpdateTimers() {
        Object.values(this.updateTimers).forEach(timer => {
            if (timer) clearInterval(timer);
        });
        this.updateTimers = {};
    }

    updateMonitoringStatus() {
        const analyticsStatus = document.getElementById('analytics-status');
        const debuggerStatus = document.getElementById('debugger-status');
        const profilerStatus = document.getElementById('profiler-status');

        const activeClass = this.state.monitoringActive ? 'active' : '';

        analyticsStatus?.classList.toggle('active', this.state.monitoringActive);
        debuggerStatus?.classList.toggle('active', this.state.monitoringActive);
        profilerStatus?.classList.toggle('active', this.state.monitoringActive);
    }

    updateConnectionStatus() {
        // Update connection indicators based on WebSocket status
        // Implementation depends on UI elements
    }

    // Utility functions
    async apiCall(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            if (data) {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(`${this.config.apiBaseUrl}${endpoint}`, options);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API call failed: ${method} ${endpoint}`, error);
            throw error;
        }
    }

    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-title">${this.capitalizeFirst(type)}</div>
            <div class="toast-message">${message}</div>
        `;

        container.appendChild(toast);

        // Auto-remove after duration
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
    }

    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString();
    }

    formatMetricName(metric) {
        return metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    getChartColor(index) {
        const colors = [
            '#2563eb', '#10b981', '#f59e0b', '#ef4444',
            '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'
        ];
        return colors[index % colors.length];
    }

    generateSampleTimeSeriesData(range) {
        const now = Date.now();
        const intervals = range === 'short' ? 60 : range === 'medium' ? 300 : 600; // seconds
        const points = range === 'short' ? 50 : range === 'medium' ? 60 : 60;

        const data = [];
        for (let i = 0; i < points; i++) {
            const timestamp = now - (points - i) * intervals * 1000;
            const value = Math.random() * 100 + Math.sin(i * 0.1) * 20;
            data.push({
                x: timestamp,
                y: Math.max(0, value)
            });
        }
        return data;
    }

    filterEvents(events) {
        const logLevel = document.getElementById('log-level')?.value || 'all';
        const filter = document.getElementById('console-filter')?.value || '';

        let filtered = events;

        // Filter by log level
        if (logLevel !== 'all') {
            const levelPriority = { error: 3, warning: 2, info: 1 };
            const minPriority = levelPriority[logLevel] || 0;
            filtered = filtered.filter(event =>
                (levelPriority[event.event_type] || 0) >= minPriority
            );
        }

        // Filter by text
        if (filter) {
            const filterLower = filter.toLowerCase();
            filtered = filtered.filter(event =>
                event.message.toLowerCase().includes(filterLower) ||
                event.component.toLowerCase().includes(filterLower)
            );
        }

        return filtered.slice(-100); // Last 100 messages
    }

    // Additional action handlers (placeholders)
    async clearCache() {
        try {
            await this.apiCall('/api/v1/system/clear-cache', 'POST');
            this.showToast('Cache cleared successfully', 'success');
        } catch (error) {
            this.showToast('Failed to clear cache', 'error');
        }
    }

    async restartComponents() {
        try {
            await this.apiCall('/api/v1/system/restart-components', 'POST');
            this.showToast('Components restarted successfully', 'success');
        } catch (error) {
            this.showToast('Failed to restart components', 'error');
        }
    }

    async runDiagnostics() {
        try {
            const result = await this.apiCall('/api/v1/system/diagnostics', 'POST');
            this.showToast('Diagnostics completed', 'success');
        } catch (error) {
            this.showToast('Diagnostics failed', 'error');
        }
    }

    async optimizePerformance() {
        try {
            await this.apiCall('/api/v1/system/optimize', 'POST');
            this.showToast('Performance optimization triggered', 'success');
        } catch (error) {
            this.showToast('Optimization failed', 'error');
        }
    }

    clearDebugConsole() {
        const console = document.getElementById('debug-console');
        if (console) {
            console.innerHTML = '';
        }
    }

    addBreakpoint() {
        const input = document.getElementById('breakpoint-input');
        const location = input?.value.trim();
        if (!location) return;

        // Add breakpoint via API
        this.apiCall('/api/v1/debug/breakpoint', 'POST', { location })
            .then(() => {
                this.showToast(`Breakpoint added: ${location}`, 'success');
                input.value = '';
                this.refreshBreakpoints();
            })
            .catch(() => {
                this.showToast('Failed to add breakpoint', 'error');
            });
    }

    addWatchVariable() {
        const input = document.getElementById('watch-input');
        const variable = input?.value.trim();
        if (!variable) return;

        // Add watch variable via API
        this.apiCall('/api/v1/debug/watch', 'POST', { variable })
            .then(() => {
                this.showToast(`Watching variable: ${variable}`, 'success');
                input.value = '';
                this.refreshWatchVariables();
            })
            .catch(() => {
                this.showToast('Failed to add watch variable', 'error');
            });
    }

    async startProfiling() {
        try {
            await this.apiCall('/api/v1/debug/profiler/start', 'POST');
            this.showToast('Profiling started', 'success');
        } catch (error) {
            this.showToast('Failed to start profiling', 'error');
        }
    }

    async stopProfiling() {
        try {
            await this.apiCall('/api/v1/debug/profiler/stop', 'POST');
            this.showToast('Profiling stopped', 'success');
        } catch (error) {
            this.showToast('Failed to stop profiling', 'error');
        }
    }

    async resetProfiler() {
        try {
            await this.apiCall('/api/v1/debug/profiler/reset', 'POST');
            this.showToast('Profiler reset', 'success');
            this.loadProfilerData();
        } catch (error) {
            this.showToast('Failed to reset profiler', 'error');
        }
    }

    // Additional methods would be implemented here...
    scheduleReconnect() {
        if (this.wsReconnectTimer) {
            clearTimeout(this.wsReconnectTimer);
        }

        this.wsReconnectTimer = setTimeout(() => {
            console.log('🔄 Attempting WebSocket reconnection...');
            this.setupWebSocket();
        }, 5000);
    }

    async exportData() {
        try {
            // Collect all current data
            const exportData = {
                timestamp: new Date().toISOString(),
                analytics: this.state.analytics,
                debug: this.state.debugData,
                profiler: this.state.profilerData,
                alerts: this.state.alerts
            };

            // Create and download file
            const blob = new Blob([JSON.stringify(exportData, null, 2)],
                { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `dashboard-export-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Data exported successfully', 'success');
        } catch (error) {
            this.showToast('Failed to export data', 'error');
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.advancedDashboard = new AdvancedDashboard();
});