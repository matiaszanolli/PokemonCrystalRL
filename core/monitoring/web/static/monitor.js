// Enhanced Pokemon Crystal RL monitoring utilities

// State
let screenUpdateInterval = null;
let statsUpdateInterval = null;
let socket = null;
let performanceCharts = {};

// Constants
const UPDATE_INTERVALS = {
    SCREEN: 200,  // 5 fps
    STATS: 1000,  // 1 second
    PERFORMANCE: 5000  // 5 seconds
};

const MAX_HISTORY_POINTS = 100;

// Initialize monitoring system
function initMonitoring() {
    initWebSocket();
    initScreenUpdates();
    initStatsUpdates();
    initCharts();
}

// WebSocket handling
function initWebSocket() {
    try {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('WebSocket connected');
            updateConnectionStatus(true);
        };
        
        socket.onclose = () => {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
            // Fallback to polling
            initPollingFallback();
        };
        
        socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                switch (msg.type) {
                    case 'screenshot_update':
                        handleScreenUpdate(msg.payload);
                        break;
                    case 'stats_update':
                        handleStatsUpdate(msg.payload);
                        break;
                    case 'text_update':
                        handleTextUpdate(msg.payload);
                        break;
                    case 'action_update':
                        handleActionUpdate(msg.payload);
                        break;
                    default:
                        break;
                }
            } catch (e) {
                console.error('WS message parse error', e);
            }
        };
        
    } catch (err) {
        console.error('WebSocket initialization error:', err);
        initPollingFallback();
    }
}

// Screen handling
function initScreenUpdates() {
    screenUpdateInterval = setInterval(requestScreenUpdate, UPDATE_INTERVALS.SCREEN);
}

function requestScreenUpdate() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'request_screenshot' }));
    } else {
        fetchScreenUpdate();
    }
}

function handleScreenUpdate(data) {
    if (!data || !data.screenshot) return;
    
    const img = document.getElementById('game-screen');
    if (img) {
        img.src = `data:image/png;base64,${data.screenshot}`;
    }
}

// Stats handling
function initStatsUpdates() {
    statsUpdateInterval = setInterval(requestStatsUpdate, UPDATE_INTERVALS.STATS);
}

function requestStatsUpdate() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'request_stats' }));
    } else {
        fetchStatsUpdate();
    }
}

function handleStatsUpdate(stats) {
    if (!stats) return;
    
    // Update basic stats
    updateStatValue('total-actions', stats.basic.total_steps);
    updateStatValue('total-episodes', stats.basic.total_episodes);
    updateStatValue('session-duration', formatDuration(stats.basic.session_duration));
    updateStatValue('actions-per-second', stats.performance.avg_actions_per_second.toFixed(1));
    
    // Update memory stats
    updateStatValue('memory-usage', `${stats.system.memory_usage_mb.toFixed(1)} MB`);
    updateStatValue('cpu-usage', `${stats.system.cpu_percent.toFixed(1)}%`);
    
    // Update charts
    updatePerformanceCharts(stats);
}

// Chart management
function initCharts() {
    performanceCharts.actions = new Chart(
        document.getElementById('actions-chart').getContext('2d'),
        createChartConfig('Actions per Second', 'rgb(75, 192, 192)')
    );
    
    performanceCharts.memory = new Chart(
        document.getElementById('memory-chart').getContext('2d'),
        createChartConfig('Memory Usage (MB)', 'rgb(255, 99, 132)')
    );
}

function createChartConfig(label, color) {
    return {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                backgroundColor: color,
                borderColor: color,
                data: [],
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second'
                    }
                }
            }
        }
    };
}

function updatePerformanceCharts(stats) {
    const timestamp = new Date();
    
    // Update actions chart
    updateChart(
        performanceCharts.actions,
        timestamp,
        stats.performance.avg_actions_per_second
    );
    
    // Update memory chart
    updateChart(
        performanceCharts.memory,
        timestamp,
        stats.system.memory_usage_mb
    );
}

function updateChart(chart, timestamp, value) {
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(value);
    
    // Keep only recent history
    if (chart.data.labels.length > MAX_HISTORY_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update('none');  // Update without animation
}

// Helper functions
function updateStatValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-status');
    if (indicator) {
        indicator.className = `status-indicator status-${connected ? 'active' : 'inactive'}`;
        indicator.title = connected ? 'Connected' : 'Disconnected';
    }
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// HTTP fallback
function initPollingFallback() {
    console.log('Falling back to HTTP polling');
}

async function fetchScreenUpdate() {
    try {
        const response = await fetch('/api/screenshot/current');
        const data = await response.json();
        handleScreenUpdate(data);
    } catch (err) {
        console.error('Screen update error:', err);
    }
}

async function fetchStatsUpdate() {
    try {
        const response = await fetch('/api/session/stats');
        const data = await response.json();
        handleStatsUpdate(data);
    } catch (err) {
        console.error('Stats update error:', err);
    }
}

// Event handlers
function handleTextUpdate(data) {
    const log = document.getElementById('text-log');
    if (!log) return;
    
    const entry = document.createElement('div');
    entry.className = 'text-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${data.text}`;
    
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
    
    // Keep log size reasonable
    while (log.childNodes.length > 100) {
        log.removeChild(log.firstChild);
    }
}

function handleActionUpdate(data) {
    const list = document.getElementById('action-list');
    if (!list) return;
    
    const item = document.createElement('li');
    item.className = 'action-item';
    item.innerHTML = `
        <span>${data.action}</span>
        <small>${new Date().toLocaleTimeString()}</small>
    `;
    
    list.appendChild(item);
    list.scrollTop = list.scrollHeight;
    
    // Keep list size reasonable
    while (list.childNodes.length > 50) {
        list.removeChild(list.firstChild);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initMonitoring);
