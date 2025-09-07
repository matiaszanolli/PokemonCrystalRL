/**
 * Shared utilities for Pokemon Crystal RL monitoring interface.
 */

const MonitoringUI = {
    // Logging Functions
    
    /**
     * Create a new log entry element.
     * @param {string} message - Log message
     * @param {string} type - Entry type (info, error, warn)
     * @returns {HTMLElement} The created log entry
     */
    createLogEntry: (message, type = 'info') => {
        const entry = document.createElement('div');
        entry.className = 'flex items-center space-x-3';
        
        const dot = document.createElement('div');
        dot.className = `w-2 h-2 rounded-full ${
            type === 'error' ? 'bg-red-500' :
            type === 'warn' ? 'bg-yellow-500' : 'bg-blue-500'
        }`;
        
        const text = document.createElement('p');
        text.className = 'text-gray-300';
        text.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        entry.appendChild(dot);
        entry.appendChild(text);
        
        return entry;
    },
    
    /**
     * Add a log entry to a container.
     * @param {string} containerId - ID of the log container element
     * @param {string} message - Log message
     * @param {string} type - Entry type (info, error, warn)
     * @param {number} maxEntries - Maximum number of entries to keep
     */
    addLogEntry: (containerId, message, type = 'info', maxEntries = 10) => {
        const log = document.getElementById(containerId);
        if (!log) return;
        
        const entry = MonitoringUI.createLogEntry(message, type);
        log.insertBefore(entry, log.firstChild);
        
        // Prune old entries
        while (log.children.length > maxEntries) {
            log.removeChild(log.lastChild);
        }
    },
    
    // Status Updates
    
    /**
     * Get a nested value from an object using dot notation.
     * @param {Object} obj - Source object
     * @param {string} path - Dot-notation path (e.g. "data.nested.value")
     * @returns {*} Value at the path or undefined
     */
    getNestedValue: (obj, path) => {
        return path.split('.').reduce((current, key) => {
            return current && current[key] !== undefined
                ? current[key]
                : undefined;
        }, obj);
    },
    
    /**
     * Update multiple elements with values from a data object.
     * @param {Object} data - Data object
     * @param {Object} mappings - Map of data paths to element IDs
     */
    updateMetrics: (data, mappings) => {
        for (const [path, elementId] of Object.entries(mappings)) {
            const element = document.getElementById(elementId);
            if (!element) continue;
            
            const value = MonitoringUI.getNestedValue(data, path);
            if (value !== undefined) {
                element.textContent = value;
            }
        }
    },
    
    // Resource Visualization
    
    /**
     * Update a progress bar element.
     * @param {string} elementId - Progress bar element ID
     * @param {number} value - Current value
     * @param {number} maxValue - Maximum value
     * @param {string} format - Value format function name
     */
    updateResourceBar: (elementId, value, maxValue = 100, format = null) => {
        const bar = document.getElementById(elementId);
        if (!bar) return;
        
        const percent = Math.min(Math.max((value / maxValue) * 100, 0), 100);
        const barElement = bar.querySelector('.progress-fill');
        const valueElement = bar.querySelector('.progress-value');
        
        if (barElement) {
            barElement.style.width = `${percent}%`;
        }
        
        if (valueElement) {
            valueElement.textContent = format ? MonitoringUI[format](value) : value;
        }
    },
    
    // Utility Formatters
    
    /**
     * Format bytes into human-readable string.
     * @param {number} bytes - Bytes to format
     * @returns {string} Formatted string (e.g. "1.5 MB")
     */
    formatBytes: (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
    },
    
    /**
     * Format duration in seconds to HH:MM:SS.
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    formatDuration: (seconds) => {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(hrs).padStart(2, '0')}:${
            String(mins).padStart(2, '0')}:${
            String(secs).padStart(2, '0')}`;
    },
    
    /**
     * Format a number to a fixed precision.
     * @param {number} value - Number to format
     * @param {number} precision - Decimal places
     * @returns {string} Formatted number
     */
    formatNumber: (value, precision = 1) => {
        return Number(value).toFixed(precision);
    },
    
    // Status Badge Updates
    
    /**
     * Update a status badge element.
     * @param {string} elementId - Badge element ID
     * @param {boolean} running - Whether the component is running
     */
    updateStatusBadge: (elementId, running) => {
        const badge = document.getElementById(elementId);
        if (!badge) return;
        
        badge.textContent = running ? 'Running' : 'Stopped';
        badge.className = `status-badge ${running ? 'running' : 'stopped'}`;
    },
    
    // WebSocket Connection Status
    
    /**
     * Update connection status indicator.
     * @param {boolean} connected - Connection state
     */
    updateConnectionStatus: (connected) => {
        const status = document.getElementById('connectionStatus');
        if (!status) return;
        
        status.textContent = connected ? 'Connected' : 'Disconnected';
        status.className = `px-2 py-1 rounded text-sm ${
            connected ? 'bg-green-600' : 'bg-red-600'
        }`;
    }
};

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringUI;
}
