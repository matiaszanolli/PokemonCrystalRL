"""
Web-based monitoring interface for Pokemon Crystal training.
"""

import json
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

class WebMonitor(BaseHTTPRequestHandler):
    """Enhanced web server with LLM decision tracking"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Pokemon Crystal LLM RL Training Dashboard</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
                        color: #fff;
                        min-height: 100vh;
                        overflow-x: hidden;
                    }
                    
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                        position: sticky;
                        top: 0;
                        z-index: 100;
                    }
                    
                    .header h1 {
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 8px;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    }
                    
                    .header p {
                        opacity: 0.9;
                        font-size: 16px;
                    }
                    
                    .status-indicator {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #00ff88;
                        margin-left: 10px;
                        box-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
                        animation: pulse 2s infinite;
                        transition: background-color 0.3s ease, box-shadow 0.3s ease;
                    }
                    
                    .status-indicator.error {
                        background: #ff4757;
                        box-shadow: 0 0 8px rgba(255, 71, 87, 0.6);
                    }
                    
                    .status-indicator.warning {
                        background: #ffa502;
                        box-shadow: 0 0 8px rgba(255, 165, 2, 0.6);
                    }
                    
                    @keyframes pulse {
                        0% { opacity: 1; transform: scale(1); }
                        50% { opacity: 0.7; transform: scale(1.1); }
                        100% { opacity: 1; transform: scale(1); }
                    }
                    
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    
                    @keyframes slideInLeft {
                        from { opacity: 0; transform: translateX(-20px); }
                        to { opacity: 1; transform: translateX(0); }
                    }
                    
                    @keyframes countUp {
                        from { transform: scale(0.8); }
                        to { transform: scale(1); }
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 0 20px;
                        display: grid;
                        grid-template-columns: 2fr 1fr;
                        gap: 20px;
                    }
                    
                    .main-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .side-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .stat-card {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .stat-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
                    }
                    
                    .stat-card::before {
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 3px;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                    }
                    
                    .stat-value {
                        font-size: 32px;
                        font-weight: 800;
                        color: #00ff88;
                        margin-bottom: 5px;
                        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
                    }
                    
                    .stat-label {
                        font-size: 14px;
                        color: #aaa;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        font-weight: 500;
                    }
                    
                    .stat-change {
                        font-size: 12px;
                        margin-top: 5px;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }
                    
                    .stat-change.positive { color: #00ff88; }
                    .stat-change.negative { color: #ff4757; }
                    
                    .game-screen-container {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        text-align: center;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .game-screen-container h3 {
                        margin-bottom: 15px;
                        font-size: 20px;
                        color: #fff;
                    }
                    
                    .game-screen {
                        border-radius: 8px;
                        border: 2px solid #00ff88;
                        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                        max-width: 100%;
                        height: auto;
                        image-rendering: pixelated;
                        image-rendering: crisp-edges;
                    }
                    
                    .screen-info {
                        display: flex;
                        justify-content: space-between;
                        margin-top: 15px;
                        font-size: 12px;
                        color: #aaa;
                    }
                    
                    .panel {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    }
                    
                    .panel h3 {
                        font-size: 18px;
                        margin-bottom: 15px;
                        color: #fff;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    .reward-bars {
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }
                    
                    .reward-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 6px;
                        padding: 8px 12px;
                        display: flex;
                        justify-content: between;
                        align-items: center;
                        font-size: 14px;
                    }
                    
                    .reward-bar.positive {
                        background: linear-gradient(90deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
                        border-left: 3px solid #00ff88;
                    }
                    
                    .reward-bar.negative {
                        background: linear-gradient(90deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
                        border-left: 3px solid #ff4757;
                    }
                    
                    .llm-decision {
                        background: rgba(255,255,255,0.05);
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        border-left: 4px solid #00ff88;
                        position: relative;
                    }
                    
                    .llm-decision::before {
                        content: '🤖';
                        position: absolute;
                        top: 12px;
                        right: 12px;
                        font-size: 16px;
                    }
                    
                    .llm-action {
                        font-weight: 700;
                        color: #00ff88;
                        font-size: 16px;
                        margin-bottom: 5px;
                    }
                    
                    .llm-reasoning {
                        color: #ccc;
                        font-size: 13px;
                        line-height: 1.4;
                    }
                    
                    .llm-timestamp {
                        color: #666;
                        font-size: 11px;
                        margin-top: 5px;
                    }
                    
                    .progress-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 10px;
                        height: 8px;
                        overflow: hidden;
                        margin-top: 8px;
                    }
                    
                    .progress-fill {
                        height: 100%;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                        transition: width 0.3s ease;
                        border-radius: 10px;
                    }
                    
                    .loading-spinner {
                        display: inline-block;
                        width: 16px;
                        height: 16px;
                        border: 2px solid #333;
                        border-radius: 50%;
                        border-top: 2px solid #00ff88;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    .error-state {
                        color: #ff4757;
                        text-align: center;
                        padding: 20px;
                    }
                    
                    @media (max-width: 1024px) {
                        .container {
                            grid-template-columns: 1fr;
                            max-width: 800px;
                        }
                        
                        .stats-grid {
                            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        }
                    }
                    
                    @media (max-width: 768px) {
                        .stats-grid {
                            grid-template-columns: repeat(2, 1fr);
                        }
                        
                        .header h1 {
                            font-size: 24px;
                        }
                        
                        .container {
                            padding: 0 15px;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Pokemon Crystal LLM RL Training Dashboard<span class="status-indicator" id="status-indicator"></span></h1>
                    <p>Advanced reinforcement learning with local LLM decision making and real-time monitoring</p>
                </div>
                
                <div class="container">
                    <div class="main-panel">
                        <!-- Stats Grid -->
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value" id="actions">-</div>
                                <div class="stat-label">Total Actions</div>
                                <div class="stat-change" id="actions-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="aps">-</div>
                                <div class="stat-label">Actions/Second</div>
                                <div class="stat-change" id="aps-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="llm-decisions">-</div>
                                <div class="stat-label">LLM Decisions</div>
                                <div class="stat-change" id="llm-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="total-reward">-</div>
                                <div class="stat-label">Total Reward</div>
                                <div class="stat-change" id="reward-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="level">-</div>
                                <div class="stat-label">Player Level</div>
                                <div class="stat-change" id="level-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="badges">-</div>
                                <div class="stat-label">Badges Earned</div>
                                <div class="progress-bar"><div class="progress-fill" id="badge-progress"></div></div>
                            </div>
                        </div>
                        
                        <!-- Game Screen -->
                        <div class="game-screen-container">
                            <h3>🖼️ Live Game Screen</h3>
                            <img id="gameScreen" class="game-screen" src="/screenshot?t=0" width="480" height="432" alt="Game Screen">
                            <div class="screen-info">
                                <span>Screen State: <span id="screen-state">-</span></span>
                                <span>Refresh Rate: <span id="refresh-rate">500ms</span></span>
                                <span>Last Updated: <span id="last-update">-</span></span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="side-panel">
                        <!-- Game State Info -->
                        <div class="panel">
                            <h3>🎮 Game State</h3>
                            <div class="reward-bars" id="game-state-info">
                                <div class="reward-bar">Map: <span id="player-map">-</span></div>
                                <div class="reward-bar">Position: <span id="player-position">-</span></div>
                                <div class="reward-bar">Money: ¥<span id="player-money">-</span></div>
                                <div class="reward-bar">Party: <span id="party-count">-</span> Pokemon</div>
                                <div class="reward-bar">HP: <span id="player-hp">-</span></div>
                            </div>
                        </div>
                        
                        <!-- Reward Breakdown -->
                        <div class="panel">
                            <h3>💰 Reward Analysis</h3>
                            <div class="reward-bars" id="reward-breakdown">
                                <div class="loading-spinner"></div> Loading...
                            </div>
                        </div>
                        
                        <!-- LLM Decisions -->
                        <div class="panel">
                            <h3>🧠 Recent LLM Decisions</h3>
                            <div id="llm-decisions-list">
                                <div class="loading-spinner"></div> Loading decisions...
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    let lastStats = {};
                    let updateCount = 0;
                    let startTime = Date.now();
                    
                    function formatNumber(num) {
                        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
                        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
                        return num.toString();
                    }
                    
                    function formatChange(current, previous) {
                        if (previous === undefined) return '';
                        const change = current - previous;
                        if (change === 0) return '';
                        const sign = change > 0 ? '+' : '';
                        const className = change > 0 ? 'positive' : 'negative';
                        return `<span class="${className}">${sign}${change.toFixed(2)}</span>`;
                    }
                    
                    function updateGameScreen() {
                        const screen = document.getElementById('gameScreen');
                        const timestamp = Date.now();
                        screen.src = `/screenshot?t=${timestamp}`;
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    }
                    
                    async function updateStats() {
                        try {
                            updateCount++;
                            const response = await fetch('/stats');
                            const stats = await response.json();
                            
                            // Update status indicator
                            document.getElementById('status-indicator').style.background = '#00ff88';
                            
                            // Update main stats with change indicators
                            document.getElementById('actions').textContent = formatNumber(stats.actions_taken || 0);
                            document.getElementById('aps').textContent = (stats.actions_per_second || 0).toFixed(1);
                            document.getElementById('llm-decisions').textContent = formatNumber(stats.llm_decision_count || 0);
                            document.getElementById('total-reward').textContent = (stats.total_reward || 0).toFixed(2);
                            document.getElementById('level').textContent = stats.player_level || 0;
                            document.getElementById('badges').textContent = `${stats.badges_total || 0}/16`;
                            
                            // Update change indicators
                            document.getElementById('actions-change').innerHTML = formatChange(stats.actions_taken, lastStats.actions_taken);
                            document.getElementById('aps-change').innerHTML = formatChange(stats.actions_per_second, lastStats.actions_per_second);
                            document.getElementById('llm-change').innerHTML = formatChange(stats.llm_decision_count, lastStats.llm_decision_count);
                            document.getElementById('reward-change').innerHTML = formatChange(stats.total_reward, lastStats.total_reward);
                            document.getElementById('level-change').innerHTML = formatChange(stats.player_level, lastStats.player_level);
                            
                            // Update badge progress
                            const badgeProgress = ((stats.badges_total || 0) / 16) * 100;
                            document.getElementById('badge-progress').style.width = badgeProgress + '%';
                            
                            // Update game state info
                            document.getElementById('player-map').textContent = stats.final_game_state?.player_map || '-';
                            document.getElementById('player-position').textContent = 
                                stats.final_game_state ? 
                                `(${stats.final_game_state.player_x}, ${stats.final_game_state.player_y})` : '-';
                            document.getElementById('player-money').textContent = formatNumber(stats.final_game_state?.money || 0);
                            document.getElementById('party-count').textContent = stats.final_game_state?.party_count || 0;
                            document.getElementById('player-hp').textContent = 
                                stats.final_game_state ? 
                                `${stats.final_game_state.player_hp}/${stats.final_game_state.player_max_hp}` : '-';
                            
                            // Update screen state
                            document.getElementById('screen-state').textContent = stats.screen_state || 'Unknown';
                            
                            // Update reward breakdown with bars
                            const rewardDiv = document.getElementById('reward-breakdown');
                            if (stats.last_reward_breakdown && stats.last_reward_breakdown !== 'no rewards') {
                                const rewards = stats.last_reward_breakdown.split(' | ');
                                rewardDiv.innerHTML = rewards.map(reward => {
                                    const [category, value] = reward.split(': ');
                                    const numValue = parseFloat(value);
                                    const className = numValue > 0 ? 'positive' : 'negative';
                                    return `<div class="reward-bar ${className}">${category}: <strong>${value}</strong></div>`;
                                }).join('');
                            } else {
                                rewardDiv.innerHTML = '<div class="reward-bar">No active rewards</div>';
                            }
                            
                            // Update LLM decisions with enhanced formatting
                            const decisionsDiv = document.getElementById('llm-decisions-list');
                            if (stats.recent_llm_decisions && stats.recent_llm_decisions.length > 0) {
                                decisionsDiv.innerHTML = stats.recent_llm_decisions.map((d, index) => {
                                    const timestamp = new Date(d.timestamp * 1000);
                                    return `<div class="llm-decision">
                                        <div class="llm-action">Action: ${d.action.toUpperCase()}</div>
                                        <div class="llm-reasoning">${d.reasoning.substring(0, 150)}${d.reasoning.length > 150 ? '...' : ''}</div>
                                        <div class="llm-timestamp">${timestamp.toLocaleTimeString()}</div>
                                    </div>`;
                                }).join('');
                            } else {
                                decisionsDiv.innerHTML = '<div class="llm-decision">No LLM decisions yet...</div>';
                            }
                            
                            lastStats = {...stats};
                            
                        } catch (e) {
                            console.error('Failed to update stats:', e);
                            const statusIndicator = document.getElementById('status-indicator');
                            statusIndicator.style.background = '#ff4757';
                            statusIndicator.className = 'status-indicator error';
                            
                            // Try to recover after a delay
                            setTimeout(() => {
                                statusIndicator.className = 'status-indicator warning';
                                statusIndicator.style.background = '#ffa502';
                            }, 2000);
                        }
                    }
                    
                    // Update refresh rate indicators
                    const STATS_REFRESH_RATE = 200;
                    const SCREEN_REFRESH_RATE = 100;
                    
                    document.getElementById('refresh-rate').textContent = `${SCREEN_REFRESH_RATE}ms`;
                    
                    // Update stats every 200ms for more responsive feedback
                    setInterval(updateStats, STATS_REFRESH_RATE);
                    
                    // Update game screen every 100ms for smooth real-time visuals
                    setInterval(updateGameScreen, SCREEN_REFRESH_RATE);
                    
                    // Performance monitoring
                    let missedUpdates = 0;
                    let totalUpdates = 0;
                    
                    setInterval(() => {
                        totalUpdates++;
                        const successRate = ((totalUpdates - missedUpdates) / totalUpdates * 100).toFixed(1);
                        
                        // Update connection quality indicator
                        const statusIndicator = document.getElementById('status-indicator');
                        if (successRate > 95) {
                            statusIndicator.className = 'status-indicator';
                            statusIndicator.style.background = '#00ff88';
                        } else if (successRate > 80) {
                            statusIndicator.className = 'status-indicator warning';
                            statusIndicator.style.background = '#ffa502';
                        }
                    }, 5000);
                    
                    // Initial load
                    updateStats();
                    updateGameScreen();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats = getattr(self.server, 'trainer_stats', {})
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path.startswith('/screenshot'):
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            screenshot_data = getattr(self.server, 'screenshot_data', None)
            if screenshot_data:
                self.wfile.write(screenshot_data)
            else:
                # Send empty 1x1 PNG if no screenshot
                empty_img = Image.new('RGB', (1, 1), (0, 0, 0))
                buf = io.BytesIO()
                empty_img.save(buf, format='PNG')
                self.wfile.write(buf.getvalue())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs
