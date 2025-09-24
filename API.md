# 🚀 Pokemon Crystal RL REST API

**Version**: 1.0.0
**Base URL**: `http://localhost:8080/api/v1`
**Content-Type**: `application/json`

## 📋 Overview

The Pokemon Crystal RL REST API provides comprehensive control over the reinforcement learning platform, including:

- **Training Session Management**: Create, configure, and control training sessions
- **Multi-Agent System**: Manage specialist agents (Battle, Explorer, Progression)
- **Plugin System**: Load, configure, and manage modular components
- **Real-time Monitoring**: Access live training metrics and system status

## 🔐 Authentication

Currently, the API operates in development mode without authentication. For production use, implement proper authentication and rate limiting.

## 📊 Response Format

All API responses follow a consistent format:

```json
{
    "success": true,
    "data": { ... },
    "message": "Optional success message",
    "error": null,
    "timestamp": 1640995200.0,
    "request_id": "optional-request-id"
}
```

## 🎮 Training Session Management

### List Training Sessions

**GET** `/api/v1/training/sessions`

List all active training sessions.

**Response:**
```json
{
    "success": true,
    "data": {
        "sessions": [
            {
                "session_id": "uuid-123",
                "status": "running",
                "config": { ... },
                "start_time": 1640995200.0,
                "current_action": 1250,
                "total_reward": 45.7,
                "active_agents": ["battle", "explorer"]
            }
        ],
        "total": 1
    }
}
```

### Create Training Session

**POST** `/api/v1/training/sessions`

Create a new training session with custom configuration.

**Request Body:**
```json
{
    "rom_path": "/path/to/pokemon_crystal.gbc",
    "save_state_path": "/path/to/save.state",
    "max_actions": 5000,
    "headless": true,
    "enable_llm": true,
    "llm_model": "smollm2:1.7b",
    "llm_interval": 15,
    "primary_agent": "hybrid",
    "enabled_agents": ["battle", "explorer", "progression"],
    "active_plugins": {
        "aggressive_battle": {"aggression": 0.8},
        "systematic_exploration": {"pattern": "spiral"}
    },
    "enable_web": true,
    "web_port": 8080
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "session_id": "uuid-456",
        "status": "stopped",
        "config": { ... },
        "start_time": 1640995200.0
    },
    "message": "Training session uuid-456 created successfully"
}
```

### Get Training Session

**GET** `/api/v1/training/sessions/{session_id}`

Get details for a specific training session.

### Control Training Session

**POST** `/api/v1/training/sessions/{session_id}/control`

Control training session lifecycle (start/stop/pause/resume).

**Request Body:**
```json
{
    "action": "start",
    "config": {
        "max_actions": 10000
    }
}
```

**Actions**: `start`, `stop`, `pause`, `resume`, `restart`

## 🤖 Multi-Agent System

### List Agents

**GET** `/api/v1/agents`

List all available agents and their status.

**Response:**
```json
{
    "success": true,
    "data": {
        "agents": [
            {
                "agent_id": "battle_agent",
                "agent_type": "battle",
                "status": "active",
                "performance_metrics": {
                    "win_rate": 0.75,
                    "actions_per_second": 2.3
                },
                "configuration": {
                    "strategy": "aggressive"
                }
            }
        ],
        "total": 3
    }
}
```

### Control Agent

**POST** `/api/v1/agents/{agent_id}/control`

Control individual agent (start/stop/configure).

**Request Body:**
```json
{
    "action": "configure",
    "configuration": {
        "strategy": "defensive",
        "priority": 0.8
    }
}
```

**Actions**: `start`, `stop`, `pause`, `resume`, `configure`

### Get Coordination Status

**GET** `/api/v1/agents/coordination`

Get multi-agent coordination status and performance.

**Response:**
```json
{
    "success": true,
    "data": {
        "coordinator_active": true,
        "coordination_strategy": "priority_based",
        "agent_priorities": {
            "battle_agent": 0.9,
            "explorer_agent": 0.7
        },
        "recent_decisions": [...],
        "conflict_resolutions": 5,
        "performance_score": 0.85
    }
}
```

## 🔧 Plugin System

### List Plugins

**GET** `/api/v1/plugins`

List all available plugins and their status.

**Response:**
```json
{
    "success": true,
    "data": {
        "plugins": [
            {
                "plugin_id": "aggressive_battle",
                "plugin_type": "battle_strategy",
                "status": "active",
                "version": "1.0.0",
                "configuration": {
                    "aggression": 0.8
                },
                "performance_metrics": {
                    "win_rate": 0.78
                }
            }
        ],
        "total": 5
    }
}
```

### Control Plugin

**POST** `/api/v1/plugins/{plugin_id}/control`

Load, unload, or configure plugins.

**Request Body:**
```json
{
    "action": "configure",
    "configuration": {
        "aggression": 0.9,
        "risk_tolerance": 0.6
    }
}
```

**Actions**: `load`, `unload`, `activate`, `deactivate`, `configure`, `reload`

## 📚 API Documentation

### Get API Documentation

**GET** `/api/v1/docs`

Get complete API documentation and endpoint specifications.

## 🛠️ Usage Examples

### Python Client Example

```python
import requests
import json

# Base configuration
API_BASE = "http://localhost:8080/api/v1"
HEADERS = {"Content-Type": "application/json"}

# Create a training session
session_config = {
    "rom_path": "/path/to/pokemon_crystal.gbc",
    "max_actions": 5000,
    "enable_llm": True,
    "llm_model": "smollm2:1.7b",
    "primary_agent": "hybrid"
}

response = requests.post(
    f"{API_BASE}/training/sessions",
    headers=HEADERS,
    data=json.dumps(session_config)
)

if response.status_code == 200:
    session_data = response.json()
    session_id = session_data["data"]["session_id"]
    print(f"Created session: {session_id}")

    # Start training
    start_request = {"action": "start"}
    response = requests.post(
        f"{API_BASE}/training/sessions/{session_id}/control",
        headers=HEADERS,
        data=json.dumps(start_request)
    )

    if response.status_code == 200:
        print("Training started successfully!")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:8080/api/v1';

async function createAndStartTraining() {
    try {
        // Create session
        const sessionConfig = {
            rom_path: '/path/to/pokemon_crystal.gbc',
            max_actions: 3000,
            enable_llm: true,
            primary_agent: 'battle'
        };

        const createResponse = await axios.post(
            `${API_BASE}/training/sessions`,
            sessionConfig
        );

        const sessionId = createResponse.data.data.session_id;
        console.log(`Created session: ${sessionId}`);

        // Start training
        await axios.post(
            `${API_BASE}/training/sessions/${sessionId}/control`,
            { action: 'start' }
        );

        console.log('Training started!');

        // Monitor agents
        const agentsResponse = await axios.get(`${API_BASE}/agents`);
        console.log('Active agents:', agentsResponse.data.data.agents);

    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

createAndStartTraining();
```

### cURL Examples

```bash
# Create training session
curl -X POST http://localhost:8080/api/v1/training/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "rom_path": "/path/to/pokemon_crystal.gbc",
    "max_actions": 5000,
    "enable_llm": true
  }'

# List all agents
curl -X GET http://localhost:8080/api/v1/agents

# Configure plugin
curl -X POST http://localhost:8080/api/v1/plugins/aggressive_battle/control \
  -H "Content-Type: application/json" \
  -d '{
    "action": "configure",
    "configuration": {"aggression": 0.9}
  }'

# Get API documentation
curl -X GET http://localhost:8080/api/v1/docs
```

## ⚠️ Error Handling

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (resource doesn't exist)
- **405**: Method Not Allowed
- **500**: Internal Server Error
- **503**: Service Unavailable

### Error Response Format

```json
{
    "success": false,
    "data": null,
    "message": null,
    "error": "Detailed error message",
    "timestamp": 1640995200.0
}
```

## 🔮 Advanced Integration

### WebSocket Integration

The REST API works alongside the existing WebSocket system for real-time updates:

- **REST API**: Control and configuration
- **WebSocket**: Real-time monitoring and live data streams

### Plugin Development

Create custom plugins that can be managed via the REST API:

```python
# Example plugin that can be controlled via API
class CustomBattleStrategy(BattleStrategyPlugin):
    def __init__(self):
        self.config = {"aggression": 0.5}

    def configure(self, config):
        self.config.update(config)
        # Plugin automatically hot-reloads via API
```

### Multi-Instance Coordination

Use the REST API to coordinate multiple training instances:

```python
# Orchestrate multiple training sessions
sessions = []
for i in range(3):
    session = create_session({
        "primary_agent": ["battle", "explorer", "progression"][i],
        "max_actions": 2000
    })
    sessions.append(session)
    start_session(session["session_id"])

# Monitor all sessions via API
for session in sessions:
    status = get_session_status(session["session_id"])
    print(f"Session {session['session_id']}: {status}")
```

## 📈 Performance & Monitoring

### Metrics Available via API

- **Training Progress**: Actions taken, rewards earned, completion rate
- **Agent Performance**: Decision speed, success rates, coordination efficiency
- **Plugin Metrics**: Resource usage, configuration effectiveness
- **System Health**: Memory usage, response times, error rates

### Rate Limiting (Production)

For production deployments, implement rate limiting:

```python
# Example rate limiting configuration
RATE_LIMITS = {
    "training/sessions": "10/minute",
    "agents/control": "30/minute",
    "plugins/control": "20/minute"
}
```

---

## 🚀 Getting Started

1. **Start the web server** with REST API enabled:
   ```bash
   python3 main.py roms/pokemon_crystal.gbc --enable-web --web-port 8080
   ```

2. **Access the API documentation**:
   ```bash
   curl http://localhost:8080/api/v1/docs
   ```

3. **Create your first training session**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/training/sessions \
     -H "Content-Type: application/json" \
     -d '{"rom_path": "roms/pokemon_crystal.gbc", "max_actions": 1000}'
   ```

The REST API provides powerful programmatic control over the entire Pokemon Crystal RL platform, enabling advanced automation, custom integrations, and scalable training orchestration! 🎮✨