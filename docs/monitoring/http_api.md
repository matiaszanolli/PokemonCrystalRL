# HTTP API Reference

The monitoring system provides a RESTful HTTP API for interacting with the training process.

## Base URL

By default, the API is available at: `http://localhost:8080/api`

## Authentication

Currently, the API does not require authentication as it is designed for local development use.

## API Endpoints

### Training Management

#### Get Training Status

```http
GET /status
```

Returns current training status and component information.

Response:
```json
{
    "status": "running",
    "components": {
        "trainer": {
            "type": "core",
            "mode": "training",
            "last_seen": "2024-08-20T02:15:30Z"
        },
        "web_server": {
            "type": "service",
            "mode": "active",
            "last_seen": "2024-08-20T02:15:35Z"
        }
    },
    "current_run": {
        "run_id": "train_20240820_021530",
        "total_episodes": 10,
        "total_steps": 1000
    }
}
```

#### Start Training

```http
POST /training/start
```

Start a new training run.

Request:
```json
{
    "config": {
        "learning_rate": 0.001,
        "batch_size": 64
    },
    "notes": "Test run with adjusted parameters"
}
```

Response:
```json
{
    "run_id": "train_20240820_021530",
    "status": "running"
}
```

#### Stop Training

```http
POST /training/stop
```

Stop the current training run.

Request:
```json
{
    "final_reward": 100.0,
    "notes": "Completed successfully"
}
```

Response:
```json
{
    "run_id": "train_20240820_021530",
    "status": "completed",
    "duration": 3600.5,
    "total_episodes": 100
}
```

#### Pause Training

```http
POST /training/pause
```

Pause the current training run.

Response:
```json
{
    "status": "paused",
    "timestamp": "2024-08-20T02:30:00Z"
}
```

#### Resume Training

```http
POST /training/resume
```

Resume a paused training run.

Response:
```json
{
    "status": "running",
    "timestamp": "2024-08-20T02:35:00Z"
}
```

### Metrics and Monitoring

#### Get Current Metrics

```http
GET /metrics/current
```

Get the most recent metrics.

Response:
```json
{
    "timestamp": "2024-08-20T02:15:35Z",
    "metrics": {
        "performance": {
            "loss": 0.5,
            "accuracy": 0.85,
            "reward": 1.0
        },
        "system": {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_usage": 70.0
        }
    }
}
```

#### Get Metric History

```http
GET /metrics/history?metric=loss&hours=1
```

Get historical metrics data.

Parameters:
- `metric`: Metric name to retrieve
- `hours`: Hours of history (default: 1)
- `interval`: Sampling interval in seconds (default: 60)

Response:
```json
{
    "metric": "loss",
    "data": [
        {
            "timestamp": "2024-08-20T01:15:35Z",
            "value": 0.8
        },
        {
            "timestamp": "2024-08-20T01:16:35Z",
            "value": 0.7
        }
    ],
    "statistics": {
        "mean": 0.75,
        "min": 0.7,
        "max": 0.8
    }
}
```

#### Get System Metrics

```http
GET /metrics/system
```

Get current system resource usage.

Response:
```json
{
    "timestamp": "2024-08-20T02:15:35Z",
    "cpu_percent": 50.0,
    "memory_percent": 60.0,
    "disk_usage": 70.0,
    "process_memory": 1024.5
}
```

### Game State

#### Get Current Screen

```http
GET /screen
```

Get the current game screen image.

Response:
```json
{
    "timestamp": "2024-08-20T02:15:35Z",
    "frame": 1000,
    "screen": "base64_encoded_image_data",
    "format": "RGB",
    "dimensions": [160, 144]
}
```

#### Get Game State

```http
GET /state
```

Get current game state information.

Response:
```json
{
    "timestamp": "2024-08-20T02:15:35Z",
    "frame": 1000,
    "map_id": 1,
    "player_x": 10,
    "player_y": 20,
    "inventory": ["POTION", "POKEBALL"],
    "pokemon": [
        {
            "species": "PIKACHU",
            "level": 5,
            "hp": 20
        }
    ]
}
```

### Training History

#### Get Training Runs

```http
GET /runs
```

List training runs.

Parameters:
- `limit`: Maximum runs to return (default: 10)
- `status`: Filter by status (optional)

Response:
```json
{
    "runs": [
        {
            "run_id": "train_20240820_021530",
            "start_time": "2024-08-20T02:15:30Z",
            "end_time": "2024-08-20T03:15:30Z",
            "status": "completed",
            "total_episodes": 100,
            "total_steps": 10000,
            "final_reward": 100.0
        }
    ],
    "total": 1
}
```

#### Get Run Details

```http
GET /runs/{run_id}
```

Get detailed information about a training run.

Response:
```json
{
    "run_id": "train_20240820_021530",
    "start_time": "2024-08-20T02:15:30Z",
    "end_time": "2024-08-20T03:15:30Z",
    "status": "completed",
    "total_episodes": 100,
    "total_steps": 10000,
    "final_reward": 100.0,
    "config": {
        "learning_rate": 0.001,
        "batch_size": 64
    },
    "statistics": {
        "mean_reward": 50.0,
        "max_reward": 100.0,
        "success_rate": 0.85
    }
}
```

#### Get Run Events

```http
GET /runs/{run_id}/events
```

Get events for a training run.

Parameters:
- `type`: Filter by event type (optional)
- `start`: Start timestamp (optional)
- `end`: End timestamp (optional)

Response:
```json
{
    "events": [
        {
            "timestamp": "2024-08-20T02:20:00Z",
            "type": "achievement",
            "data": {
                "type": "badge",
                "value": 1
            }
        },
        {
            "timestamp": "2024-08-20T02:25:00Z",
            "type": "error",
            "data": {
                "message": "Connection timeout",
                "severity": "warning"
            }
        }
    ],
    "total": 2
}
```

### Data Export

#### Export Run Data

```http
GET /runs/{run_id}/export
```

Export training run data.

Parameters:
- `format`: Export format (json/csv, default: json)
- `include_snapshots`: Include state snapshots (default: false)

Response:
```json
{
    "export_id": "export_20240820_021530",
    "status": "ready",
    "download_url": "/exports/export_20240820_021530.zip",
    "expires_at": "2024-08-21T02:15:30Z"
}
```

## Error Responses

The API uses standard HTTP status codes and includes error details in the response:

```json
{
    "error": {
        "code": "training_not_running",
        "message": "No active training run found",
        "details": {
            "last_run": "train_20240820_021530",
            "last_status": "completed"
        }
    }
}
```

Common status codes:
- `200`: Success
- `400`: Bad request
- `404`: Resource not found
- `409`: Conflict (e.g., training already running)
- `500`: Internal server error

## Rate Limiting

The API does not currently implement rate limiting, but requests should be kept reasonable to avoid performance issues:

- Screen updates: Max 30 requests/second
- Metric updates: Max 10 requests/second
- Other endpoints: Max 100 requests/minute

## Streaming Endpoints

Some endpoints support server-sent events (SSE) for real-time updates:

### Screen Stream

```http
GET /screen/stream
```

Streams game screen updates in real-time.

Events:
```json
event: screen
data: {
    "timestamp": "2024-08-20T02:15:35.123Z",
    "frame": 1000,
    "screen": "base64_encoded_image_data"
}
```

### Metrics Stream

```http
GET /metrics/stream
```

Streams metric updates in real-time.

Events:
```json
event: metrics
data: {
    "timestamp": "2024-08-20T02:15:35.123Z",
    "metrics": {
        "loss": 0.5,
        "accuracy": 0.85
    }
}
```

### Events Stream

```http
GET /events/stream
```

Streams training events in real-time.

Events:
```json
event: training
data: {
    "timestamp": "2024-08-20T02:15:35.123Z",
    "type": "achievement",
    "data": {
        "type": "badge",
        "value": 1
    }
}
```
