# Pokemon Crystal RL Test Suite

This directory contains tests for the Pokemon Crystal RL project.

## Test Setup

### Configuration and Test Mode

Tests use a special test configuration that disables non-essential features by default:

```python
@pytest.fixture
def test_config() -> TrainingConfig:
    return TrainingConfig(
        rom_path='test.gbc',
        mode=TrainingMode.SYNCHRONIZED,
        enable_web=False,     # Disable web server
        capture_screens=False, # Disable screen capture
        headless=True,        # Always run headless
        save_stats=False,     # Don't save stats
        test_mode=True       # Enable test mode
    )
```

The `test_mode=True` flag ensures that background threads and network services are disabled during tests, making tests faster and more reliable.

### Mock Components

The test suite includes mock implementations of key components:

1. **MockWebServer** (`mock_web_server.py`):
   - Simulates the web server without actual network/threads
   - Records stats and screen updates for verification
   - Same interface as TrainingWebServer

2. **MockScreenCapture** (`mock_screen_capture.py`):
   - Simulates screen capture without actual screen access
   - Provides pre-defined frames for tests
   - Thread-safe queue management for screen data

### Data Bus Testing

Data bus tests (`test_data_bus.py`) use test mode to:
- Verify data bus initialization and shutdown
- Test data publishing and subscriptions
- Handle error conditions
- Test screen data publishing

### Web Server Testing

Web server tests (`test_web_server.py`) use unittest to:
- Test all HTTP endpoints
- Test server lifecycle (start/stop)
- Test error handling
- Test stream quality control

### Screen Capture Testing

Screen capture tests (`test_screen_capture.py`) verify:
- Frame generation and queueing
- Queue overflow handling
- Mock frame management
- Capture timing and state management

## Writing Tests

### Using Test Mode

Always use `test_mode=True` when testing components that might spawn threads or network services:

```python
def test_my_feature(test_config):
    trainer = PokemonTrainer(test_config)
    # Your test code here
```

### Using Mock Components

1. For web server tests:
```python
from tests.monitoring.mock_web_server import MockWebServer

def test_web_feature(test_config):
    server = MockWebServer(test_config, mock_trainer)
    server.start()
    # Your test code here
```

2. For screen capture tests:
```python
from tests.monitoring.mock_screen_capture import MockScreenCapture

def test_capture_feature(capture_config):
    capture = MockScreenCapture(capture_config)
    capture.start()
    # Your test code here
```

### Testing with Data Bus

When testing components that use the data bus:

1. Use the `data_bus` fixture which provides a clean instance for each test
2. Mock the `get_data_bus()` function to return the test instance
3. Subscribe to relevant data types with a component ID

```python
def test_with_data_bus(data_bus, test_config):
    # Subscribe to updates
    updates = []
    def callback(data):
        updates.append(data)
    data_bus.subscribe(DataType.TRAINING_STATS, "test_subscriber", callback)
    
    # Your test code here
```

## Test Categories

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.monitoring`: Tests for monitoring components
- `@pytest.mark.integration`: Integration tests
- More markers can be added in conftest.py

## Best Practices

1. Always use test_mode for components that might create threads
2. Clean up resources in test teardown (use pytest fixtures)
3. Avoid real file I/O, network access, or screen capture in tests
4. Use mock components for testing instead of real services
5. Keep tests focused and independent
6. Use meaningful assertions that verify behavior, not implementation
7. Document test scenarios and edge cases
