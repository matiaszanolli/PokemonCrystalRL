# End-to-End Tests

## Overview

This directory contains end-to-end tests that validate complete workflows from CLI to results, including actual training sessions with real game state.

**Current Status**: Infrastructure ready, tests are templates awaiting implementation

## Test Categories

### 🟢 Smoke Tests (`e2e_smoke`)
**Duration**: 1-5 seconds each
**Purpose**: Quick validation that core workflows work
**Run on**: Every commit

```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_smoke" -v
```

### 🟡 Medium Tests (`e2e_medium`)
**Duration**: 30-60 seconds each
**Purpose**: Complete workflow validation
**Run on**: Pull requests

```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_medium" -v
```

### 🔴 Slow Tests (`e2e_slow`)
**Duration**: 2-5 minutes each
**Purpose**: Stability and performance validation
**Run on**: Nightly or manual

```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_slow" -v
```

## Files

- **`conftest.py`** - Shared fixtures for E2E tests
  - `verify_test_rom`: Ensure ROM exists or skip test
  - `verify_save_state`: Ensure save state exists or skip test
  - `isolated_training_env`: Create isolated test environment
  - `memory_monitor`: Monitor memory usage
  - `cleanup_event_bus`: Clean event bus between tests
  - `mock_ollama_for_testing`: Mock LLM for fast tests

- **`test_smoke.py`** - Quick smoke tests (5 tests, templates)
  - Training startup/shutdown
  - Save state loading
  - Web dashboard integration
  - LLM workflow basics
  - Curriculum learning startup

- **`fixtures/`** - Test data directory
  - Minimal save states for testing
  - Test configurations

## Implementation Status

### ✅ Ready
- Directory structure created
- Pytest markers configured
- Fixture infrastructure in place
- Template tests created
- Documentation complete

### ⏳ Pending Implementation
- Refactor `main.py` for programmatic access
- Implement actual test logic
- Create test helpers
- Add medium and slow test files

## Implementation Guide

### Step 1: Refactor main.py

Make training components accessible programmatically:

```python
# main.py additions
def parse_arguments_from_dict(config: dict) -> argparse.Namespace:
    """Parse arguments from dictionary for testing."""
    # Implementation

def initialize_trainer(args: argparse.Namespace) -> Trainer:
    """Initialize trainer from arguments."""
    # Implementation
```

### Step 2: Implement Test Helpers

Create `tests/e2e/helpers.py`:

```python
def create_test_config(**kwargs) -> dict:
    """Create test configuration with defaults."""
    defaults = {
        'headless': True,
        'enable_web': False,
        'max_actions': 10
    }
    defaults.update(kwargs)
    return defaults

def wait_for_server(url: str, timeout: int = 5):
    """Wait for web server to be ready."""
    # Implementation

def verify_training_output(stats: dict):
    """Verify training statistics are valid."""
    assert 'actions_taken' in stats
    assert 'total_reward' in stats
    # More validation
```

### Step 3: Implement First Test

Update `test_smoke.py` with actual implementation:

```python
@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(10)
def test_e2e_training_startup_and_shutdown(verify_test_rom, isolated_training_env):
    """Test basic training startup and shutdown."""
    from main import parse_arguments_from_dict, initialize_trainer

    # Setup
    config = create_test_config(
        rom_path=verify_test_rom,
        max_actions=10,
        output_dir=isolated_training_env['output_dir']
    )
    args = parse_arguments_from_dict(config)

    # Execute
    trainer = initialize_trainer(args)
    stats = trainer.run()

    # Verify
    assert stats['actions_taken'] == 10
    assert stats['total_reward'] is not None

    # Cleanup
    trainer.shutdown()
    assert trainer.is_shutdown
```

### Step 4: Add More Tests

Follow the pattern for remaining tests in the proposal.

## Best Practices

### 1. Isolation
- Each test runs in isolated environment
- Use `isolated_training_env` fixture
- Clean up resources in teardown

### 2. Timeout Protection
- All E2E tests must have `@pytest.mark.timeout()`
- Prevents hanging tests
- Reasonable defaults: 10s (smoke), 90s (medium), 360s (slow)

### 3. Meaningful Assertions
```python
# Bad
assert reward > 0

# Good
assert -100 < reward < 500, f"Reward {reward} outside expected range"
```

### 4. Skip When Needed
```python
@pytest.fixture
def verify_test_rom():
    if not os.path.exists(TEST_ROM_PATH):
        pytest.skip("ROM not available")
    return TEST_ROM_PATH
```

### 5. Resource Monitoring
```python
def test_with_memory_check(memory_monitor):
    # Test implementation
    pass
    # memory_monitor automatically tracks growth
```

## Running Tests

### Run All E2E Tests
```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -v
```

### Run Specific Category
```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_smoke" -v
```

### Run Single Test
```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e/test_smoke.py::test_e2e_training_startup_and_shutdown -v
```

### Run with Output
```bash
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -v -s  # -s shows print statements
```

## CI/CD Integration

See `E2E_TESTING_PROPOSAL.md` for GitHub Actions workflow examples.

## Troubleshooting

### "Test ROM not available"
- Ensure `roms/pokemon_crystal.gbc` exists
- Or set custom path in conftest.py

### "Save state not available"
- Create save state: See CLAUDE.md for instructions
- Or tests will skip automatically

### Test Hangs
- Check timeout is set: `@pytest.mark.timeout(10)`
- Verify training loop respects `max_actions`

### Memory Leaks
- Use `memory_monitor` fixture
- Check cleanup in teardown
- Look for unclosed resources

## Next Steps

1. Implement main.py refactoring (see [E2E_TESTING_PROPOSAL.md](../../E2E_TESTING_PROPOSAL.md))
2. Create test helpers
3. Implement smoke tests
4. Add medium tests
5. Add slow tests
6. Configure CI/CD

## Related Documentation

- [E2E_TESTING_PROPOSAL.md](../../E2E_TESTING_PROPOSAL.md) - Full proposal and strategy
- [TESTING_ROADMAP.md](../../TESTING_ROADMAP.md) - Current testing status
- [CLAUDE.md](../../CLAUDE.md) - Project overview

---

**Created**: 2025-10-11
**Status**: Infrastructure ready, awaiting implementation
