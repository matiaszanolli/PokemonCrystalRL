# 🧪 Testing Guide for Hybrid LLM-RL Training System

**Comprehensive guide to testing the hybrid LLM-RL training system components.**

## 🎯 Test Overview

The hybrid training system includes extensive test coverage across multiple layers:

- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: Component interaction validation  
- **End-to-End Tests**: Complete training pipeline validation
- **Performance Tests**: Benchmark and performance validation

## 📊 Test Coverage Summary

### Core Components (25+ Tests)

#### HybridLLMRLTrainer (`tests/trainer/test_hybrid_llm_rl_trainer.py`)
- ✅ **13 test methods** covering all trainer functionality
- ✅ Curriculum learning progression
- ✅ Strategy system integration
- ✅ Decision analysis integration
- ✅ Checkpoint save/load functionality
- ✅ Best model saving
- ✅ Evaluation mode testing
- ✅ Training summary generation

#### AdaptiveStrategySystem (`tests/core/test_adaptive_strategy_system_simplified.py`)
- ✅ **12 test methods** for strategy management
- ✅ Strategy type validation
- ✅ Force strategy functionality
- ✅ Statistics retrieval
- ✅ Performance history tracking
- ✅ Decision analyzer integration

#### EnhancedPyBoyPokemonCrystalEnv (`tests/core/test_enhanced_pyboy_env.py`)
- ✅ **8 test methods** for environment functionality
- ✅ Multi-modal observation space validation
- ✅ Action space configuration
- ✅ Reset and step functionality
- ✅ Action masking
- ✅ Reward calculation
- ✅ Episode simulation

## 🚀 Running Tests

### Quick Test Commands

```bash
# Run all hybrid system tests
python -m pytest tests/trainer/test_hybrid_llm_rl_trainer.py -v

# Run core component tests
python -m pytest tests/core/test_adaptive_strategy_system_simplified.py -v

# Run environment tests (with mocked dependencies)
python -m pytest tests/core/test_enhanced_pyboy_env.py -v

# Run all new tests
python -m pytest tests/trainer/test_hybrid_llm_rl_trainer.py tests/core/test_adaptive_strategy_system_simplified.py -v
```

### Test Categories

#### Unit Tests
```bash
# Individual component tests with mocked dependencies
python -m pytest tests/core/ -v -k "not integration"
python -m pytest tests/trainer/ -v -k "not integration"
```

#### Integration Tests
```bash
# Component interaction tests
python -m pytest tests/integration/ -v
```

#### Performance Tests
```bash
# Benchmark and performance validation
python -m pytest tests/performance/ -v
```

## 📋 Test Structure

### Test Naming Convention

- `test_[component]_[functionality]`: Basic functionality tests
- `test_[component]_integration`: Integration with other components
- `test_[component]_edge_cases`: Edge case and error handling
- `test_[component]_performance`: Performance and benchmark tests

### Test Categories by Component

#### Trainer Tests (`tests/trainer/`)
- **Initialization**: Basic setup and configuration
- **Training Loop**: Episode execution and curriculum learning
- **Checkpointing**: Save/load functionality
- **Evaluation**: Performance assessment
- **Integration**: Component interaction

#### Core Tests (`tests/core/`)
- **Strategy System**: Adaptive strategy selection
- **Environment**: Game environment functionality
- **Decision Analysis**: Pattern learning and history
- **Goal Planning**: Strategic goal management

#### Integration Tests (`tests/integration/`)
- **System Integration**: Full system interaction
- **End-to-End**: Complete training pipeline
- **Data Flow**: Information flow between components

## 🛠️ Writing New Tests

### Test Template

```python
"""
Tests for [Component Name].
"""

import unittest
import tempfile
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from your_module import YourComponent


class TestYourComponent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.component = YourComponent()
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test component initialization."""
        self.assertIsNotNone(self.component)
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.component.some_method()
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
```

### Mocking Guidelines

#### Mock External Dependencies
```python
@patch('your_module.ExternalDependency')
def test_with_mocked_dependency(self, mock_dependency):
    """Test with mocked external dependency."""
    # Configure mock
    mock_instance = Mock()
    mock_dependency.return_value = mock_instance
    mock_instance.method.return_value = "expected_result"
    
    # Run test
    result = your_component.use_dependency()
    
    # Verify
    self.assertEqual(result, "expected_result")
    mock_dependency.assert_called_once()
```

#### Mock Complex Objects
```python
def setUp(self):
    """Set up with complex mocked objects."""
    # Mock PyBoy instance
    self.mock_pyboy = Mock()
    self.mock_pyboy.screen.screen_ndarray.return_value = np.zeros((144, 160, 3))
    
    # Mock environment
    self.mock_env = Mock()
    self.mock_env.reset.return_value = (observation, info)
    self.mock_env.step.return_value = (next_obs, reward, done, truncated, info)
```

## 📈 Test Coverage Requirements

### Minimum Coverage Targets

- **Core Components**: 80% line coverage
- **Critical Paths**: 95% line coverage (training loop, decision making)
- **Error Handling**: 70% line coverage
- **Integration Points**: 90% line coverage

### Coverage Reporting

```bash
# Generate coverage report
python -m pytest --cov=core --cov=trainer tests/ --cov-report=html

# View coverage report
open htmlcov/index.html  # or equivalent on your system
```

## 🐛 Debugging Tests

### Common Test Failures

#### Import Errors
```python
# Fix: Ensure correct path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
```

#### Mock Configuration Issues
```python
# Fix: Proper mock setup and configuration
mock_object.configure_mock(attribute=value)
mock_object.method.return_value = expected_value
```

#### Resource Cleanup
```python
def tearDown(self):
    """Always clean up resources."""
    if hasattr(self, 'temp_dir'):
        shutil.rmtree(self.temp_dir)
    if hasattr(self, 'database_connection'):
        self.database_connection.close()
```

### Debug Mode Testing

```python
# Enable debug logging in tests
logging.basicConfig(level=logging.DEBUG)

# Add debug prints (remove before commit)
print(f"DEBUG: Variable value = {variable}")

# Use pdb for interactive debugging
import pdb; pdb.set_trace()
```

## 📊 Test Data Management

### Test Fixtures

```python
def setUp(self):
    """Create test fixtures."""
    # Create temporary directory
    self.temp_dir = tempfile.mkdtemp()
    
    # Create test database
    self.test_db = os.path.join(self.temp_dir, "test.db")
    
    # Create dummy ROM file
    self.rom_path = os.path.join(self.temp_dir, "test.gbc")
    with open(self.rom_path, 'wb') as f:
        f.write(b'\x00' * 1024)
```

### Mock Data Generation

```python
def generate_mock_game_state(self):
    """Generate realistic mock game state."""
    return {
        'player_x': random.randint(0, 255),
        'player_y': random.randint(0, 255),
        'player_hp': random.randint(1, 100),
        'max_hp': 100,
        'level': random.randint(1, 50),
        'badges': random.randint(0, 8)
    }
```

## ⚡ Performance Testing

### Benchmark Tests

```python
import time
import cProfile

def test_training_performance(self):
    """Test training performance benchmarks."""
    start_time = time.time()
    
    # Run training
    trainer.train(total_episodes=10, max_steps_per_episode=100)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Performance assertions
    self.assertLess(duration, 60.0)  # Should complete in under 1 minute
    self.assertGreater(trainer.steps_per_second, 10)  # Minimum speed
```

### Memory Usage Testing

```python
import psutil
import os

def test_memory_usage(self):
    """Test memory usage stays within bounds."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    trainer.train(total_episodes=100)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory should not increase by more than 500MB
    self.assertLess(memory_increase, 500 * 1024 * 1024)
```

## 🔧 Test Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Environment Variables for Testing

```bash
export PYTEST_TIMEOUT=300  # Test timeout in seconds
export TEST_DATABASE_URL="sqlite:///test.db"
export TEST_ROM_PATH="/tmp/test.gbc"
```

## 📚 Test Documentation Standards

### Test Docstrings

```python
def test_curriculum_advancement(self):
    """
    Test curriculum learning advancement based on performance.
    
    This test verifies that:
    1. Curriculum advances when reward thresholds are met
    2. Training statistics are updated correctly
    3. Strategy system adapts to new curriculum stage
    
    Expected behavior:
    - Episodes 1-10: Stage 0 (LLM_HEAVY)
    - High rewards trigger advancement to Stage 1
    - Strategy switches to more balanced approach
    """
```

### Test Case Organization

```python
class TestTrainerFunctionality(unittest.TestCase):
    """Test core trainer functionality."""
    
    def test_basic_training_loop(self):
        """Test basic training loop execution."""
        pass
    
    def test_curriculum_learning(self):
        """Test curriculum learning progression."""
        pass

class TestTrainerIntegration(unittest.TestCase):
    """Test trainer integration with other components."""
    
    def test_strategy_system_integration(self):
        """Test integration with adaptive strategy system."""
        pass
```

## 🎯 Test Quality Guidelines

### Test Quality Checklist

- ✅ **Clear test names** that describe what is being tested
- ✅ **Proper setup and teardown** for resource management
- ✅ **Meaningful assertions** that validate expected behavior
- ✅ **Error case coverage** for edge conditions
- ✅ **Mock usage** for external dependencies
- ✅ **Performance considerations** for long-running tests
- ✅ **Documentation** for complex test scenarios

### Common Patterns

#### Testing Exceptions
```python
def test_invalid_configuration_raises_error(self):
    """Test that invalid configuration raises appropriate error."""
    with self.assertRaises(ConfigurationError):
        trainer = HybridLLMRLTrainer(invalid_config=True)
```

#### Testing State Changes
```python
def test_strategy_switching_changes_state(self):
    """Test that strategy switching changes internal state."""
    initial_strategy = system.current_strategy
    system.force_strategy(StrategyType.LLM_HEAVY)
    self.assertNotEqual(system.current_strategy, initial_strategy)
```

## 📋 Test Maintenance

### Regular Test Maintenance Tasks

1. **Update mocks** when APIs change
2. **Review test coverage** monthly
3. **Clean up outdated tests** that no longer apply
4. **Update test data** to reflect current game state
5. **Performance regression** testing

### Test Refactoring

```python
# Extract common test setup
class BaseTrainerTest(unittest.TestCase):
    """Base class for trainer tests."""
    
    def setUp(self):
        """Common setup for trainer tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_env = self.create_mock_environment()
        self.trainer = self.create_test_trainer()
    
    def create_mock_environment(self):
        """Create standardized mock environment."""
        # Common mock setup
        pass
```

---

**Happy testing! 🧪✅**