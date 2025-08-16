# Pokemon Crystal RL Agent - Test Suite Documentation

## 🧪 Overview

This comprehensive test suite validates all aspects of the enhanced Pokemon Crystal RL system, including state-aware LLM prompting, anti-stuck logic, web monitoring, and performance optimizations.

## 📋 Test Structure

### **Test Categories**
- **Unit Tests** - Individual component testing
- **Integration Tests** - Cross-system functionality
- **Performance Tests** - Speed and memory benchmarks
- **Real-World Scenarios** - Complete gameplay workflows

### **Enhanced Features Tested**
- ✅ State-aware LLM prompting with numeric key guidance
- ✅ Temperature-based decision making (0.8/0.6)
- ✅ Enhanced anti-stuck mechanisms
- ✅ Real-time web monitoring with OCR
- ✅ Multi-model LLM backend support
- ✅ Performance benchmarking (~2.3 actions/sec)

## 🚀 Quick Start

### **Run All Tests**
```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v --tb=short
```

### **Run Specific Test Categories**
```bash
# Enhanced LLM prompting tests
pytest -m enhanced_prompting

# Performance benchmarking
pytest -m benchmarking

# Anti-stuck logic tests
pytest -m anti_stuck

# Web monitoring tests
pytest -m web_monitoring

# Real-world integration tests
pytest -m integration
```

## 📖 Test File Guide

### **Core Test Files**

| File | Purpose | Key Features |
|------|---------|--------------|
| `test_enhanced_llm_prompting.py` | LLM prompting system | Numeric keys, temperature, state-aware |
| `test_anti_stuck_logic.py` | Anti-stuck mechanisms | Screen hashing, recovery patterns |
| `test_performance_benchmarks.py` | Performance testing | 2.3 actions/sec, LLM timing |
| `test_enhanced_web_monitoring.py` | Web dashboard | OCR display, real-time charts |
| `test_unified_trainer.py` | Main trainer system | Integration, error handling |
| `test_real_world_scenarios.py` | Complete workflows | SmolLM2 gameplay cycles |

### **Configuration Files**
- `pytest.ini` - Test configuration and markers
- `conftest.py` - Shared fixtures and utilities

## 🎯 Test Examples

### **Example 1: Enhanced LLM Prompting**
```python
# Test state-aware prompting with numeric key guidance
def test_numeric_key_guidance_in_prompts(trainer):
    """Verify prompts include numeric key guidance"""
    game_state = "dialogue"
    screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
    
    with patch.object(trainer, '_detect_game_state', return_value=game_state):
        prompt = trainer._build_llm_prompt(game_state, screenshot)
    
    # Verify numeric key guidance is present
    assert "5=A" in prompt
    assert "7=START" in prompt
    assert "dialogue" in prompt.lower()
```

### **Example 2: Temperature-Based Decision Making**
```python
# Test temperature settings for varied responses
def test_temperature_configuration_by_state(trainer):
    """Verify temperature varies by game state"""
    states_to_test = {
        "dialogue": 0.8,      # High temperature for variety
        "menu": 0.6,          # Medium temperature
        "title_screen": 0.5   # Lower temperature
    }
    
    for state, expected_temp in states_to_test.items():
        with patch.object(trainer, '_detect_game_state', return_value=state):
            trainer._get_llm_action()
        
        # Verify temperature was set correctly
        call_kwargs = mock_ollama.generate.call_args[1]
        actual_temp = call_kwargs.get('options', {}).get('temperature', 0.0)
        assert abs(actual_temp - expected_temp) < 0.1
```

### **Example 3: Anti-Stuck Logic**
```python
# Test screen hash-based stuck detection
def test_stuck_detection_threshold(trainer):
    """Test stuck detection with repeated screens"""
    trainer.consecutive_same_screens = 0
    test_hash = 12345
    
    with patch.object(trainer, '_get_screen_hash', return_value=test_hash):
        # Simulate same screen repeatedly
        for i in range(25):
            trainer._get_rule_based_action(i)
        
        # Should detect being stuck
        assert trainer.consecutive_same_screens >= 20
        assert trainer.stuck_counter > 0
```

### **Example 4: Performance Benchmarking**
```python
# Test target performance of 2.3 actions/second
def test_target_actions_per_second_benchmark(trainer):
    """Verify system achieves target performance"""
    start_time = time.time()
    action_count = 100
    
    for i in range(action_count):
        action = trainer._get_rule_based_action(i)
        trainer._execute_action(action)
    
    elapsed = time.time() - start_time
    actions_per_second = action_count / elapsed
    
    # Should achieve at least 2.0 actions/second
    assert actions_per_second >= 2.0
```

### **Example 5: Web Monitoring**
```python
# Test real-time screenshot capture and queuing
def test_screen_capture_queue_management(trainer):
    """Test screenshot capture for web monitoring"""
    test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
    
    with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
        # Fill queue partially
        for i in range(15):
            trainer._capture_and_queue_screen()
        
        # Queue should contain screens but respect bounds
        assert not trainer.screen_queue.empty()
        assert trainer.screen_queue.qsize() <= 30
```

## 🔧 Writing New Tests

### **Test Structure Template**
```python
@pytest.mark.your_marker
@pytest.mark.unit  # or integration, performance
class TestYourFeature:
    """Test description"""
    
    @pytest.fixture
    def your_fixture(self):
        """Setup for tests"""
        return create_test_object()
    
    def test_specific_functionality(self, your_fixture):
        """Test a specific aspect"""
        # Arrange
        setup_test_conditions()
        
        # Act
        result = your_fixture.perform_action()
        
        # Assert
        assert result == expected_value
        assert condition_is_true
```

### **Common Patterns**

#### **Mocking PyBoy**
```python
@patch('scripts.pokemon_trainer.PyBoy')
@patch('scripts.pokemon_trainer.PYBOY_AVAILABLE', True)
def test_with_pyboy(self, mock_pyboy_class):
    mock_pyboy_instance = Mock()
    mock_pyboy_instance.frame_count = 1000
    mock_pyboy_class.return_value = mock_pyboy_instance
    
    trainer = UnifiedPokemonTrainer(config)
    # Test trainer functionality
```

#### **Mocking LLM Responses**
```python
with patch('scripts.pokemon_trainer.ollama') as mock_ollama:
    mock_ollama.generate.return_value = {'response': '5'}
    mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
    
    action = trainer._get_llm_action()
    assert action == 5
```

#### **Testing Performance**
```python
start_time = time.time()
# Perform operations
elapsed = time.time() - start_time

# Assert performance targets
assert elapsed < max_time_allowed
assert operations_per_second >= target_rate
```

## 📊 Test Markers Reference

| Marker | Purpose | Usage |
|--------|---------|-------|
| `enhanced_prompting` | LLM prompting features | `pytest -m enhanced_prompting` |
| `temperature` | Temperature-based decisions | `pytest -m temperature` |
| `anti_stuck` | Anti-stuck mechanisms | `pytest -m anti_stuck` |
| `state_detection` | Game state detection | `pytest -m state_detection` |
| `web_monitoring` | Web dashboard features | `pytest -m web_monitoring` |
| `multi_model` | Multiple LLM backends | `pytest -m multi_model` |
| `benchmarking` | Performance benchmarks | `pytest -m benchmarking` |
| `integration` | Cross-system tests | `pytest -m integration` |
| `slow` | Long-running tests | `pytest -m "not slow"` to skip |

## 🎮 Real-World Test Scenarios

### **Complete Gameplay Cycle**
```python
def test_new_game_introduction_sequence(trainer):
    """Test complete new game flow"""
    # Setup screen sequence
    screens = ["title_screen", "intro_text", "dialogue"]
    
    with patch_screen_sequence(screens):
        with patch_llm_responses():
            # Execute complete introduction
            states_seen = set()
            for step in range(20):
                state = trainer._detect_game_state(screenshot)
                states_seen.add(state)
                
                action = get_appropriate_action(state, step)
                trainer._execute_action(action)
            
            # Verify progression
            assert "title_screen" in states_seen
            assert "dialogue" in states_seen
```

### **Extended Play Session**
```python
def test_extended_play_stability(trainer):
    """Test stability during extended play"""
    start_time = time.time()
    
    for step in range(200):
        # Mixed LLM and rule-based actions
        if step % trainer.config.llm_interval == 0:
            action = trainer._get_llm_action()
        else:
            action = trainer._get_rule_based_action(step)
        
        trainer._execute_action(action)
        
        # Monitor performance
        if step % 20 == 0:
            trainer._update_stats()
    
    # Verify stability
    assert trainer.stats['actions_per_second'] >= target_rate
    assert no_errors_occurred()
```

## 🐛 Debugging Tests

### **Common Issues and Solutions**

#### **Test Timeouts**
```python
# Use shorter test runs for unit tests
config = TrainingConfig(max_actions=10)

# Mock time-consuming operations
with patch('time.sleep'):
    run_test()
```

#### **Mock Setup**
```python
# Ensure proper mock setup
@pytest.fixture
def trainer(self):
    with patch('scripts.pokemon_trainer.PyBoy'):
        trainer = UnifiedPokemonTrainer(config)
        # Additional setup
        return trainer
```

#### **Assertion Failures**
```python
# Add descriptive error messages
assert result == expected, f"Expected {expected}, got {result}"

# Use tolerances for floating point comparisons
assert abs(actual - expected) < 0.1, f"Values too different: {actual} vs {expected}"
```

### **Debugging Commands**
```bash
# Run single test with full output
pytest tests/test_file.py::test_name -v -s

# Run with Python debugger
pytest tests/test_file.py::test_name --pdb

# Run with coverage to find untested code
pytest --cov=. --cov-report=term-missing

# Run tests in parallel (faster)
pytest -n auto
```

## 📈 Performance Testing

### **Benchmarking Best Practices**
```python
def test_performance_benchmark():
    """Template for performance tests"""
    # Warm up
    for _ in range(10):
        perform_operation()
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(1000):
        perform_operation()
    elapsed = time.time() - start_time
    
    # Assert performance
    ops_per_second = 1000 / elapsed
    assert ops_per_second >= target_rate
```

### **Memory Testing**
```python
import psutil
import os

def test_memory_usage():
    """Test memory usage patterns"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform operations
    run_extended_operations()
    
    # Check memory usage
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory should be bounded
    assert memory_increase < max_allowed_mb * 1024 * 1024
```

## 🔄 Continuous Integration

### **CI Configuration Example**
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### **Test Categories for CI**
```bash
# Fast tests (run on every commit)
pytest -m "unit and not slow"

# Integration tests (run on PR)
pytest -m integration

# Performance tests (run nightly)
pytest -m benchmarking

# Complete test suite (run on release)
pytest --cov=. --cov-report=html
```

## 📚 Additional Resources

### **Testing Documentation**
- [PyTest Documentation](https://docs.pytest.org/)
- [Python Mock Library](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)

### **Project-Specific Guides**
- [Main README](../README.md) - Project overview
- [Contributing Guide](../CONTRIBUTING.md) - Development guidelines
- [Performance Guide](../docs/guides/SPEED_OPTIMIZATION_REPORT.md) - Optimization details

### **Test Data and Fixtures**
- Sample ROM data in `conftest.py`
- Mock PyBoy screens and responses
- Pre-configured trainer instances
- Performance baseline data

---

## 🤝 Contributing to Tests

When adding new features, please:

1. **Write tests first** (TDD approach)
2. **Use appropriate markers** for test categorization
3. **Include performance tests** for new functionality
4. **Add integration tests** for cross-system features
5. **Update documentation** with examples

### **Test Checklist**
- [ ] Unit tests for new components
- [ ] Integration tests for workflows
- [ ] Performance tests for bottlenecks
- [ ] Error handling tests
- [ ] Documentation updates
- [ ] CI pipeline compatibility

**Happy Testing! 🎮🧪**
