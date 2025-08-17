# 🛠️ Development Guide

This guide covers development practices, code organization, and contribution guidelines for the Pokemon Crystal RL project.

## 📁 Project Structure

```
pokemon_crystal_rl/
├── pokemon_crystal_rl/         # Main package
│   ├── agents/                # Agent implementations
│   │   ├── base.py           # Base agent class
│   │   ├── llm_agent.py      # LLM-powered agent
│   │   └── rule_agent.py     # Rule-based agent
│   ├── core/                 # Core game integration
│   │   ├── env.py           # Game environment
│   │   ├── memory_map.py    # Game memory mapping
│   │   └── game_states.py   # Game state handling
│   ├── monitoring/           # Web monitoring
│   │   ├── unified_monitor.py  # Web interface
│   │   └── metrics.py       # Performance metrics
│   ├── trainer/             # Training system
│   │   ├── trainer.py       # Main trainer
│   │   └── strategies.py    # Training strategies
│   ├── utils/              # Shared utilities
│   │   ├── rewards.py      # Reward calculation
│   │   └── state.py        # State preprocessing
│   └── vision/             # Computer vision
│       ├── processor.py    # Vision processing
│       └── font.py        # Font recognition
├── docs/                   # Documentation
├── tests/                  # Test suite
└── tools/                  # Helper scripts
```

## 🔧 Setup Development Environment

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/pokemon_crystal_rl.git
   cd pokemon_crystal_rl
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 🚀 Development Workflow

### Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Imports
import numpy as np
import torch
from typing import Dict, List, Optional

# Constants
MAX_EPISODES = 1000
DEFAULT_BATCH_SIZE = 32

# Classes
class MyClass:
    """Class docstring with description."""
    
    def __init__(self, param: int):
        """Initialize with parameters."""
        self.param = param
    
    def my_method(self) -> None:
        """Method docstring."""
        pass

# Functions
def process_data(data: np.ndarray) -> Dict[str, Any]:
    """Process data with docstring."""
    return {"result": data}
```

### Type Hints

Use type hints consistently:

```python
from typing import Dict, List, Optional, Tuple, Any

def process_game_state(
    state: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool]:
    """Process game state with type hints."""
    reward = calculate_reward(state, previous_state)
    done = check_done_condition(state)
    return reward, done
```

### Documentation

Document code using docstrings:

```python
def calculate_reward(
    current_state: Dict[str, Any],
    previous_state: Dict[str, Any]
) -> float:
    """Calculate reward based on state changes.
    
    Args:
        current_state: Current game state dictionary
        previous_state: Previous game state dictionary
        
    Returns:
        float: Calculated reward value
        
    Raises:
        ValueError: If state dictionaries are invalid
    """
    # Implementation
    pass
```

### Error Handling

Use proper error handling:

```python
class GameError(Exception):
    """Base class for game-related errors."""
    pass

def process_action(action: int) -> None:
    try:
        validate_action(action)
        execute_action(action)
    except ValueError as e:
        raise GameError(f"Invalid action: {e}")
    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        raise
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from pokemon_crystal_rl.utils import calculate_reward

def test_calculate_reward():
    # Arrange
    current_state = {"health": 100, "exp": 1000}
    previous_state = {"health": 90, "exp": 900}
    
    # Act
    reward = calculate_reward(current_state, previous_state)
    
    # Assert
    assert reward > 0
    assert isinstance(reward, float)
```

### Integration Tests

```python
import pytest
from pokemon_crystal_rl.trainer import UnifiedTrainer

@pytest.fixture
def trainer():
    return UnifiedTrainer(config=get_test_config())

def test_training_episode(trainer):
    # Run episode
    stats = trainer.run_episode()
    
    # Verify results
    assert stats["total_reward"] > 0
    assert 0 <= stats["episode_length"] <= 1000
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trainer.py

# Run with coverage
pytest --cov=pokemon_crystal_rl tests/
```

## 📝 Pull Request Process

1. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-fix
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: fix bug in trainer"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/my-feature
   ```

5. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   Describe testing done

   ## Screenshots (if applicable)
   Add screenshots

   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] All tests passing
   ```

## 📚 Additional Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints Guide](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Git Workflow](https://guides.github.com/introduction/flow/)
