# 🤝 Contributing Guide

Thank you for considering contributing to Pokemon Crystal RL! This document provides guidelines and best practices for contributing to the project.

## 📋 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and constructive
- Focus on what is best for the community

## 🚀 Getting Started

1. **Fork the Repository**
   - Fork the repository on GitHub
   - Clone your fork locally
   ```bash
   git clone https://github.com/yourusername/pokemon_crystal_rl.git
   cd pokemon_crystal_rl
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-fix
   ```

## 💻 Development Process

### 1. Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some additions:

```python
"""Module docstring with purpose."""

import standard_library
import third_party
from local_module import something

# Constants in UPPER_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

class MyClass:
    """Class docstring with description."""

    def __init__(self, param: str):
        """Initialize with parameters."""
        self.param = param

    def my_method(self) -> None:
        """Method docstring."""
        pass

def my_function(param1: int, param2: str = "default") -> bool:
    """Function docstring with parameters and return."""
    return True
```

### 2. Type Hints

Use type hints consistently:

```python
from typing import Dict, List, Optional, Tuple, Any

def process_data(
    data: Dict[str, Any],
    config: Optional[Dict[str, str]] = None
) -> Tuple[float, List[str]]:
    """Process data with proper type hints."""
    result = 0.0
    messages = []
    return result, messages
```

### 3. Documentation

Document code thoroughly:

```python
def calculate_reward(
    current_state: Dict[str, Any],
    previous_state: Dict[str, Any]
) -> float:
    """Calculate reward based on state changes.
    
    This function compares the current and previous states to determine
    the reward value for the agent's actions.
    
    Args:
        current_state: Current game state dictionary containing:
            - health (int): Current health points
            - exp (int): Current experience points
            - position (Tuple[int, int]): Current map position
        previous_state: Previous game state with same structure
        
    Returns:
        float: Calculated reward value
        
    Raises:
        ValueError: If state dictionaries are missing required keys
        TypeError: If state values are of incorrect type
    """
    # Implementation
    pass
```

### 4. Testing

Write comprehensive tests:

```python
import pytest
from pokemon_crystal_rl.utils import calculate_reward

def test_calculate_reward():
    """Test reward calculation with different states."""
    # Arrange
    current_state = {
        "health": 100,
        "exp": 1000,
        "position": (10, 20)
    }
    previous_state = {
        "health": 90,
        "exp": 900,
        "position": (10, 19)
    }
    
    # Act
    reward = calculate_reward(current_state, previous_state)
    
    # Assert
    assert reward > 0
    assert isinstance(reward, float)

@pytest.mark.parametrize("health,exp,expected", [
    (100, 1000, 10.0),
    (50, 500, 5.0),
    (0, 0, 0.0)
])
def test_reward_calculation_parametrized(health, exp, expected):
    """Test reward calculation with various inputs."""
    state = {"health": health, "exp": exp}
    reward = calculate_reward(state, state)
    assert reward == pytest.approx(expected, 0.1)
```

## 📝 Pull Request Process

1. **Update Documentation**
   - Add/update docstrings
   - Update README if needed
   - Add new documentation files
   - Update type hints

2. **Add Tests**
   - Write unit tests
   - Add integration tests
   - Update test documentation
   - Ensure CI passes

3. **Create Pull Request**
   - Use clear, descriptive title
   - Fill out PR template
   - Reference related issues
   - Request review from maintainers

4. **PR Template**
   ```markdown
   ## Description
   Clear description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   Describe testing done:
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows style guide
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] All tests passing
   ```

## 🐛 Bug Reports

When filing a bug report, please include:

1. **Description**
   - Clear, concise description
   - Expected vs actual behavior
   - Steps to reproduce

2. **Environment**
   - Python version
   - OS details
   - Dependencies versions
   - Hardware info if relevant

3. **Example Code**
   ```python
   # Minimal code to reproduce
   from pokemon_crystal_rl import Trainer
   trainer = Trainer()
   result = trainer.problematic_method()  # Error occurs here
   ```

4. **Logs/Output**
   ```
   Traceback (most recent call last):
     File "script.py", line 2, in <module>
       result = trainer.problematic_method()
   RuntimeError: Something went wrong
   ```

## 🚀 Feature Requests

When proposing new features:

1. **Description**
   - Clear feature description
   - Use cases and benefits
   - Implementation ideas

2. **Examples**
   ```python
   # Example usage of proposed feature
   trainer = Trainer()
   trainer.new_feature(param1, param2)
   ```

3. **Considerations**
   - Performance impact
   - Compatibility concerns
   - Testing requirements

## 📚 Documentation

When updating documentation:

1. **Code Documentation**
   - Clear docstrings
   - Type hints
   - Usage examples

2. **Project Documentation**
   - README updates
   - API documentation
   - Usage guides

3. **Examples**
   - Practical examples
   - Common use cases
   - Best practices

## 🔍 Code Review

When reviewing code:

1. **Code Quality**
   - Style guide compliance
   - Type hint usage
   - Documentation quality

2. **Functionality**
   - Test coverage
   - Edge cases
   - Error handling

3. **Performance**
   - Resource usage
   - Optimization opportunities
   - Bottlenecks

## 🎯 Project Goals

Keep these goals in mind when contributing:

1. **Maintainability**
   - Clean, readable code
   - Good documentation
   - Comprehensive tests

2. **Performance**
   - Efficient algorithms
   - Resource management
   - Optimization

3. **Usability**
   - Clear API design
   - Good error messages
   - Helpful documentation

## 📊 Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints Guide](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html)
- [pytest Documentation](https://docs.pytest.org/)
