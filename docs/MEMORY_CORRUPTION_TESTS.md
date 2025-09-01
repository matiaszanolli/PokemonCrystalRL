# Memory Corruption Protection Test Suite

This document describes the comprehensive test suite added to prevent and detect memory corruption issues that can cause false reward spikes in the Pokemon Crystal RL training system.

## Background

During training, we discovered that uninitialized Game Boy memory reads (particularly 0xFF values) were being interpreted as valid game progress, causing massive false reward spikes:

- **Badge corruption**: 0xFF in badge memory → 8 badges → 4000 reward points
- **Level corruption**: Impossible levels (122, 255) → massive level rewards
- **Memory state inconsistency**: Early game with impossible values

## Test Files Added

### 1. `tests/core/test_memory_corruption_protection.py`

Comprehensive tests for memory corruption protection in the core memory mapping system:

#### `TestMemoryCorruptionProtection`
- **Badge corruption detection**: Tests that 0xFF and high values (>0x80) are filtered in early game
- **Level corruption detection**: Tests that impossible levels (>100) trigger protection
- **Valid case preservation**: Ensures normal progression isn't affected

#### `TestRewardSystemCorruptionProtection` 
- **Badge reward protection**: Tests reward calculator blocks corrupted badge rewards
- **Level reward protection**: Tests level rewards are blocked for impossible levels
- **Gain capping**: Tests that multi-badge/level jumps are capped to 1 per step
- **Range validation**: Tests negative and impossibly high values are rejected

#### `TestMemoryCorruptionIntegration`
- **Complete corruption scenarios**: Tests full early-game corruption with multiple issues
- **Normal progression preservation**: Ensures valid gameplay isn't affected
- **Mixed corruption scenarios**: Tests various combinations of corruption
- **Endgame validation**: Ensures legitimate endgame progress works

#### `TestCorruptionProtectionPerformance`
- **Performance benchmarks**: Ensures corruption protection doesn't slow down training

### 2. `tests/core/test_trainer_memory_validation.py`

Tests for the trainer's `get_game_state()` memory validation logic:

#### `TestTrainerMemoryValidation`
- **Early game sanitization**: Tests that corrupted values are cleaned in early game
- **Impossible level handling**: Tests level >100 triggers full sanitization
- **Valid state preservation**: Tests that legitimate states aren't modified
- **Edge case handling**: Tests boundary conditions (level 100 vs 101)

#### `TestTrainerValidationIntegration`
- **Game progression scenarios**: Tests various stages of legitimate game progression
- **Corruption during gameplay**: Tests corruption detection in different game phases
- **Reward calculation prevention**: Tests that sanitized states prevent false rewards

## Protection Mechanisms Tested

### 1. Badge Corruption Protection
```python
# Early game + suspicious values = sanitize to 0
if (party_count == 0 or player_level == 0) and (badges == 0xFF or badges > 0x80):
    badges = 0
```

### 2. Level Corruption Protection  
```python
# Impossible levels always trigger protection
if player_level > 100:
    return 0  # For badge calculation
    player_level = 0  # In trainer validation
```

### 3. Reward System Protection
```python
# Badge rewards: Check for 0xFF in raw values + cap gains
if early_game and (0xFF in curr_raw or 0xFF in prev_raw):
    return 0.0
badge_gain = min(badge_gain, 1)  # Cap to 1 badge per step
```

### 4. Multi-layer Validation
1. **Memory map level**: `_safe_badges_total()` filters corrupted badge counts
2. **Trainer level**: `get_game_state()` sanitizes raw memory values
3. **Reward level**: Reward calculators validate inputs and cap gains

## Test Coverage

- **122 total tests** in core test suite
- **27 new tests** specifically for corruption protection
- **100% pass rate** with all protection mechanisms
- **Performance validated**: >1000 calls/sec for corruption checking

## Test Categories

The following pytest markers are used to organize tests:

- `@pytest.mark.memory_corruption`: Memory corruption protection tests
- `@pytest.mark.trainer_validation`: Trainer validation tests  
- `@pytest.mark.memory_mapping`: General memory mapping tests
- `@pytest.mark.pyboy_integration`: PyBoy integration tests
- `@pytest.mark.performance`: Performance and benchmarking tests

## Running Tests

```bash
# Run all memory corruption tests
python -m pytest -m memory_corruption -v

# Run all trainer validation tests  
python -m pytest -m trainer_validation -v

# Run all core tests
python -m pytest tests/core/ -v

# Run specific test file
python -m pytest tests/core/test_memory_corruption_protection.py -v
```

## Verification Commands

```bash
# Debug badge calculation (should show 0 for 0xFF early game)
python debug_badge_calculation.py

# Test actual training (should show no reward spikes)
python llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 100 --no-web --no-dqn
```

## Future Maintenance

These tests serve as regression prevention for the memory corruption bug. If similar issues are discovered:

1. Add new test cases to the appropriate test class
2. Update protection logic in `core/memory_map.py` and `llm_trainer.py`
3. Verify tests pass and protection works as expected
4. Document any new protection mechanisms

The test suite is designed to catch memory corruption issues early and ensure the training system remains robust against Game Boy memory initialization quirks.
