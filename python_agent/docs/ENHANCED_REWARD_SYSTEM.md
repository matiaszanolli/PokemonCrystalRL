# Enhanced Reward System for Menu and Stuck Situations

## Overview

This document describes the enhanced reward calculation system implemented to better handle menu navigation and stuck situations in the Pokemon Crystal RL training environment.

## Problem Analysis

The original reward system had several limitations when dealing with menu navigation and stuck situations:

1. **No stuck penalty**: Agent received small survival reward (+0.1) even when stuck
2. **No menu completion rewards**: No bonus for successfully navigating menus
3. **Insufficient anti-stuck motivation**: Time penalty (-0.001) was too small to discourage getting stuck
4. **No progress tracking in menus**: No rewards for advancing through menu sequences
5. **Limited action variety encouragement**: No incentive for diverse action patterns

## Enhanced Reward Features

### 1. Stuck Detection Penalties

**Escalating Stuck Penalty**:
- Triggers when `consecutive_same_screens > 10`
- Escalating penalty: `-0.1 * (consecutive_same_screens - 10)`
- Severe penalty when `consecutive_same_screens > 25`: additional `-2.0` penalty

**Implementation**:
```python
consecutive_same_screens = current_state.get('consecutive_same_screens', 0)
if consecutive_same_screens > 10:
    stuck_penalty = -0.1 * (consecutive_same_screens - 10)
    reward += stuck_penalty
    
    if consecutive_same_screens > 25:
        reward -= 2.0  # Heavy penalty for being really stuck
```

### 2. Menu Progress Rewards

**Menu Entry Reward**: `+1.0` for successfully entering a menu
**Menu Exit Reward**: `+2.0` for successfully exiting a menu (higher reward as it implies progress)

**Implementation**:
```python
curr_game_state = current_state.get('game_state', 'unknown')
prev_game_state = previous_state.get('game_state', 'unknown')

if curr_game_state == 'menu' and prev_game_state != 'menu':
    reward += 1.0  # Menu entry bonus
elif curr_game_state != 'menu' and prev_game_state == 'menu':
    reward += 2.0  # Menu exit bonus (implies progress)
```

### 3. State Transition Bonuses

Rewards for meaningful game state transitions:
- `title_screen → new_game_menu`: `+5.0`
- `title_screen → intro_sequence`: `+3.0`
- `intro_sequence → new_game_menu`: `+3.0`
- `new_game_menu → overworld`: `+10.0` (major milestone)
- `dialogue → overworld`: `+1.0`
- `menu → overworld`: `+1.5`
- `overworld → battle`: `+2.0`
- `battle → overworld`: `+3.0` (battle completion)
- `loading → overworld`: `+1.0`
- `unknown → overworld`: `+2.0` (recovered from unknown state)

### 4. Action Diversity Rewards

**Diversity Bonus**: `+0.05` for using at least 3 different actions in the last 5 actions
**Repetitive Penalty**: `-0.02` for using the same action repeatedly

**Implementation**:
```python
recent_actions = current_state.get('recent_actions', [])
if len(recent_actions) >= 5:
    unique_actions = len(set(recent_actions[-5:]))
    if unique_actions >= 3:
        reward += 0.05  # Diversity bonus
    elif unique_actions == 1:
        reward -= 0.02  # Repetitive penalty
```

### 5. Progress Momentum Rewards

**Compound Progress Bonus**: Additional `+0.1 * progress_count` when multiple types of progress occur in one step
- Progress types: level up, experience gain, money gain, map change, position movement, party growth

**Implementation**:
```python
progress_indicators = [
    curr_level > prev_level,
    curr_exp > prev_exp,
    curr_money > prev_money,
    curr_map != prev_map,
    distance_moved > 0,
    curr_party > prev_party
]

progress_count = sum(progress_indicators)
if progress_count >= 2:
    reward += 0.1 * progress_count  # Compound progress bonus
```

### 6. Enhanced Time Penalty

Increased time penalty from `-0.001` to `-0.002` to better encourage efficiency without being too punitive.

## Environment Integration

### Enhanced State Tracking

The PyBoy environment now tracks additional state information:

```python
# Enhanced state tracking for rewards
self.consecutive_same_screens = 0
self.last_screen_hash = None
self.recent_actions = collections.deque(maxlen=10)
self.game_state_history = collections.deque(maxlen=5)
```

### Screen Hash-Based Stuck Detection

```python
def _get_screen_hash(self, screenshot: np.ndarray) -> int:
    """Get a hash of the screen for stuck detection"""
    # Uses aggressive sampling for performance (every 8th pixel)
    # Combines mean, std, and spatial features (top-left, bottom-right quadrants)
    return hash((mean_val, std_val, tl, br))
```

### Game State Detection

Basic game state detection for reward calculation:
- `loading`, `intro_sequence`, `title_screen`
- `battle`, `menu`, `overworld`, `dialogue`
- Uses brightness and color variance analysis

## Testing

Comprehensive test suite with 19 tests covering:

1. **Stuck Detection Penalties** (3 tests)
   - Normal gameplay (no penalties)
   - Escalating penalties
   - Severe stuck penalties

2. **Menu Progress Rewards** (2 tests)
   - Menu entry rewards
   - Menu exit rewards

3. **State Transition Bonuses** (3 tests)
   - Title to new game transition
   - New game to overworld transition
   - Battle completion rewards

4. **Action Diversity Rewards** (3 tests)
   - Diversity bonus
   - Repetitive action penalty
   - Insufficient actions handling

5. **Progress Momentum Rewards** (2 tests)
   - Compound progress bonus
   - Single progress (no compound bonus)

6. **Integration Tests** (2 tests)
   - Environment state tracking
   - Stuck detection integration

7. **Edge Cases** (4 tests)
   - Missing state fields
   - Empty recent actions
   - Extreme stuck values
   - Unknown state transitions

## Results

All 19 tests pass successfully, validating:
- ✅ Proper stuck detection and penalties
- ✅ Menu navigation rewards
- ✅ State transition bonuses
- ✅ Action diversity tracking
- ✅ Progress momentum bonuses
- ✅ Environment integration
- ✅ Edge case handling

## Impact

The enhanced reward system provides:

1. **Better Stuck Handling**: Escalating penalties discourage getting stuck while providing clear signals for recovery
2. **Menu Navigation Incentives**: Rewards successful menu interaction and progression
3. **State Progress Recognition**: Bonuses for meaningful game state advances
4. **Action Variety Encouragement**: Incentives for diverse gameplay patterns
5. **Compound Progress Recognition**: Additional rewards for achieving multiple progress types simultaneously

This should significantly improve the agent's ability to navigate menus effectively and avoid getting stuck in repetitive patterns while maintaining the core game progression rewards.

## Files Modified

- `utils/utils.py`: Enhanced reward calculation function
- `core/pyboy_env.py`: Added enhanced state tracking and integration
- `tests/test_enhanced_reward_system.py`: Comprehensive test suite (new file)
- `docs/ENHANCED_REWARD_SYSTEM.md`: This documentation (new file)

## Usage

The enhanced reward system is automatically integrated into the environment. No changes to training scripts are required - the new reward calculations will be used automatically when the environment processes steps.
