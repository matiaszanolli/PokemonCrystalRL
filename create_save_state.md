# Creating Pokemon Crystal Save States for AI Training

## Quick Setup Guide

### 1. Create a Save State Manually

**Option A: Using mGBA GUI (Recommended)**
```bash
# Start mGBA with ROM
/usr/games/mgba-qt pokecrystal.gbc

# Play through initial setup:
# 1. Skip intro/title screens
# 2. Choose starter Pokemon
# 3. Complete first delivery quest
# 4. Get to clear overworld area

# Save state: Tools → Save state → Slot 1 (or press F1)
# This creates: pokecrystal.ss1
```

**Option B: Using mGBA CLI with debugger**
```bash
# Start with debugger
/usr/games/mgba-qt -g pokecrystal.gbc

# Use debugger commands to create save state
```

### 2. Recommended Save State Content

**Ideal Location**: New Bark Town or Route 29 (after getting starter)
- **Player**: Standing in open area (not in dialogue/menu)
- **Party**: 1 starter Pokemon (level 5-8)
- **Inventory**: Basic items (Potions, Pokeballs)
- **Progress**: Completed Professor Elm's first task
- **Status**: Clear screen, ready for movement

### 3. Save State File Locations

mGBA save states are typically saved as:
- `pokecrystal.ss1` (Slot 1)
- `pokecrystal.ss2` (Slot 2)
- etc.

### 4. Integration with AI Training

```python
# Use save state in environment
env = PokemonCrystalEnvMGBA(
    rom_path="./pokecrystal.gbc",
    save_state_path="./pokecrystal.ss1",  # Load from save state
    headless=True
)

obs, info = env.reset()  # Starts from save state
```

## Benefits of This Approach

✅ **Skip Complex Navigation**: No title screens, name entry, or intro sequences
✅ **Consistent Starting Point**: Same game state every reset
✅ **Focus on Core Mechanics**: Movement, battles, NPCs, exploration
✅ **Faster Testing**: Immediate gameplay testing
✅ **Real Game Data**: Actual Pokemon Crystal memory layout

## Creating the Save State

When you're ready, just:

1. Play the game manually to the desired point
2. Create a save state (F1 in mGBA)
3. Copy the `.ss1` file to the project directory
4. Use it in the AI environment

The AI will then start every episode from that exact point, allowing it to focus on the interesting parts of Pokemon gameplay without dealing with intro sequences!

Let me know when you have a save state ready, and I'll integrate it into the test system.
