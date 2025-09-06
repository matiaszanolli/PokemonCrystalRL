# Core Module Restructuring Plan

## Current Issues
- **Oversized modules**: `web_monitor.py` (1,236 lines) doing too many things
- **Duplicate classes**: Same functionality scattered across multiple files
- **Poor organization**: Related functionality separated, unrelated functionality combined

## Analysis Results

### Oversized Modules (>1000 lines)
- `web_monitor.py` (1,236) - Web UI, screen capture, API, WebSocket streaming

### Duplicate Classes Found
- `PyBoyGameState`: `state_machine.py` + `constants.py`
- `StateVariable`: `game_state_analyzer.py` + `state_variable_dictionary.py`  
- `GamePhase`: `game_state_analyzer.py` + `game_intelligence.py`
- `GameContext`: `semantic_context_system.py` + `game_intelligence.py`

### Memory-Related Redundancy
- `memory_reader.py` (276 lines) - Basic memory reading
- `memory_map.py` (271 lines) - Memory mapping
- `memory_map_new.py` (238 lines) - Newer mapping (redundant)

## Restructuring Plan

### Phase 1: Split Oversized Modules

#### Split `web_monitor.py` → `monitoring/` subpackage:
- `monitoring/web_server.py` - HTTP server and API endpoints
- `monitoring/screen_capture.py` - Screen capture and image processing
- `monitoring/websocket_streaming.py` - WebSocket real-time streaming
- `monitoring/dashboard_api.py` - Statistics and metrics API
- `monitoring/__init__.py` - Unified interface

### Phase 2: Remove Duplicates

#### State Management Consolidation:
- **Remove**: `PyBoyGameState` from `constants.py` (keep in `state_machine.py`)
- **Merge**: `StateVariable` classes into `state_variable_dictionary.py`
- **Consolidate**: `GamePhase` in `game_state_analyzer.py` (remove from `game_intelligence.py`)
- **Merge**: `GameContext` classes into `semantic_context_system.py`

#### Memory System Consolidation:
- **Keep**: `memory_reader.py` for core memory reading
- **Merge**: `memory_map.py` and `memory_map_new.py` → `memory_mapping.py`
- **Result**: Single source of truth for memory addresses

### Phase 3: Create Focused Subpackages

#### `state/` subpackage:
- Move: `state_machine.py` → `state/machine.py`
- Move: `state_variable_dictionary.py` → `state/variables.py`
- Move: `game_state_analyzer.py` → `state/analyzer.py`
- Create: `state/__init__.py` with unified exports

#### Memory consolidation:
- Rename: `memory_reader.py` → `memory/reader.py`
- Create: `memory/mapping.py` (merged from memory_map*.py)
- Create: `memory/__init__.py`

### Phase 4: Update Imports
- Update all imports to use new module structure
- Ensure backward compatibility where needed
- Update `core/__init__.py` with new exports

## Expected Benefits
- **Clarity**: Each file has single, clear responsibility
- **Maintainability**: Related code grouped together
- **Performance**: Reduced import overhead
- **Scalability**: Easier to extend functionality
- **Testing**: Focused test targets

## File Size Targets (After)
- No file >500 lines unless absolutely necessary
- Most files 100-300 lines
- Clear separation of concerns