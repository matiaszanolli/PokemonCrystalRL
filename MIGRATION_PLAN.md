# Web Dashboard Migration Plan

This document outlines the step-by-step plan to migrate from the current fragmented web dashboard implementations to the new unified system.

## 🎯 Migration Overview

### Current State (Before Migration)
- **4 Different Dashboards**: Multiple HTML templates with inconsistent features
- **2 Web Server Systems**: `core/web_monitor` and `monitoring/web` with overlapping functionality
- **Inconsistent APIs**: Different endpoints, data formats, and response structures
- **Fragmented Codebase**: Duplicate functionality across multiple directories

### Target State (After Migration)
- **1 Unified Dashboard**: Single, feature-complete web interface
- **1 Web Server System**: Clean, documented, maintainable server implementation
- **Consistent API**: Standardized endpoints with proper data models
- **Clean Architecture**: Well-organized code with comprehensive documentation

## 📋 Migration Phases

### Phase 1: Testing & Validation ✅ COMPLETED
- [x] Create unified web dashboard system (`web_dashboard/`)
- [x] Implement comprehensive API with proper data models
- [x] Build modern, responsive frontend with all features
- [x] Create integration test script for validation
- [x] Write comprehensive documentation

### Phase 2: Integration (NEXT STEPS)
- [ ] Update `UnifiedPokemonTrainer` to use new web dashboard
- [ ] Test with actual training sessions
- [ ] Validate all features work correctly
- [ ] Performance testing and optimization

### Phase 3: Deprecation & Cleanup
- [ ] Mark legacy systems as deprecated
- [ ] Remove obsolete code and tests
- [ ] Update all documentation references
- [ ] Clean up import statements

### Phase 4: Final Validation
- [ ] Comprehensive testing of all training modes
- [ ] Update CLAUDE.md with new instructions
- [ ] Final cleanup and optimization

## 🔧 Implementation Steps

### Step 1: Update Unified Trainer Integration

**File**: `/training/unified_pokemon_trainer.py`

Add web dashboard integration:

```python
# Add to imports
from web_dashboard import create_web_server

class UnifiedPokemonTrainer:
    def __init__(self, ...):
        # Existing initialization...

        # Initialize web dashboard if enabled
        self.web_server = None
        if web_enabled:
            self.web_server = create_web_server(
                trainer=self,
                host='localhost',
                http_port=8080,
                ws_port=8081
            )

    def start_training(self, ...):
        # Start web server if enabled
        if self.web_server:
            self.web_server.start()
            logger.info(f"🌐 Web dashboard: http://localhost:8080")

        # Existing training logic...

    def stop_training(self):
        # Stop web server
        if self.web_server:
            self.web_server.stop()

        # Existing cleanup...
```

### Step 2: Update Main Entry Point

**File**: `/main.py`

Replace web monitor initialization:

```python
# Remove old imports
# from core.web_monitor.monitor import WebMonitor

# Add new import
from web_dashboard import create_web_server

# In main function, replace:
# Old web monitor code
# if args.enable_web:
#     web_monitor = WebMonitor(trainer)
#     web_monitor.start()

# New unified dashboard
if args.enable_web:
    web_server = create_web_server(trainer)
    web_server.start()
    logger.info(f"🌐 Web dashboard: http://localhost:8080")
```

### Step 3: Test Integration

Run integration test:

```bash
# Test new unified dashboard
cd /mnt/data/src/pokemon_crystal_rl
python web_dashboard/integration_test.py

# Test with actual training
python main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 500 --enable-web --llm-interval 20
```

Verify:
- [ ] Dashboard loads at http://localhost:8080
- [ ] All sections show real data
- [ ] Real-time updates work
- [ ] Memory debug synchronization works
- [ ] LLM decisions display correctly
- [ ] No errors in console/logs

### Step 4: Deprecate Legacy Systems

Mark legacy systems as deprecated:

**File**: `/core/web_monitor/__init__.py`
```python
import warnings
warnings.warn(
    "core.web_monitor is deprecated. Use web_dashboard instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**File**: `/monitoring/web/__init__.py`
```python
import warnings
warnings.warn(
    "monitoring.web is deprecated. Use web_dashboard instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Step 5: Update Documentation

**File**: `/CLAUDE.md` - Update web monitoring section:

```markdown
### Web Monitoring
- **Unified Dashboard**: Use `--enable-web` flag to start web monitoring
- **Dashboard URL**: http://localhost:8080 (web interface)
- **WebSocket URL**: ws://localhost:8081 (real-time updates)
- **Features**: Training stats, game state, memory debug, LLM decisions
- **Documentation**: See `web_dashboard/README.md` for full documentation

### Deprecated Systems
⚠️ **Legacy web systems are deprecated**:
- `core/web_monitor/` - Use `web_dashboard` instead
- `monitoring/web/` - Use `web_dashboard` instead
- Multiple dashboard templates - Use unified dashboard at `/`
```

## 🗑️ Cleanup Plan

### Files to Remove (After Migration Complete)

#### Legacy Web Monitor System
```
core/web_monitor/
├── __init__.py           # REMOVE
├── compat.py            # REMOVE
├── http_handler.py      # REMOVE
├── monitor.py           # REMOVE
└── screen_capture.py    # REMOVE
```

#### Monitoring Web System
```
monitoring/web/
├── api/                 # REMOVE (functionality moved to web_dashboard/api/)
├── static/              # REMOVE
├── templates/           # REMOVE
├── events.py           # REMOVE
├── http_handler.py     # REMOVE
├── managers.py         # REMOVE
├── run_server.py       # REMOVE
├── server.py           # REMOVE
└── services/           # REMOVE
```

#### Legacy Dashboard Templates
```
static/
├── index.html          # REMOVE
└── templates/
    └── dashboard.html  # REMOVE

monitoring/
├── static/
│   └── index.html      # REMOVE
└── web/templates/
    └── dashboard.html  # REMOVE
```

#### Legacy Tests
```
tests/monitoring/
├── mock_web_server.py     # REMOVE
├── test_web_dashboard.py  # REMOVE
└── test_web_integration.py # REMOVE

tests/trainer/
├── mock_web_server.py     # REMOVE
└── test_web_integration.py # REMOVE
```

### Files to Update

#### Remove Legacy Imports
- `/training/unified_pokemon_trainer.py` - Remove `core.web_monitor` imports
- `/main.py` - Replace web monitor with unified dashboard
- `/trainer/compat/web_monitor.py` - Mark as deprecated
- All test files - Update web-related tests

#### Update Configuration
- `/config/constants.py` - Remove old web server constants if any
- `/requirements.txt` - Ensure websockets and pillow dependencies

## 📊 Migration Validation Checklist

### ✅ Functionality Testing
- [ ] **Training Statistics**: Actions, rewards, LLM calls display correctly
- [ ] **Game State**: Map, position, money, badges update in real-time
- [ ] **Memory Debug**: Live memory addresses show correct values
- [ ] **LLM Decisions**: Recent decisions display with reasoning
- [ ] **System Status**: Training status and connections shown
- [ ] **Screen Capture**: Live game screen updates via WebSocket

### ✅ API Testing
- [ ] **GET /api/dashboard**: Returns complete dashboard data
- [ ] **GET /api/game_state**: Returns current game state
- [ ] **GET /api/training_stats**: Returns training metrics
- [ ] **GET /api/memory_debug**: Returns memory debug data
- [ ] **GET /api/llm_decisions**: Returns recent LLM decisions
- [ ] **GET /api/system_status**: Returns system status
- [ ] **GET /health**: Returns server health status

### ✅ Real-time Testing
- [ ] **WebSocket Connection**: Establishes successfully on port 8081
- [ ] **Screen Updates**: Game screen updates in real-time
- [ ] **Stats Updates**: Training stats update automatically
- [ ] **Connection Recovery**: Reconnects after temporary disconnection
- [ ] **Error Handling**: Shows appropriate error messages

### ✅ Cross-browser Testing
- [ ] **Chrome**: All features work correctly
- [ ] **Firefox**: All features work correctly
- [ ] **Safari**: All features work correctly
- [ ] **Edge**: All features work correctly
- [ ] **Mobile**: Responsive design works on mobile devices

### ✅ Performance Testing
- [ ] **Load Time**: Dashboard loads quickly (<2 seconds)
- [ ] **Memory Usage**: Reasonable memory consumption
- [ ] **CPU Usage**: Low CPU overhead during training
- [ ] **Network Usage**: Efficient data transfer
- [ ] **Update Rate**: Smooth real-time updates

## 🚨 Risk Mitigation

### Backup Strategy
1. **Create backup branch** before starting migration
2. **Tag current working state** for easy rollback
3. **Keep legacy systems** until migration is fully validated
4. **Gradual rollout** - test with one trainer mode at a time

### Rollback Plan
If issues arise during migration:

1. **Immediate Rollback**:
   ```bash
   git checkout backup-branch
   # OR revert specific commits
   git revert <commit-hash>
   ```

2. **Partial Rollback**: Keep new system but restore old imports temporarily
3. **Debug Mode**: Run both systems in parallel for comparison

### Testing Strategy
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test with mock trainer (existing script)
3. **End-to-End Tests**: Test with actual training sessions
4. **Load Tests**: Test with long training sessions
5. **Browser Tests**: Test across different browsers and devices

## 📅 Timeline

### Week 1: Integration & Testing
- **Day 1-2**: Update trainer integration code
- **Day 3-4**: Test with actual training sessions
- **Day 5**: Performance testing and bug fixes

### Week 2: Validation & Documentation
- **Day 1-2**: Comprehensive feature testing
- **Day 3-4**: Update documentation and migration guides
- **Day 5**: Final validation and approval

### Week 3: Cleanup & Finalization
- **Day 1-2**: Mark legacy systems as deprecated
- **Day 3-4**: Remove obsolete code and tests
- **Day 5**: Final cleanup and optimization

## 🎉 Success Criteria

Migration is considered successful when:

1. **✅ Feature Parity**: All existing dashboard features work in new system
2. **✅ Performance**: No significant performance degradation
3. **✅ Reliability**: Stable operation during extended training sessions
4. **✅ Usability**: Improved user experience and interface
5. **✅ Maintainability**: Clean, documented, testable codebase
6. **✅ Documentation**: Complete documentation for users and developers

## 🔗 Dependencies

### Required Packages
```bash
pip install websockets pillow
```

### Optional Packages (for testing)
```bash
pip install requests  # For API endpoint testing
```

### System Requirements
- **Python 3.8+**: For modern async/await features
- **Modern Browser**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Network Ports**: 8080 (HTTP) and 8081 (WebSocket) available

## 📞 Support

During migration, if issues arise:

1. **Check Integration Test**: Run `python web_dashboard/integration_test.py`
2. **Check Logs**: Look for errors in console and application logs
3. **Check Browser Console**: Look for JavaScript errors
4. **Check Network Tab**: Verify API calls are working
5. **Check Documentation**: Refer to `web_dashboard/README.md`

For complex issues, the new unified system includes comprehensive debugging features:
- Performance monitoring built into dashboard
- Detailed error messages and recovery
- WebSocket connection status monitoring
- API health checks