# 🔧 Unified Trainer API Reference

Complete API documentation for the Pokemon Crystal RL unified training system.

## 📚 **Core Classes**

### **UnifiedPokemonTrainer**

The main orchestrator class that handles all training modes and configurations.

#### **Constructor**
```python
class UnifiedPokemonTrainer:
    def __init__(self, config: TrainingConfig)
```

**Parameters:**
- `config`: TrainingConfig object with all training parameters

#### **Methods**

##### **start_training()**
```python
def start_training(self) -> None
```
Starts the training process based on the configured mode.

**Behavior:**
- Initializes screen capture if enabled
- Routes to appropriate training method based on mode
- Handles cleanup and statistics generation

##### **get_performance_stats()**
```python
def get_performance_stats(self) -> Dict[str, Any]
```
Returns current performance statistics.

**Returns:**
```python
{
    'total_actions': int,
    'total_episodes': int,
    'llm_calls': int,
    'actions_per_second': float,
    'mode': str,
    'model': str
}
```

---

### **TrainingConfig**

Configuration dataclass for training parameters.

```python
@dataclass
class TrainingConfig:
    # Core settings
    rom_path: str
    mode: TrainingMode = TrainingMode.FAST_LOCAL
    llm_backend: LLMBackend = LLMBackend.SMOLLM2
    
    # Training parameters
    max_actions: int = 1000
    max_episodes: int = 10
    llm_interval: int = 10
    
    # Performance settings
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    
    # Web interface
    enable_web: bool = False
    web_port: int = 8080
    web_host: str = "localhost"
    
    # Screen capture
    capture_screens: bool = True
    capture_fps: int = 10
    screen_resize: tuple = (320, 288)
    
    # Curriculum settings
    curriculum_stages: int = 5
    stage_mastery_threshold: float = 0.7
    min_stage_episodes: int = 5
    max_stage_episodes: int = 20
    
    # Output settings
    save_stats: bool = True
    stats_file: str = "training_stats.json"
    log_level: str = "INFO"
```

---

## 🎯 **Training Modes**

### **TrainingMode Enum**
```python
class TrainingMode(Enum):
    FAST_LOCAL = "fast_local"      # Direct PyBoy, optimized capture
    CURRICULUM = "curriculum"      # Progressive skill-based training  
    ULTRA_FAST = "ultra_fast"     # Rule-based maximum speed
    MONITORED = "monitored"       # Full monitoring and analysis
    CUSTOM = "custom"             # User-defined configuration
```

### **Mode-Specific Methods**

#### **_run_fast_local_training()**
```python
def _run_fast_local_training(self) -> None
```
Executes fast local training with real-time capture.

**Features:**
- Direct PyBoy access for maximum speed
- LLM decisions at configurable intervals
- Real-time screen capture
- Web monitoring support

#### **_run_curriculum_training()**  
```python
def _run_curriculum_training(self) -> None
```
Executes progressive curriculum training.

**Features:**
- 5-stage skill progression
- Mastery validation at each stage
- Knowledge transfer between stages
- Comprehensive progress tracking

#### **_run_ultra_fast_training()**
```python
def _run_ultra_fast_training(self) -> None
```
Executes rule-based ultra-fast training.

**Features:**
- Pattern-based action selection
- No LLM overhead
- Maximum speed (600+ actions/sec)
- Minimal resource usage

#### **_run_monitored_training()**
```python
def _run_monitored_training(self) -> None
```
Executes comprehensive monitored training.

**Features:**
- Full environment wrapper usage
- Enhanced LLM agent integration
- Detailed logging and analysis
- Research-oriented features

---

## 🤖 **LLM Integration**

### **LLMBackend Enum**
```python
class LLMBackend(Enum):
    SMOLLM2 = "smollm2:1.7b"      # Ultra-fast, optimized
    LLAMA32_1B = "llama3.2:1b"    # Fastest Llama
    LLAMA32_3B = "llama3.2:3b"    # Balanced speed/quality
    QWEN25_3B = "qwen2.5:3b"      # Alternative fast option
    NONE = None                    # Rule-based only
```

### **LLM Methods**

#### **_get_llm_action()**
```python
def _get_llm_action(self, stage: str = "BASIC_CONTROLS") -> int
```
Gets action decision from the LLM.

**Parameters:**
- `stage`: Current training stage for context

**Returns:**
- Action integer (1-7) representing game input

**Prompt Format:**
```
Pokemon Crystal - Stage: {stage}
Choose action number:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Action:
```

#### **_parse_action()**
```python
def _parse_action(self, response: str) -> int
```
Parses LLM response to action integer.

**Parameters:**
- `response`: Raw LLM response text

**Returns:**
- Parsed action integer (defaults to 5 for A button)

---

## 📸 **Screen Capture**

### **Screen Capture Methods**

#### **_start_screen_capture()**
```python
def _start_screen_capture(self) -> None
```
Starts background screen capture thread.

#### **_capture_loop()**
```python
def _capture_loop(self) -> None
```
Main screen capture loop running in background thread.

**Process:**
1. Capture screen from PyBoy
2. Convert to PIL Image
3. Resize for efficiency
4. Encode as base64
5. Update latest screen buffer
6. Add to capture queue

#### **get_latest_screen()**
```python
def get_latest_screen(self) -> Optional[Dict[str, Any]]
```
Returns the most recent screen capture.

**Returns:**
```python
{
    'image_b64': str,      # Base64 encoded PNG
    'timestamp': float,    # Unix timestamp
    'size': tuple         # (width, height)
}
```

---

## 🌐 **Web Interface**

### **Web Server**

#### **_create_web_server()**
```python
def _create_web_server(self) -> HTTPServer
```
Creates HTTP server for web monitoring interface.

### **HTTP Endpoints**

#### **GET /**
Main dashboard page with real-time monitoring interface.

**Response:** HTML page with:
- Live game screen
- Performance statistics
- Training progress
- Real-time updates

#### **GET /screen**
Returns current game screen as PNG image.

**Response:** 
- Content-Type: image/png
- Body: PNG image data

#### **GET /stats**
Returns current training statistics as JSON.

**Response:**
```json
{
  "total_actions": 1247,
  "total_episodes": 5,
  "llm_calls": 124,
  "actions_per_second": 42.3,
  "mode": "fast_local",
  "model": "smollm2:1.7b",
  "start_time": 1699999999.0
}
```

---

## 📊 **Statistics and Logging**

### **Statistics Structure**
```python
stats = {
    'start_time': float,           # Training start timestamp
    'total_actions': int,          # Total actions executed
    'total_episodes': int,         # Total episodes completed
    'llm_calls': int,             # Number of LLM inference calls
    'actions_per_second': float,   # Current performance metric
    'mode': str,                  # Training mode name
    'model': str                  # LLM model name or "rule-based"
}
```

### **Statistics Methods**

#### **_update_stats()**
```python
def _update_stats(self) -> None
```
Updates performance statistics with current metrics.

#### **_finalize_training()**
```python  
def _finalize_training(self) -> None
```
Handles training cleanup and final statistics generation.

**Actions:**
- Stops screen capture
- Updates final statistics
- Saves stats to JSON file
- Prints summary report
- Cleans up resources

---

## 🎮 **Game Integration**

### **Action Mapping**
```python
actions = {
    1: WindowEvent.PRESS_ARROW_UP,
    2: WindowEvent.PRESS_ARROW_DOWN,
    3: WindowEvent.PRESS_ARROW_LEFT,
    4: WindowEvent.PRESS_ARROW_RIGHT,
    5: WindowEvent.PRESS_BUTTON_A,
    6: WindowEvent.PRESS_BUTTON_B,
    7: WindowEvent.PRESS_BUTTON_START,
    8: WindowEvent.PRESS_BUTTON_SELECT,
    0: None  # No action
}
```

### **Game Control Methods**

#### **_execute_action()**
```python
def _execute_action(self, action: int) -> None
```
Executes game action through PyBoy interface.

**Parameters:**
- `action`: Action integer (0-8)

**Process:**
1. Maps action to PyBoy WindowEvent
2. Sends input to PyBoy
3. Advances game by one frame
4. Updates action counter

---

## 📚 **Curriculum System**

### **Curriculum Methods**

#### **_run_curriculum_episode()**
```python
def _run_curriculum_episode(self, stage: int) -> bool
```
Runs single curriculum training episode.

**Parameters:**
- `stage`: Current curriculum stage (1-5)

**Returns:**
- Boolean indicating episode success

#### **_get_stage_action()**
```python
def _get_stage_action(self, stage: int) -> int
```
Gets stage-appropriate action for curriculum training.

**Parameters:**
- `stage`: Current curriculum stage

**Returns:**
- Action integer optimized for current stage

### **Stage Definitions**
```python
stage_prompts = {
    1: "BASIC_CONTROLS - Focus on navigation",
    2: "DIALOGUE - Focus on text interaction",
    3: "POKEMON_SELECTION - Focus on menu choices", 
    4: "BATTLE_FUNDAMENTALS - Focus on combat",
    5: "EXPLORATION - Focus on world navigation"
}
```

---

## 🔧 **Configuration and Utilities**

### **CLI Interface**

#### **main()**
```python
def main() -> None
```
Main CLI entry point with argument parsing.

**Supported Arguments:**
```bash
--rom PATH              # Pokemon Crystal ROM (required)
--mode MODE            # Training mode
--model MODEL          # LLM model selection
--actions N            # Maximum actions
--episodes N           # Maximum episodes
--llm-interval N       # Actions between LLM calls
--web                  # Enable web interface
--port N               # Web interface port
--no-capture          # Disable screen capture
--save-state PATH     # Save state file
--windowed            # Show game window
--debug               # Enable debug mode
--no-llm              # Disable LLM (rule-based)
```

### **Utility Methods**

#### **_print_config_summary()**
```python
def _print_config_summary(self) -> None
```
Prints training configuration summary.

#### **_detect_progress()**
```python
def _detect_progress(self, old_state: Dict, new_state: Dict) -> bool
```
Simple progress detection for PyBoy mode.

---

## 🚨 **Error Handling**

### **Common Exceptions**

#### **PyBoy Initialization**
```python
if not PYBOY_AVAILABLE:
    raise RuntimeError("PyBoy not available for fast local training")
```

#### **LLM Model Availability**
```python
try:
    ollama.show(model_name)
except:
    print(f"📥 Pulling LLM model: {model_name}")
    ollama.pull(model_name)
```

#### **Screen Capture Errors**
```python
try:
    screen_array = self.pyboy.screen.ndarray
except AttributeError:
    screen_array = np.array(self.pyboy.screen.image)
```

### **Error Recovery**

#### **LLM Fallbacks**
- Network errors → Use cached last action
- Model errors → Fall back to rule-based actions
- Parse errors → Default to A button (action 5)

#### **Training Continuity**
- Screen capture errors → Continue without capture
- Web server errors → Continue without monitoring
- Episode errors → Log and continue to next episode

---

## 📈 **Performance Optimization**

### **Speed Optimization Techniques**

1. **Direct PyBoy Access**: Bypass environment wrapper overhead
2. **LLM Caching**: Reuse decisions for multiple actions
3. **Screen Capture Threading**: Non-blocking background capture
4. **Minimal Logging**: Only essential output in fast modes
5. **Resource Pooling**: Reuse objects to minimize allocation

### **Memory Management**

1. **Queue Limits**: Bounded screen capture queues
2. **Database Cleanup**: Regular cleanup of temporary data
3. **Model Loading**: Load models once, reuse inference
4. **Buffer Management**: Efficient image buffer handling

---

## 🧪 **Testing and Debugging**

### **Debug Mode Features**

When `debug_mode=True`:
- Enhanced logging output
- LLM decision reasoning
- Performance profiling
- Error stack traces
- State transition logging

### **Performance Profiling**

```python
# Built-in performance tracking
def _profile_action(self, action_func):
    start = time.time()
    result = action_func()
    duration = time.time() - start
    self.stats['action_times'].append(duration)
    return result
```

---

**📚 This API reference covers all public methods and key internal functionality of the unified training system.**

For implementation examples and usage patterns, see the [Examples Documentation](../examples/) and [User Guides](../guides/).
