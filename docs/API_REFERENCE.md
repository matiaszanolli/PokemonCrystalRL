# 📚 Hybrid LLM-RL Training System API Reference

**Comprehensive API documentation for all components of the hybrid training system.**

## 🚀 Quick Start

```python
from trainer.hybrid_llm_rl_trainer import create_trainer_from_config

# Create and run trainer
trainer = create_trainer_from_config('config.json')
summary = trainer.train(total_episodes=1000)
```

---

## 📖 Core APIs

### HybridLLMRLTrainer

**Location:** `trainer/hybrid_llm_rl_trainer.py`

Main training orchestrator that manages the hybrid LLM-RL training process.

#### Constructor

```python
HybridLLMRLTrainer(
    env: EnhancedPyBoyPokemonCrystalEnv,
    agent: HybridAgent,
    strategy_system: AdaptiveStrategySystem,
    decision_analyzer: DecisionHistoryAnalyzer,
    llm_manager: LLMManager,
    save_dir: str = "checkpoints",
    log_level: str = "INFO"
)
```

**Parameters:**
- `env`: The training environment
- `agent`: Hybrid agent for decision making
- `strategy_system`: Adaptive strategy selection system
- `decision_analyzer`: Decision history analysis and learning
- `llm_manager`: LLM communication manager
- `save_dir`: Directory for checkpoints and saves
- `log_level`: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

#### Methods

##### `train()`
```python
def train(
    self,
    total_episodes: int = 1000,
    max_steps_per_episode: int = 10000,
    save_interval: int = 100,
    eval_interval: int = 50,
    curriculum_patience: int = 20
) -> Dict[str, Any]
```

**Description:** Main training loop with curriculum learning.

**Parameters:**
- `total_episodes`: Total number of training episodes
- `max_steps_per_episode`: Maximum steps per episode
- `save_interval`: Episodes between checkpoint saves
- `eval_interval`: Episodes between evaluations
- `curriculum_patience`: Episodes to wait for curriculum advancement

**Returns:** Training summary dictionary with metrics and results.

##### `load_checkpoint()`
```python
def load_checkpoint(self, checkpoint_path: str) -> bool
```

**Description:** Load training state from checkpoint.

**Parameters:**
- `checkpoint_path`: Path to checkpoint file

**Returns:** True if successful, False otherwise.

---

### HybridAgent

**Location:** `core/hybrid_agent.py`

Combines LLM strategic guidance with RL optimization through intelligent decision arbitration.

#### Constructor

```python
HybridAgent(
    llm_manager: LLMManager,
    adaptive_strategy: AdaptiveStrategySystem,
    action_space_size: int = 9,
    curriculum_config: Optional[Dict] = None
)
```

**Parameters:**
- `llm_manager`: Manager for LLM communications
- `adaptive_strategy`: Strategy selection system
- `action_space_size`: Number of possible actions
- `curriculum_config`: Optional curriculum learning configuration

#### Methods

##### `get_action()`
```python
def get_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> int
```

**Description:** Get action from hybrid decision making process.

**Parameters:**
- `observation`: Multi-modal environment observation
- `info`: Additional environment information

**Returns:** Selected action index.

##### `update()`
```python
def update(
    self,
    observation: Dict[str, Any],
    action: int,
    reward: float,
    next_observation: Dict[str, Any],
    done: bool
) -> None
```

**Description:** Update agent with experience for learning.

#### Agent Modes

```python
from core.hybrid_agent import AgentMode

AgentMode.LLM_ONLY     # Pure LLM decision making
AgentMode.RL_ONLY      # Pure RL decision making  
AgentMode.ADAPTIVE     # Dynamic switching based on performance
```

---

### AdaptiveStrategySystem

**Location:** `core/adaptive_strategy_system.py`

Dynamic strategy selection based on performance metrics and context.

#### Constructor

```python
AdaptiveStrategySystem(
    history_analyzer: Optional[DecisionHistoryAnalyzer] = None,
    goal_planner: Optional[GoalOrientedPlanner] = None
)
```

**Parameters:**
- `history_analyzer`: Decision history analyzer for pattern learning
- `goal_planner`: Goal-oriented planning system

#### Methods

##### `select_strategy()`
```python
def select_strategy(self, context: Dict[str, Any]) -> StrategyType
```

**Description:** Select optimal strategy based on current context.

**Parameters:**
- `context`: Game state and performance context

**Returns:** Selected strategy type.

##### `get_strategy_stats()`
```python
def get_strategy_stats(self) -> Dict[str, Any]
```

**Description:** Get comprehensive strategy performance statistics.

**Returns:** Dictionary with strategy performance metrics.

#### Strategy Types

```python
from core.adaptive_strategy_system import StrategyType

StrategyType.LLM_HEAVY    # 80% LLM, 20% rule-based
StrategyType.BALANCED     # 50% LLM, 50% rule-based
StrategyType.RULE_HEAVY   # 20% LLM, 80% rule-based
```

---

### DecisionHistoryAnalyzer

**Location:** `core/decision_history_analyzer.py`

Learns patterns from decision history with persistent SQLite storage.

#### Constructor

```python
DecisionHistoryAnalyzer(db_path: str = "decisions.db")
```

**Parameters:**
- `db_path`: Path to SQLite database file

#### Methods

##### `record_decision()`
```python
def record_decision(
    self,
    game_state: GameStateAnalysis,
    action_taken: int,
    reasoning: str,
    confidence: float,
    outcome_reward: float
) -> None
```

**Description:** Record a decision for pattern learning.

**Parameters:**
- `game_state`: Analyzed game state
- `action_taken`: Action that was taken
- `reasoning`: Decision reasoning/context
- `confidence`: Confidence score (0.0-1.0)
- `outcome_reward`: Reward received from decision

##### `analyze_patterns()`
```python
def analyze_patterns(
    self,
    min_frequency: int = 3,
    context_filter: Optional[Dict] = None
) -> List[Dict[str, Any]]
```

**Description:** Analyze decision patterns from history.

**Parameters:**
- `min_frequency`: Minimum pattern frequency to report
- `context_filter`: Optional context filtering criteria

**Returns:** List of identified patterns with metadata.

##### `get_recent_decisions()`
```python
def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]
```

**Description:** Get recent decision records.

**Parameters:**
- `limit`: Maximum number of decisions to return

**Returns:** List of recent decision records.

---

### EnhancedPyBoyPokemonCrystalEnv

**Location:** `core/enhanced_pyboy_env.py`

Advanced Gymnasium environment with multi-modal observations and action masking.

#### Constructor

```python
EnhancedPyBoyPokemonCrystalEnv(
    rom_path: str,
    headless: bool = True,
    observation_type: str = "multi_modal"
)
```

**Parameters:**
- `rom_path`: Path to Pokemon Crystal ROM file
- `headless`: Run without display window
- `observation_type`: Type of observations to provide

#### Methods

##### `reset()`
```python
def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

**Description:** Reset environment to initial state.

**Returns:** Tuple of (observation, info) dictionaries.

##### `step()`
```python
def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]
```

**Description:** Execute action in environment.

**Parameters:**
- `action`: Action index to execute

**Returns:** Tuple of (observation, reward, terminated, truncated, info).

#### Observation Space

Multi-modal observations include:

```python
{
    'screen': np.ndarray,           # Game screen (144, 160, 3)
    'state_variables': np.ndarray,  # Numerical state (25,)
    'strategic_context': {          # High-level context
        'current_goal': str,
        'context_summary': str,
        'action_suggestions': List[str]
    }
}
```

#### Action Space

Standard PyBoy actions:
- 0: A button
- 1: B button  
- 2: Start button
- 3: Select button
- 4: Up direction
- 5: Down direction
- 6: Left direction
- 7: Right direction
- 8: No action

---

## 🔧 Utility Functions

### Configuration Management

```python
from trainer.hybrid_llm_rl_trainer import create_trainer_from_config

def create_trainer_from_config(config_path: str) -> HybridLLMRLTrainer
```

**Description:** Create trainer from JSON configuration file.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:** Configured HybridLLMRLTrainer instance.

### State Analysis

```python
from core.game_state_analyzer import GameStateAnalysis

class GameStateAnalysis:
    def __init__(self, state_variables: Dict[str, Any])
    def analyze_battle_state(self) -> Dict[str, Any]
    def analyze_exploration_state(self) -> Dict[str, Any]
    def get_summary(self) -> str
```

---

## 📊 Data Structures

### Training Summary

```python
{
    'total_episodes': int,
    'total_steps': int,
    'best_reward': float,
    'final_avg_reward': float,
    'curriculum_stage_reached': int,
    'curriculum_advancements': int,
    'strategy_switches': int,
    'final_evaluation': {
        'avg_reward': float,
        'std_reward': float,
        'avg_length': float,
        'avg_llm_usage': float
    },
    'avg_episode_length': float,
    'avg_llm_usage': float
}
```

### Decision Record

```python
{
    'id': str,                      # Unique decision ID
    'timestamp': datetime,          # Decision timestamp
    'state_hash': int,              # Hashed game state
    'action': int,                  # Action taken
    'reasoning': str,               # Decision reasoning
    'confidence': float,            # Confidence score
    'outcome_reward': float,        # Reward received
    'episode_id': str,              # Episode identifier
    'step_number': int              # Step in episode
}
```

### Performance Metrics

```python
{
    'episode_reward': float,        # Current episode reward
    'average_reward': float,        # Average reward over window
    'llm_usage_rate': float,        # Percentage of LLM decisions
    'episode_length': int,          # Steps in current episode
    'strategy_switches': int,       # Number of strategy changes
    'curriculum_stage': int,        # Current curriculum stage
    'success_rate': float           # Recent success percentage
}
```

---

## 🛠️ Configuration Schema

### Training Configuration

```json
{
    "rom_path": "string",           // Path to ROM file
    "headless": "boolean",          // Run without display
    "observation_type": "string",   // "multi_modal" | "screen" | "state"
    "llm_model": "string",          // LLM model identifier
    "max_context_length": "number", // Maximum LLM context tokens
    "initial_strategy": "string",   // "llm_heavy" | "balanced" | "rule_heavy"
    "decision_db_path": "string",   // Path to decision database
    "save_dir": "string",           // Checkpoint directory
    "log_level": "string"           // "DEBUG" | "INFO" | "WARNING" | "ERROR"
}
```

### Curriculum Configuration

```python
{
    'stages': [
        {
            'reward_threshold': float,     # Reward needed to advance
            'min_episodes': int,           # Minimum episodes at stage
            'llm_confidence': float,       # LLM confidence threshold
            'strategy_preference': str     # Preferred strategy type
        }
    ],
    'advancement_patience': int,       # Episodes to wait for advancement
    'confidence_decay_rate': float     # Rate of LLM confidence reduction
}
```

---

## 🔍 Error Handling

### Common Exceptions

```python
from core.exceptions import (
    TrainingError,          # General training errors
    ConfigurationError,     # Configuration validation errors  
    EnvironmentError,       # Environment setup errors
    CheckpointError,        # Checkpoint loading/saving errors
    LLMConnectionError,     # LLM communication errors
    DecisionValidationError # Decision validation errors
)
```

### Error Recovery

Most components implement graceful error recovery:

```python
try:
    trainer.train(total_episodes=1000)
except TrainingError as e:
    print(f"Training failed: {e}")
    # Attempt recovery from last checkpoint
    if trainer.load_checkpoint("latest"):
        trainer.train(total_episodes=500)  # Resume with fewer episodes
```

---

## 🧪 Testing APIs

### Test Utilities

```python
from tests.utils import (
    create_mock_environment,
    create_test_trainer,
    generate_test_config,
    cleanup_test_files
)

# Create test environment
mock_env = create_mock_environment()

# Create trainer for testing
test_trainer = create_test_trainer(
    episodes=10,
    headless=True,
    temp_dir="/tmp/test"
)
```

### Performance Benchmarks

```python
from tests.benchmarks import (
    benchmark_training_speed,
    benchmark_memory_usage,
    benchmark_decision_latency
)

# Run benchmarks
speed_results = benchmark_training_speed(trainer, episodes=100)
memory_results = benchmark_memory_usage(trainer, duration=60)
latency_results = benchmark_decision_latency(agent, samples=1000)
```

---

## 📈 Monitoring and Logging

### Logging Configuration

```python
import logging
from trainer.hybrid_llm_rl_trainer import HybridLLMRLTrainer

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

trainer = HybridLLMRLTrainer(..., log_level="DEBUG")
```

### Metrics Collection

```python
# Access training metrics during training
def training_callback(trainer, episode, metrics):
    print(f"Episode {episode}: Reward={metrics['reward']}")
    if metrics['reward'] > best_reward:
        trainer.save_best_model()

# Register callback (if supported)
trainer.register_callback('episode_end', training_callback)
```

---

## 🔗 Integration Points

### Custom LLM Backends

```python
from trainer.llm_manager import BaseLLMManager

class CustomLLMManager(BaseLLMManager):
    def get_action(self, context: str) -> Tuple[int, Dict[str, Any]]:
        # Implement custom LLM integration
        response = your_llm_api.chat(context)
        action = self.parse_action(response)
        metadata = {'source': 'custom_llm', 'confidence': 0.8}
        return action, metadata
```

### Custom Reward Functions

```python
class CustomRewardEnvironment(EnhancedPyBoyPokemonCrystalEnv):
    def _calculate_reward(self, current_state, previous_state, action):
        base_reward = super()._calculate_reward(current_state, previous_state, action)
        
        # Add custom reward logic
        custom_reward = 0.0
        if self._check_custom_condition(current_state):
            custom_reward += 100.0
            
        return base_reward + custom_reward
```

---

**For more examples and advanced usage, see the [Hybrid Training Guide](HYBRID_TRAINING_GUIDE.md).**