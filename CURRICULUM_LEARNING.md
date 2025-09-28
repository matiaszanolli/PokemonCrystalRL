# Curriculum Learning System

## Overview

The Curriculum Learning system provides structured, progressive training for Pokemon Crystal RL agents. Instead of training on random scenarios, agents start with simple tasks and gradually advance to more complex challenges as they demonstrate competency.

## Key Features

- **Progressive Difficulty**: 5-stage curriculum from tutorial to expert level
- **Save State Integration**: Automatic selection of appropriate save states based on curriculum level
- **Adaptive Progression**: Advancement based on success rate and episode count
- **Comprehensive Tracking**: Progress persistence and detailed status reporting
- **Configurable**: Customizable curriculum stages and advancement criteria

## Architecture

### Core Components

1. **CurriculumManager**: Central orchestrator managing progression and save state selection
2. **CurriculumLevel**: Configuration for each stage with scenarios, difficulty, and advancement criteria
3. **CurriculumProgress**: Tracks episode results, success rates, and advancement history
4. **Save State Library Integration**: Automatic selection of training scenarios based on current level

### Curriculum Stages

1. **Tutorial** (Easy)
   - Scenarios: First Pokemon, Basic Exploration
   - Focus: Movement, basic interaction
   - Advancement: 60% success rate, 5-20 episodes

2. **Basic Mechanics** (Easy)
   - Scenarios: Wild Encounters, Team Building
   - Focus: Combat basics, party management
   - Advancement: 65% success rate, 8-25 episodes

3. **Intermediate** (Medium)
   - Scenarios: Gym Battles, Battle Training
   - Focus: Strategy, type advantages
   - Advancement: 70% success rate, 10-30 episodes

4. **Advanced** (Medium)
   - Scenarios: Gym Battles, Story Progression
   - Focus: Complex strategy, game progression
   - Advancement: 75% success rate, 15-40 episodes

5. **Expert** (Hard)
   - Scenarios: Elite Four, Completionist
   - Focus: Mastery, endgame content
   - Advancement: 80% success rate, 20-50 episodes

## Usage

### Basic Training

```bash
# Enable curriculum learning with existing save state library
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --max-actions 500

# With custom library path
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --library-path my_states --curriculum-episodes 10

# With web monitoring
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --enable-web --web-port 8080
```

### Standalone Curriculum Training

```bash
# Comprehensive curriculum training
python3 examples/run_curriculum_training.py roms/pokemon_crystal.gbc --episodes 15 --max-actions 1000

# With custom configuration
python3 examples/run_curriculum_training.py roms/pokemon_crystal.gbc --curriculum-config my_curriculum.json
```

### Creating Custom Curriculum

```bash
# Generate default configuration file
python3 -m training.curriculum_learning

# Edit curriculum_config.json to customize stages, scenarios, and advancement criteria
```

## Configuration

### Default Curriculum Config

```json
{
  "description": "Pokemon Crystal RL Curriculum Learning Configuration",
  "levels": [
    {
      "stage": "tutorial",
      "scenarios": ["first_pokemon", "exploration"],
      "difficulty": "easy",
      "min_success_rate": 0.6,
      "min_episodes": 5,
      "max_episodes": 20,
      "phase_filter": "tutorial",
      "tags": ["beginner", "movement"]
    }
  ]
}
```

### Save State Library Setup

1. **Create Save States**: Use the save state management CLI to add curated training scenarios

```bash
# Add tutorial save state
python3 scripts/manage_save_states.py add tutorial.state "Tutorial Start" "Beginning of game" \
  --phase tutorial --scenario first_pokemon --difficulty easy --badges 0 --location "New Bark Town"

# Add gym battle save state
python3 scripts/manage_save_states.py add gym1.state "First Gym" "Ready for Violet City Gym" \
  --phase early_game --scenario gym_battle --difficulty medium --badges 0 --level 15
```

2. **Verify Library**: Check your save state collection

```bash
python3 scripts/manage_save_states.py info
python3 scripts/manage_save_states.py list --scenario gym_battle
```

## Progress Tracking

### Status Monitoring

```python
# Get curriculum status
curriculum_manager = CurriculumManager(library)
status = curriculum_manager.get_curriculum_status()

print(f"Current Stage: {status['current_stage']}")
print(f"Episodes: {status['level_progress']['episodes']}/{status['level_progress']['max_episodes']}")
print(f"Success Rate: {status['level_progress']['success_rate']:.1%}")
```

### Advancement Criteria

- **Success-based**: Achieve minimum success rate with minimum episodes
- **Time-based**: Force advancement after maximum episodes regardless of performance
- **Progressive**: Each stage has higher success rate requirements

### Progress Persistence

Progress is automatically saved to `curriculum_progress.json` and restored between sessions.

## Integration with Existing Systems

### LLM Training

Curriculum learning works seamlessly with LLM-based training:

```bash
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --llm-model smollm2:1.7b --llm-interval 10
```

### Web Monitoring

Progress and advancement are displayed in the web dashboard:

```bash
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --enable-web
# Visit http://localhost:8080 for real-time curriculum progress
```

### Multi-Agent Training

Curriculum learning can guide specialist agent selection and coordination.

## Advanced Features

### Custom Success Criteria

Override episode success evaluation in `CurriculumTrainer._evaluate_episode_success()`:

```python
def _evaluate_episode_success(self, stats: dict) -> bool:
    # Custom logic based on specific training objectives
    badges_gained = stats.get('badges_gained', 0)
    level_gains = stats.get('level_gains', 0)
    return badges_gained > 0 or level_gains > 2
```

### Stage-Specific Configuration

Configure different advancement criteria per stage:

```json
{
  "stage": "expert",
  "scenarios": ["elite_four"],
  "difficulty": "hard",
  "min_success_rate": 0.9,
  "min_episodes": 25,
  "max_episodes": 100,
  "phase_filter": "late_game",
  "badges_range": [8, 16]
}
```

## Testing

```bash
# Run curriculum learning tests
python -m pytest tests/trainer/test_curriculum_learning.py -v

# Test specific functionality
python -m pytest tests/trainer/test_curriculum_learning.py::TestCurriculumManager::test_record_episode_result_advancement -v
```

## Best Practices

1. **Curated Save States**: Use high-quality, validated save states for consistent training
2. **Balanced Progression**: Adjust success rate thresholds to maintain challenge without frustration
3. **Scenario Diversity**: Include varied scenarios within each stage for robust learning
4. **Progress Monitoring**: Track advancement patterns and adjust curriculum based on agent performance
5. **Incremental Difficulty**: Ensure smooth transitions between curriculum stages

## Troubleshooting

### Common Issues

1. **No Suitable Save States**: Ensure save state library contains states matching curriculum requirements
2. **Stuck at Level**: Check success criteria and consider adjusting advancement thresholds
3. **Rapid Progression**: Increase minimum episode requirements or success rate thresholds

### Debugging

```bash
# Verbose logging
python3 examples/run_curriculum_training.py roms/pokemon_crystal.gbc --verbose

# Check save state recommendations
python3 scripts/manage_save_states.py recommend gym_battle --difficulty medium
```

## Future Enhancements

- **Automatic Curriculum Generation**: AI-driven curriculum creation based on agent performance
- **Multi-Modal Progression**: Combine curriculum learning with reinforcement learning techniques
- **Adaptive Thresholds**: Dynamic adjustment of advancement criteria based on learning patterns
- **Collaborative Curriculum**: Multi-agent curriculum with shared progression tracking

---

This curriculum learning system represents a significant advancement in structured AI training for complex game environments, providing a foundation for more sophisticated learning approaches.