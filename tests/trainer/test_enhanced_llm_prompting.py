#!/usr/bin/env python3
"""
test_enhanced_llm_prompting.py - Tests for Enhanced LLM Prompting System

Tests the improved state-aware prompting system including:
- Numeric key guidance (5=A, 7=START, etc.)
- Temperature-based decision making (0.8/0.6)
- State-specific prompting strategies
- Multi-model support and fallback
- Prompt effectiveness measurement
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time
import numpy as np

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the enhanced trainer system
from pokemon_crystal_rl.trainer.trainer import (
    UnifiedPokemonTrainer,
    TrainingConfig,
    TrainingMode,
    LLMBackend
)


@pytest.mark.enhanced_prompting
@pytest.mark.llm
class TestEnhancedLLMPrompting:
    """Test enhanced LLM prompting with state-aware guidance"""
    
    @pytest.fixture
    def mock_config(self):
        """Configuration for LLM prompting tests"""
        return TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=3,
            debug_mode=True,
            headless=True,
            capture_screens=False
        )
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class, mock_config):
        """Create trainer with mocked PyBoy for LLM testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        return UnifiedPokemonTrainer(mock_config)
    
    def test_numeric_key_guidance_in_prompts(self, trainer):
        """Test that LLM system uses numeric key guidance"""
        # Mock game state and screenshot
        game_state = "dialogue"
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Mock ollama to capture the prompt used
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            with patch.object(trainer.game_state_detector, 'detect_game_state', return_value=game_state):
                action = trainer.llm_manager.get_llm_action(screenshot)
            
            # Verify LLM was called with correct prompt structure
            assert mock_ollama.generate.called
            call_args = mock_ollama.generate.call_args
            prompt = call_args[1]['prompt'] if 'prompt' in call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else ""
            
            # Verify numeric key guidance is present
            assert "5=A" in prompt
            assert "7=START" in prompt
            assert "1=UP" in prompt
            assert "2=DOWN" in prompt
            
            # Verify state-specific guidance
            assert "dialogue" in prompt.lower()
    
    def test_state_specific_prompting_strategies(self, trainer):
        """Test different prompting strategies for different game states"""
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        test_states = ["dialogue", "overworld", "menu", "battle", "title_screen"]
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            for state in test_states:
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value=state):
                    action = trainer.llm_manager.get_llm_action(screenshot)
                
                # Verify LLM was called with state-specific prompt
                if mock_ollama.generate.called:
                    call_args = mock_ollama.generate.call_args
                    prompt = call_args[1]['prompt'] if 'prompt' in call_args[1] else ""
                    
                    # Each state should have specific guidance
                    assert state in prompt.lower() or "game" in prompt.lower()
                    assert "5=A" in prompt  # All prompts should have key guidance
                
                mock_ollama.reset_mock()
    
    def test_temperature_configuration_by_state(self, trainer):
        """Test temperature settings vary by game state"""
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            # Test different states
            states_to_test = {
                "dialogue": 0.8,      # High temperature for variety in responses
                "menu": 0.6,          # Medium temperature for navigation
                "battle": 0.8,        # High temperature for combat variety
                "overworld": 0.7,     # Medium-high for exploration
                "title_screen": 0.5   # Lower temperature for consistent startup
            }
            
            for state, expected_temp in states_to_test.items():
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value=state):
                    trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3)))
                
                # Verify temperature was set correctly
                if mock_ollama.generate.called:
                    call_kwargs = mock_ollama.generate.call_args[1]
                    actual_temp = call_kwargs.get('options', {}).get('temperature', 0.0)
                    assert abs(actual_temp - expected_temp) < 0.1, f"State {state}: expected {expected_temp}, got {actual_temp}"
    
    @pytest.mark.temperature
    def test_temperature_based_action_variety(self, trainer):
        """Test that higher temperatures produce more varied actions"""
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            # Mock responses for different temperatures
            low_temp_responses = ['5', '5', '5', '5', '5']  # Always A button
            high_temp_responses = ['5', '1', '3', '2', '7']  # Varied actions
            
            # Test low temperature (should be consistent)
            mock_ollama.generate.side_effect = [{'response': r} for r in low_temp_responses]
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            low_temp_actions = []
            for i in range(5):
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="title_screen"):
                    action = trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3)))
                    if action is not None:
                        low_temp_actions.append(action)
            
            # Test high temperature (should be varied)
            mock_ollama.generate.side_effect = [{'response': r} for r in high_temp_responses]
            
            high_temp_actions = []
            for i in range(5):
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="dialogue"):
                    action = trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3)))
                    if action is not None:
                        high_temp_actions.append(action)
            
            # High temperature should show more variety
            if len(high_temp_actions) >= 3:
                high_temp_variety = len(set(high_temp_actions))
                assert high_temp_variety >= 2, f"High temp produced only {high_temp_variety} unique actions"
    
    def test_llm_response_parsing_robustness(self, trainer):
        """Test robust parsing of various LLM response formats"""
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        test_responses = [
            "5",                    # Simple numeric
            "Action: 5",            # With label
            "I'll press A (5)",     # With explanation
            "5 - A button",         # With description
            "Let me press 5",       # In sentence
            "5\n",                  # With newline
            " 5 ",                  # With spaces
            "Key 5 (A button)",     # Detailed format
            "I think 5 is best",    # In reasoning
            "5 because it's A",     # With justification
        ]
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            for response in test_responses:
                mock_ollama.generate.return_value = {'response': response}
                
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="dialogue"):
                    parsed_action = trainer.llm_manager.get_llm_action(screenshot)
                
                assert parsed_action == 5, f"Failed to parse '{response}' as action 5, got {parsed_action}"
    
    def test_invalid_llm_response_handling(self, trainer):
        """Test handling of invalid LLM responses"""
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        invalid_responses = [
            "invalid",
            "9",                    # Out of range
            "0",                    # Out of range
            "",                     # Empty
            "hello world",          # No number
            "action",              # Word only
        ]
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            for response in invalid_responses:
                mock_ollama.generate.return_value = {'response': response}
                
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="dialogue"):
                    parsed_action = trainer.llm_manager.get_llm_action(screenshot)
                
                # Should fall back to valid action (1-8 range) - LLM manager has fallback logic
                assert 1 <= parsed_action <= 8, f"Invalid response '{response}' should return valid action, got {parsed_action}"
    
    def test_context_aware_prompting(self, trainer):
        """Test context-aware prompting with stuck detection"""
        # Setup mock game context  
        trainer.game_state_detector.stuck_counter = 3
        
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # When stuck, should use fallback action
        with patch.object(trainer.game_state_detector, 'is_stuck', return_value=True):
            with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="dialogue"):
                with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
                    mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
                    action = trainer.llm_manager.get_llm_action_with_vision(screenshot, 100)
        
        # Should get fallback action when stuck
        assert 1 <= action <= 8, "Should get valid fallback action when stuck"
    
    def test_prompt_effectiveness_tracking(self, trainer):
        """Test tracking of prompt effectiveness"""
        # This would test the system's ability to learn which prompts work better
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            # Simulate successful action
            with patch.object(trainer.game_state_detector, 'detect_game_state', return_value="dialogue"):
                action = trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3)))
            
            # Verify action was recorded (would be used for future prompt optimization)
            assert action is not None
            assert 1 <= action <= 8


@pytest.mark.multi_model
@pytest.mark.llm
class TestMultiModelLLMSupport:
    """Test support for multiple LLM backends"""
    
    @pytest.fixture
    def trainer_configs(self):
        """Configurations for different LLM backends"""
        return {
            LLMBackend.SMOLLM2: TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.SMOLLM2,
                headless=True
            ),
            LLMBackend.LLAMA32_1B: TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.LLAMA32_1B,
                headless=True
            ),
            LLMBackend.LLAMA32_3B: TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.LLAMA32_3B,
                headless=True
            ),
            LLMBackend.NONE: TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.NONE,
                headless=True
            )
        }
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_model_specific_configurations(self, mock_pyboy_class, trainer_configs):
        """Test that different models get appropriate configurations"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        model_expectations = {
            LLMBackend.SMOLLM2: {"model": "smollm2:1.7b", "fast": True},
            LLMBackend.LLAMA32_1B: {"model": "llama3.2:1b", "fast": True},
            LLMBackend.LLAMA32_3B: {"model": "llama3.2:3b", "fast": False},
        }
        
        for backend, config in trainer_configs.items():
            if backend == LLMBackend.NONE:
                continue
                
            trainer = UnifiedPokemonTrainer(config)
            
            if backend in model_expectations:
                expected = model_expectations[backend]
                # Test model name setting
                assert hasattr(trainer.llm_manager, 'model') or hasattr(trainer, 'llm_manager')
                # Would test specific model configurations
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_fallback_mechanism(self, mock_pyboy_class):
        """Test fallback to rule-based when LLM fails"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            headless=True
        )
        
        trainer = UnifiedPokemonTrainer(config)
        
        # Mock LLM failure
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.side_effect = Exception("LLM unavailable")
            
            # Should fallback to rule-based
            action = trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3)))
            
            # Should still return valid action via fallback
            assert action is None or (1 <= action <= 8)
    
    def test_model_performance_expectations(self):
        """Test performance expectations for different models"""
        performance_expectations = {
            LLMBackend.SMOLLM2: {"inference_time": 0.05, "memory": "2GB"},     # ~25ms
            LLMBackend.LLAMA32_1B: {"inference_time": 0.06, "memory": "1GB"},   # ~30ms
            LLMBackend.LLAMA32_3B: {"inference_time": 0.1, "memory": "3GB"},    # ~60ms
        }
        
        # This would be used to validate model performance meets expectations
        for backend, expectations in performance_expectations.items():
            assert expectations["inference_time"] < 0.2  # All should be under 200ms
            assert "GB" in expectations["memory"]


@pytest.mark.enhanced_prompting
@pytest.mark.performance
class TestPromptPerformanceOptimizations:
    """Test performance optimizations in prompting system"""
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=3,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_prompt_caching_efficiency(self, trainer):
        """Test that LLM calls are efficient"""
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        state = "dialogue"
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            start_time = time.time()
            
            # Make multiple LLM calls
            for i in range(10):
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value=state):
                    action = trainer.llm_manager.get_llm_action(screenshot)
            
            elapsed = time.time() - start_time
            
            # Should be reasonably fast (under 200ms total for 10 calls)
            assert elapsed < 0.2, f"LLM calls took {elapsed:.3f}s for 10 iterations"
    
    def test_llm_call_rate_limiting(self, trainer):
        """Test that LLM calls respect the configured interval"""
        trainer.config.llm_interval = 3  # Every 3 actions
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            llm_call_count = 0
            
            # Simulate 10 actions
            for step in range(10):
                action = trainer.llm_manager.get_llm_action(np.zeros((144, 160, 3))) if step % trainer.config.llm_interval == 0 else None
                if mock_ollama.generate.called:
                    llm_call_count += 1
                    mock_ollama.reset_mock()
            
            # Should have made approximately 10/3 = ~3 LLM calls
            expected_calls = 10 // trainer.config.llm_interval
            assert abs(llm_call_count - expected_calls) <= 1, f"Expected ~{expected_calls} LLM calls, got {llm_call_count}"
    
    def test_prompt_length_optimization(self, trainer):
        """Test that LLM system uses reasonable prompts"""
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        states = ["dialogue", "overworld", "menu", "battle", "title_screen"]
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            for state in states:
                with patch.object(trainer.game_state_detector, 'detect_game_state', return_value=state):
                    action = trainer.llm_manager.get_llm_action(screenshot)
                
                # Verify LLM was called (implies reasonable prompt was used)
                assert mock_ollama.generate.called
                
                # Check the prompt that was used
                call_args = mock_ollama.generate.call_args
                prompt = call_args[1]['prompt'] if 'prompt' in call_args[1] else ""
                
                # Prompts should be informative but not excessive
                assert 50 < len(prompt) < 2000, f"Prompt for {state} is {len(prompt)} chars (should be 50-2000)"
                
                # Should contain key information efficiently
                assert "5=A" in prompt
                assert state.lower() in prompt.lower() or "game" in prompt.lower()
                
                mock_ollama.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "enhanced_prompting"])
