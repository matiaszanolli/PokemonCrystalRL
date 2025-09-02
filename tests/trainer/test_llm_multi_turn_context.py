#!/usr/bin/env python3
"""
Unit tests for LLM Manager Multi-Turn Context features
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trainer.llm_manager import LLMManager


class TestLLMMultiTurnContext:
    """Test cases for LLM Manager Multi-Turn Context"""

    def setup_method(self):
        """Set up test fixtures"""
        # Mock LLM manager to avoid Ollama dependency
        with patch('trainer.llm_manager.OLLAMA_AVAILABLE', True), \
             patch('trainer.llm_manager.ollama.show'):
            self.llm_manager = LLMManager(model="test_model", max_context_turns=3)

    def test_conversation_memory_initialization(self):
        """Test conversation memory initialization"""
        assert hasattr(self.llm_manager, 'max_context_turns')
        assert hasattr(self.llm_manager, 'conversation_memory')
        assert hasattr(self.llm_manager, 'decision_history')
        
        assert self.llm_manager.max_context_turns == 3
        assert len(self.llm_manager.conversation_memory) == 0

    def test_add_to_conversation_memory(self):
        """Test adding decisions to conversation memory"""
        # Add a decision to memory
        self.llm_manager.add_to_conversation_memory(
            prompt="Test prompt",
            response="Action 5",
            action=5,
            game_state="overworld"
        )
        
        assert len(self.llm_manager.conversation_memory) == 1
        
        memory_entry = self.llm_manager.conversation_memory[0]
        assert memory_entry['prompt'] == "Test prompt"
        assert memory_entry['response'] == "Action 5"
        assert memory_entry['action'] == 5
        assert memory_entry['game_state'] == "overworld"
        assert 'timestamp' in memory_entry

    def test_conversation_memory_limit(self):
        """Test that conversation memory respects max_context_turns limit"""
        # Add more entries than the limit
        for i in range(5):  # Limit is 3
            self.llm_manager.add_to_conversation_memory(
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                action=i + 1,
                game_state="overworld"
            )
        
        # Should only keep the last 3
        assert len(self.llm_manager.conversation_memory) == 3
        
        # Should keep the most recent entries
        actions = [entry['action'] for entry in self.llm_manager.conversation_memory]
        assert actions == [3, 4, 5]  # Last 3 actions

    def test_build_context_prompt_empty(self):
        """Test context prompt building when memory is empty"""
        original_prompt = "Choose an action"
        
        context_prompt = self.llm_manager.build_context_prompt(original_prompt)
        
        # Should return original prompt when no memory
        assert context_prompt == original_prompt

    def test_build_context_prompt_with_history(self):
        """Test context prompt building with decision history"""
        # Add some history
        self.llm_manager.add_to_conversation_memory(
            prompt="First prompt",
            response="Move up",
            action=1,
            game_state="overworld"
        )
        self.llm_manager.add_to_conversation_memory(
            prompt="Second prompt", 
            response="Interact",
            action=5,
            game_state="dialogue"
        )
        
        original_prompt = "What should I do next?"
        context_prompt = self.llm_manager.build_context_prompt(original_prompt)
        
        # Should include previous decisions
        assert "Previous decisions for context:" in context_prompt
        assert "overworld -> chose UP (1)" in context_prompt
        assert "dialogue -> chose A (5)" in context_prompt
        assert "What should I do next?" in context_prompt
        assert "Considering your previous decisions" in context_prompt

    def test_action_to_name_conversion(self):
        """Test action number to name conversion"""
        assert self.llm_manager._action_to_name(1) == "UP"
        assert self.llm_manager._action_to_name(5) == "A"
        assert self.llm_manager._action_to_name(7) == "START"
        assert self.llm_manager._action_to_name(99) == "ACTION_99"  # Unknown action

    @patch('trainer.llm_manager.ollama.generate')
    def test_get_action_with_context(self, mock_generate):
        """Test that get_action uses conversation context"""
        # Mock successful LLM response
        mock_generate.return_value = {'response': '5'}
        
        # Add some conversation history
        self.llm_manager.add_to_conversation_memory(
            prompt="Previous prompt",
            response="Move right", 
            action=4,
            game_state="overworld"
        )
        
        # Get action
        action = self.llm_manager.get_action(game_state="overworld")
        
        # Should have called ollama.generate
        assert mock_generate.called
        
        # Check that the prompt included context
        call_args = mock_generate.call_args
        prompt_used = call_args[1]['prompt']  # keyword argument
        assert "Previous decisions for context:" in prompt_used
        assert "overworld -> chose RIGHT (4)" in prompt_used

    @patch('trainer.llm_manager.ollama.generate')
    def test_conversation_memory_tracking_on_success(self, mock_generate):
        """Test that successful actions are added to conversation memory"""
        # Mock successful LLM response
        mock_generate.return_value = {'response': '5'}
        
        initial_memory_size = len(self.llm_manager.conversation_memory)
        
        # Get action
        action = self.llm_manager.get_action(game_state="battle")
        
        # Should have added to conversation memory
        assert len(self.llm_manager.conversation_memory) == initial_memory_size + 1
        
        # Check the added entry
        latest_entry = self.llm_manager.conversation_memory[-1]
        assert latest_entry['action'] == 5
        assert latest_entry['game_state'] == "battle"

    @patch('trainer.llm_manager.ollama.generate')
    def test_conversation_memory_tracking_on_fallback(self, mock_generate):
        """Test that fallback actions are also tracked"""
        # Mock LLM failure (invalid response)
        mock_generate.return_value = {'response': 'invalid response'}
        
        initial_memory_size = len(self.llm_manager.conversation_memory)
        
        # Get action (should use fallback)
        action = self.llm_manager.get_action(game_state="dialogue")
        
        # Should have added fallback to conversation memory
        assert len(self.llm_manager.conversation_memory) == initial_memory_size + 1
        
        # Check the added entry mentions fallback
        latest_entry = self.llm_manager.conversation_memory[-1]
        assert "fallback" in latest_entry['response'].lower()

    def test_system_prompt_enhancement(self):
        """Test that system prompt is enhanced for context-aware decisions"""
        with patch('trainer.llm_manager.ollama.generate') as mock_generate:
            mock_generate.return_value = {'response': '5'}
            
            # Get action
            self.llm_manager.get_action(game_state="overworld")
            
            # Check system prompt
            call_args = mock_generate.call_args
            system_prompt = call_args[1]['system']
            assert "Learn from your previous decisions" in system_prompt
            assert "choose strategically" in system_prompt

    def test_context_prompt_length_management(self):
        """Test that context prompts don't grow too large"""
        # Add maximum history
        for i in range(5):
            self.llm_manager.add_to_conversation_memory(
                prompt=f"Very long prompt {i} " * 20,  # Long prompt
                response=f"Response {i}",
                action=i + 1,
                game_state="overworld"
            )
        
        original_prompt = "Short prompt"
        context_prompt = self.llm_manager.build_context_prompt(original_prompt)
        
        # Should only include last 3 turns (due to max_context_turns=3)
        turn_count = context_prompt.count("Turn")
        assert turn_count == 3

    def test_context_aware_temperature_selection(self):
        """Test that temperature selection works with context"""
        with patch('trainer.llm_manager.ollama.generate') as mock_generate:
            mock_generate.return_value = {'response': '5'}
            
            # Get action for dialogue state
            self.llm_manager.get_action(game_state="dialogue")
            
            # Check that correct temperature was used
            call_args = mock_generate.call_args
            options = call_args[1]['options']
            assert options['temperature'] == 0.8  # dialogue temperature


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])