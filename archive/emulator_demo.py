"""
emulator_demo.py - Simple demonstration of emulator state during training

This script shows what the emulator is doing step-by-step with detailed output.
"""

import numpy as np
import time
import json
import cv2
import os
from datetime import datetime

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from vision_processor import VisualContext


def save_screenshot_as_text(screenshot, filename):
    """Save screenshot as a text representation"""
    if screenshot is None:
        return
    
    try:
        # Create ASCII art from screenshot
        height, width = 20, 40
        resized = cv2.resize(screenshot, (width, height))
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        
        # ASCII characters from darkest to lightest
        chars = "@%#*+=-:. "
        
        ascii_art = []
        for row in gray:
            line = ""
            for pixel in row:
                char_index = min(int(pixel / 255 * (len(chars) - 1)), len(chars) - 1)
                line += chars[char_index]
            ascii_art.append(line)
        
        with open(filename, 'w') as f:
            f.write("POKEMON CRYSTAL GAME SCREEN (ASCII)\n")
            f.write("=" * 50 + "\n")
            for line in ascii_art:
                f.write(line + "\n")
        
    except Exception as e:
        print(f"Failed to save ASCII art: {e}")


def format_visual_context(visual_context: VisualContext) -> str:
    """Format visual context into readable text"""
    if not visual_context:
        return "No visual analysis available"
    
    lines = [
        f"Screen Type: {visual_context.screen_type.upper()}",
        f"Game Phase: {visual_context.game_phase}",
        f"Visual Summary: {visual_context.visual_summary}",
    ]
    
    if visual_context.detected_text:
        lines.append("\nDetected Text:")
        for text_obj in visual_context.detected_text[:3]:
            lines.append(f"  • '{text_obj.text}' (confidence: {text_obj.confidence:.2f})")
    
    if visual_context.ui_elements:
        ui_types = [elem.element_type for elem in visual_context.ui_elements]
        lines.append(f"\nUI Elements: {', '.join(set(ui_types))}")
    
    return "\n".join(lines)


def format_game_state(game_state: dict) -> str:
    """Format game state into readable text"""
    if not game_state:
        return "No game state available"
    
    player = game_state.get('player', {})
    party = game_state.get('party', [])
    
    lines = [
        "PLAYER INFO:",
        f"  Location: Map {player.get('map', 0)} at ({player.get('x', 0)}, {player.get('y', 0)})",
        f"  Money: ${player.get('money', 0)}",
        f"  Badges: {player.get('badges', 0)}",
        "",
        "POKEMON PARTY:",
    ]
    
    if party:
        for i, pokemon in enumerate(party[:3]):
            species = pokemon.get('species', 0)
            level = pokemon.get('level', 0)
            hp = pokemon.get('hp', 0)
            max_hp = pokemon.get('max_hp', 1)
            hp_percent = (hp / max_hp * 100) if max_hp > 0 else 0
            
            lines.append(f"  {i+1}. Species #{species}, Level {level}")
            lines.append(f"     HP: {hp}/{max_hp} ({hp_percent:.0f}%)")
    else:
        lines.append("  No Pokemon in party")
    
    return "\n".join(lines)


def run_emulator_demo():
    """Run a simple demonstration of the emulator"""
    print("🎮 POKEMON CRYSTAL EMULATOR DEMONSTRATION")
    print("=" * 60)
    print("This demo will show you what the emulator is doing during training.")
    print("Each step will be displayed with detailed information.")
    print("=" * 60)
    
    # Initialize components
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return
    
    print("\n🔧 Initializing emulator and agent...")
    
    env = PyBoyPokemonCrystalEnv(
        rom_path=rom_path,
        save_state_path=None,
        debug_mode=True
    )
    
    agent = EnhancedLLMPokemonAgent(
        model_name="llama3.2:3b",
        use_vision=True
    )
    
    print("✅ Initialization complete!")
    
    try:
        # Reset environment
        print("\n🚀 Starting demonstration...")
        obs = env.reset()
        
        # Run a few demonstration steps
        for step in range(5):
            print(f"\n{'='*60}")
            print(f"STEP {step + 1} - {datetime.now().strftime('%H:%M:%S')}")
            print('='*60)
            
            # Get current state
            game_state = env.get_game_state()
            print("📊 CURRENT GAME STATE:")
            print(format_game_state(game_state))
            
            # Get screenshot and analyze it
            screenshot = env.get_screenshot()
            print(f"\n🖼️  SCREENSHOT CAPTURED: {screenshot.shape}")
            
            # Save screenshot as ASCII art
            ascii_filename = f"step_{step+1}_screen.txt"
            save_screenshot_as_text(screenshot, ascii_filename)
            print(f"   ASCII representation saved to: {ascii_filename}")
            
            # Process with computer vision
            visual_context = None
            if agent.vision_processor:
                print("\n👁️  PROCESSING WITH COMPUTER VISION...")
                visual_context = agent.vision_processor.process_screenshot(screenshot)
                print(format_visual_context(visual_context))
            
            # Make decision with LLM
            print("\n🤖 LLM MAKING DECISION...")
            action = agent.decide_next_action(
                state=game_state,
                screenshot=screenshot,
                recent_history=[]
            )
            action_name = agent.action_map[action]
            print(f"   Decision: {action_name} (action #{action})")
            
            # Execute action
            print(f"\n⚡ EXECUTING ACTION: {action_name}")
            step_result = env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            print(f"   Reward received: {reward:.2f}")
            print(f"   Episode done: {done}")
            
            # Save detailed step info
            step_info = {
                "step": step + 1,
                "timestamp": datetime.now().isoformat(),
                "game_state": game_state,
                "action_taken": action_name,
                "reward": reward,
                "visual_analysis": {
                    "screen_type": visual_context.screen_type if visual_context else None,
                    "detected_text": [{"text": t.text, "confidence": t.confidence} 
                                    for t in visual_context.detected_text] if visual_context else [],
                    "visual_summary": visual_context.visual_summary if visual_context else None
                },
                "episode_done": done
            }
            
            step_filename = f"step_{step+1}_info.json"
            with open(step_filename, 'w') as f:
                json.dump(step_info, f, indent=2)
            print(f"   Detailed info saved to: {step_filename}")
            
            if done:
                print("\n🏁 Episode completed!")
                break
            
            print(f"\n⏸️  Pausing for 2 seconds...")
            time.sleep(2)
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("Files created:")
        print("  • step_N_screen.txt - ASCII art of game screens")
        print("  • step_N_info.json - Detailed step information")
        print("\nWhat you just saw:")
        print("  ✅ Emulator running Pokemon Crystal")
        print("  ✅ Computer vision analyzing screenshots")
        print("  ✅ LLM making strategic decisions")
        print("  ✅ Actions being executed in the game")
        print("  ✅ Rewards being calculated")
        print("\nThis is exactly what happens during training!")
        
    finally:
        env.close()


def main():
    """Run the emulator demonstration"""
    run_emulator_demo()


if __name__ == "__main__":
    main()
