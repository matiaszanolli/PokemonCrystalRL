"""
system_status.py - Pokemon Crystal RL System Status and Overview

This script provides an overview of the complete training system,
tests all components, and shows current status.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    dependencies = {}
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import cv2
        dependencies['opencv-python'] = True
    except ImportError:
        dependencies['opencv-python'] = False
    
    try:
        import easyocr
        dependencies['easyocr'] = True
    except ImportError:
        dependencies['easyocr'] = False
    
    try:
        import PIL
        dependencies['pillow'] = True
    except ImportError:
        dependencies['pillow'] = False
    
    try:
        import ollama
        dependencies['ollama'] = True
    except ImportError:
        dependencies['ollama'] = False
    
    try:
        from pyboy import PyBoy
        dependencies['pyboy'] = True
    except ImportError:
        dependencies['pyboy'] = False
    
    try:
        import stable_baselines3
        dependencies['stable-baselines3'] = True
    except ImportError:
        dependencies['stable-baselines3'] = False
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False
    
    return dependencies

def check_ollama_service() -> Dict[str, Any]:
    """Check Ollama service and available models"""
    ollama_status = {
        'service_running': False,
        'models_available': [],
        'recommended_model': 'llama3.2:3b'
    }
    
    try:
        import ollama
        # Try to list models to check if service is running
        models = ollama.list()
        ollama_status['service_running'] = True
        ollama_status['models_available'] = [model['name'] for model in models.get('models', [])]
    except Exception as e:
        ollama_status['error'] = str(e)
    
    return ollama_status

def check_system_components() -> Dict[str, Dict[str, Any]]:
    """Check the status of all system components"""
    components = {}
    
    # PyBoy Environment
    try:
        from pyboy_env import PyBoyPokemonCrystalEnv
        components['pyboy_env'] = {
            'status': 'available',
            'description': 'Pokemon Crystal PyBoy environment with Gym interface'
        }
    except Exception as e:
        components['pyboy_env'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Vision Processor
    try:
        from vision_processor import PokemonVisionProcessor
        components['vision_processor'] = {
            'status': 'available',
            'description': 'Computer vision for screenshot analysis'
        }
    except Exception as e:
        components['vision_processor'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Enhanced LLM Agent
    try:
        from enhanced_llm_agent import EnhancedLLMPokemonAgent
        components['enhanced_llm_agent'] = {
            'status': 'available',
            'description': 'LLM agent with computer vision integration'
        }
    except Exception as e:
        components['enhanced_llm_agent'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Training System
    try:
        from vision_enhanced_training import VisionEnhancedTrainingSession
        components['vision_enhanced_training'] = {
            'status': 'available',
            'description': 'Complete training pipeline with analytics'
        }
    except Exception as e:
        components['vision_enhanced_training'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Legacy components
    try:
        from local_llm_agent import LocalLLMPokemonAgent
        components['local_llm_agent'] = {
            'status': 'available',
            'description': 'Legacy text-only LLM agent'
        }
    except Exception as e:
        components['local_llm_agent'] = {
            'status': 'error',
            'error': str(e)
        }
    
    return components

def check_game_files() -> Dict[str, Any]:
    """Check for required game files"""
    files_status = {}
    
    # ROM files
    rom_dir = "/mnt/data/src/pokemon_crystal_rl/roms"
    if os.path.exists(rom_dir):
        rom_files = [f for f in os.listdir(rom_dir) if f.endswith('.gbc')]
        files_status['roms'] = {
            'directory_exists': True,
            'files_found': rom_files,
            'count': len(rom_files)
        }
    else:
        files_status['roms'] = {
            'directory_exists': False,
            'files_found': [],
            'count': 0
        }
    
    # Save states
    save_dir = "/mnt/data/src/pokemon_crystal_rl/save_states"
    if os.path.exists(save_dir):
        save_files = [f for f in os.listdir(save_dir) if f.endswith('.state')]
        files_status['save_states'] = {
            'directory_exists': True,
            'files_found': save_files,
            'count': len(save_files)
        }
    else:
        files_status['save_states'] = {
            'directory_exists': False,
            'files_found': [],
            'count': 0
        }
    
    return files_status

def run_component_tests() -> Dict[str, Dict[str, Any]]:
    """Run basic tests on available components"""
    test_results = {}
    
    # Test Vision Processor
    print("🧪 Testing Vision Processor...")
    try:
        from vision_processor import PokemonVisionProcessor
        import numpy as np
        
        # Create test processor
        processor = PokemonVisionProcessor()
        
        # Create mock screenshot
        test_screenshot = np.zeros((144, 160, 3), dtype=np.uint8)
        test_screenshot.fill(255)  # White background
        
        # Process screenshot
        visual_context = processor.process_screenshot(test_screenshot)
        
        test_results['vision_processor'] = {
            'status': 'passed',
            'screen_type_detected': visual_context.screen_type,
            'ui_elements_found': len(visual_context.ui_elements),
            'text_detected': len(visual_context.detected_text)
        }
        print("✅ Vision Processor test passed")
        
    except Exception as e:
        test_results['vision_processor'] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"❌ Vision Processor test failed: {e}")
    
    # Test Enhanced LLM Agent
    print("🧪 Testing Enhanced LLM Agent...")
    try:
        from enhanced_llm_agent import EnhancedLLMPokemonAgent
        
        agent = EnhancedLLMPokemonAgent(use_vision=False)  # Disable vision for quick test
        
        # Mock game state
        mock_state = {
            "player": {"x": 5, "y": 10, "map": 1, "money": 3000, "badges": 0},
            "party": [{"species": 155, "level": 10, "hp": 30, "max_hp": 30}]
        }
        
        # Test decision making
        action = agent.decide_next_action(mock_state)
        
        test_results['enhanced_llm_agent'] = {
            'status': 'passed',
            'action_decided': agent.action_map[action],
            'memory_system': 'working'
        }
        print("✅ Enhanced LLM Agent test passed")
        
    except Exception as e:
        test_results['enhanced_llm_agent'] = {
            'status': 'failed',
            'error': str(e)
        }
        print(f"❌ Enhanced LLM Agent test failed: {e}")
    
    return test_results

def print_system_status():
    """Print comprehensive system status"""
    print("🎮 Pokemon Crystal Vision-Enhanced RL Training System")
    print("=" * 60)
    print(f"Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Dependencies
    print("📦 DEPENDENCIES")
    print("-" * 20)
    deps = check_dependencies()
    for dep, status in deps.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    missing_deps = [dep for dep, status in deps.items() if not status]
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
    print()
    
    # Ollama Service
    print("🤖 OLLAMA SERVICE")
    print("-" * 20)
    ollama_status = check_ollama_service()
    if ollama_status['service_running']:
        print("  ✅ Ollama service is running")
        print(f"  📄 Models available: {len(ollama_status['models_available'])}")
        for model in ollama_status['models_available']:
            recommended = " (recommended)" if model == ollama_status['recommended_model'] else ""
            print(f"    • {model}{recommended}")
        
        if ollama_status['recommended_model'] not in ollama_status['models_available']:
            print(f"  ⚠️  Recommended model '{ollama_status['recommended_model']}' not found")
            print(f"  Install with: ollama pull {ollama_status['recommended_model']}")
    else:
        print("  ❌ Ollama service not running")
        print("  Start with: ollama serve")
        if 'error' in ollama_status:
            print(f"  Error: {ollama_status['error']}")
    print()
    
    # System Components
    print("🔧 SYSTEM COMPONENTS")
    print("-" * 20)
    components = check_system_components()
    for comp_name, comp_info in components.items():
        if comp_info['status'] == 'available':
            print(f"  ✅ {comp_name} - {comp_info['description']}")
        else:
            print(f"  ❌ {comp_name} - Error: {comp_info.get('error', 'Unknown error')}")
    print()
    
    # Game Files
    print("🎮 GAME FILES")
    print("-" * 20)
    files = check_game_files()
    
    # ROM files
    rom_status = files['roms']
    if rom_status['directory_exists']:
        if rom_status['count'] > 0:
            print(f"  ✅ ROM files: {rom_status['count']} found")
            for rom in rom_status['files_found']:
                print(f"    • {rom}")
        else:
            print("  ⚠️  ROM directory exists but no .gbc files found")
    else:
        print("  ❌ ROM directory not found")
    
    # Save states
    save_status = files['save_states']
    if save_status['directory_exists']:
        if save_status['count'] > 0:
            print(f"  ✅ Save states: {save_status['count']} found")
            for save in save_status['files_found']:
                print(f"    • {save}")
        else:
            print("  ℹ️  Save states directory exists but no .state files found (optional)")
    else:
        print("  ℹ️  Save states directory not found (optional)")
    print()
    
    # Component Tests
    print("🧪 COMPONENT TESTS")
    print("-" * 20)
    test_results = run_component_tests()
    
    for component, result in test_results.items():
        if result['status'] == 'passed':
            print(f"  ✅ {component} - Test passed")
        else:
            print(f"  ❌ {component} - Test failed: {result.get('error', 'Unknown error')}")
    print()
    
    # Overall System Status
    all_deps_ok = all(deps.values())
    ollama_ok = ollama_status['service_running']
    components_ok = all(info['status'] == 'available' for info in components.values())
    rom_ok = files['roms']['count'] > 0
    tests_ok = all(result['status'] == 'passed' for result in test_results.values())
    
    print("🏁 OVERALL STATUS")
    print("-" * 20)
    if all([all_deps_ok, ollama_ok, components_ok, rom_ok, tests_ok]):
        print("  🎉 System is ready for training!")
        print("  Run: python vision_enhanced_training.py")
    else:
        print("  ⚠️  System has issues that need to be resolved:")
        if not all_deps_ok:
            print("    • Install missing dependencies")
        if not ollama_ok:
            print("    • Start Ollama service and install models")
        if not components_ok:
            print("    • Fix component import errors")
        if not rom_ok:
            print("    • Add Pokemon Crystal ROM file")
        if not tests_ok:
            print("    • Resolve component test failures")
    print()
    
    # Usage Examples
    print("🚀 QUICK START")
    print("-" * 20)
    print("  # Test individual components:")
    print("  python vision_processor.py")
    print("  python enhanced_llm_agent.py")
    print()
    print("  # Run training session:")
    print("  python vision_enhanced_training.py")
    print()
    print("  # Check system status:")
    print("  python system_status.py")
    print()

def main():
    """Main function to run system status check"""
    print_system_status()

if __name__ == "__main__":
    main()
