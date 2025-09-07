"""
test_env_fix.py - Test the fixed PyBoyPokemonCrystalEnv

This test verifies that the environment can be created and used
without the debug_mode parameter error.
"""

import numpy as np
from environments.pyboy_env import PyBoyPokemonCrystalEnv

def test_env_creation():
    """Test environment creation with debug_mode parameter"""
    print("🧪 Testing PyBoyPokemonCrystalEnv creation...")
    
    # Test with debug_mode=True
    env = PyBoyPokemonCrystalEnv(
        rom_path="/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc",
        save_state_path=None,
        debug_mode=True,
        headless=True
    )
    print("✅ Environment created successfully with debug_mode=True")
    
    # Test basic properties
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test methods exist
    print(f"Has get_game_state: {hasattr(env, 'get_game_state')}")
    print(f"Has get_screenshot: {hasattr(env, 'get_screenshot')}")
    
    # Test basic environment properties exist
    assert hasattr(env, 'action_space'), "Environment should have action_space"
    assert hasattr(env, 'observation_space'), "Environment should have observation_space"
    
    # Test screenshot method if available
    if hasattr(env, 'get_screenshot'):
        screenshot = env.get_screenshot()
        print(f"Screenshot shape: {screenshot.shape}")
        print(f"Screenshot dtype: {screenshot.dtype}")
        assert isinstance(screenshot, np.ndarray), "Screenshot should be a numpy array"
    
    # Test game state method if available
    if hasattr(env, 'get_game_state'):
        game_state = env.get_game_state()
        print(f"Game state type: {type(game_state)}")
        print(f"Game state keys: {list(game_state.keys()) if game_state else 'Empty'}")
        assert isinstance(game_state, dict), "Game state should be a dictionary"
    
    print("✅ All environment methods work correctly")

def test_env_compatibility():
    """Test environment compatibility with training system"""
    print("\n🧪 Testing environment compatibility with training system...")
    
    from environments.pyboy_env import PyBoyPokemonCrystalEnv
    from vision.vision_enhanced_training import VisionEnhancedTrainingSession
    
    # This should not raise any parameter errors
    session = VisionEnhancedTrainingSession(
        rom_path="/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc",
        save_state_path=None,
        model_name="llama3.2:3b",
        max_steps_per_episode=100,
        screenshot_interval=10
    )
    
    print("✅ Training session created successfully")
    print("✅ No debug_mode parameter errors")
    assert session is not None, "Training session should be created successfully"
    # Check for some expected methods/attributes
    expected_attrs = ['model_name', 'max_steps', 'screenshot_interval']
    for attr in expected_attrs:
        assert hasattr(session, attr), f"Training session should have {attr} attribute"

def main():
    """Run all tests"""
    print("🔧 Testing Fixed PyBoyPokemonCrystalEnv")
    print("=" * 50)
    
    test1_passed = test_env_creation()
    test2_passed = test_env_compatibility()
    
    print("\n📋 TEST SUMMARY")
    print("-" * 20)
    print(f"Environment Creation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Training Compatibility: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The PyBoyPokemonCrystalEnv fix is working correctly.")
        print("You can now run the training system without parameter errors.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
