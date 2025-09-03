#!/usr/bin/env python3
"""
test_installation.py - Simple test script to verify BizHawk installation and setup
"""

import subprocess
import sys
import os
import shutil

def test_bizhawk_installed():
    """Test if BizHawk is installed and accessible"""
    print("Testing BizHawk installation...")
    
    # Check if bizhawk command is available
    bizhawk_path = shutil.which('bizhawk')
    if not bizhawk_path:
        print("❌ BizHawk command not found in PATH")
        return False
    
    print(f"✅ BizHawk found at: {bizhawk_path}")
    
    # Check if the actual executable exists
    if not os.path.exists('/opt/bizhawk/EmuHawk.exe'):
        print("❌ EmuHawk.exe not found in /opt/bizhawk/")
        return False
    
    print("✅ EmuHawk.exe found")
    return True

def test_mono_installed():
    """Test if Mono is installed and working"""
    print("Testing Mono installation...")
    
    try:
        result = subprocess.run(['mono', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Mono installed: {result.stdout.split()[4]}")
            return True
        else:
            print("❌ Mono not working properly")
            return False
    except FileNotFoundError:
        print("❌ Mono not found")
        return False

def test_python_dependencies():
    """Test if Python dependencies are available"""
    print("Testing Python dependencies...")
    
    required_packages = [
        'numpy',
        'torch',
        'stable_baselines3',
        'gymnasium',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_lua_scripts():
    """Test if Lua scripts are present"""
    print("Testing Lua scripts...")
    
    lua_bridge = '/mnt/data/src/pokemon_crystal_rl/lua_bridge/crystal_bridge.lua'
    json_lua = '/mnt/data/src/pokemon_crystal_rl/lua_bridge/json.lua'
    
    if not os.path.exists(lua_bridge):
        print("❌ crystal_bridge.lua not found")
        return False
    
    if not os.path.exists(json_lua):
        print("❌ json.lua not found")
        return False
    
    print("✅ Lua scripts found")
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("Testing project structure...")
    
    base_dir = '/mnt/data/src/pokemon_crystal_rl'
    required_files = [
        'README.md',
        'requirements.txt',
        'lua_bridge/crystal_bridge.lua',
        'lua_bridge/json.lua',
        'python_agent/env.py',
        'python_agent/train.py',
        'python_agent/memory_map.py',
        'python_agent/utils.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Run all tests"""
    print("🧪 Testing Pokémon Crystal RL installation\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("BizHawk Installation", test_bizhawk_installed),
        ("Mono Runtime", test_mono_installed),
        ("Lua Scripts", test_lua_scripts),
        ("Python Dependencies", test_python_dependencies),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 INSTALLATION TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Obtain a legal Pokémon Crystal ROM file")
        print("2. Run: cd python_agent && python train.py --rom-path /path/to/your/rom.gbc")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Make sure BizHawk was extracted properly to /opt/bizhawk/")
        print("- Ensure Mono is installed: sudo apt install mono-complete")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
