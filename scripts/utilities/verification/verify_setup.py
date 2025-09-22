#!/usr/bin/env python3
"""
verify_setup.py - Complete system verification before training
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print verification header"""
    print("=" * 70)
    print("🎮 POKÉMON CRYSTAL RL SYSTEM VERIFICATION")
    print("=" * 70)

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking System Requirements...")
    
    checks = []
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"❌ Python version too old: {python_version}")
        checks.append(False)
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 8:
            print(f"✅ Memory: {memory_gb:.1f}GB")
            checks.append(True)
        else:
            print(f"⚠️  Memory: {memory_gb:.1f}GB (8GB+ recommended)")
            checks.append(True)  # Warning, not failure
    except:
        print("⚠️  Could not check memory")
        checks.append(True)
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb >= 5:
            print(f"✅ Disk Space: {free_gb:.1f}GB free")
            checks.append(True)
        else:
            print(f"⚠️  Disk Space: {free_gb:.1f}GB free (5GB+ recommended)")
            checks.append(True)  # Warning, not failure
    except:
        print("⚠️  Could not check disk space")
        checks.append(True)
    
    return all(checks)

def check_dependencies():
    """Check Python dependencies"""
    print("\n📦 Checking Python Dependencies...")
    
    required_packages = {
        'stable_baselines3': 'Stable Baselines3',
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium',
        'numpy': 'NumPy',
        'flask': 'Flask',
        'plotly': 'Plotly',
        'psutil': 'PSUtil',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }
    
    checks = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
            checks.append(True)
        except ImportError:
            print(f"❌ {name} - Run: pip install {package}")
            checks.append(False)
    
    return all(checks)

def check_bizhawk():
    """Check BizHawk installation"""
    print("\n🎮 Checking BizHawk Installation...")
    
    checks = []
    
    # Check bizhawk command
    bizhawk_path = shutil.which('bizhawk')
    if bizhawk_path:
        print(f"✅ BizHawk command: {bizhawk_path}")
        checks.append(True)
    else:
        print("❌ BizHawk command not found in PATH")
        checks.append(False)
    
    # Check BizHawk directory
    bizhawk_dir = '/opt/bizhawk'
    if os.path.exists(bizhawk_dir):
        print(f"✅ BizHawk directory: {bizhawk_dir}")
        checks.append(True)
    else:
        print(f"❌ BizHawk directory not found: {bizhawk_dir}")
        checks.append(False)
    
    # Check EmuHawk executable
    emuhawk_exe = '/opt/bizhawk/EmuHawk.exe'
    if os.path.exists(emuhawk_exe):
        print(f"✅ EmuHawk executable: {emuhawk_exe}")
        checks.append(True)
    else:
        print(f"❌ EmuHawk executable not found: {emuhawk_exe}")
        checks.append(False)
    
    # Check Mono
    try:
        result = subprocess.run(['mono', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split()[4] if len(result.stdout.split()) > 4 else "unknown"
            print(f"✅ Mono runtime: {version}")
            checks.append(True)
        else:
            print("❌ Mono runtime not working")
            checks.append(False)
    except:
        print("❌ Mono runtime not found")
        checks.append(False)
    
    return all(checks)

def check_rom():
    """Check ROM file"""
    print("\n💾 Checking ROM File...")
    
    rom_path = 'pokecrystal.gbc'
    if os.path.exists(rom_path):
        # Check file size (Pokemon Crystal should be exactly 2MB)
        file_size = os.path.getsize(rom_path)
        if file_size == 2097152:  # 2MB in bytes
            print(f"✅ ROM file: {rom_path} (2MB - correct size)")
            return True
        else:
            size_mb = file_size / (1024*1024)
            print(f"⚠️  ROM file: {rom_path} ({size_mb:.1f}MB - unexpected size)")
            return True  # Still proceed, might work
    else:
        print(f"❌ ROM file not found: {rom_path}")
        return False

def check_project_structure():
    """Check project structure"""
    print("\n📁 Checking Project Structure...")
    
    required_files = [
        'lua_bridge/crystal_bridge.lua',
        'lua_bridge/json.lua',
        'python_agent/env.py',
        'python_agent/train.py',
        'python_agent/memory_map.py',
        'python_agent/utils.py',
        'monitor.py',
        'start_monitoring.py',
        'requirements.txt',
        'templates/dashboard.html'
    ]
    
    checks = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            checks.append(True)
        else:
            print(f"❌ {file_path}")
            checks.append(False)
    
    return all(checks)

def check_directories():
    """Check and create necessary directories"""
    print("\n📂 Checking Directories...")
    
    directories = [
        'models',
        'logs',
        'python_agent/models',
        'python_agent/logs',
        'templates',
        'static'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/ (created)")
    
    return True

def test_imports():
    """Test critical imports"""
    print("\n🧪 Testing Critical Imports...")
    
    checks = []
    
    tests = [
        ("Flask", "from flask import Flask"),
        ("Monitor", "from monitor import TrainingMonitor"),
        ("SB3", "from stable_baselines3 import PPO"),
    ]
    
    for test_name, import_code in tests:
        try:
            exec(import_code)
            print(f"✅ {test_name}")
            checks.append(True)
        except Exception as e:
            print(f"❌ {test_name}: {str(e)}")
            checks.append(False)
    
    # Test environment import from python_agent directory
    original_path = sys.path.copy()
    sys.path.insert(0, 'python_agent')
    
    try:
        from env import PokemonCrystalEnv
        print(f"✅ Environment")
        checks.append(True)
    except Exception as e:
        print(f"❌ Environment: {str(e)}")
        checks.append(False)
    finally:
        sys.path = original_path
    
    return all(checks)

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("ℹ️  GPU: Not available (CPU training will work)")
            return True
    except:
        print("ℹ️  GPU: Could not check (CPU training will work)")
        return True

def provide_next_steps():
    """Provide next steps"""
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    
    steps = [
        "1. Launch the monitoring system:",
        "   python3 start_monitoring.py",
        "",
        "2. Open your browser to:",
        "   Dashboard: http://localhost:5000",
        "   TensorBoard: http://localhost:6006",
        "",
        "3. Start your first training run:",
        "   - Configure parameters in the dashboard",
        "   - Click 'Start Training'",
        "   - Monitor progress in real-time!",
        "",
        "4. First training suggestions:",
        "   - Algorithm: PPO (recommended)",
        "   - Timesteps: 100,000 (for testing)",
        "   - Learning Rate: 0.0003",
        "   - Environments: 1",
        "",
        "5. Monitor the following:",
        "   - Agent learns basic movement",
        "   - Reward progression is positive",
        "   - No crashes or errors",
        "   - Memory usage stays reasonable"
    ]
    
    for step in steps:
        print(step)

def main():
    """Main verification function"""
    print_header()
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_system_requirements())
    all_checks.append(check_dependencies())
    all_checks.append(check_bizhawk())
    all_checks.append(check_rom())
    all_checks.append(check_project_structure())
    all_checks.append(check_directories())
    all_checks.append(test_imports())
    all_checks.append(check_gpu())
    
    # Final summary
    print("\n" + "=" * 70)
    if all(all_checks):
        print("🎉 VERIFICATION COMPLETE - SYSTEM READY FOR TRAINING!")
        print("=" * 70)
        provide_next_steps()
        return 0
    else:
        print("❌ VERIFICATION FAILED - PLEASE FIX ISSUES ABOVE")
        print("=" * 70)
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Install BizHawk: Follow README installation instructions")
        print("- Check ROM file is in project directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
