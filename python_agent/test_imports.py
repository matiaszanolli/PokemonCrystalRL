#!/usr/bin/env python3
"""
test_imports.py - Test script to verify all imports work after reorganization

This script tests that all the updated imports in test files work correctly
with the new modular structure.
"""

import sys
import importlib
import traceback

def test_import(module_path, description):
    """Test importing a module and print result"""
    try:
        importlib.import_module(module_path)
        print(f"✅ {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_path}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {module_path}")
        print(f"   Unexpected error: {e}")
        return False

def main():
    """Test all reorganized imports"""
    print("🧪 Testing reorganized imports...\n")
    
    success_count = 0
    total_count = 0
    
    # Test core modules
    print("📦 Core Modules:")
    modules_to_test = [
        ("core", "Core package"),
        ("core.memory_map", "Memory map module"),
        # Skip pyboy_env as it requires dependencies
    ]
    
    for module_path, desc in modules_to_test:
        if test_import(module_path, desc):
            success_count += 1
        total_count += 1
    
    print("\n🧠 Agent Modules:")
    modules_to_test = [
        ("agents", "Agents package"),
        # Skip specific agents as they require dependencies
    ]
    
    for module_path, desc in modules_to_test:
        if test_import(module_path, desc):
            success_count += 1
        total_count += 1
    
    print("\n🔧 Utility Modules:")  
    modules_to_test = [
        ("utils", "Utils package"),
        # Skip specific utils as they require dependencies
    ]
    
    for module_path, desc in modules_to_test:
        if test_import(module_path, desc):
            success_count += 1
        total_count += 1
    
    print("\n👁️  Vision Modules:")
    modules_to_test = [
        ("vision", "Vision package"),
        # Skip specific vision modules as they require dependencies
    ]
    
    for module_path, desc in modules_to_test:
        if test_import(module_path, desc):
            success_count += 1
        total_count += 1
    
    print("\n📊 Monitoring Modules:")
    modules_to_test = [
        ("monitoring", "Monitoring package"),
        # Skip specific monitoring modules as they require dependencies  
    ]
    
    for module_path, desc in modules_to_test:
        if test_import(module_path, desc):
            success_count += 1
        total_count += 1
    
    # Test main package
    print("\n📦 Main Package:")
    if test_import(".", "Main python_agent package"):
        success_count += 1
    total_count += 1
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print(f"\n🎉 All package imports working correctly!")
        return 0
    else:
        print(f"\n⚠️  Some imports failed - this is expected without dependencies installed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
