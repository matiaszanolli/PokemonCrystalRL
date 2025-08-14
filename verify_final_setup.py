#!/usr/bin/env python3
"""
Final setup verification for Pokemon Crystal Local LLM Agent
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirement(name, check_func, fix_suggestion=None):
    """Check a requirement and report status"""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name}")
            if fix_suggestion:
                print(f"   💡 Fix: {fix_suggestion}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        if fix_suggestion:
            print(f"   💡 Fix: {fix_suggestion}")
        return False

def main():
    print("🔍 Pokemon Crystal Local LLM Agent - Setup Verification")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check Python dependencies
    total_checks += 1
    def check_python_deps():
        try:
            import pyboy
            import ollama
            import sqlite3
            import numpy as np
            return True
        except ImportError as e:
            return False
    
    if check_requirement("Python dependencies (pyboy, ollama, numpy)", 
                        check_python_deps,
                        "pip install pyboy ollama numpy"):
        checks_passed += 1
    
    # Check Ollama installation
    total_checks += 1
    def check_ollama():
        return shutil.which("ollama") is not None
    
    if check_requirement("Ollama installation",
                        check_ollama,
                        "curl -fsSL https://ollama.com/install.sh | sh"):
        checks_passed += 1
    
    # Check Llama model
    total_checks += 1
    def check_llama_model():
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            return "llama3.2:3b" in result.stdout
        except:
            return False
    
    if check_requirement("Llama 3.2 3B model",
                        check_llama_model,
                        "ollama pull llama3.2:3b"):
        checks_passed += 1
    
    # Check ROM file
    total_checks += 1
    def check_rom():
        return os.path.exists("pokecrystal.gbc") and os.path.getsize("pokecrystal.gbc") > 1000000
    
    if check_requirement("Pokemon Crystal ROM",
                        check_rom,
                        "Place your Pokemon Crystal ROM as 'pokecrystal.gbc'"):
        checks_passed += 1
    
    # Check save state
    total_checks += 1
    def check_save_state():
        return os.path.exists("pokemon_crystal_intro.state")
    
    if check_requirement("PyBoy save state",
                        check_save_state,
                        "Save state exists (should be auto-created)"):
        checks_passed += 1
    
    # Check main agent files
    total_checks += 1
    def check_agent_files():
        required_files = [
            "python_agent/local_llm_agent.py",
            "python_agent/llm_play.py", 
            "python_agent/pyboy_env.py"
        ]
        return all(os.path.exists(f) for f in required_files)
    
    if check_requirement("Agent files",
                        check_agent_files,
                        "Core agent files should be present"):
        checks_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Verification Results: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL CHECKS PASSED - Ready to run!")
        print("\n🚀 Quick Start:")
        print("cd python_agent")
        print("python llm_play.py --no-headless --max-steps 500")
        print("\n🎮 Or for fast automated play:")
        print("python llm_play.py --fast --max-steps 10000")
        return True
    else:
        print("⚠️  Some requirements missing - please fix the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
