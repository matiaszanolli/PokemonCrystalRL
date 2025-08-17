#!/usr/bin/env python3
"""
Test LLM Performance Diagnostics

This script tests the LLM performance to identify bottlenecks.
"""

import time
import ollama
import sys
import json


def test_ollama_availability():
    """Test if Ollama server is running and responsive"""
    print("🔍 Testing Ollama server availability...")
    try:
        start_time = time.time()
        response = ollama.list()
        response_time = time.time() - start_time
        
        print(f"✅ Ollama server responsive in {response_time:.3f}s")
        
        # List available models
        models = response.get('models', [])
        print(f"📋 Available models: {len(models)}")
        for model in models[:5]:  # Show first 5
            name = model.get('name', 'unknown')
            size = model.get('size', 0)
            print(f"  - {name} ({size // (1024*1024)}MB)")
        
        return True, response_time
    except Exception as e:
        print(f"❌ Ollama server error: {e}")
        return False, 0.0


def test_model_availability(model_name: str):
    """Test if specific model is available and loaded"""
    print(f"🔍 Testing model availability: {model_name}")
    try:
        start_time = time.time()
        model_info = ollama.show(model_name)
        response_time = time.time() - start_time
        
        print(f"✅ Model {model_name} available in {response_time:.3f}s")
        
        # Show model details
        if 'modelfile' in model_info:
            size = len(model_info.get('modelfile', ''))
            print(f"   Model size: ~{size // 1024}KB config")
        
        return True, response_time
    except Exception as e:
        print(f"❌ Model {model_name} error: {e}")
        
        # Try to pull the model
        print(f"📥 Attempting to pull {model_name}...")
        try:
            pull_start = time.time()
            ollama.pull(model_name)
            pull_time = time.time() - pull_start
            print(f"✅ Model {model_name} pulled in {pull_time:.1f}s")
            return True, pull_time
        except Exception as pull_error:
            print(f"❌ Failed to pull {model_name}: {pull_error}")
            return False, 0.0


def test_basic_generation(model_name: str, num_tests: int = 5):
    """Test basic LLM generation performance"""
    print(f"🔍 Testing LLM generation performance ({num_tests} tests)...")
    
    test_prompt = "Pokemon Crystal Game Bot\n\nState: title_screen\nGuidance: Press 7=START to begin\n\nControls:\n1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT\n\nChoose action number (1-8):"
    
    times = []
    successful = 0
    
    for i in range(num_tests):
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=model_name,
                prompt=test_prompt,
                options={
                    'num_predict': 3,
                    'temperature': 0.7,
                    'top_k': 8,
                    'timeout': 10
                }
            )
            
            generation_time = time.time() - start_time
            times.append(generation_time)
            successful += 1
            
            # Parse response
            text = response.get('response', '').strip()
            action = None
            for char in text:
                if char.isdigit() and '1' <= char <= '8':
                    action = int(char)
                    break
            
            print(f"  Test {i+1}: {generation_time:.2f}s -> '{text[:20]}...' -> Action: {action}")
            
            if generation_time > 5.0:
                print(f"    ⚠️  SLOW CALL: {generation_time:.2f}s")
                
        except Exception as e:
            print(f"  Test {i+1}: FAILED - {str(e)[:50]}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Generation Performance Summary:")
        print(f"   Successful calls: {successful}/{num_tests}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Fastest: {min_time:.2f}s")
        print(f"   Slowest: {max_time:.2f}s")
        
        if avg_time > 3.0:
            print(f"   ⚠️  Average time is slow (>3s)")
        
        return avg_time, successful, times
    else:
        print("❌ No successful generations")
        return 0.0, 0, []


def test_concurrent_generation(model_name: str):
    """Test if model can handle rapid sequential calls"""
    print(f"🔍 Testing rapid sequential calls...")
    
    prompt = "Choose 1-8:"
    times = []
    
    for i in range(3):
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={'num_predict': 1, 'temperature': 0.5, 'timeout': 5}
            )
            
            generation_time = time.time() - start_time
            times.append(generation_time)
            
            print(f"  Rapid call {i+1}: {generation_time:.2f}s")
            
        except Exception as e:
            print(f"  Rapid call {i+1}: FAILED - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"   Rapid call average: {avg_time:.2f}s")
        return avg_time
    return 0.0


def test_adaptive_interval_simulation():
    """Simulate adaptive interval behavior based on measured performance"""
    print("🔍 Simulating adaptive LLM interval optimization...")
    
    # Test different LLM response time scenarios
    scenarios = [
        {"name": "Fast LLM", "times": [0.5, 0.6, 0.7, 0.8, 0.9], "expected_interval": "stable or decreased"},
        {"name": "Slow LLM", "times": [4.0, 4.5, 5.0, 3.8, 4.2], "expected_interval": "increased"},
        {"name": "Variable LLM", "times": [1.0, 3.5, 0.8, 4.1, 2.2], "expected_interval": "adaptive"},
    ]
    
    for scenario in scenarios:
        print(f"\n  📊 Scenario: {scenario['name']}")
        
        # Simulate adaptive logic
        llm_interval = 10  # Start with default
        response_times = []
        
        for i, response_time in enumerate(scenario['times']):
            response_times.append(response_time)
            
            # Keep last 10 for moving average (simplified)
            recent_times = response_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            old_interval = llm_interval
            
            # Apply adaptive logic
            if len(response_times) >= 10 and len(response_times) % 10 == 0:
                if avg_time > 3.0 and llm_interval < 50:
                    llm_interval = min(50, int(llm_interval * 1.5))
                elif avg_time < 1.0 and llm_interval > 10:
                    llm_interval = max(10, int(llm_interval * 0.8))
            
            print(f"    Call {i+1}: {response_time:.1f}s (avg: {avg_time:.1f}s, interval: {llm_interval})")
            
            if llm_interval != old_interval:
                direction = "↑" if llm_interval > old_interval else "↓"
                print(f"      {direction} Interval adjusted: {old_interval} → {llm_interval}")
        
        # Final assessment
        final_avg = sum(scenario['times']) / len(scenario['times'])
        print(f"    Result: Final interval {llm_interval}, avg time {final_avg:.1f}s - {scenario['expected_interval']}")


def run_diagnostics():
    """Run comprehensive LLM performance diagnostics"""
    print("🚀 LLM Performance Diagnostics")
    print("=" * 50)
    
    # Test 1: Ollama server availability
    server_ok, server_time = test_ollama_availability()
    if not server_ok:
        print("❌ Ollama server not available. Please start Ollama first.")
        return
    
    print()
    
    # Test models from fastest to slowest
    models_to_test = [
        'smollm2:1.7b',
        'llama3.2:1b', 
        'llama3.2:3b',
        'qwen2.5:3b'
    ]
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"🔍 TESTING MODEL: {model_name}")
        print(f"{'='*50}")
        
        # Test 2: Model availability
        model_ok, model_time = test_model_availability(model_name)
        if not model_ok:
            print(f"❌ Skipping {model_name} - not available")
            continue
        
        print()
        
        # Test 3: Basic generation performance
        avg_time, successful, times = test_basic_generation(model_name, num_tests=5)
        
        print()
        
        # Test 4: Rapid sequential calls
        rapid_avg = test_concurrent_generation(model_name)
        
        # Summary for this model
        print(f"\n📋 MODEL SUMMARY: {model_name}")
        print(f"   Availability check: {model_time:.3f}s")
        if successful > 0:
            print(f"   Average generation: {avg_time:.2f}s")
            print(f"   Rapid call average: {rapid_avg:.2f}s")
            
            # Performance rating
            if avg_time < 1.0:
                rating = "🟢 EXCELLENT"
            elif avg_time < 2.0:
                rating = "🟡 GOOD" 
            elif avg_time < 4.0:
                rating = "🟠 ACCEPTABLE"
            else:
                rating = "🔴 SLOW"
            
            print(f"   Performance rating: {rating}")
            
            # Recommendations
            if avg_time > 3.0:
                print(f"   💡 Consider using a smaller/faster model for real-time training")
        else:
            print(f"   ❌ No successful generations")
    
    # Test 5: Adaptive interval simulation
    print(f"\n{'='*50}")
    test_adaptive_interval_simulation()
    
    print(f"\n{'='*50}")
    print("🎯 RECOMMENDATIONS:")
    print("   - For training with LLM calls every 10 actions:")
    print("     * Target: <2s per call (allows ~5 actions/second)")  
    print("     * Minimum: <4s per call (allows ~2.5 actions/second)")
    print("   - If calls are >4s, trainer will auto-increase llm_interval")
    print("   - If calls are <1s, trainer will auto-decrease llm_interval")
    print("   - Adaptive intervals help maintain target performance automatically")
    print("   - Check system load, network, and Ollama configuration for optimal results")


if __name__ == "__main__":
    run_diagnostics()
