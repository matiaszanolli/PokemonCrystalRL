#!/usr/bin/env python3
"""
Pokemon Crystal RL Training Examples

This script provides examples of how to run different training modes.
Run this to see available options and example commands.
"""

def show_usage():
    print("🎮 Pokemon Crystal RL Training Examples")
    print("=" * 50)
    print()
    
    print("📁 Available Scripts:")
    print("  run_pokemon_trainer.py  - Main unified training script")
    print()
    
    print("🎯 Training Modes:")
    print("  fast_monitored  - Real-time training with web monitoring")
    print("  curriculum      - Progressive skill-based learning")
    print("  ultra_fast      - Maximum speed rule-based training")
    print("  custom          - User-defined configuration")
    print()
    
    print("🤖 Available LLM Models:")
    print("  smollm2:1.7b    - Fast, lightweight model (default)")
    print("  llama3.2:1b    - Compact Llama model")
    print("  llama3.2:3b    - Larger Llama model")
    print("  --no-llm        - Rule-based only (fastest)")
    print()
    
    print("💡 Example Commands:")
    print()
    
    print("1. Quick Start (Fast Monitored with Web UI):")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --actions 500 --web --debug")
    print()
    
    print("2. Ultra Fast Training (No LLM, Maximum Speed):")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --no-llm --actions 2000")
    print()
    
    print("3. Curriculum Learning (Progressive Training):")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode curriculum --episodes 10 --web")
    print()
    
    print("4. Custom Configuration:")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode custom --model llama3.2:1b --actions 1000 --llm-interval 5 --web --port 8080")
    print()
    
    print("5. Windowed Mode (See Game Window):")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --windowed --web")
    print()
    
    print("6. Load from Save State:")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --save-state ../roms/pokemon_crystal.gbc.state --web")
    print()
    
    print("🌐 Web Interface:")
    print("   When --web is enabled, open: http://localhost:8080")
    print("   Features: Live screenshots, training stats, OCR text, performance metrics")
    print()
    
    print("⚙️ Key Parameters:")
    print("   --actions N         - Maximum actions to take")
    print("   --episodes N        - Maximum episodes (for curriculum mode)")
    print("   --llm-interval N    - Actions between LLM calls (higher = faster)")
    print("   --frames-per-action - Game speed (24=standard, 16=faster, 8=legacy)")
    print("   --debug             - Enable detailed logging")
    print("   --no-capture        - Disable screen capture (faster)")
    print()
    
    print("🚨 Common Issues:")
    print("   - File not found: Use 'run_pokemon_trainer.py' not 'enhanced_monitored_training.py'")
    print("   - ROM not found: Make sure ROM is at '../roms/pokemon_crystal.gbc'")
    print("   - Core dump at end: Normal PyBoy cleanup issue, doesn't affect training")
    print("   - Permission errors: Make sure ROM file is readable")
    print()

if __name__ == "__main__":
    show_usage()
