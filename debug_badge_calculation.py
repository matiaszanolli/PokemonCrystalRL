#!/usr/bin/env python3
"""
Debug script to analyze the suspicious badge reward issue
"""

import json
from pathlib import Path
from core.memory_map import get_badges_earned, DERIVED_VALUES

def analyze_badge_calculation():
    """Analyze how badges_total is being calculated"""
    
    print("🔍 Badge Calculation Analysis")
    print("=" * 50)
    
    # Test the badge calculation function with various inputs
    test_cases = [
        (0, 0, "No badges"),
        (255, 0, "All Johto badges (invalid)"),
        (0, 255, "All Kanto badges (invalid)"),
        (255, 255, "All badges (invalid)"),
        (1, 0, "First Johto badge"),
        (0, 1, "First Kanto badge"),
        (128, 0, "Last Johto badge"),
        (0, 128, "Last Kanto badge"),
        (15, 0, "4 Johto badges"),
        (0, 15, "4 Kanto badges"),
    ]
    
    for badges, kanto, description in test_cases:
        # Test the derived value calculation
        state = {'badges': badges, 'kanto_badges': kanto}
        total = DERIVED_VALUES['badges_total'](state)
        earned_list = get_badges_earned(badges, kanto)
        
        print(f"  {description}:")
        print(f"    badges={badges:02x} ({badges:08b}), kanto={kanto:02x} ({kanto:08b})")
        print(f"    badges_total={total}, earned={len(earned_list)}, list={earned_list}")
        print()
    
    # Check for the specific combination that could cause 8 badges
    print("🔍 Looking for 8-badge combinations:")
    for badges in range(256):
        for kanto in range(256):
            state = {'badges': badges, 'kanto_badges': kanto}
            total = DERIVED_VALUES['badges_total'](state)
            if total == 8:
                earned_list = get_badges_earned(badges, kanto)
                print(f"  Found 8 badges: badges={badges:02x}, kanto={kanto:02x}, earned={earned_list}")
                break
        else:
            continue
        break
    
    # Analyze recent log files for suspicious badge data
    print("📊 Analyzing Recent Log Files")
    print("=" * 30)
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        # Find most recent LLM decisions file
        json_files = list(logs_dir.glob("llm_decisions_*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Analyzing: {latest_file}")
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                suspicious_entries = []
                for i, entry in enumerate(data):
                    game_state = entry.get('game_state', {})
                    badges = game_state.get('badges', 0)
                    kanto_badges = game_state.get('kanto_badges', 0)
                    badges_total = game_state.get('badges_total', 0)
                    
                    # Look for non-zero badge counts
                    if badges_total > 0 or badges > 0 or kanto_badges > 0:
                        suspicious_entries.append({
                            'index': i,
                            'timestamp': entry.get('timestamp'),
                            'badges': badges,
                            'kanto_badges': kanto_badges,
                            'badges_total': badges_total,
                            'action': entry.get('action')
                        })
                
                if suspicious_entries:
                    print(f"Found {len(suspicious_entries)} entries with non-zero badge data:")
                    for entry in suspicious_entries[:10]:  # Show first 10
                        print(f"  Entry {entry['index']}: badges={entry['badges']:02x}, kanto={entry['kanto_badges']:02x}, total={entry['badges_total']}, action={entry['action']}")
                else:
                    print("No suspicious badge entries found in log file")
                    
            except Exception as e:
                print(f"Error reading log file: {e}")
        else:
            print("No LLM decision log files found")
    else:
        print("Logs directory not found")

if __name__ == "__main__":
    analyze_badge_calculation()
