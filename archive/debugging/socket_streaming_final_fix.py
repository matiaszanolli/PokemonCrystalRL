
# FINAL FIX FOR SOCKET BLANK SCREEN ISSUE
# =====================================

def fix_blank_screen_streaming(trainer):
    """
    Complete fix for blank screen streaming issue.
    
    The problem was that PyBoy starts with a blank screen and needs
    to be advanced to show actual game content.
    """
    
    # Step 1: Advance PyBoy past initial blank frames
    print("🎮 Advancing PyBoy to show game content...")
    
    for i in range(100):
        trainer.pyboy.tick()
        if i % 20 == 0:
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            if variance > 10:  # Found content
                print(f"✅ Game content found at frame {i}")
                break
    
    # Step 2: Press START to begin game
    print("🕹️ Pressing START to begin game...")
    
    for i in range(20):
        trainer.strategy_manager.execute_action(7)  # START button
        screen = trainer.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        if variance > 100:
            print(f"✅ Game started successfully")
            break
    
    # Step 3: Start screen capture AFTER game has content
    print("📸 Starting screen capture with game content...")
    
    if trainer.config.capture_screens:
        trainer._start_screen_capture()
    
    print("✅ Fix applied - streaming should now work!")
    
# Usage:
# trainer = UnifiedPokemonTrainer(config)
# fix_blank_screen_streaming(trainer)
# # Now start bridge and web monitor
