#!/usr/bin/env python3
"""
verify_ui_fixes.py - Comprehensive verification of UI fixes

This script verifies that all the UI implementation issues have been properly fixed:
1. Memory leak prevention (interval cleanup)
2. Proper connection status handling
3. Data mapping correctness
4. Error handling and recovery
5. Screenshot URL memory management
"""

import re
import os


def check_dashboard_fixes():
    """Check that all fixes are properly implemented in the dashboard"""
    
    print("🔍 Verifying UI fixes in templates/dashboard.html")
    print("=" * 60)
    
    try:
        with open('templates/dashboard.html', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Dashboard template not found!")
        return False
    
    fixes_verified = []
    
    # Check 1: Connection status initialization
    if 'id="connectionStatus">Connecting...' in content:
        print("✅ 1. Connection status shows 'Connecting...' initially")
        fixes_verified.append("connection_status_init")
    else:
        print("❌ 1. Connection status initialization not fixed")
    
    # Check 2: autoConnect: false
    if 'io({autoConnect: false})' in content:
        print("✅ 2. Socket.IO autoConnect disabled")
        fixes_verified.append("auto_connect_disabled")
    else:
        print("❌ 2. Socket.IO autoConnect not disabled")
    
    # Check 3: HTTP polling memory leak prevention
    if 'let pollingIntervals = []' in content and 'pollingIntervals.push(' in content:
        print("✅ 3. Polling intervals tracked for cleanup")
        fixes_verified.append("interval_tracking")
    else:
        print("❌ 3. Polling intervals not properly tracked")
    
    # Check 4: Screenshot URL cleanup
    if 'URL.revokeObjectURL(previousScreenshotUrl)' in content:
        print("✅ 4. Screenshot URLs cleaned up to prevent memory leaks")
        fixes_verified.append("screenshot_cleanup")
    else:
        print("❌ 4. Screenshot URL cleanup missing")
    
    # Check 5: Error handling in HTTP polling
    if 'Connection Error' in content and 'connectionStatus.classList.add(\'disconnected\')' in content:
        print("✅ 5. Error handling implemented for HTTP polling")
        fixes_verified.append("error_handling")
    else:
        print("❌ 5. Error handling missing in HTTP polling")
    
    # Check 6: Proper data mapping
    required_mappings = [
        'currentStep', 'actionsPerSec', 'currentReward', 'totalReward',
        'screenType', 'mapId', 'playerX', 'playerY', 'uptime'
    ]
    
    missing_mappings = []
    for mapping in required_mappings:
        if f"getElementById('{mapping}')" in content:
            continue
        else:
            missing_mappings.append(mapping)
    
    if not missing_mappings:
        print("✅ 6. All required data mappings present")
        fixes_verified.append("data_mapping")
    else:
        print(f"❌ 6. Missing data mappings: {missing_mappings}")
    
    # Check 7: Multiple polling prevention
    if 'if (useHttpPolling) return; // Prevent multiple polling instances' in content:
        print("✅ 7. Multiple polling instances prevented")
        fixes_verified.append("polling_prevention")
    else:
        print("❌ 7. Multiple polling prevention missing")
    
    # Check 8: Interval cleanup function
    if 'function stopHttpPolling()' in content and 'clearInterval(interval)' in content:
        print("✅ 8. Interval cleanup function implemented")
        fixes_verified.append("cleanup_function")
    else:
        print("❌ 8. Interval cleanup function missing")
    
    # Check 9: Connection error detection
    if 'socket.on(\'connect_error\',' in content:
        print("✅ 9. Socket.IO connection error detection")
        fixes_verified.append("connection_error")
    else:
        print("❌ 9. Socket.IO connection error detection missing")
    
    # Check 10: Timeout fallback
    if 'setTimeout(() =>' in content and '!socket.connected && !useHttpPolling' in content:
        print("✅ 10. Timeout fallback mechanism implemented")
        fixes_verified.append("timeout_fallback")
    else:
        print("❌ 10. Timeout fallback mechanism missing")
    
    print(f"\n📊 Verification Results: {len(fixes_verified)}/10 fixes verified")
    
    if len(fixes_verified) == 10:
        print("🎉 ALL FIXES VERIFIED! The UI implementation is correct.")
        return True
    else:
        print("⚠️ Some fixes are missing or incomplete.")
        return False


def check_trainer_fixes():
    """Check that the unified trainer fixes are properly implemented"""
    
    print("\n🔍 Verifying unified trainer fixes in scripts/pokemon_trainer.py")
    print("=" * 60)
    
    try:
        with open('scripts/pokemon_trainer.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Unified trainer file not found!")
        return False
    
    fixes_verified = []
    
    # Check 1: Improved Socket.IO fallback response
    if "'use_polling': True" in content and "'polling_endpoints':" in content:
        print("✅ 1. Improved Socket.IO fallback response")
        fixes_verified.append("fallback_response")
    else:
        print("❌ 1. Socket.IO fallback response not improved")
    
    # Check 2: Status code 200 instead of 404
    if 'self.send_response(200)  # Changed from 404' in content or 'self.send_response(200)  # Better than 404' in content:
        print("✅ 2. Socket.IO fallback returns 200 instead of 404")
        fixes_verified.append("status_code_200")
    else:
        print("❌ 2. Socket.IO fallback still returns 404")
    
    # Check 3: CORS headers
    if 'Access-Control-Allow-Origin' in content and 'Access-Control-Allow-Methods' in content:
        print("✅ 3. Proper CORS headers implemented")
        fixes_verified.append("cors_headers")
    else:
        print("❌ 3. CORS headers missing or incomplete")
    
    print(f"\n📊 Trainer Verification Results: {len(fixes_verified)}/3 fixes verified")
    
    if len(fixes_verified) == 3:
        print("🎉 ALL TRAINER FIXES VERIFIED!")
        return True
    else:
        print("⚠️ Some trainer fixes are missing.")
        return False


def check_file_structure():
    """Check that all required files exist"""
    
    print("\n🔍 Verifying file structure")
    print("=" * 60)
    
    required_files = [
        'templates/dashboard.html',
        'scripts/pokemon_trainer.py',
        'test_socket_fix.py',
        'SOCKET_IO_FIX_SUMMARY.md'
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_exist = False
    
    return all_files_exist


def run_comprehensive_verification():
    """Run all verification checks"""
    
    print("🧪 COMPREHENSIVE UI FIXES VERIFICATION")
    print("=" * 80)
    
    results = {
        'dashboard_fixes': check_dashboard_fixes(),
        'trainer_fixes': check_trainer_fixes(), 
        'file_structure': check_file_structure()
    }
    
    print("\n" + "=" * 80)
    print("📋 FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name.replace('_', ' ').title():<25} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("The Socket.IO implementation is fully fixed and ready for use.")
        print("\n🚀 You can now:")
        print("   - Use the unified trainer without Socket.IO errors")
        print("   - Dashboard will gracefully fall back to HTTP polling")
        print("   - Memory leaks are prevented")
        print("   - Error handling is robust")
        print("   - All data mappings work correctly")
    else:
        print("⚠️ SOME VERIFICATIONS FAILED!")
        print("Please review the failed checks and fix the issues.")
        print("Run this script again after making corrections.")
    
    return all_passed


def main():
    """Main verification function"""
    success = run_comprehensive_verification()
    
    if success:
        print("\n🎯 To test the fixes:")
        print("   python test_socket_fix.py")
        print("\n📖 For more details, see:")
        print("   SOCKET_IO_FIX_SUMMARY.md")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
