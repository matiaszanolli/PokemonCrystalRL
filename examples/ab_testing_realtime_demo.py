#!/usr/bin/env python3
"""
A/B Testing Real-time Monitoring Demo

This demo shows the real-time WebSocket monitoring capabilities of the A/B testing framework.
It creates sample experiments and demonstrates live progress updates.
"""

import asyncio
import threading
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABTestingRealtimeDemo")

def run_realtime_demo():
    """Run a demo of real-time A/B testing monitoring"""

    logger.info("🚀 A/B Testing Real-time Monitoring Demo")
    logger.info("=" * 60)

    # Import required modules
    try:
        from core.ab_testing import ExperimentManager, ConfigurationComparator
        from web_dashboard.api.ab_testing_endpoints import ABTestingEndpoints
        from web_dashboard.server import UnifiedDashboardServer
        from web_dashboard.websocket_handler import WebSocketHandler
        from web_dashboard.api.endpoints import APIEndpoints
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you're running from the project root directory")
        return False

    # Create A/B testing components
    logger.info("🔧 Setting up A/B testing components...")

    # Create WebSocket handler first
    websocket_handler = WebSocketHandler()

    # Create experiment manager with WebSocket integration
    experiment_manager = ExperimentManager(websocket_handler=websocket_handler)
    websocket_handler.experiment_manager = experiment_manager

    config_comparator = ConfigurationComparator()
    ab_testing_api = ABTestingEndpoints(experiment_manager, config_comparator)

    # Create sample experiments for demo
    logger.info("📝 Creating sample experiments...")

    # Create multiple experiments to demonstrate real-time updates
    experiments = []

    # 1. Quick battle strategy test
    battle_config = config_comparator.create_battle_strategy_comparison()
    battle_config.name = "Real-time Demo: Battle Strategies"
    battle_config.description = "Quick battle strategy comparison with live updates"
    battle_config.sample_size_per_variant = 5  # Small for quick demo
    battle_config.max_runtime_seconds = 60    # 1 minute max

    experiment_id = experiment_manager.create_experiment(battle_config)
    experiments.append(experiment_id)
    logger.info(f"✅ Created battle strategy experiment: {experiment_id[:8]}...")

    # 2. Plugin comparison test
    plugin_config = config_comparator.create_plugin_comparison()
    plugin_config.name = "Real-time Demo: Plugin Test"
    plugin_config.description = "Plugin performance comparison with live metrics"
    plugin_config.sample_size_per_variant = 3  # Very small for demo
    plugin_config.max_runtime_seconds = 45     # 45 seconds

    experiment_id = experiment_manager.create_experiment(plugin_config)
    experiments.append(experiment_id)
    logger.info(f"✅ Created plugin comparison experiment: {experiment_id[:8]}...")

    # Setup web server components
    logger.info("🌐 Setting up web server with real-time monitoring...")

    # Create API endpoints
    api_endpoints = APIEndpoints()

    # Create unified server with A/B testing and WebSocket integration
    server = UnifiedDashboardServer(
        api_endpoints=api_endpoints,
        ab_testing_api=ab_testing_api,
        websocket_handler=websocket_handler
    )

    logger.info("📊 Real-time Monitoring Features:")
    logger.info("   • Live experiment progress updates")
    logger.info("   • Real-time performance metrics streaming")
    logger.info("   • WebSocket-based instant notifications")
    logger.info("   • Live variant comparison metrics")
    logger.info("   • Automatic experiment status updates")

    logger.info("\n🎯 Demo Instructions:")
    logger.info("1. Visit http://localhost:8080 in your browser")
    logger.info("2. Navigate to Training Visualizations → A/B Testing")
    logger.info("3. View the pre-created experiments in the 'Experiments' tab")
    logger.info("4. Click on any experiment to open the live monitoring modal")
    logger.info("5. Start an experiment and watch real-time updates:")
    logger.info("   - Progress bars update live")
    logger.info("   - Status changes instantly")
    logger.info("   - Live metrics stream in real-time")
    logger.info("   - Green 'Live Updates' indicator shows active monitoring")

    logger.info("\n⚡ Real-time Features to Observe:")
    logger.info("• Progress percentage updates every ~1 second")
    logger.info("• Live metrics show latest reward, actions/sec, win rates")
    logger.info("• Status changes (pending → running → completed) instantly")
    logger.info("• Experiment list refreshes automatically")
    logger.info("• Modal shows live variant performance comparison")

    logger.info("\n🔍 WebSocket Messages (check browser dev tools):")
    logger.info("• 'experiments_update' - General experiment list updates")
    logger.info("• 'experiment_progress' - Detailed progress for subscribed experiments")
    logger.info("• Subscribe/unsubscribe messages when opening/closing experiment modals")

    def start_demo_experiments():
        """Start some demo experiments after a delay"""
        time.sleep(5)  # Wait for server to start
        logger.info("\n🎬 Auto-starting demo experiments...")

        try:
            # Start the first experiment
            if experiment_manager.start_experiment(experiments[0]):
                logger.info("✅ Started battle strategy experiment")

            # Start the second experiment after a short delay
            time.sleep(3)
            if experiment_manager.start_experiment(experiments[1]):
                logger.info("✅ Started plugin comparison experiment")

        except Exception as e:
            logger.error(f"❌ Error starting demo experiments: {e}")

    # Start demo experiments in background
    demo_thread = threading.Thread(target=start_demo_experiments)
    demo_thread.daemon = True
    demo_thread.start()

    logger.info(f"\n🚀 Starting web server with real-time monitoring...")
    logger.info("📈 Watch the experiments progress in real-time!")

    try:
        # Start the server (this will block)
        server.run(host='localhost', port=8080)
    except KeyboardInterrupt:
        logger.info("\n👋 Demo stopped by user")
        # Clean up experiments
        experiment_manager.cleanup()
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

    return True

def main():
    """Main entry point"""
    try:
        success = run_realtime_demo()
        if success:
            logger.info("✅ Real-time A/B Testing Demo completed successfully!")
        else:
            logger.error("❌ Demo failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())