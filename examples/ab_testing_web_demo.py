#!/usr/bin/env python3
"""
A/B Testing Web Dashboard Integration Demo

This demo shows how the A/B testing framework integrates with the web dashboard.
Run this script and then visit the web dashboard to see the A/B testing interface.
"""

import asyncio
import threading
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABTestingWebDemo")

def run_web_server_demo():
    """Run a demo of the A/B testing web integration"""

    logger.info("🧪 A/B Testing Web Dashboard Integration Demo")
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

    experiment_manager = ExperimentManager()
    config_comparator = ConfigurationComparator()
    ab_testing_api = ABTestingEndpoints(experiment_manager, config_comparator)

    # Create sample experiment for demo
    logger.info("📝 Creating sample experiment...")

    # Create a battle strategy comparison experiment
    battle_config = config_comparator.create_battle_strategy_comparison()
    battle_config.name = "Demo: Battle Strategy A/B Test"
    battle_config.description = "Comparing aggressive vs defensive battle strategies"
    battle_config.sample_size_per_variant = 15
    battle_config.max_runtime_seconds = 120  # 2 minutes for demo

    sample_experiment_id = experiment_manager.create_experiment(battle_config)
    logger.info(f"✅ Created sample experiment: {sample_experiment_id[:8]}...")

    # Setup web server components
    logger.info("🌐 Setting up web server...")

    # Create API endpoints
    api_endpoints = APIEndpoints()

    # Create WebSocket handler
    websocket_handler = WebSocketHandler()

    # Create unified server with A/B testing integration
    server = UnifiedDashboardServer(
        api_endpoints=api_endpoints,
        ab_testing_api=ab_testing_api,
        websocket_handler=websocket_handler
    )

    logger.info("🚀 Starting web server...")
    logger.info("📊 A/B Testing Dashboard Features:")
    logger.info("   • Experiment Management - Create, start, stop experiments")
    logger.info("   • Template Library - Pre-built experiment configurations")
    logger.info("   • Real-time Progress - Live experiment monitoring")
    logger.info("   • Statistical Analysis - Automated significance testing")
    logger.info("   • REST API Integration - Programmatic experiment control")

    logger.info("\n🎯 Demo Instructions:")
    logger.info("1. Visit http://localhost:8080 in your browser")
    logger.info("2. Click on the 'Training Visualizations' panel")
    logger.info("3. Select the 'A/B Testing' tab")
    logger.info("4. Explore the experiment management interface:")
    logger.info("   - 'Experiments' tab: View the demo experiment")
    logger.info("   - 'Create Test' tab: Create new experiments")
    logger.info("   - 'Templates' tab: Use pre-built configurations")
    logger.info("   - 'Analytics' tab: View experiment statistics")
    logger.info("5. Click on the demo experiment to see detailed controls")

    logger.info("\n⚡ Advanced Features:")
    logger.info("• Real-time experiment progress monitoring")
    logger.info("• Statistical significance testing with multiple methods")
    logger.info("• Plugin and agent configuration comparison")
    logger.info("• Automated experiment execution and analysis")
    logger.info("• Integration with existing training pipeline")

    logger.info(f"\n🔍 API Endpoints Available:")
    logger.info("• GET  /api/v1/experiments - List all experiments")
    logger.info("• POST /api/v1/experiments - Create new experiment")
    logger.info("• GET  /api/v1/experiments/{id} - Get experiment details")
    logger.info("• POST /api/v1/experiments/{id}/control - Start/stop experiment")
    logger.info("• GET  /api/v1/experiments/templates - List templates")
    logger.info("• POST /api/v1/experiments/templates/{template} - Create from template")

    try:
        # Start the server (this will block)
        server.run(host='localhost', port=8080)
    except KeyboardInterrupt:
        logger.info("\n👋 Demo stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

    return True

def main():
    """Main entry point"""
    try:
        success = run_web_server_demo()
        if success:
            logger.info("✅ A/B Testing Web Demo completed successfully!")
        else:
            logger.error("❌ Demo failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())