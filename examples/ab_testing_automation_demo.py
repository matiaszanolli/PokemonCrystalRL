#!/usr/bin/env python3
"""
A/B Testing Automation Demo

This demo showcases the complete automated experiment execution system,
including scheduling, templates, queue management, and hands-free operation.
"""

import asyncio
import threading
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABTestingAutomationDemo")

def run_automation_demo():
    """Run a comprehensive demo of automated A/B testing"""

    logger.info("🤖 A/B Testing Automation System Demo")
    logger.info("=" * 60)

    # Import required modules
    try:
        from core.ab_testing import ExperimentManager, ConfigurationComparator
        from core.ab_testing.experiment_scheduler import ExperimentScheduler, ScheduleConfig, ScheduleType
        from core.ab_testing.automation_templates import AutomationTemplates
        from web_dashboard.api.ab_testing_endpoints import ABTestingEndpoints
        from web_dashboard.api.automation_endpoints import AutomationEndpoints
        from web_dashboard.server import UnifiedDashboardServer
        from web_dashboard.websocket_handler import WebSocketHandler
        from web_dashboard.api.endpoints import APIEndpoints
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you're running from the project root directory")
        return False

    # Create automation system
    logger.info("🔧 Setting up automated experiment system...")

    # 1. Core components
    experiment_manager = ExperimentManager()
    experiment_scheduler = ExperimentScheduler(experiment_manager)
    automation_templates = AutomationTemplates()
    config_comparator = ConfigurationComparator()

    # 2. API endpoints
    ab_testing_api = ABTestingEndpoints(experiment_manager, config_comparator)
    automation_api = AutomationEndpoints(experiment_scheduler)

    # 3. WebSocket integration
    websocket_handler = WebSocketHandler(experiment_manager=experiment_manager)
    experiment_manager.set_websocket_handler(websocket_handler)

    # 4. Start automation scheduler
    experiment_scheduler.start()

    logger.info("✅ Automation system initialized")

    # Create demo automation workflows
    logger.info("📋 Creating demo automation workflows...")

    # 1. Continuous Optimization Workflow
    logger.info("1️⃣ Setting up Continuous Optimization Workflow")
    continuous_workflows = automation_templates.create_continuous_optimization_workflow(
        base_name="Demo Continuous Optimization",
        test_interval_hours=1,  # 1 hour for demo (normally 6+)
        sample_size=10  # Small for quick demo
    )

    continuous_schedule_ids = []
    for workflow in continuous_workflows[:2]:  # Only first 2 for demo
        schedule_id = experiment_scheduler.schedule_experiment(
            workflow['experiment_config'],
            workflow['schedule_config']
        )
        continuous_schedule_ids.append(schedule_id)
        logger.info(f"   ✅ Scheduled: {workflow['name']} (ID: {schedule_id[:8]}...)")

    # 2. Performance Monitoring Workflow
    logger.info("2️⃣ Setting up Performance Monitoring Workflow")
    monitoring_workflows = automation_templates.create_performance_monitoring_workflow(
        check_interval_hours=2,  # 2 hours for demo
        performance_threshold=0.90
    )

    monitoring_schedule_ids = []
    for workflow in monitoring_workflows:
        schedule_id = experiment_scheduler.schedule_experiment(
            workflow['experiment_config'],
            workflow['schedule_config']
        )
        monitoring_schedule_ids.append(schedule_id)
        logger.info(f"   ✅ Scheduled: {workflow['name']} (ID: {schedule_id[:8]}...)")

    # 3. Regression Testing Suite
    logger.info("3️⃣ Setting up Regression Testing Suite")
    baseline_config = {
        'battle_strategy': {'aggression_level': 0.6},
        'exploration_strategy': {'coverage_threshold': 0.7}
    }
    test_configs = [
        {'name': 'Aggressive Config', 'battle_strategy': {'aggression_level': 0.9}},
        {'name': 'Conservative Config', 'battle_strategy': {'aggression_level': 0.3}}
    ]

    regression_workflows = automation_templates.create_regression_testing_suite(
        baseline_config=baseline_config,
        test_configs=test_configs,
        sample_size=8  # Small for demo
    )

    regression_schedule_ids = []
    for workflow in regression_workflows:
        schedule_id = experiment_scheduler.schedule_experiment(
            workflow['experiment_config'],
            workflow['schedule_config']
        )
        regression_schedule_ids.append(schedule_id)
        logger.info(f"   ✅ Scheduled: {workflow['name']} (ID: {schedule_id[:8]}...)")

    # 4. Weekend Stress Test
    logger.info("4️⃣ Setting up Weekend Stress Test")
    stress_workflow = automation_templates.create_weekend_stress_test(
        stress_duration_hours=2,  # 2 hours for demo (normally 48)
        sample_size=20
    )

    stress_schedule_id = experiment_scheduler.schedule_experiment(
        stress_workflow['experiment_config'],
        stress_workflow['schedule_config']
    )
    logger.info(f"   ✅ Scheduled: {stress_workflow['name']} (ID: {stress_schedule_id[:8]}...)")

    # 5. Hyperparameter Sweep
    logger.info("5️⃣ Setting up Hyperparameter Sweep")
    parameter_ranges = {
        'aggression_level': [0.3, 0.6, 0.9],
        'exploration_radius': [2, 4, 6]
    }
    base_config = {'battle_strategy': {'aggression_level': 0.5}}

    sweep_workflows = automation_templates.create_hyperparameter_sweep(
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        sample_size=5  # Very small for demo
    )

    sweep_schedule_ids = []
    for workflow in sweep_workflows[:3]:  # Only first 3 for demo
        schedule_id = experiment_scheduler.schedule_experiment(
            workflow['experiment_config'],
            workflow['schedule_config']
        )
        sweep_schedule_ids.append(schedule_id)
        logger.info(f"   ✅ Scheduled: {workflow['name']} (ID: {schedule_id[:8]}...)")

    all_schedule_ids = (continuous_schedule_ids + monitoring_schedule_ids +
                       regression_schedule_ids + [stress_schedule_id] + sweep_schedule_ids)

    logger.info(f"📊 Total scheduled experiments: {len(all_schedule_ids)}")

    # Setup web server with automation
    logger.info("🌐 Setting up web server with automation support...")

    # Create unified server with all automation components
    server = UnifiedDashboardServer(
        experiment_manager=experiment_manager,
        automation_api=automation_api,
        host='localhost',
        http_port=8080
    )

    logger.info("🚀 Automation Features Demonstrated:")
    logger.info("   • 🔄 Continuous optimization workflows")
    logger.info("   • 📊 Performance monitoring automation")
    logger.info("   • 🧪 Regression testing suites")
    logger.info("   • ⚡ Weekend stress testing")
    logger.info("   • 🎛️ Hyperparameter sweeps")
    logger.info("   • 📅 Scheduled experiment execution")
    logger.info("   • 🤖 Automated analysis and archiving")
    logger.info("   • 📈 Real-time automation monitoring")

    logger.info("\n🎯 Demo Instructions:")
    logger.info("1. Visit http://localhost:8080 in your browser")
    logger.info("2. Navigate to the A/B Testing section")
    logger.info("3. Explore automation features:")
    logger.info("   📋 View scheduled experiments: GET /api/v1/automation/schedule")
    logger.info("   📊 Check automation stats: GET /api/v1/automation/stats")
    logger.info("   🚦 Monitor queue status: GET /api/v1/automation/queue")
    logger.info("   📝 Browse templates: GET /api/v1/automation/templates")
    logger.info("   ⏯️ Control automation: POST /api/v1/automation/start|stop")

    logger.info("\n⚡ REST API Endpoints Available:")
    logger.info("• GET  /api/v1/automation/schedule - List scheduled experiments")
    logger.info("• POST /api/v1/automation/schedule - Schedule new experiment")
    logger.info("• GET  /api/v1/automation/schedule/{id} - Get schedule status")
    logger.info("• DELETE /api/v1/automation/schedule/{id} - Cancel experiment")
    logger.info("• GET  /api/v1/automation/templates - List automation templates")
    logger.info("• POST /api/v1/automation/templates/{template} - Create from template")
    logger.info("• GET  /api/v1/automation/stats - Get automation statistics")
    logger.info("• GET  /api/v1/automation/queue - Get queue status")
    logger.info("• POST /api/v1/automation/start - Start automation")
    logger.info("• POST /api/v1/automation/stop - Stop automation")

    def demo_automation_actions():
        """Demonstrate automation actions after server starts"""
        time.sleep(5)  # Wait for server to start

        logger.info("\n🎬 Demo Actions Starting...")

        # Show current queue status
        stats = experiment_scheduler.get_automation_stats()
        logger.info(f"📊 Automation Stats: {stats['total_scheduled']} scheduled, "
                   f"{stats['currently_running']} running, {stats['queue_size']} queued")

        # Start some experiments immediately for demo
        logger.info("🚀 Starting immediate demo experiments...")

        # Get first few scheduled experiments and trigger them
        scheduled_experiments = experiment_scheduler.get_scheduled_experiments()
        immediate_experiments = [exp for exp in scheduled_experiments if exp.schedule_config.schedule_type == ScheduleType.IMMEDIATE]

        if immediate_experiments:
            for exp in immediate_experiments[:2]:  # Start first 2 immediate experiments
                logger.info(f"   ▶️ Starting: {exp.experiment_config.name}")

        # Monitor progress for a while
        for i in range(12):  # Monitor for 2 minutes
            time.sleep(10)
            stats = experiment_scheduler.get_automation_stats()
            logger.info(f"⏱️ Status Update {i+1}: {stats['currently_running']} running, "
                       f"{stats['queue_size']} queued")

            if stats['currently_running'] == 0 and stats['queue_size'] == 0:
                break

        logger.info("✅ Demo automation cycle completed!")

    # Start demo actions in background
    demo_thread = threading.Thread(target=demo_automation_actions)
    demo_thread.daemon = True
    demo_thread.start()

    logger.info(f"\n🚀 Starting automation web server...")
    logger.info("🔍 Watch the automated experiments execute!")

    try:
        # Start the server (this will block)
        server.run(host='localhost', port=8080)
    except KeyboardInterrupt:
        logger.info("\n👋 Demo stopped by user")
        # Clean up
        experiment_scheduler.stop()
        experiment_manager.cleanup()
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

    return True

def main():
    """Main entry point"""
    try:
        success = run_automation_demo()
        if success:
            logger.info("✅ A/B Testing Automation Demo completed successfully!")
        else:
            logger.error("❌ Demo failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())