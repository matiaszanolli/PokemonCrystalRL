#!/usr/bin/env python3
"""
A/B Testing REST API Integration Test

Test the A/B testing REST API endpoints integration with the web dashboard.
"""

import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABTestingAPITest")

API_BASE = "http://localhost:8080/api/v1"


def test_experiment_templates():
    """Test experiment templates endpoint"""
    logger.info("Testing experiment templates...")

    try:
        response = requests.get(f"{API_BASE}/experiments/templates")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                templates = data['data']['templates']
                logger.info(f"✅ Found {len(templates)} experiment templates")
                for template in templates:
                    logger.info(f"  - {template['name']} ({template['category']})")
                return True
            else:
                logger.error(f"❌ API returned error: {data.get('error')}")
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")

    except requests.ConnectionError:
        logger.error("❌ Connection failed - is the web server running?")
        logger.info("💡 Start the web server with: python main.py --enable-web")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

    return False


def test_create_experiment_from_template():
    """Test creating experiment from template"""
    logger.info("Testing experiment creation from template...")

    try:
        # Create experiment from battle strategy template
        request_data = {
            "name": "API Test Battle Strategies",
            "sample_size_per_variant": 15,
            "max_runtime_seconds": 30
        }

        response = requests.post(
            f"{API_BASE}/experiments/templates/battle_strategy_comparison",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data)
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                experiment = data['data']
                experiment_id = experiment['experiment_id']
                logger.info(f"✅ Created experiment: {experiment['name']} (ID: {experiment_id[:8]}...)")
                return experiment_id
            else:
                logger.error(f"❌ API returned error: {data.get('error')}")
        else:
            logger.error(f"❌ HTTP error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

    return None


def test_experiment_lifecycle(experiment_id):
    """Test experiment lifecycle operations"""
    logger.info(f"Testing experiment lifecycle for {experiment_id[:8]}...")

    try:
        # Get experiment details
        response = requests.get(f"{API_BASE}/experiments/{experiment_id}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                experiment = data['data']
                logger.info(f"✅ Retrieved experiment: {experiment['name']}")
                logger.info(f"   Status: {experiment['status']}")
                logger.info(f"   Variants: {len(experiment['variants'])}")
            else:
                logger.error(f"❌ Get experiment error: {data.get('error')}")
                return False

        # Start experiment
        start_request = {"action": "start"}
        response = requests.post(
            f"{API_BASE}/experiments/{experiment_id}/control",
            headers={"Content-Type": "application/json"},
            data=json.dumps(start_request)
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                logger.info("✅ Experiment started successfully")
            else:
                logger.error(f"❌ Start experiment error: {data.get('error')}")
                return False

        # Monitor progress for a few seconds
        logger.info("Monitoring experiment progress...")
        for i in range(5):
            time.sleep(2)

            response = requests.get(f"{API_BASE}/experiments/{experiment_id}/progress")
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    progress = data['data']
                    logger.info(f"  Progress: {progress['progress_percentage']:.1f}% - Status: {progress['status']}")

                    if progress['status'] in ['completed', 'failed']:
                        break
                else:
                    logger.error(f"❌ Progress error: {data.get('error')}")

        # Stop experiment
        stop_request = {"action": "stop"}
        response = requests.post(
            f"{API_BASE}/experiments/{experiment_id}/control",
            headers={"Content-Type": "application/json"},
            data=json.dumps(stop_request)
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                logger.info("✅ Experiment stopped successfully")
            else:
                logger.error(f"❌ Stop experiment error: {data.get('error')}")

        return True

    except Exception as e:
        logger.error(f"❌ Lifecycle test failed: {e}")

    return False


def test_experiment_list():
    """Test experiment listing"""
    logger.info("Testing experiment list...")

    try:
        response = requests.get(f"{API_BASE}/experiments")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                experiments_data = data['data']
                total = experiments_data['total_count']
                active = experiments_data['active_count']
                completed = experiments_data['completed_count']

                logger.info(f"✅ Retrieved experiments list:")
                logger.info(f"   Total: {total}, Active: {active}, Completed: {completed}")

                for exp in experiments_data['experiments'][:3]:  # Show first 3
                    logger.info(f"   - {exp['name']} ({exp['status']})")

                return True
            else:
                logger.error(f"❌ List experiments error: {data.get('error')}")
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ List test failed: {e}")

    return False


def test_manager_stats():
    """Test manager statistics"""
    logger.info("Testing manager statistics...")

    try:
        response = requests.get(f"{API_BASE}/experiments/stats")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                stats = data['data']
                logger.info(f"✅ Manager stats:")
                logger.info(f"   Total experiments: {stats['total_experiments']}")
                logger.info(f"   Active experiments: {stats['active_experiments']}")
                logger.info(f"   Status distribution: {stats['status_distribution']}")
                return True
            else:
                logger.error(f"❌ Stats error: {data.get('error')}")
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Stats test failed: {e}")

    return False


def main():
    """Run all A/B testing API tests"""
    logger.info("🧪 Starting A/B Testing REST API Integration Tests")
    logger.info("=" * 60)

    tests_passed = 0
    total_tests = 5

    # Test 1: Templates
    if test_experiment_templates():
        tests_passed += 1

    # Test 2: Create experiment
    experiment_id = test_create_experiment_from_template()
    if experiment_id:
        tests_passed += 1

        # Test 3: Experiment lifecycle
        if test_experiment_lifecycle(experiment_id):
            tests_passed += 1

    # Test 4: List experiments
    if test_experiment_list():
        tests_passed += 1

    # Test 5: Manager stats
    if test_manager_stats():
        tests_passed += 1

    logger.info("=" * 60)
    logger.info(f"🎯 Tests Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        logger.info("🎉 All A/B Testing API tests passed!")
        logger.info("✅ REST API integration is working correctly")
    else:
        logger.warning(f"⚠️  Some tests failed ({total_tests - tests_passed} failures)")
        logger.info("💡 Check the web server logs for more details")

    logger.info("\n🚀 Next steps:")
    logger.info("  1. Check the web dashboard at http://localhost:8080")
    logger.info("  2. Explore the API documentation at http://localhost:8080/api/v1/docs")
    logger.info("  3. Use the API to create and manage A/B tests programmatically")


if __name__ == "__main__":
    main()