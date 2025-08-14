#!/bin/bash

# Pokémon Crystal RL Training System - Quick Start Script

echo "🎮 Starting Pokémon Crystal RL Training System..."

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  No virtual environment detected. Consider activating one first."
fi

# Start the monitoring system
echo "🚀 Launching monitoring dashboard and TensorBoard..."
python3 start_monitoring.py
