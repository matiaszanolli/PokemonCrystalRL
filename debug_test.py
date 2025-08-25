#!/usr/bin/env python3

import tempfile
from pathlib import Path
from monitoring import UnifiedMonitor, MonitorConfig

def test_metrics_debug():
    """Debug the metrics test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitorConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            data_dir=str(Path(tmpdir) / "data"),
            static_dir=str(Path(tmpdir) / "static"),
            port=8099,
            update_interval=0.1,
            snapshot_interval=0.5,
            max_events=1000,
            max_snapshots=10,
            debug=True
        )
        
        monitor = UnifiedMonitor(config=config)
        
        try:
            # Start training
            run_id = monitor.start_training()
            print(f"Started training with run_id: {run_id}")
            
            # Update metrics
            metrics = {
                "loss": 0.5,
                "accuracy": 0.8,
                "reward": 1.0
            }
            monitor.update_metrics(metrics)
            print(f"Updated metrics: {metrics}")
            
            # Check metrics were stored in memory
            for name, value in metrics.items():
                assert name in monitor.performance_metrics, f"Metric {name} not found in performance_metrics"
                assert value in monitor.performance_metrics[name], f"Value {value} not found for metric {name}"
                print(f"✓ Memory check passed for {name}: {value}")
            
            # Check metrics recorded in DB if available
            if monitor.db:
                run_metrics = monitor.db.get_run_metrics(monitor.current_run_id)
                print(f"DB metrics columns: {run_metrics.columns.tolist()}")
                print(f"DB metrics shape: {run_metrics.shape}")
                print(f"DB metrics:\n{run_metrics}")
                
                if not run_metrics.empty:
                    print("✓ Metrics found in database")
                    if "loss" in run_metrics.columns:
                        print(f"✓ 'loss' column found in DB")
                        print(f"Loss value: {run_metrics['loss'].iloc[-1]}")
                    else:
                        print(f"✗ 'loss' column NOT found in DB columns: {run_metrics.columns.tolist()}")
                else:
                    print("✗ No metrics found in database")
            else:
                print("No database available")
                
        finally:
            monitor.stop_training()

if __name__ == "__main__":
    test_metrics_debug()