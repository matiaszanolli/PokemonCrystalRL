#!/usr/bin/env python3
"""
Legacy Web System Cleanup Script

This script safely removes duplicate and obsolete web dashboard implementations,
leaving only the unified web dashboard system.

Run with --dry-run to see what would be removed without actually deleting files.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path

# Files and directories to remove
LEGACY_DASHBOARDS = [
    "monitoring/static/index.html",
    "monitoring/web/templates/dashboard.html",
    "monitoring/web/templates/status.html",
    "static/index.html",
    "static/templates/dashboard.html"
]

LEGACY_DIRECTORIES = [
    "core/web_monitor",
    "monitoring/web"
]

LEGACY_FILES = [
    "scripts/startup/start_web_monitor.py",
    "trainer/compat/web_monitor.py",
    "tests/monitoring/mock_web_server.py",
    "tests/trainer/mock_web_server.py"
]

# Files to keep (these contain the unified system)
KEEP_THESE = [
    "web_dashboard/",
    "web_dashboard/static/dashboard.html"
]


def print_banner():
    """Print cleanup banner."""
    print("🧹 Pokemon Crystal RL - Legacy Web System Cleanup")
    print("=" * 50)
    print("This script removes duplicate web dashboard implementations")
    print("and keeps only the unified web dashboard system.")
    print()


def check_unified_system():
    """Check that the unified system exists before cleanup."""
    unified_files = [
        "web_dashboard/__init__.py",
        "web_dashboard/server.py",
        "web_dashboard/static/dashboard.html",
        "web_dashboard/api/endpoints.py"
    ]

    missing = []
    for file_path in unified_files:
        if not os.path.exists(file_path):
            missing.append(file_path)

    if missing:
        print("❌ ERROR: Unified web dashboard system is incomplete!")
        print("Missing files:")
        for file_path in missing:
            print(f"  - {file_path}")
        print("\nPlease ensure the unified system is properly installed before cleanup.")
        return False

    print("✅ Unified web dashboard system verified")
    return True


def list_removals(dry_run=True):
    """List what will be removed."""
    action = "Would remove" if dry_run else "Removing"
    print(f"\n📋 {action} the following legacy files:")

    # Dashboard templates
    print("\n📱 Legacy Dashboard Templates:")
    for file_path in LEGACY_DASHBOARDS:
        if os.path.exists(file_path):
            print(f"  - {file_path}")
        else:
            print(f"  - {file_path} (already missing)")

    # Directories
    print("\n📁 Legacy Web System Directories:")
    for dir_path in LEGACY_DIRECTORIES:
        if os.path.exists(dir_path):
            file_count = sum(1 for _ in Path(dir_path).rglob("*.py"))
            print(f"  - {dir_path}/ ({file_count} Python files)")
        else:
            print(f"  - {dir_path}/ (already missing)")

    # Individual files
    print("\n📄 Legacy Support Files:")
    for file_path in LEGACY_FILES:
        if os.path.exists(file_path):
            print(f"  - {file_path}")
        else:
            print(f"  - {file_path} (already missing)")

    print(f"\n✅ Keeping the unified system:")
    for keep_path in KEEP_THESE:
        if os.path.exists(keep_path):
            print(f"  ✅ {keep_path}")
        else:
            print(f"  ❌ {keep_path} (MISSING!)")


def remove_legacy_files(dry_run=True):
    """Remove legacy dashboard files."""
    removed_count = 0

    # Remove legacy dashboard templates
    for file_path in LEGACY_DASHBOARDS:
        if os.path.exists(file_path):
            if not dry_run:
                os.remove(file_path)
                print(f"🗑️  Removed: {file_path}")
            removed_count += 1

    # Remove legacy directories
    for dir_path in LEGACY_DIRECTORIES:
        if os.path.exists(dir_path):
            if not dry_run:
                shutil.rmtree(dir_path)
                print(f"🗑️  Removed directory: {dir_path}/")
            removed_count += 1

    # Remove legacy files
    for file_path in LEGACY_FILES:
        if os.path.exists(file_path):
            if not dry_run:
                os.remove(file_path)
                print(f"🗑️  Removed: {file_path}")
            removed_count += 1

    return removed_count


def clean_empty_directories():
    """Remove empty directories after cleanup."""
    empty_dirs = []

    # Check for empty parent directories
    check_dirs = [
        "static/templates",
        "static",
        "monitoring/static",
        "scripts/startup",
        "trainer/compat"
    ]

    for dir_path in check_dirs:
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            empty_dirs.append(dir_path)

    return empty_dirs


def update_imports_file():
    """Create a guide for updating imports."""
    guide_content = """# Import Migration Guide

## Old imports to replace:

```python
# REMOVE these imports:
from core.web_monitor import WebMonitor
from core.web_monitor.monitor import WebMonitor
from monitoring.web import HttpHandler
from monitoring.web.server import run_server

# REPLACE with:
from web_dashboard import create_web_server
```

## Usage migration:

```python
# Old usage:
web_monitor = WebMonitor(trainer)
web_monitor.start()

# New usage:
web_server = create_web_server(trainer)
web_server.start()
```

See web_dashboard/README.md for complete migration instructions.
"""

    with open("IMPORT_MIGRATION_GUIDE.md", "w") as f:
        f.write(guide_content)

    print("📄 Created IMPORT_MIGRATION_GUIDE.md")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Clean up legacy web dashboard implementations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be removed without actually deleting")
    parser.add_argument("--force", action="store_true",
                       help="Skip confirmation prompt")

    args = parser.parse_args()

    print_banner()

    # Check that unified system exists
    if not check_unified_system():
        sys.exit(1)

    # List what will be removed
    list_removals(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n🔍 This was a dry run. Use --force to actually remove files.")
        return

    # Confirmation
    if not args.force:
        print(f"\n⚠️  This will permanently delete the legacy web systems!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Cleanup cancelled")
            return

    # Perform cleanup
    print(f"\n🧹 Starting cleanup...")
    removed_count = remove_legacy_files(dry_run=False)

    # Clean empty directories
    empty_dirs = clean_empty_directories()
    if empty_dirs:
        print(f"\n📁 Empty directories found (you may want to remove manually):")
        for dir_path in empty_dirs:
            print(f"  - {dir_path}/")

    # Create migration guide
    update_imports_file()

    print(f"\n✅ Cleanup complete!")
    print(f"   - Removed {removed_count} legacy files/directories")
    print(f"   - Unified web dashboard preserved at: web_dashboard/")
    print(f"   - Dashboard URL: http://localhost:8080")
    print(f"\n📖 Next steps:")
    print(f"   1. Update any imports using IMPORT_MIGRATION_GUIDE.md")
    print(f"   2. Test the unified dashboard with your trainer")
    print(f"   3. Update documentation references")


if __name__ == "__main__":
    main()