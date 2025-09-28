#!/usr/bin/env python3
"""
Save State Library Management CLI

Command-line interface for managing the save state library.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.save_state_library import (
    SaveStateLibrary, GamePhase, TrainingScenario, Difficulty,
    GameState, create_library_from_current_state
)


def list_states(args):
    """List save states with optional filtering."""
    library = SaveStateLibrary(args.library_path)

    # Parse filters
    phase = GamePhase(args.phase) if args.phase else None
    scenario = TrainingScenario(args.scenario) if args.scenario else None
    difficulty = Difficulty(args.difficulty) if args.difficulty else None
    tags = args.tags.split(',') if args.tags else None

    states = library.list_save_states(
        phase=phase,
        scenario=scenario,
        difficulty=difficulty,
        min_badges=args.min_badges,
        max_badges=args.max_badges,
        tags=tags
    )

    if not states:
        print("No save states found matching the criteria.")
        return

    print(f"Found {len(states)} save state(s):")
    print("-" * 80)

    for state in states:
        print(f"ID: {state.id}")
        print(f"Name: {state.name}")
        print(f"Description: {state.description}")
        print(f"Phase: {state.phase.value} | Scenario: {state.scenario.value} | Difficulty: {state.difficulty.value}")
        print(f"Location: {state.game_state.location} | Badges: {state.game_state.badges} | Level: {state.game_state.level}")
        print(f"Usage: {state.usage_count} times | Success Rate: {state.success_rate:.1%}" if state.success_rate else f"Usage: {state.usage_count} times")
        print(f"File: {state.file_path}")
        if state.tags:
            print(f"Tags: {', '.join(state.tags)}")
        print("-" * 80)


def add_state(args):
    """Add a new save state to the library."""
    library = SaveStateLibrary(args.library_path)

    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return

    # Parse enums
    phase = GamePhase(args.phase)
    scenario = TrainingScenario(args.scenario)
    difficulty = Difficulty(args.difficulty)

    # Create basic game state (in a real implementation, this would be extracted from the save file)
    game_state = GameState(
        player_name=args.player_name or "Player",
        location=args.location or "Unknown",
        map_id=args.map_id or 0,
        badges=args.badges or 0,
        level=args.level or 5,
        party_size=args.party_size or 1,
        money=args.money or 3000,
        playtime_hours=args.playtime or 0,
        story_progress=args.story_progress or "Beginning"
    )

    # Parse training objectives
    objectives = args.objectives.split(',') if args.objectives else [f"Train for {scenario.value}"]

    # Parse tags
    tags = args.tags.split(',') if args.tags else []

    # Parse recommended models
    models = args.models.split(',') if args.models else []

    try:
        state_id = library.add_save_state(
            source_path=args.source,
            name=args.name,
            description=args.description,
            phase=phase,
            scenario=scenario,
            difficulty=difficulty,
            game_state=game_state,
            training_objectives=objectives,
            expected_actions=args.expected_actions,
            recommended_models=models,
            tags=tags,
            created_by=args.created_by or "cli"
        )

        print(f"Successfully added save state with ID: {state_id}")

    except Exception as e:
        print(f"Error adding save state: {e}")


def info_command(args):
    """Show library information."""
    library = SaveStateLibrary(args.library_path)
    info = library.export_library_info()

    print("Save State Library Information")
    print("=" * 40)
    print(f"Total States: {info['total_states']}")
    print(f"Library Path: {info['library_path']}")
    print(f"Total Size: {info['total_size_mb']:.1f} MB")
    print()

    print("By Game Phase:")
    for phase, count in info['phases'].items():
        print(f"  {phase}: {count}")
    print()

    print("By Training Scenario:")
    for scenario, count in info['scenarios'].items():
        print(f"  {scenario}: {count}")
    print()

    print("By Difficulty:")
    for difficulty, count in info['difficulties'].items():
        print(f"  {difficulty}: {count}")
    print()

    if info['most_used']:
        print("Most Used Save States:")
        for state_id, usage_count, name in info['most_used']:
            if usage_count > 0:
                print(f"  {name} ({state_id}): {usage_count} times")


def recommend_command(args):
    """Get recommendations for a training scenario."""
    library = SaveStateLibrary(args.library_path)

    scenario = TrainingScenario(args.scenario)
    difficulty = Difficulty(args.difficulty) if args.difficulty else Difficulty.MEDIUM

    badges_range = None
    if args.min_badges is not None or args.max_badges is not None:
        badges_range = (args.min_badges or 0, args.max_badges or 16)

    recommendations = library.get_recommendations(
        target_scenario=scenario,
        target_difficulty=difficulty,
        badges_range=badges_range
    )

    if not recommendations:
        print(f"No recommendations found for scenario: {scenario.value}")
        return

    print(f"Recommended save states for {scenario.value} ({difficulty.value}):")
    print("-" * 60)

    for i, state in enumerate(recommendations, 1):
        print(f"{i}. {state.name} ({state.id})")
        print(f"   Description: {state.description}")
        print(f"   Location: {state.game_state.location} | Badges: {state.game_state.badges}")
        print(f"   Expected Actions: {state.expected_actions}")
        if state.success_rate:
            print(f"   Success Rate: {state.success_rate:.1%}")
        print(f"   File: {state.file_path}")
        print()


def verify_command(args):
    """Verify library integrity."""
    library = SaveStateLibrary(args.library_path)
    results = library.verify_integrity()

    print("Library Integrity Check Results")
    print("=" * 40)
    print(f"Valid: {len(results['valid'])}")
    print(f"Missing Files: {len(results['missing_files'])}")
    print(f"Checksum Mismatches: {len(results['checksum_mismatch'])}")

    if results['missing_files']:
        print("\nMissing Files:")
        for state_id in results['missing_files']:
            print(f"  {state_id}")

    if results['checksum_mismatch']:
        print("\nChecksum Mismatches:")
        for state_id in results['checksum_mismatch']:
            print(f"  {state_id}")


def remove_command(args):
    """Remove a save state from the library."""
    library = SaveStateLibrary(args.library_path)

    if library.remove_save_state(args.state_id):
        print(f"Successfully removed save state: {args.state_id}")
    else:
        print(f"Failed to remove save state: {args.state_id}")


def main():
    parser = argparse.ArgumentParser(description="Manage Pokemon Crystal RL Save State Library")
    parser.add_argument('--library-path', default='save_states', help='Path to save state library')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List save states')
    list_parser.add_argument('--phase', choices=[p.value for p in GamePhase], help='Filter by game phase')
    list_parser.add_argument('--scenario', choices=[s.value for s in TrainingScenario], help='Filter by scenario')
    list_parser.add_argument('--difficulty', choices=[d.value for d in Difficulty], help='Filter by difficulty')
    list_parser.add_argument('--min-badges', type=int, help='Minimum badges')
    list_parser.add_argument('--max-badges', type=int, help='Maximum badges')
    list_parser.add_argument('--tags', help='Comma-separated tags to filter by')
    list_parser.set_defaults(func=list_states)

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new save state')
    add_parser.add_argument('source', help='Path to source save state file')
    add_parser.add_argument('name', help='Name for the save state')
    add_parser.add_argument('description', help='Description of the save state')
    add_parser.add_argument('--phase', choices=[p.value for p in GamePhase], required=True, help='Game phase')
    add_parser.add_argument('--scenario', choices=[s.value for s in TrainingScenario], required=True, help='Training scenario')
    add_parser.add_argument('--difficulty', choices=[d.value for d in Difficulty], default='medium', help='Difficulty level')
    add_parser.add_argument('--player-name', help='Player name')
    add_parser.add_argument('--location', help='Current location')
    add_parser.add_argument('--map-id', type=int, help='Map ID')
    add_parser.add_argument('--badges', type=int, help='Number of badges')
    add_parser.add_argument('--level', type=int, help='Player level')
    add_parser.add_argument('--party-size', type=int, help='Party size')
    add_parser.add_argument('--money', type=int, help='Player money')
    add_parser.add_argument('--playtime', type=int, help='Playtime in hours')
    add_parser.add_argument('--story-progress', help='Story progress description')
    add_parser.add_argument('--objectives', help='Comma-separated training objectives')
    add_parser.add_argument('--expected-actions', type=int, default=1000, help='Expected number of actions')
    add_parser.add_argument('--models', help='Comma-separated recommended models')
    add_parser.add_argument('--tags', help='Comma-separated tags')
    add_parser.add_argument('--created-by', help='Creator name')
    add_parser.set_defaults(func=add_state)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show library information')
    info_parser.set_defaults(func=info_command)

    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get save state recommendations')
    recommend_parser.add_argument('scenario', choices=[s.value for s in TrainingScenario], help='Training scenario')
    recommend_parser.add_argument('--difficulty', choices=[d.value for d in Difficulty], help='Preferred difficulty')
    recommend_parser.add_argument('--min-badges', type=int, help='Minimum badges')
    recommend_parser.add_argument('--max-badges', type=int, help='Maximum badges')
    recommend_parser.set_defaults(func=recommend_command)

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify library integrity')
    verify_parser.set_defaults(func=verify_command)

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a save state')
    remove_parser.add_argument('state_id', help='ID of save state to remove')
    remove_parser.set_defaults(func=remove_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()