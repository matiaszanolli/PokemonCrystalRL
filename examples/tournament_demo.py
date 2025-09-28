#!/usr/bin/env python3
"""
Tournament Demo - Showcase tournament functionality

Demonstrates creating tournaments, adding participants, and running
competitive AI battles using the tournament management system.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.tournament import (
    TournamentManager,
    TournamentConfig,
    TournamentParticipant,
    TournamentType,
    ParticipantType
)


def setup_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_participants():
    """Create sample AI configurations for tournament"""
    participants = []

    # Battle-focused AI
    battle_ai = TournamentParticipant(
        name="Battle Master",
        description="AI focused on battle optimization",
        participant_type=ParticipantType.AI_CONFIGURATION,
        configuration={
            "plugins": ["aggressive_battle_strategy"],
            "llm_model": "smollm2:1.7b",
            "llm_interval": 5,
            "focus": "battle_optimization"
        }
    )
    participants.append(battle_ai)

    # Exploration-focused AI
    explorer_ai = TournamentParticipant(
        name="Map Explorer",
        description="AI focused on exploration and discovery",
        participant_type=ParticipantType.AI_CONFIGURATION,
        configuration={
            "plugins": ["systematic_exploration"],
            "llm_model": "smollm2:1.7b",
            "llm_interval": 10,
            "focus": "exploration_coverage"
        }
    )
    participants.append(explorer_ai)

    # Balanced AI
    balanced_ai = TournamentParticipant(
        name="Balanced Strategist",
        description="AI with balanced approach to all aspects",
        participant_type=ParticipantType.AI_CONFIGURATION,
        configuration={
            "plugins": ["balanced_strategy"],
            "llm_model": "smollm2:1.7b",
            "llm_interval": 8,
            "focus": "balanced_performance"
        }
    )
    participants.append(balanced_ai)

    # Speed-focused AI
    speed_ai = TournamentParticipant(
        name="Speed Runner",
        description="AI optimized for fast progression",
        participant_type=ParticipantType.AI_CONFIGURATION,
        configuration={
            "plugins": ["speed_optimization"],
            "llm_model": "smollm2:1.7b",
            "llm_interval": 3,
            "focus": "actions_per_second"
        }
    )
    participants.append(speed_ai)

    return participants


def run_single_elimination_demo(tournament_manager):
    """Demo single elimination tournament"""
    print("\n" + "="*60)
    print("SINGLE ELIMINATION TOURNAMENT DEMO")
    print("="*60)

    # Create tournament configuration
    config = TournamentConfig(
        name="AI Battle Royale",
        description="Single elimination tournament to find the best AI configuration",
        tournament_type=TournamentType.SINGLE_ELIMINATION,
        max_participants=8,
        min_participants=4,
        match_duration_minutes=15,
        max_actions_per_match=500,
        primary_metric="reward",
        secondary_metrics=["actions_per_second", "progress_rate"],
        auto_advance=True,
        round_delay_minutes=1
    )

    # Create tournament
    tournament_id = tournament_manager.create_tournament(config)
    print(f"✅ Created tournament: {tournament_id}")

    # Add participants
    participants = create_sample_participants()
    for participant in participants:
        success = tournament_manager.add_participant(tournament_id, participant)
        if success:
            print(f"✅ Added participant: {participant.name}")
        else:
            print(f"❌ Failed to add participant: {participant.name}")

    # Get tournament details
    tournament = tournament_manager.get_tournament(tournament_id)
    print(f"\n📊 Tournament Details:")
    print(f"   Name: {tournament.config.name}")
    print(f"   Type: {tournament.config.tournament_type.value}")
    print(f"   Participants: {len(tournament.participants)}")
    print(f"   Status: {tournament.status.value}")

    # Start tournament
    print(f"\n🚀 Starting tournament...")
    success = tournament_manager.start_tournament(tournament_id)
    if success:
        print("✅ Tournament started successfully!")

        # Show bracket structure
        tournament = tournament_manager.get_tournament(tournament_id)
        print(f"\n🏆 Tournament Bracket:")
        print(f"   Total Rounds: {tournament.total_rounds}")
        print(f"   Total Matches: {tournament.total_matches}")
        print(f"   Current Round: {tournament.current_round}")

        # Show first round matches
        first_round_matches = tournament.bracket.get_round_matches(1)
        print(f"\n🥊 Round 1 Matches:")
        for match_id in first_round_matches:
            match = tournament.matches[match_id]
            p1_name = tournament.participants[match.participant1_id].name
            p2_name = tournament.participants.get(match.participant2_id, {}).name if match.participant2_id else "BYE"
            print(f"   {p1_name} vs {p2_name}")

    else:
        print("❌ Failed to start tournament")

    return tournament_id


def run_round_robin_demo(tournament_manager):
    """Demo round robin tournament"""
    print("\n" + "="*60)
    print("ROUND ROBIN TOURNAMENT DEMO")
    print("="*60)

    # Create tournament configuration
    config = TournamentConfig(
        name="Round Robin Championship",
        description="Everyone plays everyone to determine the ultimate champion",
        tournament_type=TournamentType.ROUND_ROBIN,
        max_participants=4,
        min_participants=3,
        match_duration_minutes=10,
        max_actions_per_match=300,
        primary_metric="progress_rate",
        secondary_metrics=["reward", "battle_win_rate"],
        auto_advance=True
    )

    # Create tournament
    tournament_id = tournament_manager.create_tournament(config)
    print(f"✅ Created round robin tournament: {tournament_id}")

    # Add subset of participants
    participants = create_sample_participants()[:3]
    for participant in participants:
        tournament_manager.add_participant(tournament_id, participant)
        print(f"✅ Added participant: {participant.name}")

    # Start tournament
    print(f"\n🚀 Starting round robin tournament...")
    success = tournament_manager.start_tournament(tournament_id)
    if success:
        print("✅ Round robin tournament started!")

        tournament = tournament_manager.get_tournament(tournament_id)
        print(f"\n📋 All Matches:")
        for match in tournament.matches.values():
            p1_name = tournament.participants[match.participant1_id].name
            p2_name = tournament.participants[match.participant2_id].name
            print(f"   {p1_name} vs {p2_name}")

    return tournament_id


def monitor_tournament_progress(tournament_manager, tournament_id, check_interval=5):
    """Monitor tournament progress"""
    print(f"\n👀 Monitoring tournament {tournament_id}...")
    print("   (In real deployment, matches would execute via A/B testing framework)")
    print("   (This demo simulates match completion for demonstration)")

    last_status = None
    while True:
        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            break

        current_status = tournament.status.value
        if current_status != last_status:
            print(f"\n📊 Tournament Status: {current_status}")
            print(f"   Progress: {tournament.completed_matches}/{tournament.total_matches} matches")

            if tournament.status.value == "completed":
                print(f"🏆 Tournament completed!")
                if tournament.champion_id:
                    champion = tournament.participants[tournament.champion_id]
                    print(f"   Champion: {champion.name}")
                    print(f"   Win Rate: {champion.win_rate:.1%}")
                break

            last_status = current_status

        time.sleep(check_interval)


def main():
    """Run tournament demos"""
    setup_logging()

    print("🎯 Pokemon Crystal RL Tournament System Demo")
    print("=" * 60)

    # Initialize tournament manager
    tournament_manager = TournamentManager(
        data_dir="tournament_demo_data",
        auto_advance=True
    )

    try:
        # Run single elimination demo
        se_tournament_id = run_single_elimination_demo(tournament_manager)

        # Run round robin demo
        rr_tournament_id = run_round_robin_demo(tournament_manager)

        # Show active tournaments
        active_tournaments = tournament_manager.get_active_tournaments()
        print(f"\n📈 Active Tournaments: {len(active_tournaments)}")
        for tournament in active_tournaments:
            print(f"   - {tournament.config.name} ({tournament.status.value})")

        # Show all tournaments
        all_tournaments = tournament_manager.list_tournaments()
        print(f"\n📊 Total Tournaments Created: {len(all_tournaments)}")

        print("\n✨ Tournament Demo Complete!")
        print("\n💡 Next Steps:")
        print("   1. Start web dashboard to view tournaments in browser")
        print("   2. Use REST API endpoints for programmatic access")
        print("   3. Create custom participant configurations")
        print("   4. Run real tournaments with A/B testing integration")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        logging.exception("Demo error")
    finally:
        tournament_manager.stop()


if __name__ == "__main__":
    main()