"""
Tournament API Endpoints - REST API for tournament management

Provides RESTful API endpoints for tournament creation, management,
and real-time monitoring through the web dashboard.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, jsonify, request

from core.tournament.tournament_manager import TournamentManager
from core.tournament.tournament_models import (
    TournamentConfig,
    TournamentParticipant,
    TournamentType,
    TournamentStatus,
    ParticipantType
)
from .auth import require_auth


# Create blueprint
tournament_bp = Blueprint('tournament', __name__, url_prefix='/api/v1/tournament')

# Global tournament manager instance
tournament_manager: Optional[TournamentManager] = None

def init_tournament_api(manager: TournamentManager) -> None:
    """Initialize tournament API with manager instance"""
    global tournament_manager
    tournament_manager = manager


def _serialize_tournament(tournament) -> Dict[str, Any]:
    """Serialize tournament object for JSON response"""
    return {
        "id": tournament.id,
        "name": tournament.config.name,
        "description": tournament.config.description,
        "type": tournament.config.tournament_type.value,
        "status": tournament.status.value,
        "current_round": tournament.current_round,
        "total_rounds": tournament.total_rounds,
        "participants": len(tournament.participants),
        "matches": len(tournament.matches),
        "completed_matches": tournament.completed_matches,
        "champion_id": tournament.champion_id,
        "runner_up_id": tournament.runner_up_id,
        "created_at": tournament.created_at.isoformat() if tournament.created_at else None,
        "actual_start": tournament.actual_start.isoformat() if tournament.actual_start else None,
        "completed_at": tournament.completed_at.isoformat() if tournament.completed_at else None
    }


def _serialize_participant(participant) -> Dict[str, Any]:
    """Serialize participant object for JSON response"""
    return {
        "id": participant.id,
        "name": participant.name,
        "description": participant.description,
        "type": participant.participant_type.value,
        "matches_played": participant.matches_played,
        "wins": participant.wins,
        "losses": participant.losses,
        "draws": participant.draws,
        "win_rate": participant.win_rate,
        "points": participant.points,
        "ranking": participant.ranking,
        "configuration": participant.configuration,
        "created_at": participant.created_at.isoformat() if participant.created_at else None
    }


def _serialize_match(match) -> Dict[str, Any]:
    """Serialize match object for JSON response"""
    return {
        "id": match.id,
        "tournament_id": match.tournament_id,
        "round_number": match.round_number,
        "match_number": match.match_number,
        "participant1_id": match.participant1_id,
        "participant2_id": match.participant2_id,
        "status": match.status.value,
        "winner_id": match.winner_id,
        "participant1_score": match.participant1_score,
        "participant2_score": match.participant2_score,
        "experiment_id": match.experiment_id,
        "is_bye": match.is_bye,
        "scheduled_start": match.scheduled_start.isoformat() if match.scheduled_start else None,
        "actual_start": match.actual_start.isoformat() if match.actual_start else None,
        "completed_at": match.completed_at.isoformat() if match.completed_at else None,
        "duration_seconds": match.duration_seconds
    }


@tournament_bp.route('/tournaments', methods=['GET'])
@require_auth
def list_tournaments():
    """List all tournaments"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournaments = tournament_manager.list_tournaments()

        return jsonify({
            "tournaments": [_serialize_tournament(t) for t in tournaments],
            "total": len(tournaments)
        })

    except Exception as e:
        logging.error(f"Error listing tournaments: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/active', methods=['GET'])
@require_auth
def list_active_tournaments():
    """List active tournaments"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        active_tournaments = tournament_manager.get_active_tournaments()

        return jsonify({
            "active_tournaments": [_serialize_tournament(t) for t in active_tournaments],
            "total": len(active_tournaments)
        })

    except Exception as e:
        logging.error(f"Error listing active tournaments: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments', methods=['POST'])
@require_auth
def create_tournament():
    """Create a new tournament"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create tournament configuration
        config = TournamentConfig(
            name=data.get("name", "New Tournament"),
            description=data.get("description", ""),
            tournament_type=TournamentType(data.get("type", "single_elimination")),
            max_participants=data.get("max_participants", 16),
            min_participants=data.get("min_participants", 4),
            match_duration_minutes=data.get("match_duration_minutes", 30),
            max_actions_per_match=data.get("max_actions_per_match", 1000),
            save_state_path=data.get("save_state_path"),
            primary_metric=data.get("primary_metric", "reward"),
            secondary_metrics=data.get("secondary_metrics", []),
            auto_advance=data.get("auto_advance", True),
            round_delay_minutes=data.get("round_delay_minutes", 5)
        )

        tournament_id = tournament_manager.create_tournament(config)
        tournament = tournament_manager.get_tournament(tournament_id)

        return jsonify({
            "message": "Tournament created successfully",
            "tournament": _serialize_tournament(tournament)
        }), 201

    except ValueError as e:
        return jsonify({"error": f"Invalid tournament type: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Error creating tournament: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>', methods=['GET'])
@require_auth
def get_tournament(tournament_id: str):
    """Get tournament details"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            return jsonify({"error": "Tournament not found"}), 404

        # Include detailed information
        response = _serialize_tournament(tournament)
        response.update({
            "participants": [_serialize_participant(p) for p in tournament.participants.values()],
            "matches": [_serialize_match(m) for m in tournament.matches.values()],
            "bracket": {
                "rounds": tournament.bracket.rounds,
                "total_rounds": tournament.bracket.total_rounds,
                "current_round": tournament.bracket.current_round
            },
            "final_rankings": tournament.final_rankings,
            "statistics": tournament.tournament_statistics
        })

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error getting tournament: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/participants', methods=['POST'])
@require_auth
def add_participant(tournament_id: str):
    """Add participant to tournament"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create participant
        participant = TournamentParticipant(
            name=data.get("name", "Unnamed Participant"),
            description=data.get("description", ""),
            participant_type=ParticipantType(data.get("type", "ai_configuration")),
            configuration=data.get("configuration", {})
        )

        success = tournament_manager.add_participant(tournament_id, participant)
        if not success:
            return jsonify({"error": "Failed to add participant"}), 400

        return jsonify({
            "message": "Participant added successfully",
            "participant": _serialize_participant(participant)
        }), 201

    except ValueError as e:
        return jsonify({"error": f"Invalid participant type: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Error adding participant: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/start', methods=['POST'])
@require_auth
def start_tournament(tournament_id: str):
    """Start a tournament"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        success = tournament_manager.start_tournament(tournament_id)
        if not success:
            return jsonify({"error": "Failed to start tournament"}), 400

        tournament = tournament_manager.get_tournament(tournament_id)
        return jsonify({
            "message": "Tournament started successfully",
            "tournament": _serialize_tournament(tournament)
        })

    except Exception as e:
        logging.error(f"Error starting tournament: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/matches', methods=['GET'])
@require_auth
def get_tournament_matches(tournament_id: str):
    """Get tournament matches"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            return jsonify({"error": "Tournament not found"}), 404

        round_number = request.args.get("round", type=int)

        if round_number:
            # Get matches for specific round
            round_matches = tournament.bracket.get_round_matches(round_number)
            matches = [tournament.matches[mid] for mid in round_matches]
        else:
            # Get all matches
            matches = list(tournament.matches.values())

        return jsonify({
            "matches": [_serialize_match(m) for m in matches],
            "total": len(matches)
        })

    except Exception as e:
        logging.error(f"Error getting tournament matches: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/bracket', methods=['GET'])
@require_auth
def get_tournament_bracket(tournament_id: str):
    """Get tournament bracket structure"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            return jsonify({"error": "Tournament not found"}), 404

        # Build detailed bracket with participant names
        bracket_data = {
            "tournament_id": tournament.id,
            "total_rounds": tournament.bracket.total_rounds,
            "current_round": tournament.bracket.current_round,
            "rounds": []
        }

        for round_num, match_ids in enumerate(tournament.bracket.rounds, 1):
            round_data = {
                "round_number": round_num,
                "matches": []
            }

            for match_id in match_ids:
                match = tournament.matches[match_id]
                participant1_name = ""
                participant2_name = ""

                if match.participant1_id in tournament.participants:
                    participant1_name = tournament.participants[match.participant1_id].name

                if match.participant2_id and match.participant2_id in tournament.participants:
                    participant2_name = tournament.participants[match.participant2_id].name

                match_data = _serialize_match(match)
                match_data.update({
                    "participant1_name": participant1_name,
                    "participant2_name": participant2_name
                })

                round_data["matches"].append(match_data)

            bracket_data["rounds"].append(round_data)

        return jsonify(bracket_data)

    except Exception as e:
        logging.error(f"Error getting tournament bracket: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/standings', methods=['GET'])
@require_auth
def get_tournament_standings(tournament_id: str):
    """Get tournament standings/rankings"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            return jsonify({"error": "Tournament not found"}), 404

        # Sort participants by ranking
        participants = list(tournament.participants.values())

        if tournament.config.tournament_type in [TournamentType.ROUND_ROBIN, TournamentType.SWISS_SYSTEM]:
            # Sort by points, then by score
            participants.sort(key=lambda p: (p.points, p.total_score / max(p.matches_played, 1)), reverse=True)
        else:
            # Sort by ranking (elimination tournaments)
            participants.sort(key=lambda p: p.ranking if p.ranking else 999)

        standings = []
        for i, participant in enumerate(participants):
            standing = _serialize_participant(participant)
            standing["position"] = i + 1
            standings.append(standing)

        return jsonify({
            "standings": standings,
            "tournament_type": tournament.config.tournament_type.value,
            "champion_id": tournament.champion_id,
            "runner_up_id": tournament.runner_up_id
        })

    except Exception as e:
        logging.error(f"Error getting tournament standings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournaments/<tournament_id>/status', methods=['GET'])
@require_auth
def get_tournament_status(tournament_id: str):
    """Get tournament status summary"""
    try:
        if not tournament_manager:
            return jsonify({"error": "Tournament manager not initialized"}), 500

        tournament = tournament_manager.get_tournament(tournament_id)
        if not tournament:
            return jsonify({"error": "Tournament not found"}), 404

        # Calculate progress
        progress_percentage = 0.0
        if tournament.total_matches > 0:
            progress_percentage = (tournament.completed_matches / tournament.total_matches) * 100

        # Get current round status
        current_round_matches = tournament.bracket.get_round_matches(tournament.current_round)
        current_round_completed = 0

        for match_id in current_round_matches:
            if tournament.matches[match_id].is_completed:
                current_round_completed += 1

        return jsonify({
            "tournament_id": tournament.id,
            "status": tournament.status.value,
            "progress_percentage": progress_percentage,
            "current_round": tournament.current_round,
            "total_rounds": tournament.total_rounds,
            "current_round_progress": {
                "completed": current_round_completed,
                "total": len(current_round_matches),
                "percentage": (current_round_completed / max(len(current_round_matches), 1)) * 100
            },
            "matches": {
                "total": tournament.total_matches,
                "completed": tournament.completed_matches,
                "remaining": tournament.total_matches - tournament.completed_matches
            },
            "participants": len(tournament.participants),
            "is_active": tournament.is_running,
            "can_start": tournament.can_start
        })

    except Exception as e:
        logging.error(f"Error getting tournament status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@tournament_bp.route('/tournament/types', methods=['GET'])
@require_auth
def get_tournament_types():
    """Get available tournament types"""
    return jsonify({
        "tournament_types": [
            {
                "value": t.value,
                "name": t.value.replace("_", " ").title(),
                "description": _get_tournament_type_description(t)
            }
            for t in TournamentType
        ]
    })


def _get_tournament_type_description(tournament_type: TournamentType) -> str:
    """Get description for tournament type"""
    descriptions = {
        TournamentType.SINGLE_ELIMINATION: "Single elimination bracket - lose once and you're out",
        TournamentType.DOUBLE_ELIMINATION: "Double elimination bracket - two chances before elimination",
        TournamentType.ROUND_ROBIN: "Round robin format - everyone plays everyone",
        TournamentType.SWISS_SYSTEM: "Swiss system - dynamic pairings based on performance",
        TournamentType.LADDER: "Ladder tournament - ongoing ranking system"
    }
    return descriptions.get(tournament_type, "Tournament format")


@tournament_bp.route('/tournament/participant_types', methods=['GET'])
@require_auth
def get_participant_types():
    """Get available participant types"""
    return jsonify({
        "participant_types": [
            {
                "value": t.value,
                "name": t.value.replace("_", " ").title(),
                "description": _get_participant_type_description(t)
            }
            for t in ParticipantType
        ]
    })


def _get_participant_type_description(participant_type: ParticipantType) -> str:
    """Get description for participant type"""
    descriptions = {
        ParticipantType.AI_CONFIGURATION: "AI configuration with specific settings",
        ParticipantType.PLUGIN_COMBINATION: "Combination of plugins and strategies",
        ParticipantType.AGENT_STRATEGY: "Multi-agent strategy configuration",
        ParticipantType.HYBRID_SETUP: "Hybrid LLM-RL configuration"
    }
    return descriptions.get(participant_type, "Participant type")