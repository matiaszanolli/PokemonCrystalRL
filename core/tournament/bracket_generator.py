"""
Tournament Bracket Generator - Creates tournament structures

Generates bracket structures for different tournament formats including
single elimination, double elimination, round robin, and Swiss system.
"""

import logging
import math
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from .tournament_models import (
    Tournament,
    TournamentParticipant,
    TournamentMatch,
    TournamentBracket,
    TournamentType,
    MatchStatus
)


class BracketGenerator:
    """
    Generates tournament brackets for different formats.

    Supports:
    - Single elimination
    - Double elimination
    - Round robin
    - Swiss system
    """

    def __init__(self):
        self.logger = logging.getLogger("BracketGenerator")

    def generate_bracket(self, tournament: Tournament) -> bool:
        """
        Generate bracket structure based on tournament type.

        Args:
            tournament: Tournament to generate bracket for

        Returns:
            True if bracket generated successfully
        """
        try:
            participant_count = len(tournament.participants)

            if participant_count < tournament.config.min_participants:
                self.logger.error(f"Not enough participants: {participant_count} < {tournament.config.min_participants}")
                return False

            # Create list of participant IDs for bracket generation
            participant_ids = list(tournament.participants.keys())

            # Shuffle for random seeding (can be replaced with skill-based seeding later)
            random.shuffle(participant_ids)

            # Generate bracket based on tournament type
            if tournament.config.tournament_type == TournamentType.SINGLE_ELIMINATION:
                return self._generate_single_elimination(tournament, participant_ids)
            elif tournament.config.tournament_type == TournamentType.DOUBLE_ELIMINATION:
                return self._generate_double_elimination(tournament, participant_ids)
            elif tournament.config.tournament_type == TournamentType.ROUND_ROBIN:
                return self._generate_round_robin(tournament, participant_ids)
            elif tournament.config.tournament_type == TournamentType.SWISS_SYSTEM:
                return self._generate_swiss_system(tournament, participant_ids)
            else:
                self.logger.error(f"Unsupported tournament type: {tournament.config.tournament_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error generating bracket: {str(e)}")
            return False

    def _generate_single_elimination(self, tournament: Tournament, participant_ids: List[str]) -> bool:
        """Generate single elimination bracket"""
        participant_count = len(participant_ids)

        # Calculate tournament rounds (next power of 2)
        bracket_size = 2 ** math.ceil(math.log2(participant_count))
        tournament.total_rounds = int(math.log2(bracket_size))

        self.logger.info(f"Generating single elimination bracket for {participant_count} participants, {tournament.total_rounds} rounds")

        # Create first round with byes if needed
        first_round_matches = []
        byes_needed = bracket_size - participant_count

        # Pair participants for first round
        i = 0
        match_number = 1

        while i < len(participant_ids):
            match = TournamentMatch(
                tournament_id=tournament.id,
                round_number=1,
                match_number=match_number
            )

            match.participant1_id = participant_ids[i]

            # Check if we need a bye
            if i + 1 < len(participant_ids):
                match.participant2_id = participant_ids[i + 1]
                i += 2
            else:
                # This is a bye match
                match.participant2_id = ""
                match.status = MatchStatus.COMPLETED
                match.winner_id = match.participant1_id
                i += 1

            tournament.matches[match.id] = match
            first_round_matches.append(match.id)
            match_number += 1

        # Add first round to bracket
        tournament.bracket.rounds.append(first_round_matches)
        tournament.bracket.total_rounds = tournament.total_rounds

        # Generate subsequent rounds (placeholders)
        for round_num in range(2, tournament.total_rounds + 1):
            round_matches = []
            matches_in_round = len(first_round_matches) // (2 ** (round_num - 1))

            for match_num in range(1, matches_in_round + 1):
                match = TournamentMatch(
                    tournament_id=tournament.id,
                    round_number=round_num,
                    match_number=match_num,
                    status=MatchStatus.PENDING
                )

                tournament.matches[match.id] = match
                round_matches.append(match.id)

            tournament.bracket.rounds.append(round_matches)

        tournament.current_round = 1
        tournament.total_matches = len(tournament.matches)

        self.logger.info(f"Generated {tournament.total_matches} matches across {tournament.total_rounds} rounds")
        return True

    def _generate_double_elimination(self, tournament: Tournament, participant_ids: List[str]) -> bool:
        """Generate double elimination bracket (winners + losers brackets)"""
        # For now, implement as single elimination with plan for losers bracket
        # TODO: Implement full double elimination with winners and losers brackets
        self.logger.warning("Double elimination not fully implemented, using single elimination")
        return self._generate_single_elimination(tournament, participant_ids)

    def _generate_round_robin(self, tournament: Tournament, participant_ids: List[str]) -> bool:
        """Generate round robin bracket (everyone plays everyone)"""
        participant_count = len(participant_ids)

        # Calculate total matches: n * (n-1) / 2
        total_matches = participant_count * (participant_count - 1) // 2
        tournament.total_rounds = participant_count - 1  # Standard round robin rounds

        self.logger.info(f"Generating round robin bracket for {participant_count} participants, {total_matches} matches")

        # Generate all possible pairings
        matches = []
        match_number = 1

        for i in range(participant_count):
            for j in range(i + 1, participant_count):
                match = TournamentMatch(
                    tournament_id=tournament.id,
                    round_number=1,  # All matches in round 1 for simplicity
                    match_number=match_number,
                    participant1_id=participant_ids[i],
                    participant2_id=participant_ids[j]
                )

                tournament.matches[match.id] = match
                matches.append(match.id)
                match_number += 1

        # Add all matches to single round (can be optimized for scheduling)
        tournament.bracket.rounds.append(matches)
        tournament.bracket.total_rounds = 1
        tournament.current_round = 1
        tournament.total_matches = len(matches)

        self.logger.info(f"Generated {total_matches} matches for round robin")
        return True

    def _generate_swiss_system(self, tournament: Tournament, participant_ids: List[str]) -> bool:
        """Generate Swiss system bracket (dynamic pairing based on scores)"""
        participant_count = len(participant_ids)

        # Swiss system typically runs for log2(n) rounds
        tournament.total_rounds = max(3, int(math.log2(participant_count)))

        self.logger.info(f"Generating Swiss system bracket for {participant_count} participants, {tournament.total_rounds} rounds")

        # Generate first round with random pairings
        first_round_matches = []
        shuffled_participants = participant_ids.copy()
        random.shuffle(shuffled_participants)

        match_number = 1
        for i in range(0, len(shuffled_participants) - 1, 2):
            match = TournamentMatch(
                tournament_id=tournament.id,
                round_number=1,
                match_number=match_number,
                participant1_id=shuffled_participants[i],
                participant2_id=shuffled_participants[i + 1]
            )

            tournament.matches[match.id] = match
            first_round_matches.append(match.id)
            match_number += 1

        # Handle odd number of participants (bye)
        if len(shuffled_participants) % 2 == 1:
            bye_match = TournamentMatch(
                tournament_id=tournament.id,
                round_number=1,
                match_number=match_number,
                participant1_id=shuffled_participants[-1],
                participant2_id="",
                status=MatchStatus.COMPLETED,
                winner_id=shuffled_participants[-1]
            )
            tournament.matches[bye_match.id] = bye_match
            first_round_matches.append(bye_match.id)

        tournament.bracket.rounds.append(first_round_matches)
        tournament.bracket.total_rounds = tournament.total_rounds
        tournament.current_round = 1
        tournament.total_matches = len(first_round_matches)  # Will grow as rounds are generated

        # Subsequent rounds will be generated dynamically based on results
        self.logger.info(f"Generated first round with {len(first_round_matches)} matches")
        return True

    def advance_tournament(self, tournament: Tournament) -> bool:
        """
        Advance tournament to next round based on completed matches.

        Args:
            tournament: Tournament to advance

        Returns:
            True if successfully advanced
        """
        try:
            if tournament.config.tournament_type == TournamentType.SINGLE_ELIMINATION:
                return self._advance_single_elimination(tournament)
            elif tournament.config.tournament_type == TournamentType.ROUND_ROBIN:
                return self._advance_round_robin(tournament)
            elif tournament.config.tournament_type == TournamentType.SWISS_SYSTEM:
                return self._advance_swiss_system(tournament)
            else:
                self.logger.warning(f"Advance not implemented for {tournament.config.tournament_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error advancing tournament: {str(e)}")
            return False

    def _advance_single_elimination(self, tournament: Tournament) -> bool:
        """Advance single elimination tournament to next round"""
        current_round = tournament.current_round

        # Check if all matches in current round are completed
        current_round_matches = tournament.bracket.get_round_matches(current_round)
        if not current_round_matches:
            return False

        winners = []
        for match_id in current_round_matches:
            match = tournament.matches[match_id]
            if not match.is_completed:
                self.logger.info(f"Match {match_id} not completed, cannot advance")
                return False

            if match.winner_id:
                winners.append(match.winner_id)

        # Check if tournament is completed
        if len(winners) == 1:
            tournament.champion_id = winners[0]
            return True

        # Create next round matches
        next_round = current_round + 1
        if next_round > tournament.total_rounds:
            return False

        next_round_matches = tournament.bracket.get_round_matches(next_round)

        # Pair winners for next round
        for i, match_id in enumerate(next_round_matches):
            match = tournament.matches[match_id]

            # Assign winners to next round matches
            if i * 2 < len(winners):
                match.participant1_id = winners[i * 2]
            if i * 2 + 1 < len(winners):
                match.participant2_id = winners[i * 2 + 1]
            else:
                # Bye for odd number of winners
                match.participant2_id = ""
                match.status = MatchStatus.COMPLETED
                match.winner_id = match.participant1_id

        tournament.current_round = next_round
        self.logger.info(f"Advanced to round {next_round} with {len(winners)} winners")
        return True

    def _advance_round_robin(self, tournament: Tournament) -> bool:
        """Check if round robin is completed and calculate final standings"""
        # In round robin, all matches are in round 1
        all_matches = tournament.bracket.get_round_matches(1)

        for match_id in all_matches:
            match = tournament.matches[match_id]
            if not match.is_completed:
                return False

        # Calculate final standings
        self._calculate_round_robin_standings(tournament)
        return True

    def _advance_swiss_system(self, tournament: Tournament) -> bool:
        """Generate next round for Swiss system based on current standings"""
        current_round = tournament.current_round
        current_round_matches = tournament.bracket.get_round_matches(current_round)

        # Check if all matches in current round are completed
        for match_id in current_round_matches:
            match = tournament.matches[match_id]
            if not match.is_completed:
                return False

        # Check if tournament is completed
        if current_round >= tournament.total_rounds:
            self._calculate_swiss_standings(tournament)
            return True

        # Generate next round pairings based on current scores
        return self._generate_swiss_round(tournament, current_round + 1)

    def _generate_swiss_round(self, tournament: Tournament, round_number: int) -> bool:
        """Generate a new round for Swiss system tournament"""
        # Sort participants by score
        participants_by_score = sorted(
            tournament.participants.values(),
            key=lambda p: p.points,
            reverse=True
        )

        # Simple pairing: pair participants with similar scores
        round_matches = []
        paired_participants = set()
        match_number = 1

        for i, participant in enumerate(participants_by_score):
            if participant.id in paired_participants:
                continue

            # Find best opponent (closest score, not already played)
            opponent = None
            for j in range(i + 1, len(participants_by_score)):
                candidate = participants_by_score[j]
                if candidate.id not in paired_participants:
                    # TODO: Check if they've already played (for now, just pair them)
                    opponent = candidate
                    break

            if opponent:
                match = TournamentMatch(
                    tournament_id=tournament.id,
                    round_number=round_number,
                    match_number=match_number,
                    participant1_id=participant.id,
                    participant2_id=opponent.id
                )

                tournament.matches[match.id] = match
                round_matches.append(match.id)
                paired_participants.add(participant.id)
                paired_participants.add(opponent.id)
                match_number += 1

        # Handle bye if odd number of participants
        unpaired = [p for p in participants_by_score if p.id not in paired_participants]
        if unpaired:
            bye_match = TournamentMatch(
                tournament_id=tournament.id,
                round_number=round_number,
                match_number=match_number,
                participant1_id=unpaired[0].id,
                participant2_id="",
                status=MatchStatus.COMPLETED,
                winner_id=unpaired[0].id
            )
            tournament.matches[bye_match.id] = bye_match
            round_matches.append(bye_match.id)

        # Add round to bracket
        if len(tournament.bracket.rounds) < round_number:
            tournament.bracket.rounds.append(round_matches)
        else:
            tournament.bracket.rounds[round_number - 1] = round_matches

        tournament.current_round = round_number
        tournament.total_matches += len(round_matches)

        self.logger.info(f"Generated Swiss round {round_number} with {len(round_matches)} matches")
        return True

    def _calculate_round_robin_standings(self, tournament: Tournament) -> None:
        """Calculate final standings for round robin tournament"""
        # Sort by points, then by tiebreakers
        sorted_participants = sorted(
            tournament.participants.values(),
            key=lambda p: (p.points, p.total_score / max(p.matches_played, 1)),
            reverse=True
        )

        tournament.final_rankings = [p.id for p in sorted_participants]
        if tournament.final_rankings:
            tournament.champion_id = tournament.final_rankings[0]
            if len(tournament.final_rankings) > 1:
                tournament.runner_up_id = tournament.final_rankings[1]

        # Update rankings in participant objects
        for i, participant in enumerate(sorted_participants):
            participant.ranking = i + 1

    def _calculate_swiss_standings(self, tournament: Tournament) -> None:
        """Calculate final standings for Swiss system tournament"""
        self._calculate_round_robin_standings(tournament)  # Same logic for now