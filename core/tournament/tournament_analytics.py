"""
Tournament Analytics - Performance analysis and insights

Provides comprehensive analytics for tournament results, participant
performance, strategy effectiveness, and competitive insights.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .tournament_models import Tournament, TournamentParticipant, TournamentMatch, TournamentType


@dataclass
class PerformanceInsight:
    """Individual performance insight"""
    metric: str
    value: float
    description: str
    confidence: float
    category: str  # "strength", "weakness", "opportunity", "trend"


@dataclass
class StrategyAnalysis:
    """Analysis of strategy effectiveness"""
    strategy_name: str
    configuration: Dict[str, Any]
    tournaments_participated: int
    average_win_rate: float
    average_ranking: float
    strengths: List[str]
    weaknesses: List[str]
    recommended_improvements: List[str]
    performance_insights: List[PerformanceInsight]


@dataclass
class TournamentInsights:
    """Comprehensive tournament insights"""
    tournament_id: str
    tournament_name: str

    # Overall statistics
    total_participants: int
    total_matches: int
    average_match_duration: float
    tournament_duration_hours: float

    # Performance metrics
    most_decisive_match: Optional[str]  # Largest score difference
    closest_match: Optional[str]        # Smallest score difference
    highest_scoring_match: Optional[str]
    longest_match: Optional[str]

    # Participant insights
    dominant_participant: Optional[str]  # Best overall performance
    underdog_performer: Optional[str]   # Exceeded expectations
    most_improved: Optional[str]        # Best progression during tournament

    # Strategy insights
    most_effective_strategy: Optional[str]
    strategy_diversity_score: float     # How diverse were the strategies

    # Meta insights
    competitive_balance: float          # How balanced was the competition
    predictability_score: float        # How predictable were the results
    entertainment_value: float         # How exciting were the matches


class TournamentAnalytics:
    """
    Advanced analytics engine for tournament data.

    Provides insights into:
    - Individual participant performance
    - Strategy effectiveness across tournaments
    - Tournament competitiveness and balance
    - Meta-game trends and patterns
    """

    def __init__(self):
        self.logger = logging.getLogger("TournamentAnalytics")

    def analyze_tournament(self, tournament: Tournament) -> TournamentInsights:
        """
        Generate comprehensive insights for a completed tournament.

        Args:
            tournament: Completed tournament to analyze

        Returns:
            Detailed tournament insights
        """
        try:
            if not tournament.is_completed:
                self.logger.warning(f"Tournament {tournament.id} is not completed")

            insights = TournamentInsights(
                tournament_id=tournament.id,
                tournament_name=tournament.config.name,
                total_participants=len(tournament.participants),
                total_matches=tournament.total_matches,
                average_match_duration=self._calculate_average_match_duration(tournament),
                tournament_duration_hours=self._calculate_tournament_duration(tournament),
                most_decisive_match=self._find_most_decisive_match(tournament),
                closest_match=self._find_closest_match(tournament),
                highest_scoring_match=self._find_highest_scoring_match(tournament),
                longest_match=self._find_longest_match(tournament),
                dominant_participant=self._find_dominant_participant(tournament),
                underdog_performer=self._find_underdog_performer(tournament),
                most_improved=self._find_most_improved_participant(tournament),
                most_effective_strategy=self._find_most_effective_strategy(tournament),
                strategy_diversity_score=self._calculate_strategy_diversity(tournament),
                competitive_balance=self._calculate_competitive_balance(tournament),
                predictability_score=self._calculate_predictability_score(tournament),
                entertainment_value=self._calculate_entertainment_value(tournament)
            )

            return insights

        except Exception as e:
            self.logger.error(f"Error analyzing tournament: {str(e)}")
            return TournamentInsights(
                tournament_id=tournament.id,
                tournament_name=tournament.config.name,
                total_participants=0,
                total_matches=0,
                average_match_duration=0.0,
                tournament_duration_hours=0.0,
                strategy_diversity_score=0.0,
                competitive_balance=0.0,
                predictability_score=0.0,
                entertainment_value=0.0
            )

    def analyze_participant_performance(self, participant: TournamentParticipant,
                                      tournament: Tournament) -> List[PerformanceInsight]:
        """
        Analyze individual participant performance and generate insights.

        Args:
            participant: Participant to analyze
            tournament: Tournament context

        Returns:
            List of performance insights
        """
        insights = []

        try:
            # Win rate analysis
            if participant.win_rate > 0.75:
                insights.append(PerformanceInsight(
                    metric="win_rate",
                    value=participant.win_rate,
                    description=f"Exceptional win rate of {participant.win_rate:.1%}",
                    confidence=0.9,
                    category="strength"
                ))
            elif participant.win_rate < 0.25:
                insights.append(PerformanceInsight(
                    metric="win_rate",
                    value=participant.win_rate,
                    description=f"Low win rate of {participant.win_rate:.1%} suggests strategy needs refinement",
                    confidence=0.8,
                    category="weakness"
                ))

            # Performance consistency
            participant_matches = [m for m in tournament.matches.values()
                                 if m.participant1_id == participant.id or m.participant2_id == participant.id]

            if len(participant_matches) >= 3:
                scores = []
                for match in participant_matches:
                    if match.participant1_id == participant.id and match.participant1_score is not None:
                        scores.append(match.participant1_score)
                    elif match.participant2_id == participant.id and match.participant2_score is not None:
                        scores.append(match.participant2_score)

                if scores:
                    score_variance = statistics.variance(scores) if len(scores) > 1 else 0
                    if score_variance < 10.0:  # Low variance
                        insights.append(PerformanceInsight(
                            metric="consistency",
                            value=1.0 - (score_variance / 100.0),
                            description="Highly consistent performance across matches",
                            confidence=0.8,
                            category="strength"
                        ))
                    elif score_variance > 50.0:  # High variance
                        insights.append(PerformanceInsight(
                            metric="consistency",
                            value=1.0 - (score_variance / 100.0),
                            description="Inconsistent performance suggests unstable strategy",
                            confidence=0.7,
                            category="weakness"
                        ))

            # Average score performance
            avg_score = participant.total_score / max(participant.matches_played, 1)
            tournament_avg = self._calculate_tournament_average_score(tournament)

            if avg_score > tournament_avg * 1.2:
                insights.append(PerformanceInsight(
                    metric="average_score",
                    value=avg_score,
                    description=f"Above-average scoring performance ({avg_score:.1f} vs {tournament_avg:.1f} tournament avg)",
                    confidence=0.8,
                    category="strength"
                ))
            elif avg_score < tournament_avg * 0.8:
                insights.append(PerformanceInsight(
                    metric="average_score",
                    value=avg_score,
                    description=f"Below-average scoring performance ({avg_score:.1f} vs {tournament_avg:.1f} tournament avg)",
                    confidence=0.8,
                    category="weakness"
                ))

            return insights

        except Exception as e:
            self.logger.error(f"Error analyzing participant performance: {str(e)}")
            return []

    def analyze_strategy_across_tournaments(self, tournaments: List[Tournament],
                                          strategy_identifier: str) -> StrategyAnalysis:
        """
        Analyze strategy performance across multiple tournaments.

        Args:
            tournaments: List of tournaments to analyze
            strategy_identifier: Strategy configuration key to identify strategy

        Returns:
            Strategy analysis results
        """
        try:
            # Find participants using this strategy
            strategy_participants = []
            for tournament in tournaments:
                for participant in tournament.participants.values():
                    if strategy_identifier in str(participant.configuration):
                        strategy_participants.append((participant, tournament))

            if not strategy_participants:
                return StrategyAnalysis(
                    strategy_name=strategy_identifier,
                    configuration={},
                    tournaments_participated=0,
                    average_win_rate=0.0,
                    average_ranking=0.0,
                    strengths=[],
                    weaknesses=[],
                    recommended_improvements=[],
                    performance_insights=[]
                )

            # Calculate aggregate statistics
            win_rates = [p.win_rate for p, _ in strategy_participants]
            rankings = [p.ranking for p, _ in strategy_participants if p.ranking]

            average_win_rate = statistics.mean(win_rates) if win_rates else 0.0
            average_ranking = statistics.mean(rankings) if rankings else 0.0

            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            recommendations = []

            if average_win_rate > 0.6:
                strengths.append("High win rate across tournaments")
            elif average_win_rate < 0.4:
                weaknesses.append("Low win rate across tournaments")
                recommendations.append("Review core strategy parameters")

            if average_ranking and average_ranking <= 2.0:
                strengths.append("Consistently high tournament rankings")
            elif average_ranking and average_ranking > len(strategy_participants) * 0.7:
                weaknesses.append("Consistently low tournament rankings")
                recommendations.append("Consider strategy overhaul")

            # Generate insights
            insights = []
            for participant, tournament in strategy_participants:
                participant_insights = self.analyze_participant_performance(participant, tournament)
                insights.extend(participant_insights)

            return StrategyAnalysis(
                strategy_name=strategy_identifier,
                configuration=strategy_participants[0][0].configuration if strategy_participants else {},
                tournaments_participated=len(set(t.id for _, t in strategy_participants)),
                average_win_rate=average_win_rate,
                average_ranking=average_ranking,
                strengths=strengths,
                weaknesses=weaknesses,
                recommended_improvements=recommendations,
                performance_insights=insights[:10]  # Top 10 insights
            )

        except Exception as e:
            self.logger.error(f"Error analyzing strategy: {str(e)}")
            return StrategyAnalysis(
                strategy_name=strategy_identifier,
                configuration={},
                tournaments_participated=0,
                average_win_rate=0.0,
                average_ranking=0.0,
                strengths=[],
                weaknesses=[],
                recommended_improvements=[],
                performance_insights=[]
            )

    def generate_meta_insights(self, tournaments: List[Tournament]) -> Dict[str, Any]:
        """
        Generate meta-game insights across multiple tournaments.

        Args:
            tournaments: List of tournaments to analyze

        Returns:
            Meta-game insights and trends
        """
        try:
            insights = {
                "total_tournaments": len(tournaments),
                "total_participants": sum(len(t.participants) for t in tournaments),
                "total_matches": sum(t.total_matches for t in tournaments),
                "strategy_trends": self._analyze_strategy_trends(tournaments),
                "performance_trends": self._analyze_performance_trends(tournaments),
                "tournament_format_effectiveness": self._analyze_format_effectiveness(tournaments),
                "competitive_evolution": self._analyze_competitive_evolution(tournaments)
            }

            return insights

        except Exception as e:
            self.logger.error(f"Error generating meta insights: {str(e)}")
            return {}

    def _calculate_average_match_duration(self, tournament: Tournament) -> float:
        """Calculate average match duration in seconds"""
        durations = [m.duration_seconds for m in tournament.matches.values()
                    if m.duration_seconds is not None]
        return statistics.mean(durations) if durations else 0.0

    def _calculate_tournament_duration(self, tournament: Tournament) -> float:
        """Calculate total tournament duration in hours"""
        if tournament.actual_start and tournament.completed_at:
            return (tournament.completed_at - tournament.actual_start).total_seconds() / 3600
        return 0.0

    def _find_most_decisive_match(self, tournament: Tournament) -> Optional[str]:
        """Find match with largest score difference"""
        max_diff = 0.0
        most_decisive = None

        for match in tournament.matches.values():
            if (match.participant1_score is not None and
                match.participant2_score is not None):
                diff = abs(match.participant1_score - match.participant2_score)
                if diff > max_diff:
                    max_diff = diff
                    most_decisive = match.id

        return most_decisive

    def _find_closest_match(self, tournament: Tournament) -> Optional[str]:
        """Find match with smallest score difference"""
        min_diff = float('inf')
        closest = None

        for match in tournament.matches.values():
            if (match.participant1_score is not None and
                match.participant2_score is not None):
                diff = abs(match.participant1_score - match.participant2_score)
                if diff < min_diff:
                    min_diff = diff
                    closest = match.id

        return closest

    def _find_highest_scoring_match(self, tournament: Tournament) -> Optional[str]:
        """Find match with highest combined score"""
        max_score = 0.0
        highest = None

        for match in tournament.matches.values():
            if (match.participant1_score is not None and
                match.participant2_score is not None):
                total_score = match.participant1_score + match.participant2_score
                if total_score > max_score:
                    max_score = total_score
                    highest = match.id

        return highest

    def _find_longest_match(self, tournament: Tournament) -> Optional[str]:
        """Find match with longest duration"""
        max_duration = 0.0
        longest = None

        for match in tournament.matches.values():
            if match.duration_seconds and match.duration_seconds > max_duration:
                max_duration = match.duration_seconds
                longest = match.id

        return longest

    def _find_dominant_participant(self, tournament: Tournament) -> Optional[str]:
        """Find participant with best overall performance"""
        best_score = 0.0
        dominant = None

        for participant in tournament.participants.values():
            # Weighted score: win rate * 0.6 + (points / max_possible_points) * 0.4
            max_possible_points = participant.matches_played * 3
            points_ratio = participant.points / max(max_possible_points, 1)
            composite_score = participant.win_rate * 0.6 + points_ratio * 0.4

            if composite_score > best_score:
                best_score = composite_score
                dominant = participant.id

        return dominant

    def _find_underdog_performer(self, tournament: Tournament) -> Optional[str]:
        """Find participant who exceeded expectations"""
        # Simple heuristic: participant with lowest expected performance but good results
        # This could be enhanced with more sophisticated analysis
        participants_by_config_complexity = sorted(
            tournament.participants.values(),
            key=lambda p: len(str(p.configuration))
        )

        # Look for simple configs with good performance
        for participant in participants_by_config_complexity[:3]:
            if participant.win_rate > 0.5 and participant.ranking and participant.ranking <= 3:
                return participant.id

        return None

    def _find_most_improved_participant(self, tournament: Tournament) -> Optional[str]:
        """Find participant with best improvement during tournament"""
        # For now, return participant with best second half vs first half performance
        # This would require more detailed match-by-match tracking in real implementation
        return None

    def _find_most_effective_strategy(self, tournament: Tournament) -> Optional[str]:
        """Find most effective strategy configuration"""
        strategy_performance = defaultdict(list)

        for participant in tournament.participants.values():
            strategy_key = str(sorted(participant.configuration.items()))
            strategy_performance[strategy_key].append(participant.win_rate)

        best_strategy = None
        best_avg_performance = 0.0

        for strategy, performances in strategy_performance.items():
            avg_performance = statistics.mean(performances)
            if avg_performance > best_avg_performance:
                best_avg_performance = avg_performance
                best_strategy = strategy

        return best_strategy

    def _calculate_strategy_diversity(self, tournament: Tournament) -> float:
        """Calculate how diverse the strategies were (0.0 = all same, 1.0 = all different)"""
        unique_configs = set()
        for participant in tournament.participants.values():
            config_str = str(sorted(participant.configuration.items()))
            unique_configs.add(config_str)

        return len(unique_configs) / max(len(tournament.participants), 1)

    def _calculate_competitive_balance(self, tournament: Tournament) -> float:
        """Calculate how balanced the competition was (0.0 = one-sided, 1.0 = perfectly balanced)"""
        win_rates = [p.win_rate for p in tournament.participants.values()]
        if not win_rates:
            return 0.0

        # Lower variance in win rates = more balanced
        variance = statistics.variance(win_rates) if len(win_rates) > 1 else 0
        return max(0.0, 1.0 - variance)

    def _calculate_predictability_score(self, tournament: Tournament) -> float:
        """Calculate how predictable the results were (0.0 = unpredictable, 1.0 = very predictable)"""
        # Simplified: based on how often the "favorite" won
        # In real implementation, this would require pre-tournament predictions
        return 0.5  # Placeholder

    def _calculate_entertainment_value(self, tournament: Tournament) -> float:
        """Calculate entertainment value (0.0 = boring, 1.0 = very exciting)"""
        # Factors: close matches, upsets, duration variety
        factors = []

        # Close matches factor
        close_matches = 0
        total_completed = 0
        for match in tournament.matches.values():
            if (match.participant1_score is not None and
                match.participant2_score is not None):
                total_completed += 1
                diff = abs(match.participant1_score - match.participant2_score)
                if diff < 10.0:  # Arbitrary threshold for "close"
                    close_matches += 1

        if total_completed > 0:
            close_match_ratio = close_matches / total_completed
            factors.append(close_match_ratio)

        # Strategy diversity factor
        factors.append(self._calculate_strategy_diversity(tournament))

        # Competitive balance factor
        factors.append(self._calculate_competitive_balance(tournament))

        return statistics.mean(factors) if factors else 0.5

    def _calculate_tournament_average_score(self, tournament: Tournament) -> float:
        """Calculate average score across all tournament matches"""
        all_scores = []
        for match in tournament.matches.values():
            if match.participant1_score is not None:
                all_scores.append(match.participant1_score)
            if match.participant2_score is not None:
                all_scores.append(match.participant2_score)

        return statistics.mean(all_scores) if all_scores else 0.0

    def _analyze_strategy_trends(self, tournaments: List[Tournament]) -> Dict[str, Any]:
        """Analyze strategy trends across tournaments"""
        # Placeholder for strategy trend analysis
        return {
            "emerging_strategies": [],
            "declining_strategies": [],
            "stable_strategies": []
        }

    def _analyze_performance_trends(self, tournaments: List[Tournament]) -> Dict[str, Any]:
        """Analyze performance trends across tournaments"""
        # Placeholder for performance trend analysis
        return {
            "average_scores_trend": "stable",
            "win_rate_distribution_change": "narrowing",
            "match_duration_trend": "decreasing"
        }

    def _analyze_format_effectiveness(self, tournaments: List[Tournament]) -> Dict[str, Any]:
        """Analyze effectiveness of different tournament formats"""
        format_stats = defaultdict(list)

        for tournament in tournaments:
            format_stats[tournament.config.tournament_type.value].append({
                "entertainment_value": self._calculate_entertainment_value(tournament),
                "competitive_balance": self._calculate_competitive_balance(tournament),
                "duration_hours": self._calculate_tournament_duration(tournament)
            })

        format_analysis = {}
        for format_type, stats in format_stats.items():
            if stats:
                format_analysis[format_type] = {
                    "average_entertainment": statistics.mean([s["entertainment_value"] for s in stats]),
                    "average_balance": statistics.mean([s["competitive_balance"] for s in stats]),
                    "average_duration": statistics.mean([s["duration_hours"] for s in stats]),
                    "tournaments_count": len(stats)
                }

        return format_analysis

    def _analyze_competitive_evolution(self, tournaments: List[Tournament]) -> Dict[str, Any]:
        """Analyze how competitive landscape has evolved"""
        # Placeholder for competitive evolution analysis
        return {
            "skill_ceiling_trend": "rising",
            "meta_stability": "evolving",
            "new_entrant_success_rate": 0.3
        }