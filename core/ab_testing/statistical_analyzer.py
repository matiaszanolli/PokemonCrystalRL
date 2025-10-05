"""
Statistical Analyzer - Statistical analysis for A/B testing results

This module provides comprehensive statistical analysis for A/B testing experiments,
including significance testing, confidence intervals, and effect size calculations.
"""

import logging
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
import numpy as np

from .experiment_models import PerformanceMetrics, ExperimentResult


class StatisticalTest(Enum):
    """Types of statistical tests available"""
    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"


class EffectSizeMethod(Enum):
    """Methods for calculating effect size"""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"
    CLIFF_DELTA = "cliff_delta"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    effect_size_method: EffectSizeMethod
    confidence_interval: Tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    power: Optional[float] = None
    interpretation: str = ""


@dataclass
class AnalysisResult:
    """Comprehensive analysis result for an experiment"""
    experiment_id: str
    primary_metric_results: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    secondary_metric_results: Dict[str, StatisticalTestResult] = field(default_factory=dict)

    # Overall conclusions
    has_significant_results: bool = False
    winning_variant: Optional[str] = None
    confidence_score: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""

    # Meta-analysis
    sample_size_adequacy: Dict[str, bool] = field(default_factory=dict)
    power_analysis: Dict[str, float] = field(default_factory=dict)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis system for A/B testing experiments.

    Provides significance testing, effect size calculation, confidence intervals,
    and power analysis for experimental results.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the statistical analyzer"""
        self.config = config or {}
        self.logger = logging.getLogger("StatisticalAnalyzer")

        # Default analysis parameters
        self.default_confidence_level = self.config.get('confidence_level', 0.95)
        self.default_alpha = 1 - self.default_confidence_level
        self.minimum_sample_size = self.config.get('minimum_sample_size', 10)
        self.default_effect_size_method = EffectSizeMethod.COHENS_D

        self.logger.info("StatisticalAnalyzer initialized")

    def analyze_experiment(self, experiment_result: ExperimentResult) -> AnalysisResult:
        """
        Perform comprehensive statistical analysis on experiment results.

        Args:
            experiment_result: Results from completed experiment

        Returns:
            AnalysisResult with comprehensive statistical analysis
        """
        analysis = AnalysisResult(experiment_id=experiment_result.experiment_id)

        if len(experiment_result.variant_results) < 2:
            analysis.summary = "Insufficient variants for comparison"
            return analysis

        # Get variant names and results
        variant_names = list(experiment_result.variant_results.keys())
        control_variant = variant_names[0]  # First variant is control

        # Analyze primary metrics
        primary_metrics = ['total_reward']  # Default primary metric
        for metric_name in primary_metrics:
            analysis.primary_metric_results[metric_name] = self._analyze_metric_across_variants(
                experiment_result, metric_name, control_variant
            )

        # Analyze secondary metrics
        secondary_metrics = ['actions_per_second', 'battle_win_rate']
        for metric_name in secondary_metrics:
            if self._has_metric_data(experiment_result, metric_name):
                analysis.secondary_metric_results[metric_name] = self._analyze_metric_across_variants(
                    experiment_result, metric_name, control_variant
                )

        # Determine overall significance
        analysis.has_significant_results = any(
            result.is_significant for result in analysis.primary_metric_results.values()
        )

        # Determine winning variant
        if analysis.has_significant_results:
            analysis.winning_variant, analysis.confidence_score = self._determine_winner(
                experiment_result, analysis.primary_metric_results
            )

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis, experiment_result)

        # Create summary
        analysis.summary = self._create_analysis_summary(analysis)

        self.logger.info(f"Completed statistical analysis for experiment {experiment_result.experiment_id}")
        return analysis

    def _analyze_metric_across_variants(
        self,
        experiment_result: ExperimentResult,
        metric_name: str,
        control_variant: str
    ) -> StatisticalTestResult:
        """
        Analyze a specific metric across all variants compared to control.

        Args:
            experiment_result: Experiment results
            metric_name: Name of metric to analyze
            control_variant: Name of control variant

        Returns:
            StatisticalTestResult for the metric
        """
        # Get control data
        control_data = self._extract_metric_data(experiment_result, control_variant, metric_name)

        best_result = None
        best_p_value = 1.0

        # Compare each variant to control
        for variant_name, variant_metrics in experiment_result.variant_results.items():
            if variant_name == control_variant:
                continue

            variant_data = self._extract_metric_data(experiment_result, variant_name, metric_name)

            # Perform statistical test
            test_result = self._perform_statistical_test(
                control_data, variant_data,
                f"{control_variant}_vs_{variant_name}_{metric_name}"
            )

            # Keep the most significant result
            if test_result.p_value < best_p_value:
                best_result = test_result
                best_p_value = test_result.p_value

        return best_result if best_result else self._create_empty_test_result()

    def _extract_metric_data(self, experiment_result: ExperimentResult, variant_name: str, metric_name: str) -> List[float]:
        """Extract metric data for a specific variant"""
        variant_metrics = experiment_result.variant_results.get(variant_name)
        if not variant_metrics:
            return []

        # Map metric names to data sources
        if metric_name == 'total_reward':
            return variant_metrics.reward_samples
        elif metric_name == 'actions_per_second':
            return variant_metrics.action_times
        elif metric_name == 'battle_win_rate':
            return [1.0 if won else 0.0 for won in variant_metrics.battle_results]
        else:
            # Check custom metrics
            custom_value = variant_metrics.custom_metrics.get(metric_name)
            return [custom_value] if custom_value is not None else []

    def _perform_statistical_test(
        self,
        control_data: List[float],
        treatment_data: List[float],
        test_name: str
    ) -> StatisticalTestResult:
        """
        Perform statistical test between control and treatment groups.

        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            test_name: Name/description of the test

        Returns:
            StatisticalTestResult
        """
        if len(control_data) < self.minimum_sample_size or len(treatment_data) < self.minimum_sample_size:
            return self._create_insufficient_data_result(len(control_data), len(treatment_data))

        # Choose appropriate statistical test
        test_type = self._choose_statistical_test(control_data, treatment_data)

        # Perform the test
        if test_type == StatisticalTest.T_TEST:
            return self._perform_t_test(control_data, treatment_data, equal_var=True)
        elif test_type == StatisticalTest.WELCH_T_TEST:
            return self._perform_t_test(control_data, treatment_data, equal_var=False)
        elif test_type == StatisticalTest.MANN_WHITNEY_U:
            return self._perform_mann_whitney_test(control_data, treatment_data)
        else:
            # Fallback to Welch's t-test
            return self._perform_t_test(control_data, treatment_data, equal_var=False)

    def _choose_statistical_test(self, control_data: List[float], treatment_data: List[float]) -> StatisticalTest:
        """Choose the most appropriate statistical test based on data characteristics"""

        # Check for normality (simple check based on sample size)
        if len(control_data) < 30 or len(treatment_data) < 30:
            # Small samples - use non-parametric test
            return StatisticalTest.MANN_WHITNEY_U

        # Check for equal variances
        if self._test_equal_variances(control_data, treatment_data):
            return StatisticalTest.T_TEST
        else:
            return StatisticalTest.WELCH_T_TEST

    def _test_equal_variances(self, data1: List[float], data2: List[float]) -> bool:
        """Test if two datasets have equal variances using Levene's test"""
        if len(data1) < 3 or len(data2) < 3:
            return False

        try:
            statistic, p_value = stats.levene(data1, data2)
            return p_value > 0.05  # Assume equal variances if p > 0.05
        except:
            return False

    def _perform_t_test(
        self,
        control_data: List[float],
        treatment_data: List[float],
        equal_var: bool = True
    ) -> StatisticalTestResult:
        """Perform t-test (Student's or Welch's)"""
        try:
            statistic, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=equal_var)

            # Calculate effect size
            effect_size = self._calculate_cohens_d(control_data, treatment_data)

            # Calculate confidence interval for mean difference
            ci = self._calculate_confidence_interval_mean_diff(control_data, treatment_data)

            test_type = StatisticalTest.T_TEST if equal_var else StatisticalTest.WELCH_T_TEST

            return StatisticalTestResult(
                test_type=test_type,
                statistic=statistic,
                p_value=p_value,
                is_significant=p_value < self.default_alpha,
                confidence_level=self.default_confidence_level,
                effect_size=effect_size,
                effect_size_method=EffectSizeMethod.COHENS_D,
                confidence_interval=ci,
                sample_size_a=len(control_data),
                sample_size_b=len(treatment_data),
                interpretation=self._interpret_effect_size(effect_size)
            )
        except Exception as e:
            self.logger.error(f"Error performing t-test: {str(e)}")
            return self._create_error_result(len(control_data), len(treatment_data))

    def _perform_mann_whitney_test(
        self,
        control_data: List[float],
        treatment_data: List[float]
    ) -> StatisticalTestResult:
        """Perform Mann-Whitney U test (non-parametric)"""
        try:
            statistic, p_value = stats.mannwhitneyu(
                control_data, treatment_data, alternative='two-sided'
            )

            # Calculate Cliff's delta for effect size
            effect_size = self._calculate_cliff_delta(control_data, treatment_data)

            # Bootstrap confidence interval for median difference
            ci = self._bootstrap_confidence_interval(control_data, treatment_data)

            return StatisticalTestResult(
                test_type=StatisticalTest.MANN_WHITNEY_U,
                statistic=statistic,
                p_value=p_value,
                is_significant=p_value < self.default_alpha,
                confidence_level=self.default_confidence_level,
                effect_size=effect_size,
                effect_size_method=EffectSizeMethod.CLIFF_DELTA,
                confidence_interval=ci,
                sample_size_a=len(control_data),
                sample_size_b=len(treatment_data),
                interpretation=self._interpret_cliff_delta(effect_size)
            )
        except Exception as e:
            self.logger.error(f"Error performing Mann-Whitney test: {str(e)}")
            return self._create_error_result(len(control_data), len(treatment_data))

    def _calculate_cohens_d(self, control_data: List[float], treatment_data: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not control_data or not treatment_data:
            return 0.0

        mean1 = statistics.mean(control_data)
        mean2 = statistics.mean(treatment_data)

        if len(control_data) == 1 and len(treatment_data) == 1:
            return 0.0

        # Pooled standard deviation
        var1 = statistics.variance(control_data) if len(control_data) > 1 else 0.0
        var2 = statistics.variance(treatment_data) if len(treatment_data) > 1 else 0.0

        pooled_std = math.sqrt(((len(control_data) - 1) * var1 + (len(treatment_data) - 1) * var2) /
                              (len(control_data) + len(treatment_data) - 2))

        if pooled_std == 0:
            return 0.0

        return (mean2 - mean1) / pooled_std

    def _calculate_cliff_delta(self, control_data: List[float], treatment_data: List[float]) -> float:
        """Calculate Cliff's delta effect size"""
        if not control_data or not treatment_data:
            return 0.0

        more = sum(1 for x in treatment_data for y in control_data if x > y)
        less = sum(1 for x in treatment_data for y in control_data if x < y)

        total = len(control_data) * len(treatment_data)
        if total == 0:
            return 0.0

        return (more - less) / total

    def _calculate_confidence_interval_mean_diff(
        self,
        control_data: List[float],
        treatment_data: List[float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference"""
        if len(control_data) < 2 or len(treatment_data) < 2:
            return (0.0, 0.0)

        try:
            mean_diff = statistics.mean(treatment_data) - statistics.mean(control_data)

            # Standard error of difference
            var1 = statistics.variance(control_data)
            var2 = statistics.variance(treatment_data)
            se_diff = math.sqrt(var1/len(control_data) + var2/len(treatment_data))

            # Degrees of freedom (Welch-Satterthwaite)
            df = (var1/len(control_data) + var2/len(treatment_data))**2 / (
                (var1/len(control_data))**2/(len(control_data)-1) +
                (var2/len(treatment_data))**2/(len(treatment_data)-1)
            )

            # t-critical value
            t_crit = stats.t.ppf(1 - self.default_alpha/2, df)

            margin_error = t_crit * se_diff
            return (mean_diff - margin_error, mean_diff + margin_error)
        except:
            return (0.0, 0.0)

    def _bootstrap_confidence_interval(
        self,
        control_data: List[float],
        treatment_data: List[float],
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for median difference"""
        try:
            np.random.seed(42)  # For reproducibility

            differences = []
            for _ in range(n_bootstrap):
                control_sample = np.random.choice(control_data, len(control_data), replace=True)
                treatment_sample = np.random.choice(treatment_data, len(treatment_data), replace=True)

                diff = np.median(treatment_sample) - np.median(control_sample)
                differences.append(diff)

            # Calculate percentiles for confidence interval
            alpha = 1 - self.default_confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100

            ci_lower = np.percentile(differences, lower_percentile)
            ci_upper = np.percentile(differences, upper_percentile)

            return (float(ci_lower), float(ci_upper))
        except:
            return (0.0, 0.0)

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"

    def _interpret_cliff_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible effect"
        elif abs_delta < 0.33:
            return "small effect"
        elif abs_delta < 0.474:
            return "medium effect"
        else:
            return "large effect"

    def _has_metric_data(self, experiment_result: ExperimentResult, metric_name: str) -> bool:
        """Check if experiment has data for a specific metric"""
        for variant_metrics in experiment_result.variant_results.values():
            if metric_name == 'total_reward' and variant_metrics.reward_samples:
                return True
            elif metric_name == 'actions_per_second' and variant_metrics.action_times:
                return True
            elif metric_name == 'battle_win_rate' and variant_metrics.battle_results:
                return True
            elif metric_name in variant_metrics.custom_metrics:
                return True
        return False

    def _determine_winner(
        self,
        experiment_result: ExperimentResult,
        primary_results: Dict[str, StatisticalTestResult]
    ) -> Tuple[Optional[str], float]:
        """Determine winning variant and confidence score"""

        # Find variant with best performance on primary metrics
        variant_scores = {}

        # Get list of variants
        variant_names = list(experiment_result.variant_results.keys())
        if len(variant_names) < 2:
            return None, 0.0

        # Assume first variant is control (common convention)
        # Effect size > 0 means treatment (second variant) is better
        # Effect size < 0 means control (first variant) is better
        for variant_name, metrics in experiment_result.variant_results.items():
            # Start with actual metric values
            score = 0.0
            count = 0

            # Get the actual metric value for this variant
            if hasattr(metrics, 'total_reward') and metrics.total_reward > 0:
                score += metrics.total_reward
                count += 1

            variant_scores[variant_name] = score / max(count, 1)

        if not variant_scores:
            return None, 0.0

        # Get best variant based on actual performance
        winner = max(variant_scores.keys(), key=lambda k: variant_scores[k])

        # Calculate confidence from effect sizes
        total_effect = sum(abs(r.effect_size) for r in primary_results.values() if r.is_significant)
        confidence = min(0.99, total_effect / max(len(primary_results), 1))  # Cap at 99%

        return winner, confidence

    def _generate_recommendations(
        self,
        analysis: AnalysisResult,
        experiment_result: ExperimentResult
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        if analysis.has_significant_results:
            if analysis.winning_variant:
                recommendations.append(
                    f"Implement '{analysis.winning_variant}' configuration "
                    f"(confidence: {analysis.confidence_score:.1%})"
                )

            # Check for consistent improvements
            consistent_metrics = [
                metric for metric, result in analysis.primary_metric_results.items()
                if result.is_significant and result.effect_size > 0.2
            ]

            if consistent_metrics:
                recommendations.append(
                    f"Strong evidence for improvement in: {', '.join(consistent_metrics)}"
                )
        else:
            recommendations.append("No statistically significant differences found")
            recommendations.append("Consider increasing sample size or testing different configurations")

        # Sample size recommendations
        min_samples = max(analysis.primary_metric_results.values(),
                         key=lambda r: r.sample_size_a, default=None)
        if min_samples and min_samples.sample_size_a < 50:
            recommendations.append("Increase sample size to at least 50 per variant for more reliable results")

        return recommendations

    def _create_analysis_summary(self, analysis: AnalysisResult) -> str:
        """Create summary of analysis results"""
        if analysis.has_significant_results:
            summary = f"Significant results found. "
            if analysis.winning_variant:
                summary += f"Recommended configuration: {analysis.winning_variant} "
                summary += f"(confidence: {analysis.confidence_score:.1%})"
        else:
            summary = "No significant differences detected between variants. "
            summary += "Additional testing may be needed."

        return summary

    def _create_empty_test_result(self) -> StatisticalTestResult:
        """Create empty test result for cases with no data"""
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=self.default_confidence_level,
            effect_size=0.0,
            effect_size_method=self.default_effect_size_method,
            confidence_interval=(0.0, 0.0),
            sample_size_a=0,
            sample_size_b=0,
            interpretation="no data"
        )

    def _create_insufficient_data_result(self, sample_size_a: int, sample_size_b: int) -> StatisticalTestResult:
        """Create test result for insufficient data"""
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=self.default_confidence_level,
            effect_size=0.0,
            effect_size_method=self.default_effect_size_method,
            confidence_interval=(0.0, 0.0),
            sample_size_a=sample_size_a,
            sample_size_b=sample_size_b,
            interpretation="insufficient data"
        )

    def _create_error_result(self, sample_size_a: int, sample_size_b: int) -> StatisticalTestResult:
        """Create test result for analysis errors"""
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=self.default_confidence_level,
            effect_size=0.0,
            effect_size_method=self.default_effect_size_method,
            confidence_interval=(0.0, 0.0),
            sample_size_a=sample_size_a,
            sample_size_b=sample_size_b,
            interpretation="analysis error"
        )