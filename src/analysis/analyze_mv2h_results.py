#!/usr/bin/env python3
"""
MV2H Results Analysis Script

Analyzes MV2H evaluation results from CSV files and generates tables/statistics
for research papers. Supports comparison between different models and methods.

This script ensures reproducibility of results - reviewers can regenerate
all tables and figures by running this script on the raw evaluation CSV.

Usage:
    # Analyze MT3 chunk evaluation results
    python scripts/analyze_mv2h_results.py results/chunks.csv

    # Compare multiple models
    python scripts/analyze_mv2h_results.py results/mt3_chunks.csv results/zeng_chunks.csv

    # Export to markdown table format
    python scripts/analyze_mv2h_results.py results/chunks.csv --format markdown
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ChunkResult:
    """Single chunk evaluation result."""

    task_id: str
    chunk_index: int
    piece_id: str
    performance: str
    status: str
    multi_pitch: float = 0.0
    voice: float = 0.0
    meter: float = 0.0
    value: float = 0.0
    harmony: float = 0.0
    mv2h: float = 0.0
    mv2h_custom: float = 0.0


@dataclass
class AnalysisSummary:
    """Summary statistics for analysis."""

    total: int = 0
    success: int = 0
    failed: int = 0
    success_rate: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_results_csv(csv_path: str) -> List[ChunkResult]:
    """
    Load chunk evaluation results from CSV.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of ChunkResult objects
    """
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse chunk_index, piece_id, performance from task_id if not in CSV
            # task_id format: Bach#Prelude#bwv_875#Ahfat01M.10
            chunk_index = 0
            piece_id = row.get("piece_id", "")
            performance = row.get("performance", "")

            if row.get("chunk_index"):
                try:
                    chunk_index = int(row["chunk_index"])
                except ValueError:
                    pass

            task_id = row.get("task_id", "")
            if task_id and "#" in task_id:
                # Parse from task_id: Bach#Prelude#bwv_875#Ahfat01M.10
                parts = task_id.rsplit("#", 1)  # ["Bach#Prelude#bwv_875", "Ahfat01M.10"]
                if len(parts) == 2:
                    if not piece_id:
                        piece_id = parts[0]
                    perf_chunk = parts[1]  # "Ahfat01M.10"
                    if "." in perf_chunk:
                        perf, idx = perf_chunk.rsplit(".", 1)
                        if not performance:
                            performance = perf
                        if chunk_index == 0:
                            try:
                                chunk_index = int(idx)
                            except ValueError:
                                pass

            # Parse metrics
            def safe_float(val: str, default: float = 0.0) -> float:
                try:
                    return float(val) if val else default
                except ValueError:
                    return default

            results.append(ChunkResult(
                task_id=task_id,
                chunk_index=chunk_index,
                piece_id=piece_id,
                performance=performance,
                status=row.get("status", "unknown"),
                multi_pitch=safe_float(row.get("Multi-pitch")),
                voice=safe_float(row.get("Voice")),
                meter=safe_float(row.get("Meter")),
                value=safe_float(row.get("Value")),
                harmony=safe_float(row.get("Harmony")),
                mv2h=safe_float(row.get("MV2H")),
                mv2h_custom=safe_float(row.get("MV2H_custom")),
            ))

    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def compute_overall_summary(results: List[ChunkResult]) -> Dict[str, Any]:
    """
    Compute overall summary statistics.

    Two methods are computed:
    1. Zeng method: Only average successful chunks (excludes failures)
    2. Strict method: Include failures as 0

    Args:
        results: List of chunk results

    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    successful = [r for r in results if r.status == "success"]
    n_success = len(successful)
    n_failed = total - n_success

    # Status breakdown
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r.status] += 1

    # Zeng method: only successful samples
    zeng_metrics = {}
    if successful:
        zeng_metrics = {
            "Multi-pitch": np.mean([r.multi_pitch for r in successful]),
            "Voice": np.mean([r.voice for r in successful]),
            "Meter": np.mean([r.meter for r in successful]),
            "Value": np.mean([r.value for r in successful]),
            "Harmony": np.mean([r.harmony for r in successful]),
            "MV2H": np.mean([r.mv2h for r in successful]),
        }
        zeng_metrics["MV2H_custom"] = (
            zeng_metrics["Multi-pitch"] + zeng_metrics["Voice"] +
            zeng_metrics["Value"] + zeng_metrics["Harmony"]
        ) / 4

    # Strict method: failures as 0
    strict_metrics = {}
    if total > 0:
        strict_metrics = {
            "Multi-pitch": sum(r.multi_pitch for r in successful) / total,
            "Voice": sum(r.voice for r in successful) / total,
            "Meter": sum(r.meter for r in successful) / total,
            "Value": sum(r.value for r in successful) / total,
            "Harmony": sum(r.harmony for r in successful) / total,
            "MV2H": sum(r.mv2h for r in successful) / total,
        }
        strict_metrics["MV2H_custom"] = (
            strict_metrics["Multi-pitch"] + strict_metrics["Voice"] +
            strict_metrics["Value"] + strict_metrics["Harmony"]
        ) / 4

    return {
        "total": total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / total if total > 0 else 0,
        "status_breakdown": dict(status_counts),
        "zeng_method": zeng_metrics,
        "strict_method": strict_metrics,
    }


def analyze_by_chunk_position(
    results: List[ChunkResult],
    bins: List[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze success rate by chunk position.

    This reveals how transcription quality varies with song position,
    e.g., whether MT3 drifts over time.

    Args:
        results: List of chunk results
        bins: List of (start, end) tuples for position bins
              Default: [(1,10), (11,20), (21,30), (31,40), (41,50),
                       (51,100), (101,200), (201,500)]

    Returns:
        List of analysis dictionaries per bin
    """
    if bins is None:
        # Default bins based on ASAP test set distribution:
        # - Shortest piece: 43 measures (Chopin Etude op10#2)
        # - Longest piece: 296 measures (Beethoven Sonata 21-1)
        # - Most pieces have 50-200 measures
        bins = [
            (1, 25), (26, 50), (51, 75), (76, 100),
            (101, 150), (151, 200), (201, 300),
        ]

    analysis = []
    for start, end in bins:
        bin_results = [r for r in results if start <= r.chunk_index <= end]
        if not bin_results:
            continue

        successful = [r for r in bin_results if r.status == "success"]
        n_total = len(bin_results)
        n_success = len(successful)

        # Compute average metrics for successful chunks
        avg_metrics = {}
        if successful:
            avg_metrics = {
                "Multi-pitch": np.mean([r.multi_pitch for r in successful]),
                "Voice": np.mean([r.voice for r in successful]),
                "Meter": np.mean([r.meter for r in successful]),
                "Value": np.mean([r.value for r in successful]),
                "Harmony": np.mean([r.harmony for r in successful]),
                "MV2H": np.mean([r.mv2h for r in successful]),
                "MV2H_custom": np.mean([r.mv2h_custom for r in successful]),
            }

        analysis.append({
            "position_range": f"{start}-{end}",
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": n_success / n_total if n_total > 0 else 0,
            "metrics": avg_metrics,
        })

    return analysis


def analyze_by_piece(results: List[ChunkResult]) -> List[Dict[str, Any]]:
    """
    Analyze results grouped by piece.

    Args:
        results: List of chunk results

    Returns:
        List of per-piece analysis dictionaries
    """
    # Group by piece_id
    by_piece = defaultdict(list)
    for r in results:
        by_piece[r.piece_id].append(r)

    analysis = []
    for piece_id, piece_results in sorted(by_piece.items()):
        successful = [r for r in piece_results if r.status == "success"]
        n_total = len(piece_results)
        n_success = len(successful)

        avg_metrics = {}
        if successful:
            avg_metrics = {
                "Multi-pitch": np.mean([r.multi_pitch for r in successful]),
                "MV2H_custom": np.mean([r.mv2h_custom for r in successful]),
            }

        analysis.append({
            "piece_id": piece_id,
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": n_success / n_total if n_total > 0 else 0,
            "metrics": avg_metrics,
        })

    return analysis


def analyze_by_composer(results: List[ChunkResult]) -> List[Dict[str, Any]]:
    """
    Analyze results grouped by composer.

    Args:
        results: List of chunk results

    Returns:
        List of per-composer analysis dictionaries
    """
    # Group by composer (first part of piece_id)
    by_composer = defaultdict(list)
    for r in results:
        composer = r.piece_id.split("#")[0] if "#" in r.piece_id else r.piece_id
        by_composer[composer].append(r)

    analysis = []
    for composer, composer_results in sorted(by_composer.items()):
        successful = [r for r in composer_results if r.status == "success"]
        n_total = len(composer_results)
        n_success = len(successful)

        avg_metrics = {}
        if successful:
            avg_metrics = {
                "Multi-pitch": np.mean([r.multi_pitch for r in successful]),
                "Voice": np.mean([r.voice for r in successful]),
                "Value": np.mean([r.value for r in successful]),
                "Harmony": np.mean([r.harmony for r in successful]),
                "MV2H_custom": np.mean([r.mv2h_custom for r in successful]),
            }

        analysis.append({
            "composer": composer,
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": n_success / n_total if n_total > 0 else 0,
            "metrics": avg_metrics,
        })

    return analysis


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def print_overall_summary(summary: Dict[str, Any]) -> None:
    """Print overall summary in formatted output."""
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total chunks:  {summary['total']}")
    print(f"Successful:    {summary['n_success']} ({summary['success_rate']*100:.1f}%)")
    print(f"Failed:        {summary['n_failed']} ({(1-summary['success_rate'])*100:.1f}%)")

    print("\nStatus Breakdown:")
    for status, count in sorted(summary["status_breakdown"].items()):
        pct = count / summary["total"] * 100 if summary["total"] > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")

    print("\n" + "-" * 60)
    print("Zeng Method (exclude failures)")
    print("-" * 60)
    if summary["zeng_method"]:
        for metric, value in summary["zeng_method"].items():
            print(f"  {metric}: {value*100:.2f}%")

    print("\n" + "-" * 60)
    print("Strict Method (failures as 0)")
    print("-" * 60)
    if summary["strict_method"]:
        for metric, value in summary["strict_method"].items():
            print(f"  {metric}: {value*100:.2f}%")


def print_position_analysis(analysis: List[Dict[str, Any]]) -> None:
    """Print chunk position analysis."""
    print("\n" + "=" * 60)
    print("SUCCESS RATE BY CHUNK POSITION")
    print("=" * 60)

    for item in analysis:
        mv2h = item["metrics"].get("MV2H_custom", 0) * 100 if item["metrics"] else 0
        print(f"  {item['position_range']:>8}: {item['success_rate']*100:5.1f}% "
              f"(n={item['n_total']:4d}, MV2H_custom={mv2h:5.1f}%)")


def print_composer_analysis(analysis: List[Dict[str, Any]]) -> None:
    """Print per-composer analysis."""
    print("\n" + "=" * 60)
    print("RESULTS BY COMPOSER")
    print("=" * 60)

    for item in analysis:
        mv2h = item["metrics"].get("MV2H_custom", 0) * 100 if item["metrics"] else 0
        print(f"  {item['composer']:12s}: {item['success_rate']*100:5.1f}% "
              f"(n={item['n_total']:4d}, MV2H_custom={mv2h:5.1f}%)")


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_evaluability_comparison(
    mt3_results: List[ChunkResult],
    output_path: Optional[str] = None,
    zeng_csv_path: Optional[str] = None,
) -> None:
    """
    Plot Evaluability (Success Rate) vs Chunk Position comparison.

    Creates a scatter plot with fitted lines comparing:
    - MT3 + MuseScore: Weibull decay fit (shows decline over position)
    - Zeng: Scatter points + constant fit (no position dependence)

    Args:
        mt3_results: List of MT3 chunk results
        output_path: Path to save the plot
        zeng_csv_path: Path to Zeng's per-chunk results CSV
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        from scipy.stats import linregress
    except ImportError:
        print("matplotlib and scipy required: pip install matplotlib scipy")
        return

    # =========================================================================
    # Process MT3 data
    # =========================================================================
    by_index_mt3 = defaultdict(list)
    for r in mt3_results:
        by_index_mt3[r.chunk_index].append(r)

    mt3_indices = sorted(by_index_mt3.keys())
    mt3_success_rates = []
    mt3_sample_counts = []

    for idx in mt3_indices:
        chunk_results = by_index_mt3[idx]
        successful = [r for r in chunk_results if r.status == "success"]
        success_rate = len(successful) / len(chunk_results) if chunk_results else 0
        mt3_success_rates.append(success_rate)
        mt3_sample_counts.append(len(chunk_results))

    mt3_indices = np.array(mt3_indices)
    mt3_success_rates = np.array(mt3_success_rates)
    mt3_sample_counts = np.array(mt3_sample_counts)

    # =========================================================================
    # Process Zeng data (if provided)
    # =========================================================================
    zeng_indices = None
    zeng_success_rates = None
    zeng_mean = None

    if zeng_csv_path and Path(zeng_csv_path).exists():
        # Load Zeng per-chunk results
        zeng_results = []
        with open(zeng_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get measure_idx (chunk position)
                measure_idx = 0
                if row.get("measure_idx"):
                    try:
                        measure_idx = int(row["measure_idx"])
                    except ValueError:
                        pass
                elif row.get("task_id") and "." in row["task_id"]:
                    # Parse from task_id: Bach#Prelude#bwv_875#Ahfat01M.10
                    try:
                        measure_idx = int(row["task_id"].rsplit(".", 1)[1])
                    except (ValueError, IndexError):
                        pass

                status = row.get("status", "unknown")
                zeng_results.append((measure_idx, status))

        # Group by measure index
        by_index_zeng = defaultdict(list)
        for idx, status in zeng_results:
            by_index_zeng[idx].append(status)

        zeng_indices_list = sorted(by_index_zeng.keys())
        zeng_success_rates_list = []

        for idx in zeng_indices_list:
            statuses = by_index_zeng[idx]
            success_rate = sum(1 for s in statuses if s == "success") / len(statuses)
            zeng_success_rates_list.append(success_rate)

        zeng_indices = np.array(zeng_indices_list)
        zeng_success_rates = np.array(zeng_success_rates_list)

        # Calculate overall mean for Zeng
        total_success = sum(1 for idx, status in zeng_results if status == "success")
        zeng_mean = total_success / len(zeng_results)
        print(f"Zeng: {total_success}/{len(zeng_results)} = {zeng_mean*100:.1f}% overall success rate")

    # =========================================================================
    # Curve Fitting for MT3: Compare Exponential vs Weibull
    # =========================================================================

    def exponential_decay(x, a, b):
        """Exponential decay: f(x) = a * exp(-x/b)

        Parameters:
        - a: amplitude (initial success rate at x=0)
        - b: decay constant (characteristic length where rate drops to a*e^(-1))
        """
        return a * np.exp(-x / b)

    def weibull_decay(x, a, b, c):
        """Weibull decay: f(x) = a * exp(-(x/b)^c)

        Parameters:
        - a: amplitude (initial success rate at x=0)
        - b: scale parameter (characteristic position where rate drops to a*e^(-1))
        - c: shape parameter (c>1: accelerating decay, c<1: decelerating decay)
        """
        return a * np.exp(-((x / b) ** c))

    print(f"\n{'='*60}")
    print("MT3 Curve Fitting Comparison")
    print(f"{'='*60}")
    print(f"  Data: {len(mt3_indices)} positions, {sum(mt3_sample_counts)} total chunks")
    print(f"  Samples per position: min={min(mt3_sample_counts)}, max={max(mt3_sample_counts)}, mean={np.mean(mt3_sample_counts):.1f}")

    # --- Exponential Fit ---
    try:
        popt_exp, _ = curve_fit(
            exponential_decay, mt3_indices, mt3_success_rates,
            p0=[0.5, 100],
            bounds=([0, 1], [1, 1000]),
            sigma=1/np.sqrt(mt3_sample_counts + 1),
            maxfev=5000
        )
        mt3_fitted_exp = exponential_decay(mt3_indices, *popt_exp)

        ss_res_exp = np.sum((mt3_success_rates - mt3_fitted_exp) ** 2)
        ss_tot = np.sum((mt3_success_rates - np.mean(mt3_success_rates)) ** 2)
        r2_exp = 1 - (ss_res_exp / ss_tot)
        rmse_exp = np.sqrt(np.mean((mt3_success_rates - mt3_fitted_exp) ** 2))
        # AIC for model comparison (lower is better)
        n = len(mt3_success_rates)
        aic_exp = n * np.log(ss_res_exp / n) + 2 * 2  # 2 parameters

        print(f"\n  [1] Exponential: f(x) = a * exp(-x/b)")
        print(f"      Parameters: a={popt_exp[0]:.4f}, b={popt_exp[1]:.2f}")
        print(f"      R² = {r2_exp:.4f}, RMSE = {rmse_exp:.4f}, AIC = {aic_exp:.2f}")
    except Exception as e:
        print(f"  Exponential fit failed: {e}")
        r2_exp = -999
        aic_exp = 999

    # --- Weibull Fit ---
    try:
        popt_weibull, _ = curve_fit(
            weibull_decay, mt3_indices, mt3_success_rates,
            p0=[0.5, 100, 1],
            bounds=([0, 1, 0.1], [1, 500, 5]),
            sigma=1/np.sqrt(mt3_sample_counts + 1),
            maxfev=5000
        )
        mt3_fitted_weibull = weibull_decay(mt3_indices, *popt_weibull)

        ss_res_weibull = np.sum((mt3_success_rates - mt3_fitted_weibull) ** 2)
        r2_weibull = 1 - (ss_res_weibull / ss_tot)
        rmse_weibull = np.sqrt(np.mean((mt3_success_rates - mt3_fitted_weibull) ** 2))
        aic_weibull = n * np.log(ss_res_weibull / n) + 2 * 3  # 3 parameters

        print(f"\n  [2] Weibull: f(x) = a * exp(-(x/b)^c)")
        print(f"      Parameters: a={popt_weibull[0]:.4f}, b={popt_weibull[1]:.2f}, c={popt_weibull[2]:.4f}")
        print(f"      R² = {r2_weibull:.4f}, RMSE = {rmse_weibull:.4f}, AIC = {aic_weibull:.2f}")
    except Exception as e:
        print(f"  Weibull fit failed: {e}")
        r2_weibull = -999
        aic_weibull = 999

    # --- Model Selection ---
    print(f"\n  {'─'*50}")
    print(f"  Model Comparison:")
    print(f"  {'─'*50}")
    print(f"  | Model       | R²     | RMSE   | AIC     |")
    print(f"  |-------------|--------|--------|---------|")
    print(f"  | Exponential | {r2_exp:.4f} | {rmse_exp:.4f} | {aic_exp:7.2f} |")
    print(f"  | Weibull     | {r2_weibull:.4f} | {rmse_weibull:.4f} | {aic_weibull:7.2f} |")

    if aic_weibull < aic_exp:
        print(f"\n  ✓ BEST MODEL: Weibull (lower AIC by {aic_exp - aic_weibull:.2f})")
        mt3_fitted = mt3_fitted_weibull
        mt3_fit_label = f"Weibull (a={popt_weibull[0]:.2f}, b={popt_weibull[1]:.0f}, c={popt_weibull[2]:.2f})"
        popt = popt_weibull
        r_squared = r2_weibull
    else:
        print(f"\n  ✓ BEST MODEL: Exponential (lower AIC by {aic_weibull - aic_exp:.2f})")
        mt3_fitted = mt3_fitted_exp
        mt3_fit_label = f"Exponential (a={popt_exp[0]:.2f}, b={popt_exp[1]:.0f})"
        popt = popt_exp
        r_squared = r2_exp

    # =========================================================================
    # Zeng trend test (Linear regression to prove NO position dependence)
    # =========================================================================
    if zeng_indices is not None and len(zeng_indices) > 2:
        # Get sample counts for Zeng
        zeng_sample_counts = []
        for idx in zeng_indices:
            zeng_sample_counts.append(len(by_index_zeng[idx]))
        zeng_sample_counts = np.array(zeng_sample_counts)

        # Linear regression: evaluability ~ position
        slope, intercept, r_value, p_value, std_err = linregress(zeng_indices, zeng_success_rates)

        print(f"\n{'='*60}")
        print("Zeng Linear Regression (H0: no position dependence)")
        print(f"{'='*60}")
        print(f"  Slope = {slope*100:.4f}%/measure")
        print(f"  Intercept = {intercept*100:.2f}%")
        print(f"  R² = {r_value**2:.6f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  95% CI for slope: [{(slope-1.96*std_err)*100:.4f}%, {(slope+1.96*std_err)*100:.4f}%]")
        print(f"  Sample counts per position: min={min(zeng_sample_counts)}, max={max(zeng_sample_counts)}, mean={np.mean(zeng_sample_counts):.1f}")

        if p_value > 0.05:
            print(f"  ✓ CONCLUSION: p={p_value:.3f} > 0.05, NO significant trend (constant model valid)")
        else:
            print(f"  ✗ CONCLUSION: p={p_value:.3f} < 0.05, significant trend detected")

    # =========================================================================
    # Plot
    # =========================================================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.5,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    # Colors
    color_mt3 = "#648FFF"      # Blue
    color_zeng = "#228B22"     # Green (Forest Green)

    # Plot MT3: scatter + Weibull fit
    ax.scatter(mt3_indices, mt3_success_rates * 100, alpha=0.3, s=15,
               color=color_mt3, edgecolors="none", rasterized=True,
               label="_nolegend_")
    ax.plot(mt3_indices, mt3_fitted * 100, color=color_mt3, linewidth=2.5,
            label=f"MT3 + MuseScore ({mt3_fit_label})")

    # Plot Zeng: scatter + constant line
    if zeng_indices is not None and zeng_success_rates is not None:
        ax.scatter(zeng_indices, zeng_success_rates * 100, alpha=0.3, s=15,
                   color=color_zeng, edgecolors="none", rasterized=True,
                   label="_nolegend_")
        ax.axhline(y=zeng_mean * 100, color=color_zeng, linewidth=2.5,
                   linestyle="-", label=f"Zeng (constant = {zeng_mean*100:.1f}%)")
    else:
        # Fallback: just draw constant line at default value
        ax.axhline(y=88.2, color=color_zeng, linewidth=2.5,
                   linestyle="-", label="Zeng (constant = 88.2%)")

    # Formatting
    ax.set_xlabel("Chunk Position (measure number)")
    ax.set_ylabel("Evaluability / Success Rate (%)")
    ax.set_title("Evaluability vs Chunk Position: Pipeline vs E2E", fontweight="bold")

    max_idx = max(mt3_indices)
    if zeng_indices is not None:
        max_idx = max(max_idx, max(zeng_indices))
    ax.set_xlim(0, max_idx + 10)
    ax.set_ylim(0, 105)

    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {pdf_path} and {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_cumulative_error(
    results: List[ChunkResult],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze whether MT3+MuseScore exhibits cumulative error.

    Three complementary analyses:

    1. Transition Matrix: P(fail|prev_fail) vs P(fail|prev_success)
       - If cumulative error: P(fail|prev_fail) >> P(fail|prev_success)
       - Failure should be "sticky"

    2. Run Length Analysis: Distribution of consecutive success/fail runs
       - If cumulative error: long fail runs, short recovery

    3. Time to First Failure: Kaplan-Meier survival analysis
       - If cumulative error: Weibull c > 1 (accelerating hazard)

    Args:
        results: List of chunk results
        output_path: Path to save analysis plots

    Returns:
        Dictionary with analysis results
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
    except ImportError:
        print("matplotlib and scipy required: pip install matplotlib scipy")
        return {}

    # =========================================================================
    # Group by (piece_id, performance) and build sequences
    # =========================================================================
    by_performance = defaultdict(list)
    for r in results:
        key = (r.piece_id, r.performance)
        by_performance[key].append(r)

    print(f"\n{'='*70}")
    print("CUMULATIVE ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Total performances: {len(by_performance)}")

    # Build success/fail sequences for each performance
    # sequence[i] = 1 if success, 0 if fail
    sequences = []
    for (piece_id, perf), chunks in by_performance.items():
        chunks_sorted = sorted(chunks, key=lambda x: x.chunk_index)
        seq = [1 if c.status == "success" else 0 for c in chunks_sorted]
        sequences.append({
            "piece_id": piece_id,
            "performance": perf,
            "sequence": seq,
            "length": len(seq),
        })

    avg_length = np.mean([s["length"] for s in sequences])
    print(f"Average sequence length: {avg_length:.1f} chunks")

    # =========================================================================
    # Analysis 1: Transition Matrix
    # =========================================================================
    # Count transitions: success->success, success->fail, fail->success, fail->fail
    transitions = {"SS": 0, "SF": 0, "FS": 0, "FF": 0}

    for s in sequences:
        seq = s["sequence"]
        for i in range(len(seq) - 1):
            prev, curr = seq[i], seq[i + 1]
            if prev == 1 and curr == 1:
                transitions["SS"] += 1
            elif prev == 1 and curr == 0:
                transitions["SF"] += 1
            elif prev == 0 and curr == 1:
                transitions["FS"] += 1
            else:  # prev == 0 and curr == 0
                transitions["FF"] += 1

    # Calculate conditional probabilities
    total_after_success = transitions["SS"] + transitions["SF"]
    total_after_fail = transitions["FS"] + transitions["FF"]

    p_fail_given_success = transitions["SF"] / total_after_success if total_after_success > 0 else 0
    p_fail_given_fail = transitions["FF"] / total_after_fail if total_after_fail > 0 else 0
    p_success_given_fail = transitions["FS"] / total_after_fail if total_after_fail > 0 else 0

    print(f"\n{'-'*50}")
    print("Analysis 1: Transition Matrix")
    print(f"{'-'*50}")
    print(f"\nTransition counts:")
    print(f"  Success -> Success: {transitions['SS']:,}")
    print(f"  Success -> Fail:    {transitions['SF']:,}")
    print(f"  Fail -> Success:    {transitions['FS']:,}")
    print(f"  Fail -> Fail:       {transitions['FF']:,}")
    print(f"\nConditional probabilities:")
    print(f"  P(fail | prev_success) = {p_fail_given_success*100:.2f}%")
    print(f"  P(fail | prev_fail)    = {p_fail_given_fail*100:.2f}%")
    print(f"  P(success | prev_fail) = {p_success_given_fail*100:.2f}%  <- 'Recovery rate'")

    # Sticky ratio: how much more likely to fail after a fail vs after a success
    sticky_ratio = p_fail_given_fail / p_fail_given_success if p_fail_given_success > 0 else float('inf')
    print(f"\n  Sticky ratio = P(fail|fail) / P(fail|success) = {sticky_ratio:.2f}")

    if sticky_ratio > 1.5:
        print(f"  *** Sticky ratio > 1.5: Failure is STICKY (cumulative error SUPPORTED) ***")
    else:
        print(f"  Sticky ratio <= 1.5: Failure is NOT particularly sticky")

    # =========================================================================
    # Analysis 2: Run Length Analysis
    # =========================================================================
    success_runs = []  # lengths of consecutive success runs
    fail_runs = []     # lengths of consecutive fail runs

    for s in sequences:
        seq = s["sequence"]
        if not seq:
            continue

        current_val = seq[0]
        current_run = 1

        for i in range(1, len(seq)):
            if seq[i] == current_val:
                current_run += 1
            else:
                # End of run
                if current_val == 1:
                    success_runs.append(current_run)
                else:
                    fail_runs.append(current_run)
                current_val = seq[i]
                current_run = 1

        # Don't forget the last run
        if current_val == 1:
            success_runs.append(current_run)
        else:
            fail_runs.append(current_run)

    print(f"\n{'-'*50}")
    print("Analysis 2: Run Length Analysis")
    print(f"{'-'*50}")
    print(f"\nSuccess runs: n={len(success_runs)}")
    print(f"  Mean length: {np.mean(success_runs):.2f}")
    print(f"  Median:      {np.median(success_runs):.1f}")
    print(f"  Max:         {max(success_runs)}")
    print(f"  Distribution: {np.percentile(success_runs, [25, 50, 75, 90, 95])}")

    print(f"\nFail runs: n={len(fail_runs)}")
    print(f"  Mean length: {np.mean(fail_runs):.2f}")
    print(f"  Median:      {np.median(fail_runs):.1f}")
    print(f"  Max:         {max(fail_runs)}")
    print(f"  Distribution: {np.percentile(fail_runs, [25, 50, 75, 90, 95])}")

    # If cumulative error: fail runs should be longer than expected from random
    # Expected run length under independence = 1 / P(transition)
    expected_fail_run = 1 / p_success_given_fail if p_success_given_fail > 0 else float('inf')
    print(f"\n  Expected fail run length (if independent): {expected_fail_run:.2f}")
    print(f"  Actual mean fail run length: {np.mean(fail_runs):.2f}")

    if np.mean(fail_runs) > expected_fail_run * 1.2:
        print(f"  *** Fail runs are LONGER than expected (cumulative error SUPPORTED) ***")
    else:
        print(f"  Fail runs are close to expected (no strong cumulative effect)")

    # =========================================================================
    # Analysis 3: Time to First Failure (Kaplan-Meier style)
    # =========================================================================
    # For each performance, find position of first failure
    first_failures = []
    for s in sequences:
        seq = s["sequence"]
        first_fail_pos = None
        for i, val in enumerate(seq):
            if val == 0:
                first_fail_pos = i
                break
        if first_fail_pos is not None:
            first_failures.append(first_fail_pos)
        else:
            # Never failed - censored at end of sequence
            first_failures.append(len(seq))  # right-censored

    print(f"\n{'-'*50}")
    print("Analysis 3: Time to First Failure")
    print(f"{'-'*50}")
    print(f"\nFirst failure position:")
    print(f"  Mean: {np.mean(first_failures):.2f}")
    print(f"  Median: {np.median(first_failures):.1f}")
    print(f"  Min: {min(first_failures)}, Max: {max(first_failures)}")

    # Compute survival curve: S(t) = P(first failure > t)
    max_t = max(first_failures)
    survival_t = list(range(max_t + 1))
    survival_prob = []
    for t in survival_t:
        n_survived = sum(1 for ff in first_failures if ff > t)
        survival_prob.append(n_survived / len(first_failures))

    survival_t = np.array(survival_t)
    survival_prob = np.array(survival_prob)

    # Fit Weibull survival: S(t) = exp(-(t/b)^c)
    def weibull_survival(t, b, c):
        return np.exp(-((t / b) ** c))

    # Filter out zeros for fitting
    mask = survival_prob > 0.01
    t_fit = survival_t[mask]
    s_fit = survival_prob[mask]

    try:
        popt, _ = curve_fit(
            weibull_survival, t_fit, s_fit,
            p0=[50, 1],
            bounds=([1, 0.1], [500, 10]),
            maxfev=5000
        )
        weibull_b, weibull_c = popt

        # Goodness of fit
        fitted = weibull_survival(t_fit, *popt)
        ss_res = np.sum((s_fit - fitted) ** 2)
        ss_tot = np.sum((s_fit - np.mean(s_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"\nWeibull survival fit: S(t) = exp(-(t/b)^c)")
        print(f"  b (scale) = {weibull_b:.2f}")
        print(f"  c (shape) = {weibull_c:.4f}")
        print(f"  R² = {r2:.4f}")
        print(f"\n  *** Shape parameter c = {weibull_c:.4f} ***")

        if weibull_c > 1:
            print(f"  c > 1: ACCELERATING hazard (cumulative error SUPPORTED)")
            print(f"         The longer you survive, the MORE likely to fail next")
        elif weibull_c < 1:
            print(f"  c < 1: DECELERATING hazard (infant mortality)")
            print(f"         Early failures, then stabilizes")
        else:
            print(f"  c = 1: CONSTANT hazard (memoryless/exponential)")

    except Exception as e:
        print(f"Weibull fit failed: {e}")
        weibull_b, weibull_c, r2 = None, None, None

    # =========================================================================
    # Plot all three analyses
    # =========================================================================
    if output_path:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        })

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot 1: Transition probabilities
        ax1 = axes[0]
        labels = ["P(F|S)", "P(F|F)", "P(S|F)"]
        values = [p_fail_given_success * 100, p_fail_given_fail * 100, p_success_given_fail * 100]
        colors = ["#648FFF", "#DC267F", "#228B22"]
        bars = ax1.bar(labels, values, color=colors, edgecolor="black", linewidth=1)
        ax1.set_ylabel("Probability (%)")
        ax1.set_title("(a) Transition Probabilities", fontweight="bold")
        ax1.set_ylim(0, 100)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{val:.1f}%", ha="center", fontsize=9)

        # Plot 2: Run length distribution
        ax2 = axes[1]
        ax2.hist(success_runs, bins=30, alpha=0.6, label=f"Success (mean={np.mean(success_runs):.1f})",
                 color="#228B22", edgecolor="black", linewidth=0.5)
        ax2.hist(fail_runs, bins=30, alpha=0.6, label=f"Fail (mean={np.mean(fail_runs):.1f})",
                 color="#DC267F", edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Run Length")
        ax2.set_ylabel("Frequency")
        ax2.set_title("(b) Run Length Distribution", fontweight="bold")
        ax2.legend()
        ax2.set_xlim(0, min(100, max(max(success_runs), max(fail_runs))))

        # Plot 3: Survival curve
        ax3 = axes[2]
        ax3.step(survival_t, survival_prob * 100, where="post", color="#648FFF",
                 linewidth=2, label="Empirical")
        if weibull_c is not None:
            t_smooth = np.linspace(0, max(t_fit), 200)
            ax3.plot(t_smooth, weibull_survival(t_smooth, weibull_b, weibull_c) * 100,
                     color="#DC267F", linewidth=2, linestyle="--",
                     label=f"Weibull (c={weibull_c:.2f})")
        ax3.set_xlabel("Position (chunks from start)")
        ax3.set_ylabel("Survival Probability (%)")
        ax3.set_title("(c) Time to First Failure", fontweight="bold")
        ax3.legend()
        ax3.set_xlim(0, min(150, max(survival_t)))
        ax3.set_ylim(0, 105)

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()

        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {pdf_path} and {output_path}")
        plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Cumulative Error Hypothesis")
    print(f"{'='*70}")

    evidence_for = 0
    evidence_against = 0

    print("\n| Test | Result | Interpretation |")
    print("|------|--------|----------------|")

    # Test 1: Sticky ratio
    if sticky_ratio > 1.5:
        print(f"| Sticky ratio | {sticky_ratio:.2f} | Failure is sticky (supports) |")
        evidence_for += 1
    else:
        print(f"| Sticky ratio | {sticky_ratio:.2f} | Not sticky (against) |")
        evidence_against += 1

    # Test 2: Fail run length
    if np.mean(fail_runs) > expected_fail_run * 1.2:
        print(f"| Fail run length | {np.mean(fail_runs):.1f} > {expected_fail_run:.1f} | Longer than expected (supports) |")
        evidence_for += 1
    else:
        print(f"| Fail run length | {np.mean(fail_runs):.1f} ~ {expected_fail_run:.1f} | As expected (against) |")
        evidence_against += 1

    # Test 3: Weibull shape
    if weibull_c is not None:
        if weibull_c > 1.2:
            print(f"| Weibull c | {weibull_c:.2f} | Accelerating hazard (supports) |")
            evidence_for += 1
        elif weibull_c < 0.8:
            print(f"| Weibull c | {weibull_c:.2f} | Decelerating hazard (against) |")
            evidence_against += 1
        else:
            print(f"| Weibull c | {weibull_c:.2f} | Near constant (neutral) |")

    print(f"\nEvidence for cumulative error: {evidence_for}/3")
    print(f"Evidence against: {evidence_against}/3")

    if evidence_for >= 2:
        print("\n*** CONCLUSION: Cumulative error hypothesis is SUPPORTED ***")
    elif evidence_against >= 2:
        print("\n*** CONCLUSION: Cumulative error hypothesis is NOT SUPPORTED ***")
    else:
        print("\n*** CONCLUSION: Mixed evidence, no strong conclusion ***")

    return {
        "transitions": transitions,
        "p_fail_given_success": p_fail_given_success,
        "p_fail_given_fail": p_fail_given_fail,
        "sticky_ratio": sticky_ratio,
        "mean_success_run": np.mean(success_runs),
        "mean_fail_run": np.mean(fail_runs),
        "weibull_b": weibull_b,
        "weibull_c": weibull_c,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }


def plot_pipeline_vs_e2e_mode_locking(
    mt3_results: List[ChunkResult],
    zeng_csv_path: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Compare Mode Locking between Pipeline (MT3) and E2E (Zeng) methods.

    Key insight: Pipeline methods get "locked" into failure mode early,
    while E2E methods can recover from failures.

    Creates a figure with:
    (a) Recovery Rate comparison: P(success | prev_fail)
    (b) First-chunk effect comparison

    Args:
        mt3_results: List of MT3 chunk results
        zeng_csv_path: Path to Zeng results CSV
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    # =========================================================================
    # Process MT3 data
    # =========================================================================
    mt3_by_perf = defaultdict(list)
    for r in mt3_results:
        key = (r.piece_id, r.performance)
        mt3_by_perf[key].append(r)

    mt3_transitions = {"SS": 0, "SF": 0, "FS": 0, "FF": 0}
    mt3_songs = []

    for (piece, perf), chunks in mt3_by_perf.items():
        chunks_sorted = sorted(chunks, key=lambda x: x.chunk_index)
        seq = [1 if c.status == "success" else 0 for c in chunks_sorted]
        if len(seq) < 2:
            continue
        mt3_songs.append({
            "sequence": seq,
            "first_success": seq[0] == 1,
        })
        for i in range(len(seq) - 1):
            prev, curr = seq[i], seq[i + 1]
            key = ("S" if prev else "F") + ("S" if curr else "F")
            mt3_transitions[key] += 1

    # =========================================================================
    # Process Zeng data
    # =========================================================================
    zeng_by_perf = defaultdict(list)
    with open(zeng_csv_path, "r") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "")
            if "#" in task_id:
                parts = task_id.rsplit("#", 1)
                if len(parts) == 2 and "." in parts[1]:
                    piece = parts[0]
                    perf = parts[1].rsplit(".", 1)[0]
                    chunk_idx = int(parts[1].rsplit(".", 1)[1])
                    zeng_by_perf[(piece, perf)].append({
                        "chunk_idx": chunk_idx,
                        "status": row.get("status", "unknown"),
                    })

    zeng_transitions = {"SS": 0, "SF": 0, "FS": 0, "FF": 0}
    zeng_songs = []

    for (piece, perf), chunks in zeng_by_perf.items():
        chunks_sorted = sorted(chunks, key=lambda x: x["chunk_idx"])
        seq = [1 if c["status"] == "success" else 0 for c in chunks_sorted]
        if len(seq) < 2:
            continue
        zeng_songs.append({
            "sequence": seq,
            "first_success": seq[0] == 1,
        })
        for i in range(len(seq) - 1):
            prev, curr = seq[i], seq[i + 1]
            key = ("S" if prev else "F") + ("S" if curr else "F")
            zeng_transitions[key] += 1

    # =========================================================================
    # Calculate metrics
    # =========================================================================
    # Recovery rate: P(success | prev_fail)
    mt3_recovery = mt3_transitions["FS"] / (mt3_transitions["FS"] + mt3_transitions["FF"]) \
        if (mt3_transitions["FS"] + mt3_transitions["FF"]) > 0 else 0
    zeng_recovery = zeng_transitions["FS"] / (zeng_transitions["FS"] + zeng_transitions["FF"]) \
        if (zeng_transitions["FS"] + zeng_transitions["FF"]) > 0 else 0

    # =========================================================================
    # Plot: Single panel showing Recovery Rate
    # =========================================================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    color_pipeline = "#DC267F"  # Magenta for Pipeline
    color_e2e = "#228B22"       # Green for E2E

    # Recovery Rate comparison
    methods = ["MT3+MuseScore\n(Pipeline)", "Zeng\n(E2E)"]
    recovery_rates = [mt3_recovery * 100, zeng_recovery * 100]
    n_samples = [
        mt3_transitions["FS"] + mt3_transitions["FF"],
        zeng_transitions["FS"] + zeng_transitions["FF"],
    ]
    colors = [color_pipeline, color_e2e]

    bars = ax.bar(methods, recovery_rates, color=colors, edgecolor="black", linewidth=1.5, width=0.6)

    for bar, val, n in zip(bars, recovery_rates, n_samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%\n(n={n:,})", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Recovery Rate: P(success | prev_fail)", fontsize=13)
    ax.set_title("Mode Locking: Pipeline vs E2E", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(1.35, 52, "Random (50%)", fontsize=10, color="gray", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()

    if output_path:
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Pipeline vs E2E plot saved to: {pdf_path} and {output_path}")
    else:
        plt.show()

    plt.close()


def plot_mode_locking(
    results: List[ChunkResult],
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize Mode Locking phenomenon in MT3+MuseScore pipeline.

    Shows that success/failure is determined early and persists throughout the song,
    rather than accumulating errors over time.

    Creates a figure with:
    (a) Per-song success pattern heatmap (sorted by success rate)
    (b) First-chunk conditional analysis
    (c) Transition probability diagram

    Args:
        results: List of chunk results
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    # =========================================================================
    # Prepare data
    # =========================================================================
    by_performance = defaultdict(list)
    for r in results:
        key = (r.piece_id, r.performance)
        by_performance[key].append(r)

    # Build per-song data
    song_data = []
    for (piece_id, perf), chunks in by_performance.items():
        chunks_sorted = sorted(chunks, key=lambda x: x.chunk_index)
        seq = [1 if c.status == "success" else 0 for c in chunks_sorted]
        success_rate = np.mean(seq)
        first_success = seq[0] == 1 if seq else False
        song_data.append({
            "piece_id": piece_id,
            "performance": perf,
            "sequence": seq,
            "length": len(seq),
            "success_rate": success_rate,
            "first_success": first_success,
        })

    # Sort by success rate for heatmap
    song_data_sorted = sorted(song_data, key=lambda x: -x["success_rate"])

    # =========================================================================
    # Calculate statistics
    # =========================================================================
    # First-chunk conditional
    first_success_songs = [s for s in song_data if s["first_success"]]
    first_fail_songs = [s for s in song_data if not s["first_success"]]

    rest_rate_after_success = np.mean([
        np.mean(s["sequence"][1:]) if len(s["sequence"]) > 1 else 0
        for s in first_success_songs
    ]) if first_success_songs else 0

    rest_rate_after_fail = np.mean([
        np.mean(s["sequence"][1:]) if len(s["sequence"]) > 1 else 0
        for s in first_fail_songs
    ]) if first_fail_songs else 0

    # Transition probabilities
    transitions = {"SS": 0, "SF": 0, "FS": 0, "FF": 0}
    for s in song_data:
        seq = s["sequence"]
        for i in range(len(seq) - 1):
            prev, curr = seq[i], seq[i + 1]
            key = ("S" if prev else "F") + ("S" if curr else "F")
            transitions[key] += 1

    total_after_s = transitions["SS"] + transitions["SF"]
    total_after_f = transitions["FS"] + transitions["FF"]
    p_ss = transitions["SS"] / total_after_s if total_after_s > 0 else 0
    p_ff = transitions["FF"] / total_after_f if total_after_f > 0 else 0

    # =========================================================================
    # Plot
    # =========================================================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    fig = plt.figure(figsize=(12, 5))

    # Layout: heatmap takes 60%, other two share 40%
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.3)

    # -------------------------------------------------------------------------
    # (a) Heatmap: Per-song success pattern
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    # Normalize sequences to same length (0-100%)
    n_bins = 50
    heatmap_data = []
    for s in song_data_sorted:
        seq = s["sequence"]
        if len(seq) < 2:
            continue
        # Resample to n_bins
        indices = np.linspace(0, len(seq) - 1, n_bins).astype(int)
        resampled = [seq[i] for i in indices]
        heatmap_data.append(resampled)

    heatmap_array = np.array(heatmap_data)

    # Custom colormap: red for fail, green for success
    cmap = LinearSegmentedColormap.from_list("rg", ["#DC267F", "#228B22"])

    im = ax1.imshow(heatmap_array, aspect="auto", cmap=cmap,
                    extent=[0, 100, len(heatmap_data), 0])

    ax1.set_xlabel("Position (% of song)")
    ax1.set_ylabel("Songs (sorted by success rate)")
    ax1.set_title("(a) Per-Song Success Pattern", fontweight="bold")

    # Add colorbar legend
    legend_elements = [
        mpatches.Patch(facecolor="#228B22", label="Success"),
        mpatches.Patch(facecolor="#DC267F", label="Fail"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # (b) First-chunk conditional
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    categories = ["First chunk\nSUCCESS", "First chunk\nFAIL"]
    values = [rest_rate_after_success * 100, rest_rate_after_fail * 100]
    colors = ["#228B22", "#DC267F"]
    counts = [len(first_success_songs), len(first_fail_songs)]

    bars = ax2.bar(categories, values, color=colors, edgecolor="black", linewidth=1)

    for bar, val, n in zip(bars, values, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.1f}%\n(n={n})", ha="center", fontsize=9)

    ax2.set_ylabel("Rest of Song Success Rate (%)")
    ax2.set_title("(b) First Chunk Determines Fate", fontweight="bold")
    ax2.set_ylim(0, 70)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # (c) Transition diagram
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])

    # Draw two circles: Success state and Fail state
    circle_s = plt.Circle((0.3, 0.5), 0.15, color="#228B22", alpha=0.8)
    circle_f = plt.Circle((0.7, 0.5), 0.15, color="#DC267F", alpha=0.8)
    ax3.add_patch(circle_s)
    ax3.add_patch(circle_f)

    ax3.text(0.3, 0.5, "S", ha="center", va="center", fontsize=16,
             fontweight="bold", color="white")
    ax3.text(0.7, 0.5, "F", ha="center", va="center", fontsize=16,
             fontweight="bold", color="white")

    # Self-loops (arrows back to same state)
    # S -> S
    ax3.annotate("", xy=(0.2, 0.65), xytext=(0.4, 0.65),
                 arrowprops=dict(arrowstyle="->", color="#228B22", lw=2,
                                connectionstyle="arc3,rad=0.5"))
    ax3.text(0.3, 0.82, f"{p_ss*100:.0f}%", ha="center", fontsize=10,
             fontweight="bold", color="#228B22")

    # F -> F
    ax3.annotate("", xy=(0.8, 0.65), xytext=(0.6, 0.65),
                 arrowprops=dict(arrowstyle="->", color="#DC267F", lw=2,
                                connectionstyle="arc3,rad=-0.5"))
    ax3.text(0.7, 0.82, f"{p_ff*100:.0f}%", ha="center", fontsize=10,
             fontweight="bold", color="#DC267F")

    # Cross transitions (smaller, less prominent)
    # S -> F
    ax3.annotate("", xy=(0.55, 0.45), xytext=(0.45, 0.45),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax3.text(0.5, 0.38, f"{(1-p_ss)*100:.0f}%", ha="center", fontsize=9, color="gray")

    # F -> S
    ax3.annotate("", xy=(0.45, 0.55), xytext=(0.55, 0.55),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax3.text(0.5, 0.62, f"{(1-p_ff)*100:.0f}%", ha="center", fontsize=9, color="gray")

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.set_title("(c) Sticky Transitions", fontweight="bold")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()

    if output_path:
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mode locking plot saved to: {pdf_path} and {output_path}")
    else:
        plt.show()

    plt.close()


def plot_phase_drift(
    results: List[ChunkResult],
    output_path: Optional[str] = None,
    window_size: int = 20,
) -> None:
    """
    Plot Phase Drift visualization: Success Rate and MV2H vs Chunk Position.

    Uses a sliding window to smooth the data and show the trend clearly.

    Args:
        results: List of chunk results
        output_path: Path to save the plot (if None, displays interactively)
        window_size: Window size for rolling average
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting: pip install matplotlib")
        return

    # Group results by chunk_index
    by_index = defaultdict(list)
    for r in results:
        by_index[r.chunk_index].append(r)

    # Compute per-index metrics
    indices = sorted(by_index.keys())
    success_rates = []
    mv2h_scores = []

    for idx in indices:
        chunk_results = by_index[idx]
        successful = [r for r in chunk_results if r.status == "success"]
        success_rate = len(successful) / len(chunk_results) if chunk_results else 0
        success_rates.append(success_rate)

        # MV2H for successful chunks only
        if successful:
            mv2h = np.mean([r.mv2h_custom for r in successful])
        else:
            mv2h = 0
        mv2h_scores.append(mv2h)

    # Convert to numpy arrays
    indices = np.array(indices)
    success_rates = np.array(success_rates)
    mv2h_scores = np.array(mv2h_scores)

    # Compute rolling average
    def rolling_mean(arr, window):
        cumsum = np.cumsum(np.insert(arr, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    if len(indices) > window_size:
        roll_indices = indices[window_size - 1:]
        roll_success = rolling_mean(success_rates, window_size)
        roll_mv2h = rolling_mean(mv2h_scores, window_size)
    else:
        roll_indices = indices
        roll_success = success_rates
        roll_mv2h = mv2h_scores

    # Publication quality settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "figure.dpi": 300,
    })

    # Figure size: 7 inch wide (double column), 5 inch tall
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Colors: colorblind-friendly (IBM Design)
    color_scatter = "#648FFF"  # blue
    color_line = "#DC267F"     # magenta

    # Plot 1: Success Rate
    ax1.scatter(indices, success_rates * 100, alpha=0.25, s=8,
                color=color_scatter, edgecolors="none", rasterized=True)
    ax1.plot(roll_indices, roll_success * 100, color=color_line, linewidth=2,
             label=f"Rolling mean (n={window_size})")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("(a) Evaluation Success Rate", loc="left", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(indices) + 10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot 2: MV2H Score
    ax2.scatter(indices, mv2h_scores * 100, alpha=0.25, s=8,
                color=color_scatter, edgecolors="none", rasterized=True)
    ax2.plot(roll_indices, roll_mv2h * 100, color=color_line, linewidth=2,
             label=f"Rolling mean (n={window_size})")
    ax2.set_xlabel("Chunk Position (measure number)")
    ax2.set_ylabel("MV2H Score (%)")
    ax2.set_title("(b) MV2H Score (successful chunks)", loc="left", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        # Save both PDF (vector) and PNG (raster)
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {pdf_path} (vector) and {output_path} (300 DPI)")
    else:
        plt.show()


# =============================================================================
# CLI
# =============================================================================

# Default paths
DEFAULT_CSV = "data/experiments/mt3/results/chunks_song.csv"
DEFAULT_PIPELINE_VS_E2E_PLOT = "results/pipeline_vs_e2e.png"

# Zeng per-chunk results (from piano-a2s evaluation)
ZENG_CHUNK_CSV = "/home/bloggerwang/piano-a2s/results/chunk_results.csv"


def main():
    """
    Analyze MT3 MV2H results with sensible defaults.

    Just run: poetry run python src/analysis/analyze_mv2h_results.py
    """
    # Use command line arg if provided, otherwise default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        print(f"Expected: {DEFAULT_CSV}")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# Analyzing: {csv_path}")
    print(f"{'#' * 70}")

    # Load results
    results = load_results_csv(csv_path)
    print(f"Loaded {len(results)} chunk results")

    # Run analyses
    summary = compute_overall_summary(results)
    print_overall_summary(summary)

    position_analysis = analyze_by_chunk_position(results)
    print_position_analysis(position_analysis)

    composer_analysis = analyze_by_composer(results)
    print_composer_analysis(composer_analysis)

    # Generate plots
    Path(DEFAULT_PIPELINE_VS_E2E_PLOT).parent.mkdir(parents=True, exist_ok=True)

    # Pipeline vs E2E Mode Locking comparison (the key figure!)
    plot_pipeline_vs_e2e_mode_locking(
        mt3_results=results,
        zeng_csv_path=ZENG_CHUNK_CSV,
        output_path=DEFAULT_PIPELINE_VS_E2E_PLOT,
    )


if __name__ == "__main__":
    main()
