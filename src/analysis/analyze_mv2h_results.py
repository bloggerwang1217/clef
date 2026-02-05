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


def print_fair_comparison_table(
    csv_paths: Dict[str, str],
    filter_task_ids: set,
    system_types: Dict[str, str],
) -> None:
    """
    Print fair comparison table for all systems on the same sample set.

    Args:
        csv_paths: Dict mapping system name to CSV path
        filter_task_ids: Set of task_ids to filter to (for fair comparison)
        system_types: Dict mapping system name to type (Pipeline/E2E)
    """
    import pandas as pd

    print("\n" + "=" * 80)
    print("FAIR COMPARISON TABLE (Strict Method: failures as 0)")
    print(f"Sample set: n = {len(filter_task_ids):,}")
    print("=" * 80)

    results = []
    for name, csv_path in csv_paths.items():
        if not Path(csv_path).exists():
            print(f"Warning: {csv_path} not found, skipping {name}")
            continue

        df = pd.read_csv(csv_path)

        # Filter to fair comparison set
        df = df[df["task_id"].isin(filter_task_ids)].copy()
        total = len(df)
        successful = df[df["status"] == "success"]
        n_success = len(successful)

        # Evaluability
        eval_rate = n_success / total if total > 0 else 0

        # Strict metrics: sum of successful / total (failures as 0)
        multi_pitch = successful["Multi-pitch"].sum() / total if total > 0 else 0
        voice = successful["Voice"].sum() / total if total > 0 else 0
        value = successful["Value"].sum() / total if total > 0 else 0
        harmony = successful["Harmony"].sum() / total if total > 0 else 0
        mv2h = successful["MV2H"].sum() / total if total > 0 else 0

        # MV2H_custom (4-metric average)
        mv2h_custom = (multi_pitch + voice + value + harmony) / 4

        results.append({
            "name": name,
            "type": system_types.get(name, "Unknown"),
            "n": total,
            "eval": eval_rate,
            "multi_pitch": multi_pitch,
            "voice": voice,
            "value": value,
            "harmony": harmony,
            "mv2h": mv2h,
            "mv2h_custom": mv2h_custom,
        })

        print(f"\n{name}:")
        print(f"  n = {total}")
        print(f"  Evaluability = {eval_rate*100:.1f}%")
        print(f"  Multi-pitch  = {multi_pitch*100:.1f}")
        print(f"  Voice        = {voice*100:.1f}")
        print(f"  Value        = {value*100:.1f}")
        print(f"  Harmony      = {harmony*100:.1f}")
        print(f"  MV2H         = {mv2h*100:.1f}")
        print(f"  MV2H_custom  = {mv2h_custom*100:.1f}")

    # Print LaTeX table format
    print("\n" + "=" * 80)
    print("LaTeX Table Rows (copy-paste ready)")
    print("=" * 80)
    for r in results:
        # Format: System & Type & n & Eval & F_p & F_voi & F_val & F_harm & F_MV2H \\
        print(f"{r['name']} & {r['type']} & {r['n']:,} & "
              f"{r['eval']*100:.1f} & {r['multi_pitch']*100:.1f} & "
              f"{r['voice']*100:.1f} & {r['value']*100:.1f} & "
              f"{r['harmony']*100:.1f} & {r['mv2h_custom']*100:.1f} \\\\")


# =============================================================================
# VISUALIZATION
# =============================================================================


# =============================================================================
# LOGISTIC TRANSITION MODEL
# =============================================================================


def run_transition_model(
    csv_path: str,
    name: str,
) -> Dict[str, Any]:
    """
    Fit Logistic Transition Model to analyze Mode Locking and Phase Drift.

    Model specification:
        logit(P(Y_t = 1)) = beta_0 + beta_1 * position + beta_2 * Y_{t-1}

    Where:
        - Y_t: success (1) or failure (0) at chunk t
        - position: normalized position in song (0-1)
        - Y_{t-1}: lagged outcome (previous chunk's success)

    Parameters:
        - beta_1: Phase Drift effect (negative = success decreases with position)
        - beta_2: Mode Locking effect (positive = sticky success/failure)

    Interpretation:
        - OR(position) < 1: Phase Drift present (success odds decrease from start to end)
        - OR(prev_success) >> 1: Mode Locking present (failure is "sticky")
        - Recovery Rate = P(success | prev_fail) at mid-song

    Note:
        GEE with exchangeable correlation was tested but Independence structure
        performed better (lower QIC), indicating that the transition term (Y_{t-1})
        already captures most within-song dependence. Simple logistic regression
        is therefore preferred for interpretability.

    Args:
        csv_path: Path to chunk results CSV
        name: Name of the system for display

    Returns:
        Dictionary with model results including coefficients, odds ratios, and CIs
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.read_csv(csv_path)
    df["success"] = (df["status"] == "success").astype(int)

    # Parse song_id from different CSV formats
    if "piece_id" in df.columns and "performance" in df.columns:
        df["song_id"] = df["piece_id"] + "#" + df["performance"]
    elif "task_id" in df.columns:
        df["song_id"] = df["task_id"].apply(
            lambda x: x.rsplit("#", 1)[0] + "#" + x.rsplit("#", 1)[1].rsplit(".", 1)[0]
            if "#" in x and "." in x.rsplit("#", 1)[1]
            else x
        )

    # Get chunk_index from different CSV formats
    if "chunk_index" not in df.columns:
        if "measure_idx" in df.columns:
            df["chunk_index"] = df["measure_idx"]
        elif "task_id" in df.columns:
            df["chunk_index"] = df["task_id"].apply(
                lambda x: int(x.rsplit(".", 1)[1]) if "." in x else 0
            )

    # Normalize position within song (0-1 scale)
    df["song_length"] = df.groupby("song_id")["chunk_index"].transform("max") + 1
    df["position_norm"] = df["chunk_index"] / df["song_length"]

    # Sort and create lagged Y
    df_sorted = df.sort_values(["song_id", "chunk_index"]).reset_index(drop=True)
    df_sorted["prev_success"] = df_sorted.groupby("song_id")["success"].shift(1)

    # Drop first observation of each song (no prev)
    df_model = df_sorted.dropna(subset=["prev_success"]).copy()
    df_model["prev_success"] = df_model["prev_success"].astype(int)

    # Fit logistic regression
    model = smf.logit("success ~ position_norm + prev_success", data=df_model)
    result = model.fit(disp=0)

    # Extract results
    b0 = result.params["Intercept"]
    b1 = result.params["position_norm"]
    b2 = result.params["prev_success"]
    se1 = result.bse["position_norm"]
    se2 = result.bse["prev_success"]
    p1 = result.pvalues["position_norm"]
    p2 = result.pvalues["prev_success"]

    # Odds ratios with 95% CI
    or1 = np.exp(b1)
    or1_ci = (np.exp(b1 - 1.96 * se1), np.exp(b1 + 1.96 * se1))
    or2 = np.exp(b2)
    or2_ci = (np.exp(b2 - 1.96 * se2), np.exp(b2 + 1.96 * se2))

    # Calculate probabilities at mid-song
    avg_pos = 0.5
    logit_after_fail = b0 + b1 * avg_pos + b2 * 0
    logit_after_success = b0 + b1 * avg_pos + b2 * 1
    p_after_fail = 1 / (1 + np.exp(-logit_after_fail))
    p_after_success = 1 / (1 + np.exp(-logit_after_success))

    return {
        "name": name,
        "n_obs": len(df_model),
        "n_songs": df_model["song_id"].nunique(),
        "success_rate": df_model["success"].mean(),
        "b0_intercept": b0,
        "b1_position": b1,
        "se1_position": se1,
        "p1_position": p1,
        "b2_prev": b2,
        "se2_prev": se2,
        "p2_prev": p2,
        "or_position": or1,
        "or_position_ci": or1_ci,
        "or_prev": or2,
        "or_prev_ci": or2_ci,
        "p_after_fail": p_after_fail,
        "p_after_success": p_after_success,
        "recovery_rate": p_after_fail,
        "pseudo_r2": result.prsquared,
        "aic": result.aic,
        "has_phase_drift": p1 < 0.05 and b1 < 0,
        "has_mode_locking": p2 < 0.05 and b2 > 0,
    }


def print_transition_model_results(results: List[Dict[str, Any]]) -> None:
    """Print Logistic Transition Model results in formatted table."""
    print("\n" + "=" * 95)
    print("LOGISTIC TRANSITION MODEL RESULTS")
    print("Model: logit(P(Y_t)) = b0 + b1*position + b2*prev_success")
    print("=" * 95)

    # Coefficients table
    print(f"\n{'System':<20} {'b1(pos)':>8} {'OR1':>8} {'95% CI':>16} {'b2(prev)':>10} {'OR2':>8}")
    print("-" * 95)

    for r in results:
        sig1 = "***" if r["p1_position"] < 0.001 else "**" if r["p1_position"] < 0.01 else "*" if r["p1_position"] < 0.05 else ""
        sig2 = "***" if r["p2_prev"] < 0.001 else ""
        or1_ci = r.get("or_position_ci", (0, 0))
        print(f"{r['name']:<20} {r['b1_position']:>7.3f}{sig1:<1} {r['or_position']:>8.2f} "
              f"[{or1_ci[0]:.2f}, {or1_ci[1]:.2f}] {r['b2_prev']:>9.3f}{sig2:<1} {r['or_prev']:>8.1f}")

    # Interpretation
    print("\n" + "-" * 95)
    print("INTERPRETATION")
    print("-" * 95)
    print(f"{'System':<20} {'Phase Drift':>12} {'Mode Locking':>14} {'Recovery':>10} {'R²':>8}")
    print("-" * 95)

    for r in results:
        pd_str = "YES" if r["has_phase_drift"] else "NO"
        ml_str = "SEVERE" if r["or_prev"] > 10 else "MODERATE" if r["or_prev"] > 3 else "MINIMAL"
        recovery = r.get("recovery_rate", r["p_after_fail"])
        pseudo_r2 = r.get("pseudo_r2", 0)
        print(f"{r['name']:<20} {pd_str:>12} {ml_str:>14} {recovery*100:>9.1f}% {pseudo_r2:>8.3f}")

    # Recovery rate details
    print("\n" + "-" * 95)
    print("Recovery Rate: P(success | prev_fail) at mid-song position")
    print("-" * 95)

    for r in results:
        print(f"{r['name']:<20} P(S|F)={r['p_after_fail']*100:>5.1f}%  "
              f"P(S|S)={r['p_after_success']*100:>5.1f}%  "
              f"OR(prev)={r['or_prev']:>5.1f}x")


def compute_markov_marginal(b0: float, b1: float, b2: float, pi_0: float, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical marginal P(success | position) by iterating Markov chain.

    The transition model gives conditional probabilities:
        P(S | position, prev=S) = expit(b0 + b1*position + b2*1)
        P(S | position, prev=F) = expit(b0 + b1*position + b2*0)

    The marginal at each position is computed by:
        pi_{t+1} = P(S|S, pos) * pi_t + P(S|F, pos) * (1 - pi_t)

    This captures the self-reinforcing nature of Mode Locking:
    early failures propagate forward, causing exponential-like decay.

    Args:
        b0, b1, b2: Transition model parameters
        pi_0: Initial P(success) at position 0
        n_steps: Number of position steps

    Returns:
        positions: Array of normalized positions (0 to 1)
        marginals: Array of P(success) at each position
    """
    positions = np.linspace(0, 1, n_steps)
    marginals = np.zeros(n_steps)
    marginals[0] = pi_0

    for t in range(1, n_steps):
        pos = positions[t]

        # Transition probabilities at this position
        p_s_given_s = 1 / (1 + np.exp(-(b0 + b1 * pos + b2 * 1)))  # P(S | prev=S)
        p_s_given_f = 1 / (1 + np.exp(-(b0 + b1 * pos + b2 * 0)))  # P(S | prev=F)

        # Marginal update: weighted by probability of being in each state
        pi_prev = marginals[t - 1]
        marginals[t] = p_s_given_s * pi_prev + p_s_given_f * (1 - pi_prev)

    return positions, marginals


def plot_success_rate_by_position(
    csv_paths: Dict[str, str],
    output_path: Optional[str] = None,
    n_empirical_bins: int = 50,
    filter_to_task_ids: Optional[set] = None,
) -> None:
    """
    Plot MV2H Evaluability vs normalized position with Markov chain theoretical curve.

    Shows:
    - Theoretical marginal from Markov chain simulation (solid line)
    - Empirical data points (scatter)

    The theoretical curve is derived from the transition model parameters
    by iterating the Markov chain forward from the initial success rate.

    Args:
        csv_paths: Dict mapping system name to CSV path
        output_path: Path to save the plot
        n_empirical_bins: Number of bins for empirical data visualization
        filter_to_task_ids: If provided, filter all systems to only these task_ids
                           (for fair comparison across systems with different sample sizes)
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("matplotlib, pandas, and statsmodels required")
        return

    # Publication quality settings (larger fonts for print)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.5,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    # Colors for each system (formal names for publication)
    # Using IBM/Wong colorblind-safe palette
    styles = {
        "MT3 + MuseScore": {"color": "#DC267F", "label": "MT3 + MuseScore (Pipeline)"},
        "Transkun + Beyer": {"color": "#FE6100", "label": "Transkun + Beyer & Dai (Pipeline)"},
        "Zeng (E2E)": {"color": "#648FFF", "label": "Zeng et al. (E2E*)"},
    }

    for name, csv_path in csv_paths.items():
        if not Path(csv_path).exists():
            print(f"Warning: {csv_path} not found, skipping {name}")
            continue

        df = pd.read_csv(csv_path)

        # Filter to specified task_ids for fair comparison
        if filter_to_task_ids is not None:
            original_len = len(df)
            df = df[df["task_id"].isin(filter_to_task_ids)].copy()
            print(f"  {name}: filtered {original_len} -> {len(df)} samples")

        df["success"] = (df["status"] == "success").astype(int)

        # Parse song_id and chunk_index from task_id
        if "task_id" in df.columns:
            df["song_id"] = df["task_id"].apply(
                lambda x: x.rsplit(".", 1)[0] if "." in x else x
            )
            df["chunk_index"] = df["task_id"].apply(
                lambda x: int(x.rsplit(".", 1)[1]) if "." in x else 0
            )

        # Compute normalized position within each song
        song_lengths = df.groupby("song_id")["chunk_index"].max()
        df["song_length"] = df["song_id"].map(song_lengths)
        df["position_norm"] = df["chunk_index"] / df["song_length"]

        # Get initial success rate (first chunk of each song)
        first_chunks = df[df["chunk_index"] == 1]
        pi_0 = first_chunks["success"].mean() if len(first_chunks) > 0 else df["success"].mean()

        # Fit transition model
        df_sorted = df.sort_values(["song_id", "chunk_index"]).reset_index(drop=True)
        df_sorted["prev_success"] = df_sorted.groupby("song_id")["success"].shift(1)
        df_model = df_sorted.dropna(subset=["prev_success"]).copy()
        df_model["prev_success"] = df_model["prev_success"].astype(int)

        try:
            model = smf.logit("success ~ position_norm + prev_success", data=df_model)
            result = model.fit(disp=0)
            b0 = result.params["Intercept"]
            b1 = result.params["position_norm"]
            b2 = result.params["prev_success"]
        except Exception as e:
            print(f"Could not fit model for {name}: {e}")
            continue

        # Compute theoretical marginal via Markov chain
        positions, marginals = compute_markov_marginal(b0, b1, b2, pi_0)

        style = styles.get(name, {"color": "gray"})
        color = style["color"]

        # Compute measure-level evaluability
        # Each chunk covers 5 measures with stride=1, so measures overlap across chunks
        # For measure m: evaluability = mean(success) for all chunks containing m
        measure_data = []

        for song_id in df["song_id"].unique():
            song_df = df[df["song_id"] == song_id].copy()
            if len(song_df) == 0:
                continue

            # Determine measure range for this song
            # chunk_index corresponds to start_measure (1-indexed, 5-bar chunks)
            min_chunk = song_df["chunk_index"].min()
            max_chunk = song_df["chunk_index"].max()
            # Last chunk covers measures [max_chunk, max_chunk+4]
            max_measure = max_chunk + 4

            # For each measure, find chunks that contain it
            for measure in range(min_chunk, max_measure + 1):
                # Chunk with start_measure=s covers measures [s, s+4]
                # So measure m is in chunk s if s <= m <= s+4, i.e., m-4 <= s <= m
                containing_chunks = song_df[
                    (song_df["chunk_index"] >= measure - 4) &
                    (song_df["chunk_index"] <= measure)
                ]

                if len(containing_chunks) > 0:
                    evaluability = containing_chunks["success"].mean()
                    # Normalized position: measure / total_measures
                    pos_norm = measure / max_measure
                    measure_data.append({
                        "song_id": song_id,
                        "measure": measure,
                        "position_norm": pos_norm,
                        "evaluability": evaluability,
                    })

        if measure_data:
            measure_df = pd.DataFrame(measure_data)

            # Compute empirical mean with bootstrap CI using position bins
            n_bins = 50
            measure_df["pos_bin"] = pd.cut(
                measure_df["position_norm"], bins=n_bins, labels=False
            )

            # Compute mean and bootstrap CI for each bin
            bin_stats = []
            for bin_idx in range(n_bins):
                bin_data = measure_df[measure_df["pos_bin"] == bin_idx]["evaluability"].values
                if len(bin_data) >= 5:
                    mean_val = np.mean(bin_data)
                    # Bootstrap CI
                    n_bootstrap = 1000
                    np.random.seed(42)
                    boot_means = [
                        np.mean(np.random.choice(bin_data, size=len(bin_data), replace=True))
                        for _ in range(n_bootstrap)
                    ]
                    ci_low = np.percentile(boot_means, 2.5)
                    ci_high = np.percentile(boot_means, 97.5)
                    bin_stats.append({
                        "pos": (bin_idx + 0.5) / n_bins,
                        "mean": mean_val,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    })

            if bin_stats:
                stats_df = pd.DataFrame(bin_stats)

                # Plot empirical CI ribbon
                ax.fill_between(
                    stats_df["pos"],
                    stats_df["ci_low"] * 100,
                    stats_df["ci_high"] * 100,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

                # Plot empirical mean line (dashed)
                ax.plot(
                    stats_df["pos"],
                    stats_df["mean"] * 100,
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )

        # Plot theoretical marginal
        ax.plot(
            positions,
            marginals * 100,
            color=color,
            linewidth=2.5,
            label=style.get("label", name),
            alpha=0.9,
        )

    # Plot formatting
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Normalized Position within Song")
    ax.set_ylabel("MV2H Evaluability (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="-")

    plt.tight_layout()

    if output_path:
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Success rate plot saved to: {pdf_path} and {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# CLI
# =============================================================================

# Default paths
DEFAULT_CSV = "data/experiments/mt3/results/chunks_song.csv"
TRANSKUN_BEYER_CSV = "data/experiments/transkun_beyer/results/chunks.csv"
ZENG_CSV = "data/experiments/zeng/results/chunks.csv"
DEFAULT_SUCCESS_RATE_PLOT = "results/success_rate_by_position.png"


def main():
    """
    Analyze MV2H results with Mixed Transition Model.

    Usage:
        # Run full analysis with all systems
        poetry run python src/analysis/analyze_mv2h_results.py

        # Analyze specific CSV
        poetry run python src/analysis/analyze_mv2h_results.py path/to/results.csv
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Parse arguments (simple flag parsing)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    # Use command line arg if provided, otherwise default
    csv_path = args[0] if args else DEFAULT_CSV

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        print(f"Expected: {DEFAULT_CSV}")
        sys.exit(1)

    fair_comparison = "--fair" in flags
    generate_table = "--table" in flags

    print(f"\n{'#' * 70}")
    print(f"# MV2H Results Analysis")
    print(f"{'#' * 70}")

    # Load and analyze MT3 results
    results = load_results_csv(csv_path)
    print(f"Loaded {len(results)} chunk results from {csv_path}")

    # Run basic analyses
    summary = compute_overall_summary(results)
    print_overall_summary(summary)

    position_analysis = analyze_by_chunk_position(results)
    print_position_analysis(position_analysis)

    # =========================================================================
    # Logistic Transition Model Analysis (the key analysis!)
    # =========================================================================
    print(f"\n{'#' * 70}")
    print("# Logistic Transition Model: Phase Drift + Mode Locking")
    print(f"{'#' * 70}")

    transition_results = []

    # MT3
    if Path(DEFAULT_CSV).exists():
        try:
            r = run_transition_model(DEFAULT_CSV, "MT3 + MuseScore")
            transition_results.append(r)
            print(f"Analyzed: MT3 + MuseScore (n={r['n_obs']})")
        except Exception as e:
            print(f"Failed to analyze MT3: {e}")

    # Transkun + Beyer
    if Path(TRANSKUN_BEYER_CSV).exists():
        try:
            r = run_transition_model(TRANSKUN_BEYER_CSV, "Transkun + Beyer")
            transition_results.append(r)
            print(f"Analyzed: Transkun + Beyer (n={r['n_obs']})")
        except Exception as e:
            print(f"Failed to analyze Transkun + Beyer: {e}")

    # Zeng
    if Path(ZENG_CSV).exists():
        try:
            r = run_transition_model(ZENG_CSV, "Zeng (E2E)")
            transition_results.append(r)
            print(f"Analyzed: Zeng (n={r['n_obs']})")
        except Exception as e:
            print(f"Failed to analyze Zeng: {e}")

    if transition_results:
        print_transition_model_results(transition_results)

        # Generate success rate by position plot (main figure)
        csv_paths = {
            "MT3 + MuseScore": DEFAULT_CSV,
            "Transkun + Beyer": TRANSKUN_BEYER_CSV,
            "Zeng (E2E)": ZENG_CSV,
        }

        # Use Zeng's task_ids for fair comparison if --fair flag is set
        filter_task_ids = None

        if fair_comparison and Path(ZENG_CSV).exists():
            import pandas as pd
            zeng_df = pd.read_csv(ZENG_CSV)
            filter_task_ids = set(zeng_df["task_id"].tolist())
            print(f"\n[Fair Comparison Mode] Filtering all systems to Zeng's {len(filter_task_ids):,} task_ids")

        # Generate table data if requested
        if generate_table and filter_task_ids is not None:
            system_types = {
                "MT3 + MuseScore": "Pipeline",
                "Transkun + Beyer": "Pipeline",
                "Zeng (E2E)": "E2E",
            }
            print_fair_comparison_table(csv_paths, filter_task_ids, system_types)

        plot_success_rate_by_position(
            csv_paths,
            output_path=DEFAULT_SUCCESS_RATE_PLOT,
            filter_to_task_ids=filter_task_ids,
        )


if __name__ == "__main__":
    main()
