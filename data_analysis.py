"""
Data Analysis for PMD Mixing Trees - Flushing and Loading Reduction

This script analyzes mixing trees to compare:
1. ORIGINAL (Unmodified) - baseline tree
2. MODIFIED (Scaled + Merged) - optimized using Scaling & Merging algorithms

Based on research paper:
"Reducing the Number of Flushing by Scaling Mixers on PMDs"

The analysis:
1. Generates or loads random trees (heights 2-7, 300 each for heights 3-5)
2. Transforms each tree using Scaling & Merging algorithms
3. Runs PMD grid placement simulation on both versions
4. Calculates flush and loading counts
5. Generates comparison statistics and paper-style outputs

Author: PMD Analysis Script
"""

import sys
import os
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from importlib import reload
import csv

# Add Codes folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Codes"))

# Import NTM modules
from NTM.tree import node
from NTM.scaling import transform_tree, get_tree_total_reagent_volume
from NTM.pmd_simulation import (
    simulate_placement, 
    analyze_tree, 
    count_loading_operations,
    count_volume_based_flushing,
    DEFAULT_GRID_SIZE
)

# Import tree data with reload
import trees_data.trees_data as trees_data_module
reload(trees_data_module)
from trees_data.trees_data import get_all_trees, TREES
from trees_data.generate_trees import generate_trees_by_height_counts

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 10  # PMD grid size (configurable, default 10)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_results")
FLUSH_TIME_SECONDS = 15.2  # 2 x 7.6s loading time per flush (paper)

# Dataset generation (paper replication defaults)
USE_GENERATED_TREES = True
COUNTS_BY_HEIGHT = {
    2: 100,
    3: 300,
    4: 300,
    5: 300,
    6: 100,
    7: 100,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tree_height(tree):
    """Calculate tree height (max depth from root to leaf)."""
    if not tree.children:
        return 0
    return 1 + max(get_tree_height(child) for child in tree.children)


def count_mixers(tree):
    """Count number of mixer nodes in tree."""
    count = 0
    def traverse(n):
        nonlocal count
        if n.children:
            count += 1
            for child in n.children:
                traverse(child)
    traverse(tree)
    return count


def count_reagents(tree):
    """Count number of reagent (leaf) nodes in tree."""
    count = 0
    def traverse(n):
        nonlocal count
        if not n.children:
            count += 1
        else:
            for child in n.children:
                traverse(child)
    traverse(tree)
    return count


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Created output directory: {OUTPUT_DIR}")


def load_trees_dataset():
    """Load or generate trees for analysis based on configuration."""
    if USE_GENERATED_TREES:
        print("Generating trees on the fly (paper replication dataset)...")
        trees_by_height = generate_trees_by_height_counts(COUNTS_BY_HEIGHT)
        all_trees = []
        for h in sorted(trees_by_height.keys()):
            all_trees.extend(trees_by_height[h])
        return all_trees

    return get_all_trees()


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_single_tree(tree, tree_index, grid_size=GRID_SIZE):
    """
    Analyze a single tree: original vs. modified (scaled+merged).
    
    The paper's key metric is PLACEMENT FLUSHES - conflicts during grid placement
    that require flushing the grid to continue. Scaling reduces these by converting
    Group A (high-conflict) mixers to Group B (low-conflict) mixers.
    
    Args:
        tree_index: Index of tree in TREES list
        grid_size: PMD grid size for simulation
    
    Returns:
        dict: Analysis results for this tree
    """
    # Get original tree
    original_tree = tree
    
    # Transform to get modified tree
    modified_tree, transform_stats = transform_tree(original_tree)
    
    # Run placement simulations - THIS IS THE KEY METRIC from the paper
    original_sim = simulate_placement(original_tree, grid_size)
    modified_sim = simulate_placement(modified_tree, grid_size)
    
    # Placement flushes are what the paper optimizes
    original_placement_flushes = original_sim['placement_flushes']
    modified_placement_flushes = modified_sim['placement_flushes']
    
    # Calculate placement flush reduction
    placement_reduction = original_placement_flushes - modified_placement_flushes
    placement_reduction_pct = (placement_reduction / original_placement_flushes * 100) if original_placement_flushes > 0 else (100.0 if modified_placement_flushes == 0 else 0.0)
    
    # Volume-based metrics (for reference)
    original_volume_waste = count_volume_based_flushing(original_tree)
    modified_volume_waste = count_volume_based_flushing(modified_tree)
    
    # Loading metrics
    original_loading = count_loading_operations(original_tree)
    modified_loading = count_loading_operations(modified_tree)
    
    return {
        'tree_index': tree_index,
        'height': get_tree_height(original_tree),
        'num_mixers': count_mixers(original_tree),
        'num_reagents': count_reagents(original_tree),
        
        # PLACEMENT FLUSHES - the paper's key metric
        'original_flushing': original_placement_flushes,  # Renamed for clarity in stats
        'modified_flushing': modified_placement_flushes,
        'flushing_reduction': placement_reduction,
        'flushing_reduction_pct': placement_reduction_pct,
        
        # Detailed placement metrics
        'original_placement_flushes': original_placement_flushes,
        'modified_placement_flushes': modified_placement_flushes,
        'placement_reduction': placement_reduction,
        'placement_reduction_pct': placement_reduction_pct,
        
        # Volume waste (increases with scaling, but placement conflicts decrease)
        'original_volume_waste': original_volume_waste,
        'modified_volume_waste': modified_volume_waste,
        
        # Loading metrics
        'original_loading': original_loading,
        'modified_loading': modified_loading,
        'loading_change': modified_loading - original_loading,
        'loading_change_pct': (modified_loading - original_loading) / original_loading * 100 if original_loading > 0 else 0,

        # Time metrics (seconds)
        'original_time': original_placement_flushes * FLUSH_TIME_SECONDS,
        'modified_time': modified_placement_flushes * FLUSH_TIME_SECONDS,
        
        # Total flushes (for reference)
        'original_total_flushes': original_sim['total_flushes'],
        'modified_total_flushes': modified_sim['total_flushes'],
        
        # Transformation stats
        'nodes_scaled': transform_stats['nodes_scaled'],
        'nodes_merged': transform_stats['nodes_merged'],
    }


def run_full_analysis(grid_size=GRID_SIZE, progress_interval=50):
    """
    Run analysis on all trees in the dataset.
    
    Args:
        grid_size: PMD grid size for simulation
        progress_interval: Print progress every N trees
    
    Returns:
        list: List of results for each tree
    """
    all_trees = load_trees_dataset()
    num_trees = len(all_trees)
    results = []
    
    print(f"\n{'='*60}")
    print(f"PMD MIXING TREE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total trees: {num_trees}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"{'='*60}\n")
    
    for i, tree in enumerate(all_trees):
        result = analyze_single_tree(tree, i, grid_size)
        results.append(result)
        
        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{num_trees} trees...")
    
    print(f"\n✓ Analysis complete for {num_trees} trees")
    
    return results


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def compute_statistics_by_height(results):
    """
    Compute statistics grouped by tree height.
    
    Args:
        results: List of analysis results
    
    Returns:
        dict: Statistics by height
    """
    # Group by height
    by_height = defaultdict(list)
    for r in results:
        by_height[r['height']].append(r)
    
    stats_by_height = {}
    
    for height in sorted(by_height.keys()):
        trees = by_height[height]
        n = len(trees)
        
        # Flushing statistics
        orig_flush = [t['original_flushing'] for t in trees]
        mod_flush = [t['modified_flushing'] for t in trees]
        flush_reduction = [t['flushing_reduction'] for t in trees]
        flush_reduction_pct = [t['flushing_reduction_pct'] for t in trees]
        
        # Loading statistics
        orig_load = [t['original_loading'] for t in trees]
        mod_load = [t['modified_loading'] for t in trees]
        load_change = [t['loading_change'] for t in trees]

        # Time statistics (Figure 11)
        orig_time = [t['original_flushing'] * FLUSH_TIME_SECONDS for t in trees]
        mod_time = [t['modified_flushing'] * FLUSH_TIME_SECONDS for t in trees]
        
        # Placement flush statistics
        orig_place_flush = [t['original_placement_flushes'] for t in trees]
        mod_place_flush = [t['modified_placement_flushes'] for t in trees]
        
        stats_by_height[height] = {
            'count': n,
            
            # Flushing
            'original_flushing_mean': np.mean(orig_flush),
            'original_flushing_std': np.std(orig_flush),
            'modified_flushing_mean': np.mean(mod_flush),
            'modified_flushing_std': np.std(mod_flush),
            'flushing_reduction_mean': np.mean(flush_reduction),
            'flushing_reduction_std': np.std(flush_reduction),
            'flushing_reduction_pct_mean': np.mean(flush_reduction_pct),
            'flushing_reduction_pct_std': np.std(flush_reduction_pct),
            
            # Loading
            'original_loading_mean': np.mean(orig_load),
            'modified_loading_mean': np.mean(mod_load),
            'loading_change_mean': np.mean(load_change),

            # Time (seconds)
            'original_time_mean': np.mean(orig_time),
            'modified_time_mean': np.mean(mod_time),
            
            # Placement flushes
            'original_placement_flush_mean': np.mean(orig_place_flush),
            'modified_placement_flush_mean': np.mean(mod_place_flush),
        }
        
        # Perform paired t-test for flushing reduction significance
        if n > 1:
            t_stat, p_value = stats.ttest_rel(orig_flush, mod_flush)
            stats_by_height[height]['ttest_statistic'] = t_stat
            stats_by_height[height]['ttest_pvalue'] = p_value
    
    return stats_by_height


def compute_overall_statistics(results):
    """
    Compute overall statistics across all trees.
    
    Args:
        results: List of analysis results
    
    Returns:
        dict: Overall statistics
    """
    n = len(results)
    
    orig_flush = [r['original_flushing'] for r in results]
    mod_flush = [r['modified_flushing'] for r in results]
    flush_reduction = [r['flushing_reduction'] for r in results]
    flush_reduction_pct = [r['flushing_reduction_pct'] for r in results]
    
    orig_load = [r['original_loading'] for r in results]
    mod_load = [r['modified_loading'] for r in results]

    orig_time = [r['original_flushing'] * FLUSH_TIME_SECONDS for r in results]
    mod_time = [r['modified_flushing'] * FLUSH_TIME_SECONDS for r in results]
    
    overall_stats = {
        'total_trees': n,
        
        # Flushing
        'original_flushing_total': sum(orig_flush),
        'modified_flushing_total': sum(mod_flush),
        'original_flushing_mean': np.mean(orig_flush),
        'modified_flushing_mean': np.mean(mod_flush),
        'flushing_reduction_mean': np.mean(flush_reduction),
        'flushing_reduction_std': np.std(flush_reduction),
        'flushing_reduction_pct_mean': np.mean(flush_reduction_pct),
        
        # Loading
        'original_loading_total': sum(orig_load),
        'modified_loading_total': sum(mod_load),
        'original_loading_mean': np.mean(orig_load),
        'modified_loading_mean': np.mean(mod_load),
        'loading_increase_pct': (sum(mod_load) - sum(orig_load)) / sum(orig_load) * 100 if sum(orig_load) > 0 else 0,

        # Time (seconds)
        'original_time_mean': np.mean(orig_time),
        'modified_time_mean': np.mean(mod_time),
        
        # Transformation stats
        'total_nodes_scaled': sum(r['nodes_scaled'] for r in results),
        'total_nodes_merged': sum(r['nodes_merged'] for r in results),
    }
    
    # Overall t-test
    if n > 1:
        t_stat, p_value = stats.ttest_rel(orig_flush, mod_flush)
        overall_stats['ttest_statistic'] = t_stat
        overall_stats['ttest_pvalue'] = p_value
    
    return overall_stats


def create_table2_dataframe(stats_by_height):
    """
    Create a DataFrame resembling Table 2 from the paper.

    Columns:
    - Height
    - Avg Overlaps (proxy for flushes)
    - Avg Reduction %
    - Avg Loading Count (Unscaled)
    - Avg Loading Count (Scaled)
    """
    rows = []
    for height in sorted(stats_by_height.keys()):
        stats = stats_by_height[height]
        rows.append({
            'Height': height,
            'Avg Overlaps': stats['original_flushing_mean'],
            'Avg Reduction %': stats['flushing_reduction_pct_mean'],
            'Avg Loading (Unscaled)': stats['original_loading_mean'],
            'Avg Loading (Scaled)': stats['modified_loading_mean'],
        })

    return pd.DataFrame(rows).round(3)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_flushing_comparison(stats_by_height, save_path=None):
    """
    Create bar chart comparing original vs. modified flushing by height.
    """
    heights = sorted(stats_by_height.keys())
    orig_means = [stats_by_height[h]['original_flushing_mean'] for h in heights]
    mod_means = [stats_by_height[h]['modified_flushing_mean'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, orig_means, width, label='Original (Unmodified)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, mod_means, width, label='Modified (Scaled+Merged)', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Mean Flushing Count', fontsize=12)
    ax.set_title('Flushing Comparison: Original vs. Modified Trees', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_loading_comparison(stats_by_height, save_path=None):
    """
    Create bar chart comparing original vs. modified loading by height.
    """
    heights = sorted(stats_by_height.keys())
    orig_means = [stats_by_height[h]['original_loading_mean'] for h in heights]
    mod_means = [stats_by_height[h]['modified_loading_mean'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, orig_means, width, label='Original (Unmodified)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, mod_means, width, label='Modified (Scaled+Merged)', color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Mean Loading Count', fontsize=12)
    ax.set_title('Loading Comparison: Original vs. Modified Trees', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_flushing_time_by_height(stats_by_height, save_path=None):
    """
    Create bar chart of average flushing time vs height (Figure 11 style).
    Time is computed as T = F * 15.2 seconds.
    """
    heights = sorted(stats_by_height.keys())
    orig_times = [stats_by_height[h]['original_time_mean'] for h in heights]
    mod_times = [stats_by_height[h]['modified_time_mean'] for h in heights]

    x = np.arange(len(heights))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, orig_times, width, label='Original (Unmodified)', color='#e67e22', alpha=0.85)
    ax.bar(x + width/2, mod_times, width, label='Modified (Scaled+Merged)', color='#1abc9c', alpha=0.85)

    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Flushing Time (seconds)', fontsize=12)
    ax.set_title('Average Flushing Time by Height (T = F × 15.2s)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def plot_loading_boxplot(results, save_path=None):
    """
    Create box plot comparing loading counts (Unscaled vs Scaled).
    """
    orig_load = [r['original_loading'] for r in results]
    mod_load = [r['modified_loading'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([orig_load, mod_load], labels=['Unscaled', 'Scaled'], patch_artist=True,
               boxprops=dict(facecolor='#bdc3c7', color='#7f8c8d'),
               medianprops=dict(color='#2c3e50'))

    ax.set_ylabel('Loading Count (Mixer Nodes)', fontsize=12)
    ax.set_title('Loading Count Reduction via Merging', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def plot_reduction_percentages(stats_by_height, save_path=None):
    """
    Create line plot showing flushing reduction percentage by height.
    """
    heights = sorted(stats_by_height.keys())
    reduction_pcts = [stats_by_height[h]['flushing_reduction_pct_mean'] for h in heights]
    reduction_stds = [stats_by_height[h]['flushing_reduction_pct_std'] for h in heights]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(heights, reduction_pcts, yerr=reduction_stds, 
                marker='o', markersize=10, capsize=5, capthick=2,
                linewidth=2, color='#2ecc71', ecolor='#27ae60')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Flushing Reduction (%)', fontsize=12)
    ax.set_title('Flushing Reduction Percentage by Tree Height', fontsize=14, fontweight='bold')
    ax.set_xticks(heights)
    ax.grid(alpha=0.3)
    
    # Add data labels
    for h, pct in zip(heights, reduction_pcts):
        ax.annotate(f'{pct:.1f}%', xy=(h, pct), xytext=(5, 10),
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_scatter_comparison(results, save_path=None):
    """
    Create scatter plot comparing original vs modified flushing for each tree.
    """
    orig_flush = [r['original_flushing'] for r in results]
    mod_flush = [r['modified_flushing'] for r in results]
    heights = [r['height'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by height
    scatter = ax.scatter(orig_flush, mod_flush, c=heights, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (y=x)
    max_val = max(max(orig_flush), max(mod_flush))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No Change (y=x)')
    
    ax.set_xlabel('Original Flushing Count', fontsize=12)
    ax.set_ylabel('Modified Flushing Count', fontsize=12)
    ax.set_title('Original vs. Modified Flushing (Each Point = One Tree)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Tree Height', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_summary_dashboard(results, stats_by_height, overall_stats, save_path=None):
    """
    Create a comprehensive dashboard with multiple plots.
    """
    fig = plt.figure(figsize=(16, 12))
    
    heights = sorted(stats_by_height.keys())
    
    # 1. Flushing comparison bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    orig_means = [stats_by_height[h]['original_flushing_mean'] for h in heights]
    mod_means = [stats_by_height[h]['modified_flushing_mean'] for h in heights]
    x = np.arange(len(heights))
    width = 0.35
    ax1.bar(x - width/2, orig_means, width, label='Original', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, mod_means, width, label='Modified', color='#27ae60', alpha=0.8)
    ax1.set_xlabel('Tree Height')
    ax1.set_ylabel('Mean Flushing Count')
    ax1.set_title('Flushing Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(heights)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Loading comparison
    ax2 = fig.add_subplot(2, 2, 2)
    orig_load = [stats_by_height[h]['original_loading_mean'] for h in heights]
    mod_load = [stats_by_height[h]['modified_loading_mean'] for h in heights]
    ax2.bar(x - width/2, orig_load, width, label='Original', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, mod_load, width, label='Modified', color='#9b59b6', alpha=0.8)
    ax2.set_xlabel('Tree Height')
    ax2.set_ylabel('Mean Loading Count')
    ax2.set_title('Loading Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(heights)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Reduction percentage line plot
    ax3 = fig.add_subplot(2, 2, 3)
    reduction_pcts = [stats_by_height[h]['flushing_reduction_pct_mean'] for h in heights]
    ax3.plot(heights, reduction_pcts, marker='o', markersize=10, linewidth=2, color='#2ecc71')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Tree Height')
    ax3.set_ylabel('Flushing Reduction (%)')
    ax3.set_title('Flushing Reduction by Height')
    ax3.set_xticks(heights)
    ax3.grid(alpha=0.3)
    for h, pct in zip(heights, reduction_pcts):
        ax3.annotate(f'{pct:.1f}%', xy=(h, pct), xytext=(3, 8), textcoords='offset points', fontsize=9)
    
    # 4. Scatter plot
    ax4 = fig.add_subplot(2, 2, 4)
    orig_flush = [r['original_flushing'] for r in results]
    mod_flush = [r['modified_flushing'] for r in results]
    tree_heights = [r['height'] for r in results]
    scatter = ax4.scatter(orig_flush, mod_flush, c=tree_heights, cmap='viridis', alpha=0.6, s=30)
    max_val = max(max(orig_flush), max(mod_flush))
    ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    ax4.set_xlabel('Original Flushing')
    ax4.set_ylabel('Modified Flushing')
    ax4.set_title('Flushing: Original vs. Modified')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Height')
    
    # Overall title
    fig.suptitle(
        f"PMD Analysis Summary: {overall_stats['total_trees']} Trees | "
        f"Mean Flushing Reduction: {overall_stats['flushing_reduction_pct_mean']:.1f}%",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_results_csv(results, filepath):
    """Export per-tree results to CSV."""
    fieldnames = [
        'tree_index', 'height', 'num_mixers', 'num_reagents',
        'original_flushing', 'modified_flushing', 'flushing_reduction', 'flushing_reduction_pct',
        'original_loading', 'modified_loading', 'loading_change', 'loading_change_pct',
        'original_time', 'modified_time',
        'original_placement_flushes', 'modified_placement_flushes',
        'nodes_scaled', 'nodes_merged'
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Saved: {filepath}")


def export_summary_csv(stats_by_height, overall_stats, filepath):
    """Export summary statistics to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(['PMD Mixing Tree Analysis Summary'])
        writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        
        # Overall statistics
        writer.writerow(['OVERALL STATISTICS'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Trees', overall_stats['total_trees']])
        writer.writerow(['Total Original Flushing', overall_stats['original_flushing_total']])
        writer.writerow(['Total Modified Flushing', overall_stats['modified_flushing_total']])
        writer.writerow(['Mean Flushing Reduction (%)', f"{overall_stats['flushing_reduction_pct_mean']:.2f}"])
        writer.writerow(['Total Original Loading', overall_stats['original_loading_total']])
        writer.writerow(['Total Modified Loading', overall_stats['modified_loading_total']])
        writer.writerow(['Loading Increase (%)', f"{overall_stats['loading_increase_pct']:.2f}"])
        writer.writerow(['Mean Original Flushing Time (s)', f"{overall_stats['original_time_mean']:.2f}"])
        writer.writerow(['Mean Modified Flushing Time (s)', f"{overall_stats['modified_time_mean']:.2f}"])
        writer.writerow(['Nodes Scaled', overall_stats['total_nodes_scaled']])
        writer.writerow(['Nodes Merged', overall_stats['total_nodes_merged']])
        if 'ttest_pvalue' in overall_stats:
            writer.writerow(['T-Test P-Value', f"{overall_stats['ttest_pvalue']:.6f}"])
        writer.writerow([])
        
        # Statistics by height
        writer.writerow(['STATISTICS BY HEIGHT'])
        writer.writerow([
            'Height', 'Count',
            'Orig Flush Mean', 'Orig Flush Std',
            'Mod Flush Mean', 'Mod Flush Std',
            'Reduction Mean', 'Reduction %',
            'Orig Load Mean', 'Mod Load Mean',
            'Orig Time Mean (s)', 'Mod Time Mean (s)'
        ])
        
        for h in sorted(stats_by_height.keys()):
            s = stats_by_height[h]
            writer.writerow([
                h, s['count'],
                f"{s['original_flushing_mean']:.2f}", f"{s['original_flushing_std']:.2f}",
                f"{s['modified_flushing_mean']:.2f}", f"{s['modified_flushing_std']:.2f}",
                f"{s['flushing_reduction_mean']:.2f}", f"{s['flushing_reduction_pct_mean']:.1f}%",
                f"{s['original_loading_mean']:.2f}", f"{s['modified_loading_mean']:.2f}",
                f"{s['original_time_mean']:.2f}", f"{s['modified_time_mean']:.2f}"
            ])
    
    print(f"✓ Saved: {filepath}")


def export_html_report(results, stats_by_height, overall_stats, filepath):
    """Export comprehensive HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PMD Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f1f1f1; }}
        .highlight {{ background: #e8f8f5; font-weight: bold; }}
        .metric-card {{ display: inline-block; background: #ecf0f1; padding: 20px; margin: 10px; border-radius: 8px; min-width: 180px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .reduction {{ color: #27ae60; }}
        .increase {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 PMD Mixing Tree Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analysis comparing Original (Unmodified) vs. Modified (Scaled + Merged) mixing trees.</p>
        
        <h2>📊 Overall Summary</h2>
        <div class="metric-card">
            <div class="metric-value">{overall_stats['total_trees']}</div>
            <div class="metric-label">Total Trees Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value reduction">{overall_stats['flushing_reduction_pct_mean']:.1f}%</div>
            <div class="metric-label">Mean Flushing Reduction</div>
        </div>
        <div class="metric-card">
            <div class="metric-value increase">{overall_stats['loading_increase_pct']:.1f}%</div>
            <div class="metric-label">Loading Increase</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overall_stats['total_nodes_scaled']}</div>
            <div class="metric-label">Nodes Scaled</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{overall_stats['total_nodes_merged']}</div>
            <div class="metric-label">Nodes Merged</div>
        </div>
        
        <h2>📈 Flushing Statistics by Height</h2>
        <table>
            <tr>
                <th>Height</th>
                <th>Count</th>
                <th>Original Flushing (Mean)</th>
                <th>Modified Flushing (Mean)</th>
                <th>Reduction</th>
                <th>Reduction %</th>
            </tr>"""
    
    for h in sorted(stats_by_height.keys()):
        s = stats_by_height[h]
        html += f"""
            <tr>
                <td>{h}</td>
                <td>{s['count']}</td>
                <td>{s['original_flushing_mean']:.2f} ± {s['original_flushing_std']:.2f}</td>
                <td>{s['modified_flushing_mean']:.2f} ± {s['modified_flushing_std']:.2f}</td>
                <td class="reduction">{s['flushing_reduction_mean']:.2f}</td>
                <td class="reduction highlight">{s['flushing_reduction_pct_mean']:.1f}%</td>
            </tr>"""
    
    html += """
        </table>
        
        <h2>📦 Loading Statistics by Height</h2>
        <table>
            <tr>
                <th>Height</th>
                <th>Original Loading (Mean)</th>
                <th>Modified Loading (Mean)</th>
                <th>Change</th>
            </tr>"""
    
    for h in sorted(stats_by_height.keys()):
        s = stats_by_height[h]
        change = s['loading_change_mean']
        change_class = 'increase' if change > 0 else 'reduction'
        html += f"""
            <tr>
                <td>{h}</td>
                <td>{s['original_loading_mean']:.2f}</td>
                <td>{s['modified_loading_mean']:.2f}</td>
                <td class="{change_class}">{change:+.2f}</td>
            </tr>"""
    
    html += f"""
        </table>
        
        <h2>📉 Statistical Significance</h2>
        <p>Paired T-Test for flushing reduction:</p>
        <ul>
            <li><strong>T-Statistic:</strong> {overall_stats.get('ttest_statistic', 'N/A'):.4f}</li>
            <li><strong>P-Value:</strong> {overall_stats.get('ttest_pvalue', 'N/A'):.6f}</li>
            <li><strong>Significant (p < 0.05)?:</strong> {'Yes ✓' if overall_stats.get('ttest_pvalue', 1) < 0.05 else 'No'}</li>
        </ul>
        
        <h2>ℹ️ Methodology</h2>
        <p>The analysis follows the methodology from "Reducing the Number of Flushing by Scaling Mixers on PMDs":</p>
        <ol>
            <li><strong>Scaling Algorithm:</strong> Group A mixers (output ≥ 3, odd) are scaled by 2x to convert to Group B</li>
            <li><strong>Constraint Propagation:</strong> Children are scaled if parent demand exceeds their capacity</li>
            <li><strong>Merging Algorithm:</strong> Redundant mixers (output = total volume) are merged into parents</li>
            <li><strong>PMD Simulation:</strong> Placement simulation on {GRID_SIZE}×{GRID_SIZE} grid counts flush operations</li>
        </ol>
        
        <hr>
        <p style="color: #7f8c8d; font-size: 12px;">
            Generated by PMD Analysis Script | Grid Size: {GRID_SIZE}×{GRID_SIZE}
        </p>
    </div>
</body>
</html>"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    print(f"✓ Saved: {filepath}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  PMD MIXING TREE ANALYSIS")
    print("  Comparing Original vs. Modified (Scaled + Merged) Trees")
    print("="*70)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Run analysis
    results = run_full_analysis(grid_size=GRID_SIZE)
    
    # Compute statistics
    print("\n📊 Computing statistics...")
    stats_by_height = compute_statistics_by_height(results)
    overall_stats = compute_overall_statistics(results)
    
    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY RESULTS")
    print("="*70)
    print(f"\nTotal Trees Analyzed: {overall_stats['total_trees']}")
    print(f"\nFLUSHING:")
    print(f"  Original Total: {overall_stats['original_flushing_total']}")
    print(f"  Modified Total: {overall_stats['modified_flushing_total']}")
    print(f"  Mean Reduction: {overall_stats['flushing_reduction_mean']:.2f}")
    print(f"  Mean Reduction %: {overall_stats['flushing_reduction_pct_mean']:.1f}%")
    
    print(f"\nLOADING:")
    print(f"  Original Total: {overall_stats['original_loading_total']}")
    print(f"  Modified Total: {overall_stats['modified_loading_total']}")
    print(f"  Loading Increase: {overall_stats['loading_increase_pct']:.1f}%")
    
    print(f"\nTRANSFORMATION:")
    print(f"  Nodes Scaled: {overall_stats['total_nodes_scaled']}")
    print(f"  Nodes Merged: {overall_stats['total_nodes_merged']}")
    
    if 'ttest_pvalue' in overall_stats:
        sig = "YES ✓" if overall_stats['ttest_pvalue'] < 0.05 else "NO"
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"  T-Test P-Value: {overall_stats['ttest_pvalue']:.6f}")
        print(f"  Significant (p < 0.05): {sig}")
    
    # By height summary
    print("\n" + "-"*70)
    print("BY HEIGHT:")
    print("-"*70)
    print(f"{'Height':<8} {'Count':<8} {'Orig Flush':<12} {'Mod Flush':<12} {'Reduction %':<12}")
    for h in sorted(stats_by_height.keys()):
        s = stats_by_height[h]
        print(f"{h:<8} {s['count']:<8} {s['original_flushing_mean']:<12.2f} {s['modified_flushing_mean']:<12.2f} {s['flushing_reduction_pct_mean']:<12.1f}%")
    
    # Generate plots
    print("\n📈 Generating plots...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_flushing_comparison(stats_by_height, os.path.join(OUTPUT_DIR, f'flushing_comparison_{timestamp}.png'))
    plot_loading_comparison(stats_by_height, os.path.join(OUTPUT_DIR, f'loading_comparison_{timestamp}.png'))
    plot_flushing_time_by_height(stats_by_height, os.path.join(OUTPUT_DIR, f'flushing_time_{timestamp}.png'))
    plot_loading_boxplot(results, os.path.join(OUTPUT_DIR, f'loading_boxplot_{timestamp}.png'))
    plot_reduction_percentages(stats_by_height, os.path.join(OUTPUT_DIR, f'reduction_percentages_{timestamp}.png'))
    plot_scatter_comparison(results, os.path.join(OUTPUT_DIR, f'scatter_comparison_{timestamp}.png'))
    plot_summary_dashboard(results, stats_by_height, overall_stats, os.path.join(OUTPUT_DIR, f'summary_dashboard_{timestamp}.png'))
    
    # Export data
    print("\n📁 Exporting data...")
    export_results_csv(results, os.path.join(OUTPUT_DIR, f'per_tree_results_{timestamp}.csv'))
    export_summary_csv(stats_by_height, overall_stats, os.path.join(OUTPUT_DIR, f'summary_statistics_{timestamp}.csv'))
    export_html_report(results, stats_by_height, overall_stats, os.path.join(OUTPUT_DIR, f'analysis_report_{timestamp}.html'))

    # Table 2 replication
    table2 = create_table2_dataframe(stats_by_height)
    table2_path = os.path.join(OUTPUT_DIR, f'table2_replication_{timestamp}.csv')
    table2.to_csv(table2_path, index=False)
    print(f"✓ Saved: {table2_path}")
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("="*70)
    
    return results, stats_by_height, overall_stats


if __name__ == "__main__":
    results, stats_by_height, overall_stats = main()
