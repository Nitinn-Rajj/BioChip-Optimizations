"""
Data Analysis for Loading & Flushing Operations on Mixing Trees

This script analyzes mixing trees to compare:
1. WITHOUT Scaling - baseline (W.O. Scaling)
2. WITH Scaling - optimized using ECN-based heuristic (W. Scaling)

Based on papers:
- "Reducing the Number of Flushing by Scaling Mixers on PMDs" (Miku_v1)
- "Transport-Free Module Binding for Sample Preparation" (DATE 2020)

Author: Data Analysis Script
"""

import sys
import os
from copy import deepcopy
from collections import defaultdict
import datetime
import math
import random
import csv

# Add Codes folder to path for correct NTM module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Codes"))

from NTM import *
from NTM.tree import node, leftistTree, viewTree
from NTM.hda import hda
from NTM.ntm import ntm
from NTM.ntm_helper import getTimeCount, getCellCount, getValveCount

# For visualization
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ============================================================
# TREE ANALYSIS HELPER FUNCTIONS
# ============================================================

def get_tree_height(tree):
    """
    Calculate the height of a mixing tree.
    Height = maximum depth from root to any leaf node.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Height of the tree
    """
    if not tree.children:
        return 0
    return 1 + max(get_tree_height(child) for child in tree.children)


def count_overlaps(tree):
    """
    Count the number of cell overlaps (reuse) in the mixing tree.
    
    An overlap occurs when a mixer's input volume exceeds its output volume,
    meaning cells need to be reused/flushed. In the context of the Miku paper,
    overlaps represent the cells that need to be cleaned between operations.
    
    For each mixer: overlap = sum(children_volumes) - mixer_output_volume
    This is essentially the same as flushing cells.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Total number of overlaps (cells requiring reuse)
    """
    total_overlaps = 0
    
    def traverse(n):
        nonlocal total_overlaps
        if n.children:  # Mixer node
            input_volume = sum(child.volume for child in n.children)
            output_volume = n.volume
            overlap = input_volume - output_volume
            if overlap > 0:
                total_overlaps += overlap
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return total_overlaps


# ============================================================
# LOADING & FLUSHING CALCULATION FUNCTIONS (Tree-Based)
# For Variable-Sized Mixers as per Miku Paper
# ============================================================

def get_mixer_size(mixer_node):
    """
    Get the size (number of cells) of a mixer.
    
    For variable-sized mixers, the size is the sum of all children's volumes.
    Mixer size = sum of (volume of each child)
    
    Args:
        mixer_node: A node representing a mixer
    
    Returns:
        int: Number of cells in this mixer
    """
    if not mixer_node.children:
        # Leaf node (reagent) - size is its volume
        return mixer_node.volume
    
    # Mixer size = sum of children volumes (what they contribute)
    return sum(child.volume for child in mixer_node.children)


def count_loading_from_tree(tree):
    """
    Count loading operations directly from tree structure.
    
    Loading = placing reagents in cells.
    Each leaf node (reagent) contributes its volume as loading operations.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Total loading operations (total reagent cells)
    """
    total_loading = 0
    
    def traverse(n):
        nonlocal total_loading
        if not n.children:
            # Leaf node - reagent loading
            total_loading += n.volume
        else:
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return total_loading


def count_flushing_from_tree(tree):
    """
    Count flushing operations directly from tree structure.
    
    Flushing = cells that need to be washed out.
    For each mixer: flushing = mixer_size - volume_passed_to_parent
    
    Based on Miku paper formula:
    Flushing(Mn) = CellCount(Mn) - VolumeToParent(Mn)
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Total flushing operations (cells to wash)
    """
    total_flushing = 0
    
    def traverse(n):
        nonlocal total_flushing
        if n.children:  # Mixer node
            # Mixer size = sum of children volumes
            mixer_size = sum(child.volume for child in n.children)
            # Volume passed to parent = n.volume
            volume_to_parent = n.volume
            # Flushing = what doesn't go to parent
            flushing = mixer_size - volume_to_parent
            if flushing > 0:
                total_flushing += flushing
            
            # Recurse to children
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return total_flushing


def count_flushing_events_from_tree(tree):
    """
    Count number of mixers that require flushing (events, not cells).
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Number of flushing events
    """
    flushing_events = 0
    
    def traverse(n):
        nonlocal flushing_events
        if n.children:  # Mixer node
            mixer_size = sum(child.volume for child in n.children)
            volume_to_parent = n.volume
            if mixer_size > volume_to_parent:
                flushing_events += 1
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return flushing_events


def count_mixers(tree):
    """
    Count total number of mixer nodes.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Number of mixer nodes
    """
    mixer_count = 0
    
    def traverse(n):
        nonlocal mixer_count
        if n.children:
            mixer_count += 1
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return mixer_count


def count_reagents(tree):
    """
    Count total number of reagent (leaf) nodes.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Number of reagent nodes
    """
    reagent_count = 0
    
    def traverse(n):
        nonlocal reagent_count
        if not n.children:
            reagent_count += 1
        else:
            for child in n.children:
                traverse(child)
    
    traverse(tree)
    return reagent_count


def calculate_ecn(tree):
    """
    Calculate ECN (Estimated Cell Use Number) for the tree.
    
    From Miku paper:
    ECN(Mn) = ReagentVol(Mn) + sum(ECN(child_mixers))
    
    For a leaf (reagent): ECN = volume
    For a mixer: ECN = sum of children ECNs
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: ECN value
    """
    def calc_ecn(n):
        if not n.children:
            return n.volume
        return sum(calc_ecn(child) for child in n.children)
    
    return calc_ecn(tree)


def get_total_cells_used(tree):
    """
    Calculate total cells used in the mixing process.
    
    This equals the loading count (all cells that get loaded).
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        int: Total cells used
    """
    return count_loading_from_tree(tree)


def get_all_tree_metrics(tree):
    """
    Get all metrics from tree structure.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        dict: All metrics
    """
    loading = count_loading_from_tree(tree)
    flushing = count_flushing_from_tree(tree)
    
    return {
        'loading_count': loading,
        'flushing_cells': flushing,
        'flushing_events': count_flushing_events_from_tree(tree),
        'mixer_count': count_mixers(tree),
        'reagent_count': count_reagents(tree),
        'total_cells_used': loading,
        'ecn': calculate_ecn(tree),
        'total_loading_with_reloads': loading + flushing,  # After flushing, cells must be reloaded
        'efficiency_ratio': tree.volume / loading if loading > 0 else 0,  # Output / Input
    }


# ============================================================
# ECN-BASED OPTIMIZATION FUNCTIONS
# ============================================================

def optimize_volumes_with_ecn(tree):
    """
    Apply ECN-based Scaling optimization to adjust mixer volumes.
    
    The Scaling method from the Miku paper adjusts each mixer's output
    volume to match what the parent mixer actually needs, thus reducing
    the waste (flushing).
    
    Key insight: If a mixer only needs to provide X units to its parent,
    we can "scale" its output to X instead of producing more than needed.
    
    This propagates demand downward: each mixer knows exactly how much
    it needs to provide, and can adjust volumes accordingly.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        node: Optimized tree (new copy with scaled volumes)
    """
    optimized = deepcopy(tree)
    _apply_scaling_optimization(optimized)
    return optimized


def _apply_scaling_optimization(node, required_volume=None):
    """
    Recursively apply scaling optimization to reduce flushing.
    
    The scaling method works by propagating the required volume from
    parent to child, allowing each mixer to produce only what's needed.
    
    Args:
        node: Current node being processed
        required_volume: Volume required by parent (None for root)
    """
    if not node.children:
        # Leaf node (reagent) - cannot scale, but could potentially reduce volume
        if required_volume is not None and required_volume < node.volume:
            node.volume = required_volume
        return
    
    # For mixer nodes:
    # Current input volume = sum of children volumes
    current_input = sum(child.volume for child in node.children)
    
    # If this is root or no constraint, keep current volume
    if required_volume is None:
        required_volume = node.volume
    
    # Scale this mixer's output to match what parent needs
    # But can't output more than we have input
    new_output = min(required_volume, current_input)
    node.volume = new_output
    
    # Now we need to distribute the required input among children
    # The key optimization: we want children to provide exactly what we need
    # Total input needed = new_output (ideal case - no flushing)
    
    # Calculate how much each child should contribute
    # We use a proportional scaling based on original volumes
    if current_input > 0:
        scale_factor = new_output / current_input
        
        # Apply scaling to each child
        for child in node.children:
            # Scale down the volume this child needs to provide
            new_child_vol = max(1, int(round(child.volume * scale_factor)))
            
            # Recursively optimize children
            _apply_scaling_optimization(child, new_child_vol)


def apply_scaling_method(tree):
    """
    Apply the Scaling method from the Miku paper.
    
    The scaling method adjusts volumes at each level to minimize flushing
    by ensuring mixers only produce what their parents need.
    
    This is a bottom-up approach that calculates optimal volumes.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        node: Optimized tree with scaled volumes
    """
    optimized = deepcopy(tree)
    _scale_tree_bottom_up(optimized)
    return optimized


def _scale_tree_bottom_up(node):
    """
    Bottom-up scaling: adjust children volumes first, then current node.
    
    The idea is to make each mixer's output equal to what parent needs,
    thus eliminating flushing at that mixer.
    """
    if not node.children:
        return
    
    # First, recursively optimize all children
    for child in node.children:
        _scale_tree_bottom_up(child)
    
    # Now scale this node's children to match what we need
    # The mixer needs 'node.volume' as output
    # Children together need to provide >= node.volume
    
    current_input = sum(child.volume for child in node.children)
    output_needed = node.volume
    
    if current_input > output_needed:
        # We have excess - need to scale down children
        # This reduces flushing
        scale_factor = output_needed / current_input
        
        # Scale children volumes proportionally
        for child in node.children:
            new_vol = max(1, int(round(child.volume * scale_factor)))
            child.volume = new_vol


def sort_children_by_volume(children):
    """
    Sort children nodes by volume in descending order.
    This is the ECN heuristic - process larger volumes first.
    
    Args:
        children: List of child nodes
    
    Returns:
        list: Sorted children list
    """
    # Separate mixers and reagents
    mixers = [(i, c) for i, c in enumerate(children) if c.children]
    reagents = [(i, c) for i, c in enumerate(children) if not c.children]
    
    # Sort mixers by volume (descending)
    mixers_sorted = sorted(mixers, key=lambda x: x[1].volume, reverse=True)
    
    # Return sorted mixers followed by reagents
    return [c for _, c in mixers_sorted] + [c for _, c in reagents]


def apply_random_volume_assignment(tree):
    """
    Randomly adjust volumes (baseline/random strategy).
    
    This represents the non-optimized case where volumes
    are not strategically chosen.
    
    Args:
        tree: Root node of mixing tree
    
    Returns:
        node: Tree with shuffled children order (affects execution)
    """
    randomized = deepcopy(tree)
    
    def randomize_subtree(n):
        if not n.children:
            return
        
        # Shuffle children order (affects mixing sequence)
        random.shuffle(n.children)
        
        # Recursively randomize children
        for child in n.children:
            randomize_subtree(child)
    
    randomize_subtree(randomized)
    return randomized


# ============================================================
# TREE ANALYSIS FUNCTIONS
# ============================================================

def analyze_tree(tree, apply_ecn_optimization=False):
    """
    Analyze a single tree with or without ECN optimization.
    
    Args:
        tree: A node object representing the mixing tree
        apply_ecn_optimization: Whether to apply ECN optimization
    
    Returns:
        dict: Analysis results
    """
    # Make a deep copy to avoid modifying original
    tree_copy = deepcopy(tree)
    
    # Apply ECN optimization if requested
    if apply_ecn_optimization:
        tree_copy = optimize_volumes_with_ecn(tree_copy)
    
    # Get all metrics from tree structure
    metrics = get_all_tree_metrics(tree_copy)
    
    return metrics


def compare_with_without_ecn(tree, tree_index=None):
    """
    Compare metrics for a tree with and without ECN optimization.
    
    Args:
        tree: A node object representing the mixing tree
        tree_index: Optional index for identification
    
    Returns:
        dict: Comparison results
    """
    try:
        # Analyze with original (variable-sized) configuration
        metrics_original = analyze_tree(tree, apply_ecn_optimization=False)
        
        # Analyze with ECN optimization
        metrics_ecn = analyze_tree(tree, apply_ecn_optimization=True)
        
        # Calculate improvements
        comparison = {
            'tree_index': tree_index,
            'without_ecn': metrics_original,
            'with_ecn': metrics_ecn,
            'improvements': {}
        }
        
        # Calculate improvement for each metric (positive reduction = good)
        for key in metrics_original:
            original_val = metrics_original[key]
            ecn_val = metrics_ecn[key]
            
            if original_val > 0:
                reduction_pct = ((original_val - ecn_val) / original_val) * 100
            else:
                reduction_pct = 0
            
            comparison['improvements'][key] = {
                'absolute_change': ecn_val - original_val,
                'reduction_percent': reduction_pct
            }
        
        return comparison
    
    except Exception as e:
        return {
            'tree_index': tree_index,
            'error': str(e)
        }


# ============================================================
# BATCH ANALYSIS FUNCTIONS
# ============================================================

def analyze_all_trees(trees, verbose=True):
    """
    Analyze all trees and collect statistics.
    
    Args:
        trees: List of node objects (mixing trees)
        verbose: Whether to print progress
    
    Returns:
        list: Analysis results for each tree
    """
    results = []
    
    if verbose:
        print(f"Analyzing {len(trees)} variable-sized mixer trees...")
    
    for i, tree in enumerate(trees):
        comparison = compare_with_without_ecn(tree, tree_index=i)
        results.append(comparison)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(trees)} trees")
    
    return results


def aggregate_results(results):
    """
    Aggregate results from multiple tree analyses.
    
    Args:
        results: List of comparison results from analyze_all_trees
    
    Returns:
        dict: Aggregated statistics
    """
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    if not valid_results:
        return {'error': 'No valid results', 'errors': error_results}
    
    # Initialize aggregation
    metrics_keys = list(valid_results[0]['without_ecn'].keys())
    
    aggregated = {
        'total_trees': len(results),
        'valid_trees': len(valid_results),
        'error_count': len(error_results),
        'without_ecn': {key: [] for key in metrics_keys},
        'with_ecn': {key: [] for key in metrics_keys},
    }
    
    # Collect values
    for r in valid_results:
        for key in metrics_keys:
            aggregated['without_ecn'][key].append(r['without_ecn'][key])
            aggregated['with_ecn'][key].append(r['with_ecn'][key])
    
    # Calculate statistics
    def calc_stats(values):
        if not values:
            return {'mean': 0, 'min': 0, 'max': 0, 'total': 0, 'std': 0}
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 1 else 0
        return {
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'total': sum(values),
            'std': variance ** 0.5
        }
    
    stats = {
        'total_trees': len(results),
        'valid_trees': len(valid_results),
        'error_count': len(error_results),
        'without_ecn_stats': {},
        'with_ecn_stats': {},
        'improvement_stats': {}
    }
    
    for key in metrics_keys:
        stats['without_ecn_stats'][key] = calc_stats(aggregated['without_ecn'][key])
        stats['with_ecn_stats'][key] = calc_stats(aggregated['with_ecn'][key])
        
        # Calculate average improvement
        without_mean = stats['without_ecn_stats'][key]['mean']
        with_mean = stats['with_ecn_stats'][key]['mean']
        
        if without_mean > 0:
            reduction_pct = ((without_mean - with_mean) / without_mean) * 100
        else:
            reduction_pct = 0
        
        stats['improvement_stats'][key] = {
            'mean_absolute_change': with_mean - without_mean,
            'mean_reduction_percent': reduction_pct
        }
    
    return stats


def print_analysis_summary(stats):
    """
    Print a formatted summary of the analysis results.
    
    Args:
        stats: Aggregated statistics from aggregate_results
    """
    # Check for errors
    if 'error' in stats:
        print(f"\n❌ Error: {stats['error']}")
        return
    
    print("\n" + "=" * 80)
    print("DATA ANALYSIS SUMMARY: Variable-Sized Mixers")
    print("Loading & Flushing Operations with ECN Optimization")
    print("=" * 80)
    print(f"\nTotal trees analyzed: {stats.get('valid_trees', 0)}/{stats.get('total_trees', 0)}")
    if stats.get('error_count', 0) > 0:
        print(f"Errors: {stats['error_count']}")
    
    print("\n" + "-" * 80)
    print(f"{'Metric':<30} {'Without ECN':<15} {'With ECN':<15} {'Change %':<12}")
    print("-" * 80)
    
    # Key metrics to display
    display_metrics = [
        ('loading_count', 'Loading Operations'),
        ('flushing_cells', 'Flushing (Cells)'),
        ('flushing_events', 'Flushing Events'),
        ('total_loading_with_reloads', 'Total Load Cycles'),
        ('mixer_count', 'Mixer Count'),
        ('reagent_count', 'Reagent Inputs'),
        ('total_cells_used', 'Total Cells Used'),
        ('ecn', 'ECN Value'),
        ('efficiency_ratio', 'Efficiency Ratio'),
    ]
    
    for metric_key, metric_name in display_metrics:
        without = stats['without_ecn_stats'].get(metric_key, {}).get('mean', 0)
        with_ecn = stats['with_ecn_stats'].get(metric_key, {}).get('mean', 0)
        reduction = stats['improvement_stats'].get(metric_key, {}).get('mean_reduction_percent', 0)
        
        # Format based on whether it's a percentage or count
        if metric_key == 'efficiency_ratio':
            print(f"{metric_name:<30} {without:<15.4f} {with_ecn:<15.4f} {reduction:>+11.2f}%")
        else:
            print(f"{metric_name:<30} {without:<15.2f} {with_ecn:<15.2f} {reduction:>+11.2f}%")
    
    print("-" * 80)
    
    # Highlight key findings
    flushing_change = stats['improvement_stats'].get('flushing_cells', {}).get('mean_reduction_percent', 0)
    loading_change = stats['improvement_stats'].get('total_loading_with_reloads', {}).get('mean_reduction_percent', 0)
    ecn_change = stats['improvement_stats'].get('ecn', {}).get('mean_reduction_percent', 0)
    
    print(f"\n📊 KEY FINDINGS (Variable-Sized Mixers with ECN):")
    print(f"   • Flushing cells change: {flushing_change:+.2f}%")
    print(f"   • Total loading cycles change: {loading_change:+.2f}%")
    print(f"   • ECN value change: {ecn_change:+.2f}%")
    
    if flushing_change > 0:
        print(f"\n   ✅ ECN optimization REDUCES flushing by {flushing_change:.2f}%!")
    elif flushing_change < 0:
        print(f"\n   ⚠️ ECN optimization increases flushing by {-flushing_change:.2f}%")
    else:
        print(f"\n   ⚪ No change in flushing operations")
    
    # Additional insight about variable-sized mixers
    avg_flushing = stats['without_ecn_stats'].get('flushing_cells', {}).get('mean', 0)
    avg_loading = stats['without_ecn_stats'].get('loading_count', {}).get('mean', 0)
    if avg_loading > 0:
        flush_ratio = (avg_flushing / avg_loading) * 100
        print(f"\n   📈 Average flush-to-load ratio: {flush_ratio:.1f}%")
        print(f"      (Lower is better - less waste)")
    
    print("=" * 80)


def save_results(results, stats, output_dir='.'):
    """
    Save analysis results to files.
    
    Args:
        results: Raw results from analyze_all_trees
        stats: Aggregated statistics
        output_dir: Directory to save results
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory
    results_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(results_dir, f'ecn_analysis_detailed_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary statistics
    stats_file = os.path.join(results_dir, f'ecn_analysis_summary_{timestamp}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to: {stats_file}")
    
    return results_file, stats_file


# ============================================================
# COMPREHENSIVE PAPER ANALYSIS FUNCTIONS (Miku_v1)
# ============================================================

def analyze_tree_comprehensive(tree, tree_index=None):
    """
    Comprehensive analysis of a single tree for paper reproduction.
    
    Args:
        tree: A node object representing the mixing tree
        tree_index: Optional index for identification
    
    Returns:
        dict: Complete analysis including overlaps and flushing W/WO scaling
    """
    try:
        # Create copies for analysis
        tree_original = deepcopy(tree)
        tree_scaled = optimize_volumes_with_ecn(deepcopy(tree))
        
        # Basic tree metrics
        height = get_tree_height(tree_original)
        mixer_count = count_mixers(tree_original)
        reagent_count = count_reagents(tree_original)
        
        # WITHOUT Scaling (original tree)
        overlaps_wo_scaling = count_overlaps(tree_original)
        flushing_wo_scaling = count_flushing_from_tree(tree_original)
        loading_wo_scaling = count_loading_from_tree(tree_original)
        
        # WITH Scaling (ECN optimized)
        overlaps_w_scaling = count_overlaps(tree_scaled)
        flushing_w_scaling = count_flushing_from_tree(tree_scaled)
        loading_w_scaling = count_loading_from_tree(tree_scaled)
        
        # Calculate reductions
        overlap_reduction = 0 if overlaps_wo_scaling == 0 else \
            ((overlaps_wo_scaling - overlaps_w_scaling) / overlaps_wo_scaling) * 100
        flushing_reduction = 0 if flushing_wo_scaling == 0 else \
            ((flushing_wo_scaling - flushing_w_scaling) / flushing_wo_scaling) * 100
        loading_reduction = 0 if loading_wo_scaling == 0 else \
            ((loading_wo_scaling - loading_w_scaling) / loading_wo_scaling) * 100
        
        # Total operations (loading + flushing)
        total_ops_wo = loading_wo_scaling + flushing_wo_scaling
        total_ops_w = loading_w_scaling + flushing_w_scaling
        total_reduction = 0 if total_ops_wo == 0 else \
            ((total_ops_wo - total_ops_w) / total_ops_wo) * 100
        
        return {
            'tree_index': tree_index,
            'height': height,
            'mixer_count': mixer_count,
            'reagent_count': reagent_count,
            'overlaps_wo_scaling': overlaps_wo_scaling,
            'overlaps_w_scaling': overlaps_w_scaling,
            'overlap_reduction_pct': overlap_reduction,
            'flushing_wo_scaling': flushing_wo_scaling,
            'flushing_w_scaling': flushing_w_scaling,
            'flushing_reduction_pct': flushing_reduction,
            'loading_wo_scaling': loading_wo_scaling,
            'loading_w_scaling': loading_w_scaling,
            'loading_reduction_pct': loading_reduction,
            'total_ops_wo_scaling': total_ops_wo,
            'total_ops_w_scaling': total_ops_w,
            'total_reduction_pct': total_reduction,
        }
    except Exception as e:
        return {
            'tree_index': tree_index,
            'error': str(e)
        }


def generate_table2_statistics(analysis_results):
    """
    Generate Table 2 from the Miku paper - Descriptive Statistics.
    
    Groups results by tree height and calculates aggregates.
    
    Args:
        analysis_results: List of comprehensive analysis results
    
    Returns:
        dict: Table 2 data structure
    """
    # Filter valid results
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Group by height
    by_height = defaultdict(list)
    for r in valid_results:
        by_height[r['height']].append(r)
    
    table2 = {}
    for height in sorted(by_height.keys()):
        trees = by_height[height]
        n = len(trees)
        
        if n == 0:
            continue
        
        # Calculate averages
        avg_mixers = sum(t['mixer_count'] for t in trees) / n
        avg_overlaps_wo = sum(t['overlaps_wo_scaling'] for t in trees) / n
        avg_overlaps_w = sum(t['overlaps_w_scaling'] for t in trees) / n
        avg_flushing_wo = sum(t['flushing_wo_scaling'] for t in trees) / n
        avg_flushing_w = sum(t['flushing_w_scaling'] for t in trees) / n
        avg_loading_wo = sum(t['loading_wo_scaling'] for t in trees) / n
        avg_loading_w = sum(t['loading_w_scaling'] for t in trees) / n
        avg_total_wo = sum(t['total_ops_wo_scaling'] for t in trees) / n
        avg_total_w = sum(t['total_ops_w_scaling'] for t in trees) / n
        
        # Calculate percentage reductions from averages
        avg_overlap_reduction = 0 if avg_overlaps_wo == 0 else \
            ((avg_overlaps_wo - avg_overlaps_w) / avg_overlaps_wo) * 100
        avg_flushing_reduction = 0 if avg_flushing_wo == 0 else \
            ((avg_flushing_wo - avg_flushing_w) / avg_flushing_wo) * 100
        avg_loading_reduction = 0 if avg_loading_wo == 0 else \
            ((avg_loading_wo - avg_loading_w) / avg_loading_wo) * 100
        avg_total_reduction = 0 if avg_total_wo == 0 else \
            ((avg_total_wo - avg_total_w) / avg_total_wo) * 100
        
        # Also calculate mean of individual reductions
        mean_overlap_reduction = sum(t['overlap_reduction_pct'] for t in trees) / n
        mean_flushing_reduction = sum(t['flushing_reduction_pct'] for t in trees) / n
        mean_loading_reduction = sum(t['loading_reduction_pct'] for t in trees) / n
        mean_total_reduction = sum(t['total_reduction_pct'] for t in trees) / n
        
        table2[height] = {
            'num_trees': n,
            'avg_num_mixers': avg_mixers,
            'avg_overlaps_wo_scaling': avg_overlaps_wo,
            'avg_overlaps_w_scaling': avg_overlaps_w,
            'avg_overlap_reduction_pct': avg_overlap_reduction,
            'mean_overlap_reduction_pct': mean_overlap_reduction,
            'avg_flushing_wo_scaling': avg_flushing_wo,
            'avg_flushing_w_scaling': avg_flushing_w,
            'avg_flushing_reduction_pct': avg_flushing_reduction,
            'mean_flushing_reduction_pct': mean_flushing_reduction,
            'avg_loading_wo_scaling': avg_loading_wo,
            'avg_loading_w_scaling': avg_loading_w,
            'avg_loading_reduction_pct': avg_loading_reduction,
            'mean_loading_reduction_pct': mean_loading_reduction,
            'avg_total_wo_scaling': avg_total_wo,
            'avg_total_w_scaling': avg_total_w,
            'avg_total_reduction_pct': avg_total_reduction,
            'mean_total_reduction_pct': mean_total_reduction,
        }
    
    # Calculate overall statistics
    n_total = len(valid_results)
    if n_total > 0:
        avg_overlaps_wo_total = sum(t['overlaps_wo_scaling'] for t in valid_results) / n_total
        avg_overlaps_w_total = sum(t['overlaps_w_scaling'] for t in valid_results) / n_total
        avg_flushing_wo_total = sum(t['flushing_wo_scaling'] for t in valid_results) / n_total
        avg_flushing_w_total = sum(t['flushing_w_scaling'] for t in valid_results) / n_total
        avg_loading_wo_total = sum(t['loading_wo_scaling'] for t in valid_results) / n_total
        avg_loading_w_total = sum(t['loading_w_scaling'] for t in valid_results) / n_total
        avg_total_wo_total = sum(t['total_ops_wo_scaling'] for t in valid_results) / n_total
        avg_total_w_total = sum(t['total_ops_w_scaling'] for t in valid_results) / n_total
        
        table2['overall'] = {
            'num_trees': n_total,
            'avg_num_mixers': sum(t['mixer_count'] for t in valid_results) / n_total,
            'avg_overlaps_wo_scaling': avg_overlaps_wo_total,
            'avg_overlaps_w_scaling': avg_overlaps_w_total,
            'avg_overlap_reduction_pct': 0 if avg_overlaps_wo_total == 0 else \
                ((avg_overlaps_wo_total - avg_overlaps_w_total) / avg_overlaps_wo_total) * 100,
            'avg_flushing_wo_scaling': avg_flushing_wo_total,
            'avg_flushing_w_scaling': avg_flushing_w_total,
            'avg_flushing_reduction_pct': 0 if avg_flushing_wo_total == 0 else \
                ((avg_flushing_wo_total - avg_flushing_w_total) / avg_flushing_wo_total) * 100,
            'avg_loading_wo_scaling': avg_loading_wo_total,
            'avg_loading_w_scaling': avg_loading_w_total,
            'avg_loading_reduction_pct': 0 if avg_loading_wo_total == 0 else \
                ((avg_loading_wo_total - avg_loading_w_total) / avg_loading_wo_total) * 100,
            'avg_total_wo_scaling': avg_total_wo_total,
            'avg_total_w_scaling': avg_total_w_total,
            'avg_total_reduction_pct': 0 if avg_total_wo_total == 0 else \
                ((avg_total_wo_total - avg_total_w_total) / avg_total_wo_total) * 100,
            'mean_overlap_reduction_pct': sum(t['overlap_reduction_pct'] for t in valid_results) / n_total,
            'mean_flushing_reduction_pct': sum(t['flushing_reduction_pct'] for t in valid_results) / n_total,
            'mean_loading_reduction_pct': sum(t['loading_reduction_pct'] for t in valid_results) / n_total,
            'mean_total_reduction_pct': sum(t['total_reduction_pct'] for t in valid_results) / n_total,
        }
    
    return table2


def print_table2(table2):
    """
    Print Table 2 in a formatted way - Flushing Statistics.
    """
    print("\n" + "=" * 100)
    print("TABLE 2A: FLUSHING Statistics (Replicating Miku Paper)")
    print("=" * 100)
    
    print(f"{'Height':<8} {'N':<6} {'Avg':<10} {'Overlaps':<12} {'Overlaps':<12} {'Overlap':<12} "
          f"{'Flushing':<12} {'Flushing':<12} {'Flushing':<12}")
    print(f"{'':<8} {'':<6} {'Mixers':<10} {'(W.O.)':<12} {'(W.)':<12} {'Red. %':<12} "
          f"{'(W.O.)':<12} {'(W.)':<12} {'Red. %':<12}")
    print("-" * 100)
    
    for height in sorted([h for h in table2.keys() if h != 'overall']):
        data = table2[height]
        print(f"{height:<8} {data['num_trees']:<6} {data['avg_num_mixers']:<10.2f} "
              f"{data['avg_overlaps_wo_scaling']:<12.2f} {data['avg_overlaps_w_scaling']:<12.2f} "
              f"{data['avg_overlap_reduction_pct']:<12.2f} "
              f"{data['avg_flushing_wo_scaling']:<12.2f} {data['avg_flushing_w_scaling']:<12.2f} "
              f"{data['avg_flushing_reduction_pct']:<12.2f}")
    
    print("-" * 100)
    if 'overall' in table2:
        data = table2['overall']
        print(f"{'Overall':<8} {data['num_trees']:<6} {data['avg_num_mixers']:<10.2f} "
              f"{data['avg_overlaps_wo_scaling']:<12.2f} {data['avg_overlaps_w_scaling']:<12.2f} "
              f"{data['mean_overlap_reduction_pct']:<12.2f} "
              f"{data['avg_flushing_wo_scaling']:<12.2f} {data['avg_flushing_w_scaling']:<12.2f} "
              f"{data['mean_flushing_reduction_pct']:<12.2f}")
    print("=" * 100)


def print_loading_table(table2):
    """
    Print Loading Statistics Table.
    """
    print("\n" + "=" * 100)
    print("TABLE 2B: LOADING Statistics")
    print("=" * 100)
    
    print(f"{'Height':<8} {'N':<6} {'Loading':<12} {'Loading':<12} {'Loading':<12} "
          f"{'Total Ops':<12} {'Total Ops':<12} {'Total':<12}")
    print(f"{'':<8} {'':<6} {'(W.O.)':<12} {'(W.)':<12} {'Red. %':<12} "
          f"{'(W.O.)':<12} {'(W.)':<12} {'Red. %':<12}")
    print("-" * 100)
    
    for height in sorted([h for h in table2.keys() if h != 'overall']):
        data = table2[height]
        print(f"{height:<8} {data['num_trees']:<6} "
              f"{data['avg_loading_wo_scaling']:<12.2f} {data['avg_loading_w_scaling']:<12.2f} "
              f"{data['avg_loading_reduction_pct']:<12.2f} "
              f"{data['avg_total_wo_scaling']:<12.2f} {data['avg_total_w_scaling']:<12.2f} "
              f"{data['avg_total_reduction_pct']:<12.2f}")
    
    print("-" * 100)
    if 'overall' in table2:
        data = table2['overall']
        print(f"{'Overall':<8} {data['num_trees']:<6} "
              f"{data['avg_loading_wo_scaling']:<12.2f} {data['avg_loading_w_scaling']:<12.2f} "
              f"{data['mean_loading_reduction_pct']:<12.2f} "
              f"{data['avg_total_wo_scaling']:<12.2f} {data['avg_total_w_scaling']:<12.2f} "
              f"{data['mean_total_reduction_pct']:<12.2f}")
    print("=" * 100)


def calculate_time_estimation(analysis_results, time_per_flush=15.2):
    """
    Calculate time estimation for flushing operations.
    One flushing operation = 15.2 seconds (2 × 7.6 seconds per loading/wash cycle).
    
    Args:
        analysis_results: List of comprehensive analysis results
        time_per_flush: Time in seconds per flushing operation (default 15.2)
    
    Returns:
        dict: Time estimation data grouped by height
    """
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Add time estimates to each result
    for r in valid_results:
        r['time_wo_scaling'] = r['flushing_wo_scaling'] * time_per_flush
        r['time_w_scaling'] = r['flushing_w_scaling'] * time_per_flush
    
    # Group by height
    by_height = defaultdict(list)
    for r in valid_results:
        by_height[r['height']].append(r)
    
    time_data = {}
    for height in sorted(by_height.keys()):
        trees = by_height[height]
        n = len(trees)
        
        avg_time_wo = sum(t['time_wo_scaling'] for t in trees) / n
        avg_time_w = sum(t['time_w_scaling'] for t in trees) / n
        
        time_data[height] = {
            'num_trees': n,
            'avg_time_wo_scaling': avg_time_wo,
            'avg_time_w_scaling': avg_time_w,
            'time_reduction_pct': 0 if avg_time_wo == 0 else \
                ((avg_time_wo - avg_time_w) / avg_time_wo) * 100
        }
    
    return time_data


def plot_figure11(time_data, output_dir='./analysis_results'):
    """
    Plot Figure 11 - Average Flushing Time vs Tree Height (grouped bar chart).
    
    Args:
        time_data: Time estimation data from calculate_time_estimation
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in time_data.keys()])
    times_wo = [time_data[h]['avg_time_wo_scaling'] for h in heights]
    times_w = [time_data[h]['avg_time_w_scaling'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, times_wo, width, label='W.O. Scaling', color='#ff6b6b', edgecolor='black')
    bars2 = ax.bar(x + width/2, times_w, width, label='W. Scaling', color='#4ecdc4', edgecolor='black')
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Flushing Time (seconds)', fontsize=12)
    ax.set_title('Figure 11: Average Flushing Time vs Tree Height\n(1 flush = 15.2 seconds)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure11_flushing_time.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'figure11_flushing_time.pdf'))
    print(f"✓ Figure 11 saved to {output_dir}/figure11_flushing_time.png")
    plt.close()


def calculate_loading_time_estimation(analysis_results, time_per_load=7.6):
    """
    Calculate time estimation for loading operations.
    One loading operation = 7.6 seconds per cell loading.
    
    Args:
        analysis_results: List of comprehensive analysis results
        time_per_load: Time in seconds per loading operation (default 7.6)
    
    Returns:
        dict: Time estimation data grouped by height
    """
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Add time estimates to each result
    for r in valid_results:
        r['loading_time_wo_scaling'] = r['loading_wo_scaling'] * time_per_load
        r['loading_time_w_scaling'] = r['loading_w_scaling'] * time_per_load
    
    # Group by height
    by_height = defaultdict(list)
    for r in valid_results:
        by_height[r['height']].append(r)
    
    time_data = {}
    for height in sorted(by_height.keys()):
        trees = by_height[height]
        n = len(trees)
        
        avg_time_wo = sum(t['loading_time_wo_scaling'] for t in trees) / n
        avg_time_w = sum(t['loading_time_w_scaling'] for t in trees) / n
        
        time_data[height] = {
            'num_trees': n,
            'avg_loading_time_wo_scaling': avg_time_wo,
            'avg_loading_time_w_scaling': avg_time_w,
            'loading_time_reduction_pct': 0 if avg_time_wo == 0 else \
                ((avg_time_wo - avg_time_w) / avg_time_wo) * 100
        }
    
    return time_data


def plot_loading_time(loading_time_data, output_dir='./analysis_results'):
    """
    Plot Loading Time vs Tree Height (grouped bar chart).
    
    Args:
        loading_time_data: Time estimation data from calculate_loading_time_estimation
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in loading_time_data.keys()])
    times_wo = [loading_time_data[h]['avg_loading_time_wo_scaling'] for h in heights]
    times_w = [loading_time_data[h]['avg_loading_time_w_scaling'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, times_wo, width, label='W.O. Scaling', color='#9b59b6', edgecolor='black')
    bars2 = ax.bar(x + width/2, times_w, width, label='W. Scaling', color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Loading Time (seconds)', fontsize=12)
    ax.set_title('Average Loading Time vs Tree Height\n(1 loading = 7.6 seconds)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loading_time_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'loading_time_comparison.pdf'))
    print(f"✓ Loading time chart saved to {output_dir}/loading_time_comparison.png")
    plt.close()


def plot_combined_operations(table2, output_dir='./analysis_results'):
    """
    Plot combined comparison of Loading, Flushing, and Total Operations.
    
    Args:
        table2: Table 2 statistics
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in table2.keys() if h != 'overall'])
    
    loading_wo = [table2[h]['avg_loading_wo_scaling'] for h in heights]
    loading_w = [table2[h]['avg_loading_w_scaling'] for h in heights]
    flushing_wo = [table2[h]['avg_flushing_wo_scaling'] for h in heights]
    flushing_w = [table2[h]['avg_flushing_w_scaling'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - 1.5*width, loading_wo, width, label='Loading W.O.', color='#9b59b6', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, loading_w, width, label='Loading W.', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, flushing_wo, width, label='Flushing W.O.', color='#e74c3c', edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, flushing_w, width, label='Flushing W.', color='#27ae60', edgecolor='black')
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Operations Count', fontsize=12)
    ax.set_title('Loading vs Flushing Operations by Tree Height\n(With vs Without Scaling)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loading_flushing_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'loading_flushing_comparison.pdf'))
    print(f"✓ Combined operations chart saved to {output_dir}/loading_flushing_comparison.png")
    plt.close()


def plot_total_operations(table2, output_dir='./analysis_results'):
    """
    Plot Total Operations (Loading + Flushing) comparison.
    
    Args:
        table2: Table 2 statistics
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in table2.keys() if h != 'overall'])
    
    total_wo = [table2[h]['avg_total_wo_scaling'] for h in heights]
    total_w = [table2[h]['avg_total_w_scaling'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, total_wo, width, label='W.O. Scaling', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, total_w, width, label='W. Scaling', color='#27ae60', edgecolor='black')
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Total Operations (Loading + Flushing)', fontsize=12)
    ax.set_title('Total Operations by Tree Height\n(With vs Without Scaling)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add reduction percentage labels
    for i, (h, wo, w) in enumerate(zip(heights, total_wo, total_w)):
        if wo > 0:
            reduction = ((wo - w) / wo) * 100
            ax.annotate(f'-{reduction:.1f}%',
                       xy=(i, max(wo, w) + 5),
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_operations_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'total_operations_comparison.pdf'))
    print(f"✓ Total operations chart saved to {output_dir}/total_operations_comparison.png")
    plt.close()


def plot_loading_box_plots(analysis_results, output_dir='./analysis_results'):
    """
    Plot box plots showing distribution of loading operations by height.
    
    Args:
        analysis_results: List of comprehensive analysis results
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Group by height
    by_height = defaultdict(lambda: {'wo': [], 'w': []})
    for r in valid_results:
        by_height[r['height']]['wo'].append(r['loading_wo_scaling'])
        by_height[r['height']]['w'].append(r['loading_w_scaling'])
    
    heights = sorted(by_height.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot for W.O. Scaling
    data_wo = [by_height[h]['wo'] for h in heights]
    bp1 = axes[0].boxplot(data_wo, tick_labels=heights, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#9b59b6')
    axes[0].set_xlabel('Tree Height', fontsize=12)
    axes[0].set_ylabel('Loading Operations', fontsize=12)
    axes[0].set_title('Loading Distribution (W.O. Scaling)', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot for W. Scaling
    data_w = [by_height[h]['w'] for h in heights]
    bp2 = axes[1].boxplot(data_w, tick_labels=heights, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#3498db')
    axes[1].set_xlabel('Tree Height', fontsize=12)
    axes[1].set_ylabel('Loading Operations', fontsize=12)
    axes[1].set_title('Loading Distribution (W. Scaling)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_loading_distribution.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'boxplot_loading_distribution.pdf'))
    print(f"✓ Loading box plots saved to {output_dir}/boxplot_loading_distribution.png")
    plt.close()


def calculate_correlation(analysis_results):
    """
    Calculate Pearson correlation between overlap reduction and flushing reduction.
    
    Args:
        analysis_results: List of comprehensive analysis results
    
    Returns:
        dict: Correlation results
    """
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Extract data for correlation
    overlap_reduction = [r['overlap_reduction_pct'] for r in valid_results]
    flushing_reduction = [r['flushing_reduction_pct'] for r in valid_results]
    
    overlaps_wo = [r['overlaps_wo_scaling'] for r in valid_results]
    flushing_wo = [r['flushing_wo_scaling'] for r in valid_results]
    
    # Pearson correlation between reductions
    corr_reduction, p_reduction = stats.pearsonr(overlap_reduction, flushing_reduction)
    
    # Pearson correlation between raw overlaps and flushing (W.O. Scaling)
    corr_raw, p_raw = stats.pearsonr(overlaps_wo, flushing_wo)
    
    return {
        'reduction_correlation': {
            'pearson_r': corr_reduction,
            'p_value': p_reduction,
            'interpretation': 'Strong positive correlation' if corr_reduction > 0.7 else \
                'Moderate correlation' if corr_reduction > 0.4 else 'Weak correlation'
        },
        'raw_correlation': {
            'pearson_r': corr_raw,
            'p_value': p_raw,
            'interpretation': 'Strong positive correlation' if corr_raw > 0.7 else \
                'Moderate correlation' if corr_raw > 0.4 else 'Weak correlation'
        },
        'data': {
            'overlap_reduction': overlap_reduction,
            'flushing_reduction': flushing_reduction,
            'overlaps_wo': overlaps_wo,
            'flushing_wo': flushing_wo
        }
    }


def plot_scatter_correlation(correlation_results, output_dir='./analysis_results'):
    """
    Plot scatter plot: Overlaps (W.O. Scaling) vs Flushing (W.O. Scaling).
    
    Args:
        correlation_results: Results from calculate_correlation
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    overlaps = correlation_results['data']['overlaps_wo']
    flushing = correlation_results['data']['flushing_wo']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(overlaps, flushing, alpha=0.6, c='#3498db', edgecolors='black', s=50)
    
    # Fit regression line
    z = np.polyfit(overlaps, flushing, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(overlaps), max(overlaps), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, 
            label=f'Trend line (r={correlation_results["raw_correlation"]["pearson_r"]:.3f})')
    
    ax.set_xlabel('Number of Overlaps (W.O. Scaling)', fontsize=12)
    ax.set_ylabel('Number of Flush Operations (W.O. Scaling)', fontsize=12)
    ax.set_title('Hypothesis Validation: Overlaps vs Flushing Operations\n'
                 f'Pearson r = {correlation_results["raw_correlation"]["pearson_r"]:.4f}, '
                 f'p = {correlation_results["raw_correlation"]["p_value"]:.2e}', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_overlaps_vs_flushing.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'scatter_overlaps_vs_flushing.pdf'))
    print(f"✓ Scatter plot saved to {output_dir}/scatter_overlaps_vs_flushing.png")
    plt.close()


def plot_box_plots(analysis_results, output_dir='./analysis_results'):
    """
    Plot box plots showing distribution of flushing operations by height.
    
    Args:
        analysis_results: List of comprehensive analysis results
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Group by height
    by_height = defaultdict(lambda: {'wo': [], 'w': []})
    for r in valid_results:
        by_height[r['height']]['wo'].append(r['flushing_wo_scaling'])
        by_height[r['height']]['w'].append(r['flushing_w_scaling'])
    
    heights = sorted(by_height.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot for W.O. Scaling
    data_wo = [by_height[h]['wo'] for h in heights]
    bp1 = axes[0].boxplot(data_wo, tick_labels=heights, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#ff6b6b')
    axes[0].set_xlabel('Tree Height', fontsize=12)
    axes[0].set_ylabel('Flushing Operations', fontsize=12)
    axes[0].set_title('Flushing Distribution (W.O. Scaling)', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot for W. Scaling
    data_w = [by_height[h]['w'] for h in heights]
    bp2 = axes[1].boxplot(data_w, tick_labels=heights, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#4ecdc4')
    axes[1].set_xlabel('Tree Height', fontsize=12)
    axes[1].set_ylabel('Flushing Operations', fontsize=12)
    axes[1].set_title('Flushing Distribution (W. Scaling)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_flushing_distribution.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'boxplot_flushing_distribution.pdf'))
    print(f"✓ Box plots saved to {output_dir}/boxplot_flushing_distribution.png")
    plt.close()


def calculate_zero_flush_rate(analysis_results):
    """
    Calculate the percentage of trees that require zero flushing after scaling.
    
    Args:
        analysis_results: List of comprehensive analysis results
    
    Returns:
        dict: Zero-flush statistics by height and overall
    """
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    # Group by height
    by_height = defaultdict(list)
    for r in valid_results:
        by_height[r['height']].append(r)
    
    zero_flush_stats = {}
    for height in sorted(by_height.keys()):
        trees = by_height[height]
        n = len(trees)
        
        zero_wo = sum(1 for t in trees if t['flushing_wo_scaling'] == 0)
        zero_w = sum(1 for t in trees if t['flushing_w_scaling'] == 0)
        
        zero_flush_stats[height] = {
            'num_trees': n,
            'zero_flush_wo_scaling': zero_wo,
            'zero_flush_w_scaling': zero_w,
            'zero_flush_rate_wo_scaling': (zero_wo / n) * 100 if n > 0 else 0,
            'zero_flush_rate_w_scaling': (zero_w / n) * 100 if n > 0 else 0,
        }
    
    # Overall
    n_total = len(valid_results)
    zero_wo_total = sum(1 for r in valid_results if r['flushing_wo_scaling'] == 0)
    zero_w_total = sum(1 for r in valid_results if r['flushing_w_scaling'] == 0)
    
    zero_flush_stats['overall'] = {
        'num_trees': n_total,
        'zero_flush_wo_scaling': zero_wo_total,
        'zero_flush_w_scaling': zero_w_total,
        'zero_flush_rate_wo_scaling': (zero_wo_total / n_total) * 100 if n_total > 0 else 0,
        'zero_flush_rate_w_scaling': (zero_w_total / n_total) * 100 if n_total > 0 else 0,
    }
    
    return zero_flush_stats


def print_zero_flush_stats(zero_flush_stats):
    """
    Print zero-flush success rate statistics.
    """
    print("\n" + "=" * 70)
    print("ZERO-FLUSH SUCCESS RATE ANALYSIS")
    print("=" * 70)
    print(f"{'Height':<10} {'N':<8} {'Zero W.O.':<12} {'Rate W.O. %':<14} {'Zero W.':<10} {'Rate W. %':<12}")
    print("-" * 70)
    
    for height in sorted([h for h in zero_flush_stats.keys() if h != 'overall']):
        data = zero_flush_stats[height]
        print(f"{height:<10} {data['num_trees']:<8} {data['zero_flush_wo_scaling']:<12} "
              f"{data['zero_flush_rate_wo_scaling']:<14.2f} {data['zero_flush_w_scaling']:<10} "
              f"{data['zero_flush_rate_w_scaling']:<12.2f}")
    
    print("-" * 70)
    if 'overall' in zero_flush_stats:
        data = zero_flush_stats['overall']
        print(f"{'Overall':<10} {data['num_trees']:<8} {data['zero_flush_wo_scaling']:<12} "
              f"{data['zero_flush_rate_wo_scaling']:<14.2f} {data['zero_flush_w_scaling']:<10} "
              f"{data['zero_flush_rate_w_scaling']:<12.2f}")
    print("=" * 70)


# ============================================================
# TABLE EXPORT FUNCTIONS (CSV & HTML)
# ============================================================

def export_table2_csv(table2, output_dir='./analysis_results'):
    """
    Export Table 2 statistics to CSV files for better visualization.
    
    Args:
        table2: Table 2 statistics dictionary
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in table2.keys() if h != 'overall'])
    
    # === Table 2A: Flushing Statistics ===
    flushing_csv_path = os.path.join(output_dir, 'table2a_flushing_stats.csv')
    with open(flushing_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Height', 'N', 'Avg Mixers', 'Overlaps (W.O.)', 'Overlaps (W.)', 
                        'Overlap Red. %', 'Flushing (W.O.)', 'Flushing (W.)', 'Flushing Red. %'])
        
        for h in heights:
            d = table2[h]
            writer.writerow([h, d['num_trees'], f"{d['avg_num_mixers']:.2f}",
                           f"{d['avg_overlaps_wo_scaling']:.2f}", f"{d['avg_overlaps_w_scaling']:.2f}",
                           f"{d['avg_overlap_reduction_pct']:.2f}",
                           f"{d['avg_flushing_wo_scaling']:.2f}", f"{d['avg_flushing_w_scaling']:.2f}",
                           f"{d['avg_flushing_reduction_pct']:.2f}"])
        
        # Overall row
        d = table2['overall']
        writer.writerow(['Overall', d['num_trees'], f"{d['avg_num_mixers']:.2f}",
                       f"{d['avg_overlaps_wo_scaling']:.2f}", f"{d['avg_overlaps_w_scaling']:.2f}",
                       f"{d['avg_overlap_reduction_pct']:.2f}",
                       f"{d['avg_flushing_wo_scaling']:.2f}", f"{d['avg_flushing_w_scaling']:.2f}",
                       f"{d['avg_flushing_reduction_pct']:.2f}"])
    
    print(f"✓ Table 2A (Flushing) saved to: {flushing_csv_path}")
    
    # === Table 2B: Loading Statistics ===
    loading_csv_path = os.path.join(output_dir, 'table2b_loading_stats.csv')
    with open(loading_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Height', 'N', 'Loading (W.O.)', 'Loading (W.)', 'Loading Red. %',
                        'Total Ops (W.O.)', 'Total Ops (W.)', 'Total Red. %'])
        
        for h in heights:
            d = table2[h]
            writer.writerow([h, d['num_trees'],
                           f"{d['avg_loading_wo_scaling']:.2f}", f"{d['avg_loading_w_scaling']:.2f}",
                           f"{d['avg_loading_reduction_pct']:.2f}",
                           f"{d['avg_total_wo_scaling']:.2f}", f"{d['avg_total_w_scaling']:.2f}",
                           f"{d['avg_total_reduction_pct']:.2f}"])
        
        # Overall row
        d = table2['overall']
        writer.writerow(['Overall', d['num_trees'],
                       f"{d['avg_loading_wo_scaling']:.2f}", f"{d['avg_loading_w_scaling']:.2f}",
                       f"{d['avg_loading_reduction_pct']:.2f}",
                       f"{d['avg_total_wo_scaling']:.2f}", f"{d['avg_total_w_scaling']:.2f}",
                       f"{d['avg_total_reduction_pct']:.2f}"])
    
    print(f"✓ Table 2B (Loading) saved to: {loading_csv_path}")
    
    # === Combined Summary CSV ===
    summary_csv_path = os.path.join(output_dir, 'analysis_summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Height', 'N', 'Avg Mixers', 
                        'Flushing (W.O.)', 'Flushing (W.)', 'Flushing Red. %',
                        'Loading (W.O.)', 'Loading (W.)', 'Loading Red. %',
                        'Total Ops (W.O.)', 'Total Ops (W.)', 'Total Red. %'])
        
        for h in heights:
            d = table2[h]
            writer.writerow([h, d['num_trees'], f"{d['avg_num_mixers']:.2f}",
                           f"{d['avg_flushing_wo_scaling']:.2f}", f"{d['avg_flushing_w_scaling']:.2f}",
                           f"{d['avg_flushing_reduction_pct']:.2f}",
                           f"{d['avg_loading_wo_scaling']:.2f}", f"{d['avg_loading_w_scaling']:.2f}",
                           f"{d['avg_loading_reduction_pct']:.2f}",
                           f"{d['avg_total_wo_scaling']:.2f}", f"{d['avg_total_w_scaling']:.2f}",
                           f"{d['avg_total_reduction_pct']:.2f}"])
        
        d = table2['overall']
        writer.writerow(['Overall', d['num_trees'], f"{d['avg_num_mixers']:.2f}",
                       f"{d['avg_flushing_wo_scaling']:.2f}", f"{d['avg_flushing_w_scaling']:.2f}",
                       f"{d['avg_flushing_reduction_pct']:.2f}",
                       f"{d['avg_loading_wo_scaling']:.2f}", f"{d['avg_loading_w_scaling']:.2f}",
                       f"{d['avg_loading_reduction_pct']:.2f}",
                       f"{d['avg_total_wo_scaling']:.2f}", f"{d['avg_total_w_scaling']:.2f}",
                       f"{d['avg_total_reduction_pct']:.2f}"])
    
    print(f"✓ Summary table saved to: {summary_csv_path}")


def export_table2_html(table2, time_data, loading_time_data, output_dir='./analysis_results'):
    """
    Export comprehensive analysis tables to a single HTML file with styling.
    
    Args:
        table2: Table 2 statistics dictionary
        time_data: Flushing time estimation data
        loading_time_data: Loading time estimation data
        output_dir: Directory to save HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in table2.keys() if h != 'overall'])
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PMD Scaling Analysis Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px 10px;
            text-align: center;
            font-weight: 600;
        }
        td {
            padding: 12px 10px;
            text-align: center;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f4f8;
        }
        tr.overall-row {
            background-color: #e8f6f3 !important;
            font-weight: bold;
            border-top: 2px solid #3498db;
        }
        .reduction-high {
            color: #27ae60;
            font-weight: bold;
        }
        .reduction-medium {
            color: #f39c12;
            font-weight: bold;
        }
        .summary-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        .summary-box h3 {
            margin: 0 0 10px 0;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .timestamp {
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <h1>🔬 PMD Scaling Method Analysis Results</h1>
    <p style="text-align: center; color: #7f8c8d;">Programmable Microfluidic Devices - Reducing Flushing & Loading Operations</p>
"""
    
    # Summary Box
    overall = table2['overall']
    html_content += f"""
    <div class="summary-box">
        <h3>📊 Overall Performance Summary</h3>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{overall['num_trees']}</div>
                <div class="stat-label">Trees Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall['avg_flushing_reduction_pct']:.1f}%</div>
                <div class="stat-label">Flushing Reduction</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall['avg_loading_reduction_pct']:.1f}%</div>
                <div class="stat-label">Loading Reduction</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall['avg_total_reduction_pct']:.1f}%</div>
                <div class="stat-label">Total Ops Reduction</div>
            </div>
        </div>
    </div>
"""
    
    # Table 2A: Flushing Statistics
    html_content += """
    <h2>📋 Table 2A: Flushing Statistics</h2>
    <table>
        <tr>
            <th>Height</th>
            <th>N</th>
            <th>Avg Mixers</th>
            <th>Overlaps (W.O.)</th>
            <th>Overlaps (W.)</th>
            <th>Overlap Red. %</th>
            <th>Flushing (W.O.)</th>
            <th>Flushing (W.)</th>
            <th>Flushing Red. %</th>
        </tr>
"""
    
    for h in heights:
        d = table2[h]
        reduction_class = 'reduction-high' if d['avg_flushing_reduction_pct'] >= 80 else 'reduction-medium'
        html_content += f"""        <tr>
            <td>{h}</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_num_mixers']:.2f}</td>
            <td>{d['avg_overlaps_wo_scaling']:.2f}</td>
            <td>{d['avg_overlaps_w_scaling']:.2f}</td>
            <td>{d['avg_overlap_reduction_pct']:.2f}%</td>
            <td>{d['avg_flushing_wo_scaling']:.2f}</td>
            <td>{d['avg_flushing_w_scaling']:.2f}</td>
            <td class="{reduction_class}">{d['avg_flushing_reduction_pct']:.2f}%</td>
        </tr>
"""
    
    d = table2['overall']
    html_content += f"""        <tr class="overall-row">
            <td>Overall</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_num_mixers']:.2f}</td>
            <td>{d['avg_overlaps_wo_scaling']:.2f}</td>
            <td>{d['avg_overlaps_w_scaling']:.2f}</td>
            <td>{d['avg_overlap_reduction_pct']:.2f}%</td>
            <td>{d['avg_flushing_wo_scaling']:.2f}</td>
            <td>{d['avg_flushing_w_scaling']:.2f}</td>
            <td class="reduction-high">{d['avg_flushing_reduction_pct']:.2f}%</td>
        </tr>
    </table>
"""
    
    # Table 2B: Loading Statistics
    html_content += """
    <h2>📋 Table 2B: Loading Statistics</h2>
    <table>
        <tr>
            <th>Height</th>
            <th>N</th>
            <th>Loading (W.O.)</th>
            <th>Loading (W.)</th>
            <th>Loading Red. %</th>
            <th>Total Ops (W.O.)</th>
            <th>Total Ops (W.)</th>
            <th>Total Red. %</th>
        </tr>
"""
    
    for h in heights:
        d = table2[h]
        reduction_class = 'reduction-high' if d['avg_loading_reduction_pct'] >= 80 else 'reduction-medium'
        html_content += f"""        <tr>
            <td>{h}</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_loading_wo_scaling']:.2f}</td>
            <td>{d['avg_loading_w_scaling']:.2f}</td>
            <td class="{reduction_class}">{d['avg_loading_reduction_pct']:.2f}%</td>
            <td>{d['avg_total_wo_scaling']:.2f}</td>
            <td>{d['avg_total_w_scaling']:.2f}</td>
            <td>{d['avg_total_reduction_pct']:.2f}%</td>
        </tr>
"""
    
    d = table2['overall']
    html_content += f"""        <tr class="overall-row">
            <td>Overall</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_loading_wo_scaling']:.2f}</td>
            <td>{d['avg_loading_w_scaling']:.2f}</td>
            <td>{d['avg_loading_reduction_pct']:.2f}%</td>
            <td>{d['avg_total_wo_scaling']:.2f}</td>
            <td>{d['avg_total_w_scaling']:.2f}</td>
            <td class="reduction-high">{d['avg_total_reduction_pct']:.2f}%</td>
        </tr>
    </table>
"""
    
    # Time Estimation Tables
    html_content += """
    <h2>⏱️ Time Estimation Analysis</h2>
    <p style="color: #7f8c8d;">Flushing: 15.2 seconds per operation (2 × 7.6s) | Loading: 7.6 seconds per operation</p>
    
    <h3>Flushing Time</h3>
    <table>
        <tr>
            <th>Height</th>
            <th>N</th>
            <th>Avg Time W.O. (s)</th>
            <th>Avg Time W. (s)</th>
            <th>Time Reduction %</th>
        </tr>
"""
    
    for h in sorted(time_data.keys()):
        d = time_data[h]
        html_content += f"""        <tr>
            <td>{h}</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_time_wo_scaling']:.2f}</td>
            <td>{d['avg_time_w_scaling']:.2f}</td>
            <td class="reduction-high">{d['time_reduction_pct']:.2f}%</td>
        </tr>
"""
    
    html_content += """    </table>
    
    <h3>Loading Time</h3>
    <table>
        <tr>
            <th>Height</th>
            <th>N</th>
            <th>Avg Time W.O. (s)</th>
            <th>Avg Time W. (s)</th>
            <th>Time Reduction %</th>
        </tr>
"""
    
    for h in sorted(loading_time_data.keys()):
        d = loading_time_data[h]
        html_content += f"""        <tr>
            <td>{h}</td>
            <td>{d['num_trees']}</td>
            <td>{d['avg_loading_time_wo_scaling']:.2f}</td>
            <td>{d['avg_loading_time_w_scaling']:.2f}</td>
            <td class="reduction-high">{d['loading_time_reduction_pct']:.2f}%</td>
        </tr>
"""
    
    # Close HTML
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content += f"""    </table>
    
    <p class="timestamp">Generated on: {timestamp}</p>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, 'analysis_results.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Interactive HTML report saved to: {html_path}")


def export_per_tree_csv(analysis_results, output_dir='./analysis_results'):
    """
    Export per-tree analysis results to CSV for detailed inspection.
    
    Args:
        analysis_results: List of comprehensive analysis results
        output_dir: Directory to save CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'per_tree_analysis.csv')
    
    valid_results = [r for r in analysis_results if 'error' not in r]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tree Index', 'Height', 'Mixer Count', 
                        'Overlaps (W.O.)', 'Overlaps (W.)', 'Overlap Red. %',
                        'Flushing (W.O.)', 'Flushing (W.)', 'Flushing Red. %',
                        'Loading (W.O.)', 'Loading (W.)', 'Loading Red. %',
                        'Total Ops (W.O.)', 'Total Ops (W.)', 'Total Red. %'])
        
        for r in valid_results:
            writer.writerow([
                r['tree_index'], r['height'], r['mixer_count'],
                r['overlaps_wo_scaling'], r['overlaps_w_scaling'], f"{r['overlap_reduction_pct']:.2f}",
                r['flushing_wo_scaling'], r['flushing_w_scaling'], f"{r['flushing_reduction_pct']:.2f}",
                r['loading_wo_scaling'], r['loading_w_scaling'], f"{r['loading_reduction_pct']:.2f}",
                r['total_ops_wo_scaling'], r['total_ops_w_scaling'], f"{r['total_reduction_pct']:.2f}"
            ])
    
    print(f"✓ Per-tree analysis saved to: {csv_path}")


def plot_combined_barplot(table2, output_dir='./analysis_results'):
    """
    Create combined bar plot for flushing comparison across heights.
    
    Args:
        table2: Table 2 statistics
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    heights = sorted([h for h in table2.keys() if h != 'overall'])
    
    flushing_wo = [table2[h]['avg_flushing_wo_scaling'] for h in heights]
    flushing_w = [table2[h]['avg_flushing_w_scaling'] for h in heights]
    
    x = np.arange(len(heights))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, flushing_wo, width, label='W.O. Scaling', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, flushing_w, width, label='W. Scaling', color='#27ae60', edgecolor='black')
    
    ax.set_xlabel('Tree Height', fontsize=12)
    ax.set_ylabel('Average Flushing Operations', fontsize=12)
    ax.set_title('Average Flushing Operations by Tree Height\n(With vs Without Scaling Method)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(heights)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add reduction percentage labels
    for i, (h, wo, w) in enumerate(zip(heights, flushing_wo, flushing_w)):
        if wo > 0:
            reduction = ((wo - w) / wo) * 100
            ax.annotate(f'-{reduction:.1f}%',
                       xy=(i, max(wo, w) + 1),
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplot_flushing_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'barplot_flushing_comparison.pdf'))
    print(f"✓ Flushing comparison bar plot saved to {output_dir}/barplot_flushing_comparison.png")
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main function to run the comprehensive data analysis for Miku_v1 paper.
    """
    print("=" * 80)
    print("COMPREHENSIVE DATA ANALYSIS FOR MIKU_V1 PAPER")
    print("Programmable Microfluidic Devices - Scaling Method Analysis")
    print("=" * 80)
    
    # Import trees from trees_data
    try:
        from trees_data.trees_data import get_all_trees
        all_trees = get_all_trees()
        print(f"\n✓ Loaded {len(all_trees)} mixing trees from trees_data")
    except ImportError as e:
        print(f"Error importing trees: {e}")
        print("Make sure trees_data/trees_data.py exists")
        print("Run: python trees_data/generate_trees.py to generate the dataset")
        return None, None
    
    # Create output directory
    output_dir = './analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # 1. DATA CLEANING & SEGMENTATION
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 1: Data Loading & Segmentation")
    print("-" * 80)
    
    start_time = datetime.datetime.now()
    
    # Comprehensive analysis of all trees
    print(f"Analyzing {len(all_trees)} trees...")
    analysis_results = []
    for i, tree in enumerate(all_trees):
        result = analyze_tree_comprehensive(tree, tree_index=i)
        analysis_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_trees)} trees")
    
    valid_results = [r for r in analysis_results if 'error' not in r]
    print(f"\n✓ Analyzed {len(valid_results)}/{len(all_trees)} trees successfully")
    
    # Display height distribution
    height_dist = defaultdict(int)
    for r in valid_results:
        height_dist[r['height']] += 1
    print(f"\nTree Height Distribution:")
    for h in sorted(height_dist.keys()):
        print(f"  Height {h}: {height_dist[h]} trees")
    
    # ============================================================
    # 2. REPLICATION OF TABLE 2 (Descriptive Statistics)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 2: Replication of Table 2 (Descriptive Statistics)")
    print("-" * 80)
    
    table2 = generate_table2_statistics(analysis_results)
    print_table2(table2)
    
    # Check paper claim for height 5 (~81.34% reduction)
    if 5 in table2:
        flushing_red_h5 = table2[5]['avg_flushing_reduction_pct']
        print(f"\n📋 PAPER CLAIM CHECK (Height 5):")
        print(f"   Paper claims ~81.34% flushing reduction for height 5")
        print(f"   Our result: {flushing_red_h5:.2f}% reduction")
        diff = abs(flushing_red_h5 - 81.34)
        if diff < 10:
            print(f"   ✅ Claim validated! (Difference: {diff:.2f}%)")
        else:
            print(f"   ⚠️ Deviation from paper claim: {diff:.2f}%")
            print(f"      (May be due to different tree samples or generation parameters)")
    else:
        print(f"\n⚠️ No height 5 trees found for paper claim validation")
    
    # ============================================================
    # 3. TIME ESTIMATION ANALYSIS (Figure 11)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 3: Time Estimation Analysis (Replicating Figure 11)")
    print("-" * 80)
    
    TIME_PER_FLUSH = 15.2  # seconds (2 × 7.6 seconds)
    print(f"Time per flushing operation: {TIME_PER_FLUSH} seconds (2 × 7.6s loading/wash)")
    
    time_data = calculate_time_estimation(analysis_results, TIME_PER_FLUSH)
    
    print(f"\n{'Height':<10} {'N':<8} {'Avg Time W.O. (s)':<20} {'Avg Time W. (s)':<18} {'Reduction %':<12}")
    print("-" * 70)
    for height in sorted([h for h in time_data.keys()]):
        data = time_data[height]
        print(f"{height:<10} {data['num_trees']:<8} {data['avg_time_wo_scaling']:<20.2f} "
              f"{data['avg_time_w_scaling']:<18.2f} {data['time_reduction_pct']:<12.2f}")
    
    # Generate Figure 11
    plot_figure11(time_data, output_dir)
    
    # ============================================================
    # 4. HYPOTHESIS VALIDATION (Correlation)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 4: Hypothesis Validation (Correlation Analysis)")
    print("-" * 80)
    
    correlation_results = calculate_correlation(analysis_results)
    
    print("\n📊 CORRELATION ANALYSIS:")
    print(f"\n1. Overlaps (W.O.) vs Flushing (W.O.) - Raw Data:")
    print(f"   Pearson r = {correlation_results['raw_correlation']['pearson_r']:.4f}")
    print(f"   p-value = {correlation_results['raw_correlation']['p_value']:.2e}")
    print(f"   Interpretation: {correlation_results['raw_correlation']['interpretation']}")
    
    print(f"\n2. Overlap Reduction % vs Flushing Reduction %:")
    print(f"   Pearson r = {correlation_results['reduction_correlation']['pearson_r']:.4f}")
    print(f"   p-value = {correlation_results['reduction_correlation']['p_value']:.2e}")
    print(f"   Interpretation: {correlation_results['reduction_correlation']['interpretation']}")
    
    if correlation_results['raw_correlation']['pearson_r'] > 0.7:
        print("\n✅ HYPOTHESIS VALIDATED: Strong correlation confirms that")
        print("   'reducing overlaps is crucial in reducing flush operations'")
    
    # Generate scatter plot
    plot_scatter_correlation(correlation_results, output_dir)
    
    # ============================================================
    # 5. ADVANCED DISTRIBUTION ANALYSIS
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 5: Advanced Distribution Analysis")
    print("-" * 80)
    
    # Box plots
    print("\nGenerating box plots for flushing distribution...")
    plot_box_plots(analysis_results, output_dir)
    
    # Zero-flush success rate
    zero_flush_stats = calculate_zero_flush_rate(analysis_results)
    print_zero_flush_stats(zero_flush_stats)
    
    # Additional bar plot
    plot_combined_barplot(table2, output_dir)
    
    # ============================================================
    # 6. LOADING ANALYSIS
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 6: Loading Operations Analysis")
    print("-" * 80)
    
    # Print Loading Table (Table 2B)
    print_loading_table(table2)
    
    # Loading time estimation
    TIME_PER_LOADING = 7.6  # seconds per loading operation
    print(f"\n📊 LOADING TIME ESTIMATION:")
    print(f"Time per loading operation: {TIME_PER_LOADING} seconds")
    
    loading_time_data = calculate_loading_time_estimation(analysis_results, TIME_PER_LOADING)
    
    print(f"\n{'Height':<10} {'N':<8} {'Avg Time W.O. (s)':<20} {'Avg Time W. (s)':<18} {'Reduction %':<12}")
    print("-" * 70)
    for height in sorted([h for h in loading_time_data.keys()]):
        data = loading_time_data[height]
        print(f"{height:<10} {data['num_trees']:<8} {data['avg_loading_time_wo_scaling']:<20.2f} "
              f"{data['avg_loading_time_w_scaling']:<18.2f} {data['loading_time_reduction_pct']:<12.2f}")
    
    # Generate loading plots
    print("\nGenerating loading operation visualizations...")
    plot_loading_time(loading_time_data, output_dir)
    plot_combined_operations(table2, output_dir)
    plot_total_operations(table2, output_dir)
    plot_loading_box_plots(analysis_results, output_dir)
    
    # ============================================================
    # SUMMARY & EXPORT
    # ============================================================
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"\nOutput files saved to: {output_dir}/")
    print("\n📊 Flushing Analysis Plots:")
    print("  • figure11_flushing_time.png/pdf")
    print("  • scatter_overlaps_vs_flushing.png/pdf")
    print("  • boxplot_flushing_distribution.png/pdf")
    print("  • barplot_flushing_comparison.png/pdf")
    print("\n📊 Loading Analysis Plots:")
    print("  • loading_time_comparison.png/pdf")
    print("  • combined_operations.png/pdf")
    print("  • total_operations.png/pdf")
    print("  • boxplot_loading_distribution.png/pdf")
    
    # ============================================================
    # 7. EXPORT TABLES (CSV & HTML)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 7: Exporting Tables (CSV & HTML)")
    print("-" * 80)
    
    # Export CSV tables
    export_table2_csv(table2, output_dir)
    
    # Export HTML report
    export_table2_html(table2, time_data, loading_time_data, output_dir)
    
    # Export per-tree analysis CSV
    export_per_tree_csv(analysis_results, output_dir)
    
    print("\n📋 Table Files Generated:")
    print("  • table2a_flushing_stats.csv")
    print("  • table2b_loading_stats.csv")
    print("  • analysis_summary.csv")
    print("  • per_tree_analysis.csv")
    print("  • analysis_results.html (Interactive Report)")
    
    return analysis_results, table2


def print_per_tree_breakdown(results, max_trees=10):
    """
    Print detailed breakdown for individual trees.
    """
    valid = [r for r in results[:max_trees] if 'error' not in r]
    
    if not valid:
        return
    
    print(f"\n{'='*80}")
    print(f"PER-TREE BREAKDOWN (First {len(valid)} trees)")
    print(f"{'='*80}")
    print(f"{'Tree':<6} {'Height':<8} {'Mixers':<8} {'Flush WO':<10} {'Flush W':<10} {'Reduction %':<12}")
    print("-" * 60)
    
    for r in valid:
        idx = r['tree_index']
        height = r['height']
        mixers = r['mixer_count']
        flush_wo = r['flushing_wo_scaling']
        flush_w = r['flushing_w_scaling']
        reduction = r['flushing_reduction_pct']
        
        print(f"{idx:<6} {height:<8} {mixers:<8} {flush_wo:<10} {flush_w:<10} {reduction:<12.2f}")
    
    print("-" * 60)


if __name__ == "__main__":
    results, stats = main()
    if results:
        print_per_tree_breakdown(results[:10])
