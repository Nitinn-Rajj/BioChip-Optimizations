
"""
PMD Grid Placement Simulation

Based on the research paper:
"Reducing the Number of Flushing by Scaling Mixers on PMDs"

This module implements:
1. ECN (Estimated Cell Use Number) calculation - bottom-up metric for scheduling
2. PMD Grid representation with FREE / ACTIVE / DIRTY states
3. Mixing Window search for valid placements (Algorithm 1)
4. Full placement simulation with flush counting

Algorithm 1 (Section 4.3) is a geometric tiling puzzle:
- Root is placed at center and marked ACTIVE
- Children are placed in order of ECN (highest first)
- Each child must overlap with parent by exactly output_to_parent cells
- Siblings CANNOT share any cells
- If every combination touches DIRTY cells, a FLUSH is required
- After placing children, parent becomes DIRTY and children become ACTIVE

Author: PMD Analysis Script
"""

from copy import deepcopy
from collections import deque
import math


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_GRID_SIZE = 10  # Default PMD grid is 10x10


# ============================================================
# ECN CALCULATION (Phase 1: Pre-processing)
# ============================================================

def calculate_ecn(tree):
    """
    Calculate Estimated Cell Use Number (ECN) for each node in the tree.
    
    ECN is calculated bottom-up (from leaves to root):
    - For a reagent (leaf): ECN = volume
    - For a mixer: ECN = sum of all reagent volumes in subtree
    
    Args:
        tree: Root node of the mixing tree
    
    Returns:
        dict: Mapping from node id to its ECN value
    """
    ecn_values = {}
    
    def compute_ecn(node):
        if not node.children:
            ecn_values[id(node)] = node.volume
            return node.volume
        
        total_ecn = 0
        for child in node.children:
            child_ecn = compute_ecn(child)
            total_ecn += child_ecn
        
        ecn_values[id(node)] = total_ecn
        return total_ecn
    
    compute_ecn(tree)
    return ecn_values


def get_ecn(node, ecn_values):
    """Get ECN value for a node."""
    return ecn_values.get(id(node), 0)


# ============================================================
# PMD GRID CLASS
# ============================================================

class PMDGrid:
    """
    Represents the PMD (Programmable Microfluidic Device) grid.

    Each cell is in one of three states:
    - FREE: available
    - ACTIVE: currently occupied by the parent or its placed children
    - DIRTY: used by previous mixers and cannot be touched without flushing
    """

    def __init__(self, size=DEFAULT_GRID_SIZE):
        self.size = size
        self.active_cells = set()
        self.dirty_cells = set()
        self.mixer_cells = {}

    def is_valid_position(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def is_free(self, cell):
        return cell not in self.active_cells and cell not in self.dirty_cells

    def is_dirty(self, cell):
        return cell in self.dirty_cells

    def is_active(self, cell):
        return cell in self.active_cells

    def place_mixer(self, mixer_id, cells):
        cells_set = set(cells)
        self.mixer_cells[mixer_id] = cells_set
        self.active_cells |= cells_set
        # If a mixer is placed on a previously dirty cell after a flush,
        # it is now ACTIVE.
        self.dirty_cells -= cells_set

    def get_mixer_cells(self, mixer_id):
        return list(self.mixer_cells.get(mixer_id, set()))

    def flush(self, preserve_active=True):
        if preserve_active:
            self.dirty_cells.clear()
        else:
            self.active_cells.clear()
            self.dirty_cells.clear()
            self.mixer_cells.clear()

    def set_parent_dirty(self, parent_id, child_ids):
        parent_cells = self.mixer_cells.get(parent_id, set())
        child_cells = set()
        for cid in child_ids:
            child_cells |= self.mixer_cells.get(cid, set())

        # Parent cells not occupied by children become DIRTY
        to_dirty = parent_cells - child_cells
        self.dirty_cells |= to_dirty

        # ACTIVE cells should include children (not the parent anymore)
        self.active_cells -= parent_cells
        self.active_cells |= child_cells


# ============================================================
# MIXING WINDOW SEARCH
# ============================================================

def get_mixer_dimensions(volume):
    """Get all valid rectangular dimensions for a mixer of given volume."""
    if volume <= 0:
        return [(1, 1)]

    dimensions = set()
    for w in range(1, int(math.sqrt(volume)) + 1):
        if volume % w == 0:
            h = volume // w
            dimensions.add((w, h))
            dimensions.add((h, w))

    dims_list = list(dimensions)
    dims_list.sort(key=lambda d: abs(d[0] - d[1]) + max(d))
    return dims_list


def find_root_placement(grid, volume):
    """Find placement for the root mixer at or near center of grid."""
    center = grid.size // 2
    dimensions = get_mixer_dimensions(volume)

    for width, height in dimensions:
        start_row = max(0, min(center - height // 2, grid.size - height))
        start_col = max(0, min(center - width // 2, grid.size - width))

        cells = []
        valid = True

        for r in range(height):
            for c in range(width):
                row, col = start_row + r, start_col + c
                if not grid.is_valid_position(row, col):
                    valid = False
                    break
                if not grid.is_free((row, col)):
                    valid = False
                    break
                cells.append((row, col))
            if not valid:
                break

        if valid and len(cells) == volume:
            return cells

    return None


def find_valid_placements(grid, child_volume, parent_cells, overlap_required, allow_dirty=False):
    """
    Return all valid placements for a child mixer (Algorithm 1).

    A valid placement must:
    1. Fit within grid bounds (exact rectangle area == child_volume)
    2. Overlap with parent by exactly overlap_required cells
    3. Never overlap ACTIVE cells that are not the parent
    4. DIRTY cells are allowed only when allow_dirty=True (flagged for flush cost)

    Returns list of tuples: (cells, touches_dirty)
    """
    valid_placements = []
    parent_cells_set = set(parent_cells)
    dimensions = get_mixer_dimensions(child_volume)

    for width, height in dimensions:
        for start_row in range(grid.size - height + 1):
            for start_col in range(grid.size - width + 1):
                cells = []
                valid = True
                touches_dirty = False

                for r in range(height):
                    for c in range(width):
                        row, col = start_row + r, start_col + c
                        cell = (row, col)

                        if grid.is_active(cell) and cell not in parent_cells_set:
                            valid = False
                            break

                        if grid.is_dirty(cell):
                            if not allow_dirty:
                                valid = False
                                break
                            touches_dirty = True

                        cells.append(cell)

                    if not valid:
                        break

                if not valid:
                    continue

                cells_set = set(cells)
                overlap_cells = cells_set & parent_cells_set
                if len(overlap_cells) != overlap_required:
                    continue

                valid_placements.append((cells, touches_dirty))

    return valid_placements


# ============================================================
# PLACEMENT SIMULATION
# ============================================================

def get_mixer_total_volume(node):
    """Get the total volume (size) of a mixer = sum of children volumes."""
    if not node.children:
        return node.volume
    return sum(child.volume for child in node.children)


def simulate_placement(tree, grid_size=DEFAULT_GRID_SIZE, debug=False):
    """
    Run the placement simulation to count flush operations.

    Implements Algorithm 1 (Section 4.3) exactly:
    - Uses FREE / ACTIVE / DIRTY grid states
    - Places root at center
    - Places children in ECN order
    - Finds all valid placements for each child
    - Searches a clean combination (no DIRTY overlap, no sibling overlap)
    - Flushes only when DIRTY prevents a clean placement
    """
    grid = PMDGrid(grid_size)
    flush_count = 0
    placements_made = 0

    ecn_values = calculate_ecn(tree)

    root_volume = get_mixer_total_volume(tree)
    root_cells = find_root_placement(grid, root_volume)
    if root_cells is None:
        if debug:
            print(f"Warning: Cannot place root (volume={root_volume}) on {grid_size}x{grid_size} grid")
        return {
            'flush_count': 1,
            'placement_flushes': 1,
            'total_flushes': 1,
            'placements_made': 0,
            'grid_size': grid_size
        }

    root_id = id(tree)
    grid.place_mixer(root_id, root_cells)
    placements_made += 1

    if debug:
        print(f"Placed root {tree.value} at cells: {root_cells}")
        print(f"Root volume: {root_volume}")

    process_queue = deque([tree])

    while process_queue:
        parent = process_queue.popleft()
        parent_id = id(parent)
        parent_cells = grid.get_mixer_cells(parent_id)

        if not parent_cells:
            continue

        children_to_place = [c for c in parent.children if c.children]

        if not children_to_place:
            continue

        children_to_place.sort(key=lambda c: get_ecn(c, ecn_values), reverse=True)

        if debug:
            print(f"\nProcessing parent {parent.value}, children: {[c.value for c in children_to_place]}")

        def compute_candidates(allow_dirty=False):
            candidates = []
            for child in children_to_place:
                child_volume = get_mixer_total_volume(child)
                overlap_required = child.volume
                placements = find_valid_placements(
                    grid, child_volume, parent_cells, overlap_required, allow_dirty=allow_dirty
                )
                candidates.append(placements)
                if debug:
                    print(
                        f"  Child {child.value}: volume={child_volume}, overlap_required={overlap_required}, placements={len(placements)}"
                    )
            return candidates

        def find_best_combination(candidates):
            if any(len(c) == 0 for c in candidates):
                return None

            best = None
            best_cost = None

            chosen_by_index = {}
            order = sorted(range(len(candidates)), key=lambda i: len(candidates[i]))

            def has_overlap(placements):
                for i in range(len(placements)):
                    for j in range(i + 1, len(placements)):
                        if set(placements[i]) & set(placements[j]):
                            return True
                return False

            def backtrack(order_idx, touches_dirty_any):
                nonlocal best, best_cost
                if order_idx == len(order):
                    ordered = [chosen_by_index[i] for i in range(len(candidates))]
                    overlap_cost = 1 if has_overlap(ordered) else 0
                    dirty_cost = 1 if touches_dirty_any else 0
                    cost = dirty_cost + overlap_cost
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best = (ordered, touches_dirty_any, overlap_cost)
                    return

                child_idx = order[order_idx]
                for placement, touches_dirty in candidates[child_idx]:
                    chosen_by_index[child_idx] = placement
                    backtrack(order_idx + 1, touches_dirty_any or touches_dirty)
                    chosen_by_index.pop(child_idx, None)

            backtrack(0, False)
            return best

        candidates = compute_candidates(allow_dirty=True)
        best = find_best_combination(candidates)

        if best is None:
            flush_count += 1
            if debug:
                print("  IMPOSSIBLE: no valid placements for this parent.")
            continue

        chosen, touches_dirty, overlap_cost = best

        if touches_dirty:
            flush_count += 1
            if debug:
                print("  FLUSH REQUIRED: dirty cells block clean placement. Flushing DIRTY cells.")
            grid.flush(preserve_active=True)
            candidates = compute_candidates(allow_dirty=False)
            best = find_best_combination(candidates)
            if best is None:
                flush_count += 1
                if debug:
                    print("  IMPOSSIBLE AFTER FLUSH: grid too small or geometry infeasible.")
                continue
            chosen, _, overlap_cost = best

        if overlap_cost > 0:
            flush_count += 1
            if debug:
                print("  SIBLING OVERLAP COST: 1")

        # Place chosen children
        child_ids = []
        for child, placement in zip(children_to_place, chosen):
            child_id = id(child)
            grid.place_mixer(child_id, placement)
            child_ids.append(child_id)
            placements_made += 1
            process_queue.append(child)

        # Parent becomes DIRTY; children become ACTIVE
        grid.set_parent_dirty(parent_id, child_ids)

    return {
        'flush_count': flush_count,
        'placement_flushes': flush_count,
        'total_flushes': flush_count,
        'placements_made': placements_made,
        'grid_size': grid_size
    }


# ============================================================
# ADDITIONAL METRICS
# ============================================================

def count_volume_based_flushing(tree):
    """Count flushing based on volume waste at each mixer."""
    total_waste = 0
    
    def traverse(node):
        nonlocal total_waste
        if node.children:
            input_volume = sum(child.volume for child in node.children)
            output_volume = node.volume
            waste = input_volume - output_volume
            if waste > 0:
                total_waste += waste
            for child in node.children:
                traverse(child)
    
    traverse(tree)
    return total_waste


def count_loading_operations(tree):
    """Count loading operations as the number of mixer nodes."""
    return count_mixers(tree)


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


def count_group_a_mixers(tree):
    """Count number of Group A mixers (odd output >= 3)."""
    count = 0
    def traverse(n):
        nonlocal count
        if n.children:
            if n.volume >= 3 and n.volume % 2 == 1:
                count += 1
            for child in n.children:
                traverse(child)
    traverse(tree)
    return count


def analyze_tree(tree, grid_size=DEFAULT_GRID_SIZE, debug=False):
    """Run complete analysis on a tree."""
    sim_results = simulate_placement(tree, grid_size, debug=debug)
    loading = count_loading_operations(tree)
    volume_waste = count_volume_based_flushing(tree)
    num_mixers = count_mixers(tree)
    num_reagents = count_reagents(tree)
    group_a_count = count_group_a_mixers(tree)
    
    return {
        'flush_count': sim_results['flush_count'],
        'placements_made': sim_results['placements_made'],
        'volume_waste': volume_waste,
        'loading_count': loading,
        'num_mixers': num_mixers,
        'num_reagents': num_reagents,
        'group_a_count': group_a_count,
        'grid_size': grid_size
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    from tree import node
    
    # Create a test tree with multiple Group A children competing for space
    test_tree = node('M1', vol=2, children=[
        node('M2', vol=3, children=[  # Group A: needs 3-cell overlap
            node('R1', vol=2, children=[]),
            node('R2', vol=2, children=[]),
        ]),
        node('M3', vol=3, children=[  # Group A: needs 3-cell overlap  
            node('R3', vol=2, children=[]),
            node('R4', vol=2, children=[]),
        ]),
        node('M4', vol=3, children=[  # Group A: needs 3-cell overlap
            node('R5', vol=2, children=[]),
            node('R6', vol=2, children=[]),
        ]),
    ])
    
    print("="*60)
    print("TEST: 3 GROUP A SIBLINGS COMPETING FOR SPACE")
    print("="*60)
    print(f"Parent total volume: {sum(c.volume for c in test_tree.children)}")
    print("Each child needs 3-cell overlap (Group A)")
    print()
    
    result = simulate_placement(test_tree, debug=True)
    print()
    print(f"Flush count: {result['flush_count']}")
