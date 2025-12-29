import collections
import sys
from z3 import *


# ==========================================
# PART 1: DATA STRUCTURES & TREE UTILS
# ==========================================

class MixingNode:
    def __init__(self, name, id_num, children=None, edge_weight=1):
        self.name = name
        self.id = id_num
        self.children = children if children else []
        self.edge_weight = edge_weight
        self.parent = None
        
        # Scheduling properties
        self.subtree_weight = 0
        self.coords = None       # (x, y)
        self.quadrant = 0
        
        # Link children to self
        for child in self.children:
            child.parent = self

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"{self.name} (w={self.edge_weight})"

def construct_tree_from_nested_list(structure, counter=None):
    """
    Recursively builds a MixingNode tree from a nested list.
    Format: [Name, Weight, [Child1, Child2, ...]]
    Or:     [Name, [Child1, Child2, ...]] (Default Weight=1)
    Leaf:   [Name, Weight] or just Name (Default Weight=1)
    """
    if counter is None:
        counter = [0]
        
    # Case 1: Simple String Leaf "R1"
    if isinstance(structure, str):
        return MixingNode(structure, -1, edge_weight=1)
    
    # Parse Name
    name = structure[0]
    
    # Parse Weight and Children
    weight = 1
    children_data = []
    
    if len(structure) == 2:
        # Could be [Name, Weight] (Leaf) or [Name, Children] (Node)
        if isinstance(structure[1], (int, float)):
            weight = structure[1]
        elif isinstance(structure[1], list):
            children_data = structure[1]
    elif len(structure) == 3:
        # [Name, Weight, Children]
        weight = structure[1]
        children_data = structure[2]
    
    children_nodes = [construct_tree_from_nested_list(c, counter) for c in children_data]
    
    counter[0] += 1
    return MixingNode(name, counter[0], children_nodes, edge_weight=weight)

def parse_ratio_string(ratio_str):
    """
    Parses a ratio string (e.g., "2:4:6") into a list of normalized float ratios.
    Example: "2:4:6" -> [2/12, 4/12, 6/12] -> [0.166, 0.333, 0.5]
    """
    try:
        parts = [int(x) for x in ratio_str.split(':')]
        total = sum(parts)
        if total == 0:
            raise ValueError("Sum of ratios cannot be zero.")
        return [x / total for x in parts]
    except ValueError as e:
        print(f"Error parsing ratio string '{ratio_str}': {e}")
        sys.exit(1)

def print_tree(node, prefix="", is_last=True):
    """Prints the tree structure in a visual format."""
    connector = "└── " if is_last else "├── "
    print(prefix + connector + str(node))
    
    prefix += "    " if is_last else "│   "
    
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        is_last_child = (i == child_count - 1)
        print_tree(child, prefix, is_last_child)

def gen_mixing_tree_from_ratio(target_ratios, depth=4):
    """
    Generates a Standard Mixing Tree using Base-4 logic.
    Ref: Paper 2 (2020), Fig 3a logic.
    """
    total_units = 4**depth
    integer_ratios = [int(r * total_units) for r in target_ratios]
    
    # Fix rounding errors
    diff = total_units - sum(integer_ratios)
    integer_ratios[0] += diff
    
    # Create leaf queue
    leaf_queue = collections.deque()
    for r_idx, count in enumerate(integer_ratios):
        for _ in range(count):
            leaf_queue.append(MixingNode(f"R{r_idx+1}", -1))
            
    # Build tree bottom-up
    layer_nodes = list(leaf_queue)
    node_counter = 1
    
    while len(layer_nodes) > 1:
        next_layer = []
        for i in range(0, len(layer_nodes), 4):
            children = layer_nodes[i : i+4]
            if not children: break
            
            new_node = MixingNode(f"M{node_counter}", node_counter, children)
            node_counter += 1
            next_layer.append(new_node)
        layer_nodes = next_layer
        
    return layer_nodes[0]

# ==========================================
# PART 2: 2020 BASELINE (NTM + HDA)
# ==========================================

def calculate_weights(node):
    """Calculates subtree weight for Left Factoring."""
    if node.is_leaf():
        node.subtree_weight = 1
        return 1
    w = 1 + sum(calculate_weights(c) for c in node.children)
    node.subtree_weight = w
    return w

def left_factoring(node):
    """
    [cite_start]Algorithm: Left Factoring [cite: 712-713]
    Sorts heavy branches to the left.
    """
    if node.is_leaf(): return
    node.children.sort(key=lambda x: x.subtree_weight, reverse=True)
    for child in node.children:
        left_factoring(child)

def hda_heuristic(node):
    """
    [cite_start]Algorithm: HDA Heuristic [cite: 745-746]
    Simple check to balance heavy branches.
    """
    if node.is_leaf() or len(node.children) < 2: return
    
    left = node.children[0]
    right = node.children[-1]
    
    # If left is >2x heavier, move a grandchild to right
    if left.subtree_weight > 2 * right.subtree_weight:
        if not left.is_leaf() and len(left.children) > 0:
            heavy_grandchild = left.children[0]
            left.children.pop(0)
            right.children.append(heavy_grandchild)
            heavy_grandchild.parent = right
            calculate_weights(node) # Recalculate

    for child in node.children:
        hda_heuristic(child)

def get_quadrant(x, y):
    if x >= 0 and y >= 0: return 1
    if x < 0 and y >= 0: return 2
    if x < 0 and y < 0: return 3
    if x >= 0 and y < 0: return 4
    return 0

def ntm_placement(root, fpva_n):
    """
    [cite_start]Algorithm: NTM Placement [cite: 730-734]
    Places modules on an N x N grid using BFS.
    """
    # Calculate boundary (e.g., if N=10, limit is +/- 5)
    limit = fpva_n // 2
    
    # 1. Initialize Root at Center
    root.coords = (0,0)
    root.quadrant = 1
    
    queue = collections.deque([root])
    occupied = set([(0,0)])
    
    print(f"\n[NTM] Placing Modules on {fpva_n}x{fpva_n} Grid (Limit: +/-{limit})...")
    
    while queue:
        parent = queue.popleft()
        if parent.is_leaf(): continue
        
        px, py = parent.coords
        
        # Valid moves (Adjacent cells)
        moves = [
            (px+1, py), (px-1, py), (px, py+1), (px, py-1),
            (px+1, py+1), (px-1, py-1), (px+1, py-1), (px-1, py+1)
        ]
        
        # Check available spots (Empty AND Inside Bounds)
        available_spots = []
        for m in moves:
            mx, my = m
            if m not in occupied:
                if abs(mx) <= limit and abs(my) <= limit:
                    available_spots.append(m)
        
        # Place children
        for child in parent.children:
            if child.is_leaf(): continue
            
            if not available_spots:
                print(f"  [!] CRITICAL FAILURE: Gridlock! No valid spot for {child.name} near {parent.coords}")
                print(f"      (The 2020 parallel tree is too big for this chip size)")
                return
            
            # Heuristic: Try to find disjoint quadrant
            best_spot = available_spots[0]
            parent_q = get_quadrant(px, py)
            
            for spot in available_spots:
                if get_quadrant(spot[0], spot[1]) != parent_q:
                    best_spot = spot
                    break
            
            # Bind
            child.coords = best_spot
            child.quadrant = get_quadrant(best_spot[0], best_spot[1])
            occupied.add(best_spot)
            available_spots.remove(best_spot)
            
            print(f"  > Placed {child.name} at {child.coords} (Quad {child.quadrant})")
            queue.append(child)

def run_simulation(root):
    """Simulates the NTM schedule [cite: 720-721]"""
    time = 0
    finished_nodes = set()
    all_mixers = []
    
    q = [root]
    while q:
        curr = q.pop(0)
        if not curr.is_leaf():
            all_mixers.append(curr)
            q.extend(curr.children)
            
    print(f"\n[NTM] Simulation Start:")
    
    # Loop until root is finished
    while root not in finished_nodes:
        time += 1
        active = []
        
        for node in all_mixers:
            if node in finished_nodes: continue
            
            # Check inputs ready
            ready = True
            for child in node.children:
                if not child.is_leaf() and child not in finished_nodes:
                    ready = False
                    break
            if ready:
                active.append(node)
        
        # Execute
        for node in active:
            finished_nodes.add(node)
            
        if active:
            print(f"  Time {time}: Mixing {[n.name for n in active]}")
        
        if time > 50: # Safety break
            print("  [!] Simulation stuck or taking too long.")
            break
            
    if root in finished_nodes:
        print(f"[NTM] Protocol Complete. Time: {time} cycles. Cells Used: {len(finished_nodes)*4}")
    else:
        print(f"[NTM] Protocol Failed.")

# ==========================================
# PART 3: 2025 OPTIMIZATION (SMT SKEWED TREE)
# ==========================================

def solve_skewed_tree_z3(target_ratios, fpva_size):
    """
    [cite_start]Algorithm: AdaptiveMixingTree [cite: 267-276]
    Uses Z3 to find a vertical stack recipe.
    """
    print(f"\n[SMT] Searching for Skewed Tree (2025 Approach)...")
    
    # Boundary Check: 2025 Algo only needs a 2x2 mixer.
    if fpva_size < 2:
        print("Error: FPVA too small. Minimum size 2x2 required.")
        return

    k = len(target_ratios)
    epsilon = 0.01
    
    # Search for depth d from 2 to 9
    for d in range(2, 10):
        solver = Solver()
        
        # --- Variables ---
        # r[i][j]: Amount of Reagent j added at level i
        r = [[Int(f'r_{i}_{j}') for j in range(k)] for i in range(d)]
        # W[i]: Fluid shared from level i+1 to level i
        W = [Int(f'W_{i}') for i in range(d)]
        # R_accum[i][j]: Accumulated concentration tracking
        R_accum = [[Int(f'R_acc_{i}_{j}') for j in range(k)] for i in range(d)]
        
        # --- Constraints ---
        for i in range(d):
            # [cite_start]Domain Constraints [cite: 251-252]
            # Sharing is 1-3 units (unless bottom leaf)
            if i < d-1: 
                solver.add(And(W[i] >= 1, W[i] <= 3))
            
            reagent_sum = 0
            for j in range(k):
                # Reagent inputs 0-3
                solver.add(And(r[i][j] >= 0, r[i][j] <= 3))
                reagent_sum += r[i][j]
            
            # [cite_start]Mixer Consistency: Sum must be 4 [cite: 253-260]
            if i == d-1: # Leaf
                solver.add(reagent_sum == 4)
            else: # Internal
                solver.add(reagent_sum + W[i] == 4)

        # [cite_start]Mixing Ratio Constraints (Recursive) [cite: 261-263]
        for i in reversed(range(d)):
            for j in range(k):
                if i == d-1:
                    solver.add(R_accum[i][j] == r[i][j])
                else:
                    # Scaling factor accounts for base-4 dilution
                    scale = 4 ** ((d-1)-i)
                    solver.add(R_accum[i][j] == (r[i][j] * scale) + (W[i] * R_accum[i+1][j]))
        
        # [cite_start]Final Target Constraint [cite: 266]
        total_units = 4**d
        for j in range(k):
            final = R_accum[0][j]
            target = int(target_ratios[j] * total_units)
            err = int(epsilon * total_units)
            solver.add(final >= target - err)
            solver.add(final <= target + err)
            
        # --- Solve & Extract Recipe ---
        if solver.check() == sat:
            print(f"  > Solution FOUND at Depth {d}!")
            print(f"  > Cells Used: 4 (Vertical Stack)")
            
            m = solver.model()
            
            print(f"\n  [RECIPE] Execution Plan (Bottom-Up):")
            print(f"  -------------------------------------")
            
            # Extract and print recipe from Level d-1 (Bottom) to 0 (Top)
            for i in reversed(range(d)):
                # Extract integer values from Z3 model
                inputs = [m.evaluate(r[i][j]).as_long() for j in range(k)]
                
                if i == d-1:
                    # Bottom Leaf
                    print(f"  STEP 1 (Level {i}):")
                    print(f"    - Load Mixer with Reagents: {inputs}")
                    print(f"    - MIX.")
                else:
                    # Internal Node
                    shared_from_prev = m.evaluate(W[i]).as_long()
                    print(f"  STEP {d-i} (Level {i}):")
                    print(f"    - Retain {shared_from_prev} units from previous mix.")
                    print(f"    - Add new Reagents: {inputs}")
                    print(f"    - MIX.")
            
            print(f"  -------------------------------------")
            print(f"  FINAL RESULT: Target ratio achieved.\n")
            return
            
    print("  > No solution found within depth limits.")

# ==========================================
# MAIN EXECUTION 
# ==========================================

if __name__ == "__main__":
    # --- USER SETTINGS ---
    # FPVA Size (NxN grid). 
    FPVA_N = 6 
    
    # Target Ratio Input
    ratio_input = "26:63:47:43:7:70"
    
    # Convert input string to normalized target vector
    target = parse_ratio_string(ratio_input)
    print(f"Target Ratios: {target}")
    
    # --- RUN 2020 BASELINE ---
    print(f"--- Running 2020 Baseline (NTM/HDA) on {FPVA_N}x{FPVA_N} Grid ---")
    
    # OPTION 1: Generate Tree from Ratios (Commented out)
    # tree_root = gen_mixing_tree_from_ratio(target, depth=3)
    
    # OPTION 2: Manual Tree Input
    # Define your tree structure here: [Name, Weight, [Child1, Child2, ...]]
    # Note: Root weight is typically ignored or set to 1.
    custom_tree_structure = [
        "M1", 1, [
            ["M2", 1, [["R1", 2], ["R2", 2]]],
            ["M4", 1, [
                ["M5", 1, [["R1", 2], ["R2", 2]]],
                ["M6", 1, [
                    ["M9", 1, [["R1", 2], ["R2", 2]]],
                    ["M10", 1, [["R2", 1], ["R3", 3]]],
                    ["M11", 1, [["R4", 3], ["R5", 1]]],
                    ["M12", 1, [["R5", 2], ["R6", 2]]]
                ]],
                ["M7", 1, [["R2", 1], ["R3", 3]]],
                ["M8", 1, [["R4", 2], ["R5", 1], ["R6", 1]]]
            ]],
            ["M3", 1, [["R3", 2], ["R4", 2]]],
            ["m1", 1, [["R6", 1], ["R6", 1], ["R6", 1], ["R6", 1]]]
        ]
    ]
    tree_root = construct_tree_from_nested_list(custom_tree_structure)
    
    print("\n[Tree Structure] Initial Input Tree:")
    print_tree(tree_root)

    calculate_weights(tree_root)
    left_factoring(tree_root)
    hda_heuristic(tree_root)
    
    print("\n[Tree Structure] HDA Optimized Tree:")
    print_tree(tree_root)
    
    # Run placement with explicit size limit
    ntm_placement(tree_root, FPVA_N)
    run_simulation(tree_root)
    
    # --- RUN 2025 OPTIMIZATION ---
    print(f"\n--- Running 2025 Optimization (SMT) ---")
    solve_skewed_tree_z3(target, FPVA_N)