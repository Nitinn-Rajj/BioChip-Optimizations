"""
Scaling and Merging Algorithms for Mixing Trees

Based on the research paper:
"Reducing the Number of Flushing by Scaling Mixers on PMDs"

This module implements:
1. Scaling Algorithm - Convert Group A mixers to Group B by doubling volumes
2. Merging Algorithm - Remove redundant mixers where output == total_volume

Author: PMD Analysis Script
"""

from copy import deepcopy


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_output_to_parent(node):
    """
    Get the volume this node sends to its parent.
    This is simply the node's volume attribute.
    """
    return node.volume


def get_total_volume(node):
    """
    Get the total volume (sum of inputs) for a mixer node.
    For a mixer, this is the sum of children's volumes.
    For a reagent (leaf), this is its own volume.
    """
    if not node.children:
        return node.volume
    return sum(child.volume for child in node.children)


def is_mixer(node):
    """Check if a node is a mixer (has children)."""
    return len(node.children) > 0


def is_reagent(node):
    """Check if a node is a reagent/leaf (no children)."""
    return len(node.children) == 0


# ============================================================
# PART 1: SCALING ALGORITHM
# ============================================================

def is_group_a(node):
    """
    Check if a mixer node belongs to Group A.
    
    Condition: The node provides 3 or more droplets to its parent,
    AND the number of droplets provided is an ODD integer.
    
    Group A mixers are high-risk for causing overlaps.
    
    Args:
        node: A mixer node to check
    
    Returns:
        bool: True if the node is Group A, False otherwise
    """
    if not is_mixer(node):
        return False
    
    output_to_parent = get_output_to_parent(node)
    
    # Condition: output >= 3 AND output is ODD
    return output_to_parent >= 3 and output_to_parent % 2 == 1


def identify_group_a_nodes(tree):
    """
    Traverse the tree and identify all Group A mixer nodes.
    
    Args:
        tree: Root node of the mixing tree
    
    Returns:
        list: List of Group A mixer nodes (references)
    """
    group_a_nodes = []
    
    def traverse(node):
        if is_mixer(node):
            if is_group_a(node):
                group_a_nodes.append(node)
            for child in node.children:
                traverse(child)
    
    traverse(tree)
    return group_a_nodes


def scale_node(node, factor=2):
    """
    Apply scaling factor to a node.
    
    Multiplies the node's volume and all its children's volumes by the factor.
    This effectively doubles the mixer size.
    
    Args:
        node: The mixer node to scale
        factor: Scaling factor (default 2)
    """
    # Scale this node's output volume
    node.volume *= factor
    
    # Scale all children's volumes (inputs to this mixer)
    for child in node.children:
        child.volume *= factor


def propagate_constraints(node, parent_demand=None):
    """
    Propagate constraints down the tree after scaling.
    
    Constraint: A child mixer cannot provide more droplets to a parent
    than the child mixer's own total size.
    
    If the demand from parent exceeds child's capacity, scale the child.
    
    Args:
        node: Current node being checked
        parent_demand: The volume the parent expects from this node
    """
    if not is_mixer(node):
        # Reagent nodes don't have children to propagate to
        return
    
    # Check if this node can satisfy parent demand
    if parent_demand is not None:
        total_volume = get_total_volume(node)
        if parent_demand > total_volume:
            # Need to scale this node to meet demand
            # Calculate required scaling factor
            factor = 2
            while total_volume * factor < parent_demand:
                factor *= 2
            scale_node(node, factor)
    
    # Now propagate to children
    for child in node.children:
        if is_mixer(child):
            # The demand from this node to its child is child.volume
            child_demand = child.volume
            child_capacity = get_total_volume(child)
            
            if child_demand > child_capacity:
                # Scale the child
                factor = 2
                while child_capacity * factor < child_demand:
                    factor *= 2
                scale_node(child, factor)
            
            # Continue propagating down
            propagate_constraints(child, child.volume)


def apply_scaling(tree):
    """
    Apply the Scaling Algorithm to the tree.
    
    Step 1: Identify all Group A nodes
    Step 2: Scale them (multiply volumes by 2)
    Step 3: Propagate constraints downward
    
    Args:
        tree: Root node of the mixing tree (will be modified in-place)
    
    Returns:
        int: Number of nodes that were scaled
    """
    # Step 1: Identify Group A nodes
    group_a_nodes = identify_group_a_nodes(tree)
    
    # Step 2: Scale each Group A node
    for node in group_a_nodes:
        scale_node(node, factor=2)
    
    # Step 3: Propagate constraints from root downward
    # We need to check the entire tree for constraint violations
    def propagate_from_root(node):
        if not is_mixer(node):
            return
        
        for child in node.children:
            if is_mixer(child):
                # Check if child can provide what node expects
                child_demand = child.volume  # What parent expects
                child_capacity = get_total_volume(child)
                
                while child_demand > child_capacity:
                    # Scale the child
                    scale_node(child, factor=2)
                    child_capacity = get_total_volume(child)
                
                # Continue down the tree
                propagate_from_root(child)
    
    propagate_from_root(tree)
    
    return len(group_a_nodes)


# ============================================================
# PART 2: MERGING ALGORITHM
# ============================================================

def is_merge_candidate(node):
    """
    Check if a mixer node is a candidate for merging.
    
    Condition: output_to_parent == total_volume
    
    This means the mixer empties completely into the parent,
    so there's no need for a separate mixing step.
    
    Args:
        node: A mixer node to check
    
    Returns:
        bool: True if the node should be merged, False otherwise
    """
    if not is_mixer(node):
        return False
    
    output = get_output_to_parent(node)
    total = get_total_volume(node)
    
    return output == total


def merge_node_into_parent(parent, child_to_merge):
    """
    Merge a child node into its parent.
    
    Remove the child from parent's children list and add all of
    the child's children directly to the parent.
    
    Args:
        parent: The parent mixer node
        child_to_merge: The child node to be merged (removed)
    
    Returns:
        bool: True if merge was successful
    """
    if child_to_merge not in parent.children:
        return False
    
    # Get index of child to merge
    idx = parent.children.index(child_to_merge)
    
    # Remove the child
    parent.children.remove(child_to_merge)
    
    # Insert child's children at the same position
    for i, grandchild in enumerate(child_to_merge.children):
        parent.children.insert(idx + i, grandchild)
    
    return True


def apply_merging(tree):
    """
    Apply the Merging Algorithm to the tree.
    
    Iterate through the tree and merge nodes where output == total_volume.
    Continue until no more merges are possible.
    
    Args:
        tree: Root node of the mixing tree (will be modified in-place)
    
    Returns:
        int: Number of nodes that were merged
    """
    merge_count = 0
    changed = True
    
    while changed:
        changed = False
        
        def find_and_merge(node):
            nonlocal changed, merge_count
            
            if not is_mixer(node):
                return
            
            # Check each child for merge candidacy
            children_to_check = list(node.children)  # Copy to avoid modification during iteration
            
            for child in children_to_check:
                if is_mixer(child) and is_merge_candidate(child):
                    # Merge this child
                    if merge_node_into_parent(node, child):
                        merge_count += 1
                        changed = True
                        # Restart from this node since children changed
                        find_and_merge(node)
                        return
            
            # Continue to children
            for child in node.children:
                find_and_merge(child)
        
        find_and_merge(tree)
    
    return merge_count


# ============================================================
# MAIN TRANSFORMATION FUNCTION
# ============================================================

def count_mixers(tree):
    """Count total number of mixer nodes in a tree."""
    count = 0
    def traverse(node):
        nonlocal count
        if node.children:
            count += 1
            for child in node.children:
                traverse(child)
    traverse(tree)
    return count


def count_group_a(tree):
    """Count number of Group A mixers in a tree."""
    count = 0
    def traverse(node):
        nonlocal count
        if node.children:
            if is_group_a(node):
                count += 1
            for child in node.children:
                traverse(child)
    traverse(tree)
    return count


def transform_tree(original_tree):
    """
    Transform an original mixing tree using Scaling and Merging algorithms.
    
    Execution Flow:
    1. Deep copy the original tree
    2. Identify and scale Group A nodes (multiply volumes by 2)
    3. Propagate constraints downward (scale children if demand > capacity)
    4. Merge redundant nodes (output == total_volume)
    5. Return the modified tree
    
    Args:
        original_tree: Root node of the original mixing tree
    
    Returns:
        tuple: (modified_tree, stats_dict)
            - modified_tree: The transformed tree
            - stats_dict: Dictionary with transformation statistics
    """
    # Collect original stats before transformation
    original_mixer_count = count_mixers(original_tree)
    original_group_a = count_group_a(original_tree)
    original_total_volume = get_tree_total_reagent_volume(original_tree)
    
    # Step 1: Deep copy
    modified_tree = deepcopy(original_tree)
    
    # Step 2 & 3: Apply Scaling Algorithm
    num_scaled = apply_scaling(modified_tree)
    
    # Step 4: Apply Merging Algorithm
    num_merged = apply_merging(modified_tree)
    
    # Collect modified stats after transformation
    modified_mixer_count = count_mixers(modified_tree)
    modified_group_a = count_group_a(modified_tree)
    modified_total_volume = get_tree_total_reagent_volume(modified_tree)
    
    # Collect statistics
    stats = {
        'nodes_scaled': num_scaled,
        'nodes_merged': num_merged,
        'original_mixer_count': original_mixer_count,
        'modified_mixer_count': modified_mixer_count,
        'group_a_original': original_group_a,
        'group_a_modified': modified_group_a,
        'original_total_volume': original_total_volume,
        'modified_total_volume': modified_total_volume,
    }
    
    return modified_tree, stats


def get_tree_total_reagent_volume(tree):
    """
    Calculate total reagent volume in a tree (sum of all leaf volumes).
    
    Args:
        tree: Root node of the tree
    
    Returns:
        int: Total reagent volume
    """
    total = 0
    
    def traverse(node):
        nonlocal total
        if not node.children:
            total += node.volume
        else:
            for child in node.children:
                traverse(child)
    
    traverse(tree)
    return total


# ============================================================
# UTILITY FUNCTIONS FOR DEBUGGING
# ============================================================

def print_tree_info(tree, label="Tree"):
    """Print detailed information about a tree for debugging."""
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    
    def traverse(node, depth=0):
        indent = "  " * depth
        if not node.children:
            print(f"{indent}[R] {node.value}: vol={node.volume}")
        else:
            total = get_total_volume(node)
            output = get_output_to_parent(node)
            group = "A" if is_group_a(node) else "B"
            merge = " [MERGE]" if is_merge_candidate(node) else ""
            print(f"{indent}[M] {node.value}: total={total}, out={output}, Group {group}{merge}")
            for child in node.children:
                traverse(child, depth + 1)
    
    traverse(tree)


if __name__ == "__main__":
    # Test with a simple example
    from tree import node
    
    # Create a test tree
    test_tree = node('M1', vol=3, children=[
        node('R1', vol=2, children=[]),
        node('R2', vol=3, children=[]),
        node('M2', vol=3, children=[
            node('R3', vol=2, children=[]),
            node('R4', vol=2, children=[])
        ])
    ])
    
    print_tree_info(test_tree, "Original Tree")
    
    modified, stats = transform_tree(test_tree)
    
    print_tree_info(modified, "Modified Tree")
    print(f"\nTransformation Stats: {stats}")
