"""
Random Mixing Tree Generator

Generates 100 unique random mixing trees with the following constraints:
- Height <= 4
- Reagents: {R1, R2, R3, R4}
- Edge weights: random natural numbers from 1 to 10
- Fluid physics: mixer output volume <= sum of children volumes
"""

import sys
import os
import random
import json
import pickle


# Define node class locally to avoid pydot dependency during generation
class node(object):
    """
    Node class for mixing trees.
    Compatible with NTM.tree.node
    """
    def __init__(self, value, vol=1, children=None):
        self.value = value
        self.volume = vol
        self.children = children if children is not None else []
        self.hash = random.randint(1, 100000001)

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<{}>'.format(self.value)

REAGENTS = ['R1', 'R2', 'R3', 'R4']
MAX_HEIGHT = 7  # Increased to support height 7 trees
MAX_VOLUME = 10
MIN_VOLUME = 1
NUM_TREES = 600  # 100 per height (2-7)


def generate_random_tree(max_depth=5, current_depth=0, force_depth=False):
    """
    Recursively generate a random mixing tree.
    
    Args:
        max_depth: Maximum allowed depth (height) of the tree
        current_depth: Current depth in the recursion
        force_depth: If True, forces at least one path to reach max_depth
    
    Returns:
        A node object representing the tree/subtree
    """
    # At max depth, must be a leaf (reagent) node
    if current_depth >= max_depth:
        reagent = random.choice(REAGENTS)
        volume = random.randint(MIN_VOLUME, MAX_VOLUME)
        return node(reagent, vol=volume, children=[])
    
    # Decide if this should be a leaf or mixer node
    # Higher probability of leaf at deeper levels (but not if forcing depth)
    leaf_probability = current_depth / max_depth if not force_depth else 0.3
    
    if current_depth > 0 and random.random() < leaf_probability and not force_depth:
        # Create a leaf (reagent) node
        reagent = random.choice(REAGENTS)
        volume = random.randint(MIN_VOLUME, MAX_VOLUME)
        return node(reagent, vol=volume, children=[])
    
    # Create a mixer node with 2-4 children
    num_children = random.randint(2, 4)
    children = []
    
    # If forcing depth, ensure at least one child continues to max depth
    force_one_child = force_depth and (current_depth < max_depth - 1)
    
    for i in range(num_children):
        # Force the first child to continue if needed
        child_force = force_one_child and (i == 0)
        child = generate_random_tree(max_depth, current_depth + 1, force_depth=child_force)
        children.append(child)
    
    # Calculate sum of children volumes
    total_children_volume = sum(child.volume for child in children)
    
    # Mixer output volume must be <= sum of children volumes and <= MAX_VOLUME
    max_allowed_volume = min(total_children_volume, MAX_VOLUME)
    volume = random.randint(MIN_VOLUME, max_allowed_volume)
    
    return node(f'M{current_depth}', vol=volume, children=children)


def tree_to_dict(n):
    """
    Convert a node tree to a dictionary representation for serialization.
    """
    return {
        'value': n.value,
        'volume': n.volume,
        'children': [tree_to_dict(child) for child in n.children]
    }


def dict_to_tree(d):
    """
    Convert a dictionary representation back to a node tree.
    """
    n = node(d['value'], vol=d['volume'], children=[])
    n.children = [dict_to_tree(child) for child in d['children']]
    return n


def tree_to_code_string(n, var_name='tree', indent=0):
    """
    Convert a tree to Python code string that can be executed to recreate the tree.
    """
    prefix = '    ' * indent
    
    if not n.children:
        # Leaf node
        return f"node('{n.value}', vol={n.volume}, children=[])"
    else:
        # Mixer node
        children_code = ',\n'.join([
            prefix + '        ' + tree_to_code_string(child, indent=indent+1)
            for child in n.children
        ])
        return f"node('{n.value}', vol={n.volume}, children=[\n{children_code}\n{prefix}    ])"


def get_tree_signature(n):
    """
    Get a unique signature for a tree to check for duplicates.
    Uses structure + values + volumes.
    """
    if not n.children:
        return f"({n.value},{n.volume})"
    else:
        children_sigs = sorted([get_tree_signature(child) for child in n.children])
        return f"({n.value},{n.volume},[{','.join(children_sigs)}])"


def generate_unique_trees(num_trees=100, max_attempts=10000):
    """
    Generate a specified number of unique random trees.
    """
    trees = []
    signatures = set()
    attempts = 0
    
    while len(trees) < num_trees and attempts < max_attempts:
        tree = generate_random_tree(max_depth=MAX_HEIGHT)
        sig = get_tree_signature(tree)
        
        if sig not in signatures:
            signatures.add(sig)
            trees.append(tree)
            if len(trees) % 50 == 0:
                print(f"Generated tree {len(trees)}/{num_trees}")
        
        attempts += 1
    
    if len(trees) < num_trees:
        print(f"Warning: Could only generate {len(trees)} unique trees after {max_attempts} attempts")
    
    return trees


def generate_trees_by_height(trees_per_height=100, heights=[1, 2, 3, 4, 5], max_attempts_per_tree=100):
    """
    Generate a balanced set of trees with specific heights.
    
    Args:
        trees_per_height: Number of trees to generate for each height
        heights: List of heights to generate
        max_attempts_per_tree: Max attempts per tree generation
    
    Returns:
        dict: Trees grouped by height
    """
    trees_by_height = {h: [] for h in heights}
    signatures = set()
    
    for target_height in heights:
        print(f"\nGenerating {trees_per_height} trees with height {target_height}...")
        attempts = 0
        max_attempts = trees_per_height * max_attempts_per_tree
        
        while len(trees_by_height[target_height]) < trees_per_height and attempts < max_attempts:
            # Generate tree with max_depth = target_height, forcing depth
            tree = generate_random_tree(max_depth=target_height, force_depth=True)
            
            # Verify actual height
            actual_height = get_tree_height(tree)
            
            if actual_height == target_height:
                sig = get_tree_signature(tree)
                if sig not in signatures:
                    signatures.add(sig)
                    trees_by_height[target_height].append(tree)
                    if len(trees_by_height[target_height]) % 25 == 0:
                        print(f"  Height {target_height}: {len(trees_by_height[target_height])}/{trees_per_height}")
            
            attempts += 1
        
        actual_count = len(trees_by_height[target_height])
        if actual_count < trees_per_height:
            print(f"  Warning: Only generated {actual_count} trees for height {target_height}")
        else:
            print(f"  ✓ Generated {actual_count} trees for height {target_height}")
    
    return trees_by_height


def get_tree_height(tree):
    """Calculate tree height."""
    if not tree.children:
        return 0
    return 1 + max(get_tree_height(child) for child in tree.children)


def save_trees_as_python(trees, filepath):
    """
    Save trees as a Python file that can be imported directly.
    """
    with open(filepath, 'w') as f:
        f.write('"""\nAuto-generated mixing trees data.\n')
        f.write('100 unique random mixing trees with:\n')
        f.write('- Height <= 4\n')
        f.write('- Reagents: {R1, R2, R3, R4}\n')
        f.write('- Edge weights: 1-10\n')
        f.write('- Fluid physics constraint satisfied\n')
        f.write('"""\n\n')
        f.write('import sys\n')
        f.write('import os\n')
        f.write('sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Codes"))\n')
        f.write('from NTM.tree import node\n\n')
        
        f.write('def get_tree(index):\n')
        f.write('    """Get a specific tree by index (0-99)."""\n')
        f.write('    return TREES[index]\n\n')
        
        f.write('def get_all_trees():\n')
        f.write('    """Get all 100 trees as a list."""\n')
        f.write('    return TREES.copy()\n\n')
        
        f.write('# Tree definitions\n')
        f.write('TREES = [\n')
        
        for i, tree in enumerate(trees):
            code = tree_to_code_string(tree)
            f.write(f'    # Tree {i}\n')
            f.write(f'    {code},\n\n')
        
        f.write(']\n')
        
        f.write('\n# Number of trees available\n')
        f.write('NUM_TREES = len(TREES)\n')


def save_trees_as_json(trees, filepath):
    """
    Save trees as JSON for portability.
    """
    trees_data = [tree_to_dict(tree) for tree in trees]
    with open(filepath, 'w') as f:
        json.dump(trees_data, f, indent=2)


def save_trees_as_pickle(trees, filepath):
    """
    Save trees as pickle for direct Python object serialization.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(trees, f)


def load_trees_from_json(filepath):
    """
    Load trees from JSON file.
    """
    with open(filepath, 'r') as f:
        trees_data = json.load(f)
    return [dict_to_tree(d) for d in trees_data]


def load_trees_from_pickle(filepath):
    """
    Load trees from pickle file.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Balanced Mixing Trees Dataset")
    print(f"Constraints: Heights 2-{MAX_HEIGHT}, Reagents: {REAGENTS}")
    print(f"Volumes: {MIN_VOLUME}-{MAX_VOLUME}")
    print("=" * 60)
    
    # Generate balanced trees by height (100 per height, heights 2-7)
    TREES_PER_HEIGHT = 100
    trees_by_height = generate_trees_by_height(
        trees_per_height=TREES_PER_HEIGHT, 
        heights=[2, 3, 4, 5, 6, 7]
    )
    
    # Combine all trees
    all_trees = []
    for h in sorted(trees_by_height.keys()):
        all_trees.extend(trees_by_height[h])
    
    print(f"\n✓ Total trees generated: {len(all_trees)}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save as Python file (directly importable)
    python_path = os.path.join(script_dir, 'trees_data.py')
    save_trees_as_python(all_trees, python_path)
    print(f"\nSaved as Python file: {python_path}")
    
    # Save as JSON (portable format)
    json_path = os.path.join(script_dir, 'trees_data.json')
    save_trees_as_json(all_trees, json_path)
    print(f"Saved as JSON file: {json_path}")
    
    # Save as pickle (fast loading)
    pickle_path = os.path.join(script_dir, 'trees_data.pkl')
    save_trees_as_pickle(all_trees, pickle_path)
    print(f"Saved as pickle file: {pickle_path}")
    
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    for h in sorted(trees_by_height.keys()):
        print(f"  Height {h}: {len(trees_by_height[h])} trees")
    print("=" * 60)
    print("\nDone! You can now import trees using:")
    print("  from trees_data.trees_data import get_tree, get_all_trees, TREES")
    print("  tree = get_tree(0)  # Get first tree")
    print("  all_trees = get_all_trees()  # Get all trees")
