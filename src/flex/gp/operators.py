import random


def _compute_node_depths(individual):
    """Compute depth for each node in a prefix-encoded GP tree."""
    depths = []
    # Stack contains the number of children left to visit at each level.
    pending_children = []

    for node in individual:
        depth = len(pending_children)
        depths.append(depth)

        # Consume one expected child from the current parent (if any).
        if pending_children:
            pending_children[-1] -= 1
            while pending_children and pending_children[-1] == 0:
                pending_children.pop()

        # Add this node's children to the stack.
        arity = getattr(node, "arity", 0)
        if arity > 0:
            pending_children.append(arity)

    return depths


def cxOnePointSameDepth(ind1, ind2):
    """One-point crossover constrained to matching node depth.

    This variant behaves like DEAP's ``gp.cxOnePoint``, but crossover points are
    sampled so that both trees swap subtrees rooted at the same depth. For
    strongly-typed trees, return type compatibility is also enforced.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    depths1 = _compute_node_depths(ind1)
    depths2 = _compute_node_depths(ind2)

    # (ret_type, depth) -> candidate node indices, excluding root index 0.
    types_depths1 = {}
    for idx in range(1, len(ind1)):
        key = (getattr(ind1[idx], "ret", None), depths1[idx])
        types_depths1.setdefault(key, []).append(idx)

    types_depths2 = {}
    for idx in range(1, len(ind2)):
        key = (getattr(ind2[idx], "ret", None), depths2[idx])
        types_depths2.setdefault(key, []).append(idx)

    common_keys = list(set(types_depths1.keys()).intersection(types_depths2.keys()))
    if not common_keys:
        return ind1, ind2

    chosen_key = random.choice(common_keys)
    index1 = random.choice(types_depths1[chosen_key])
    index2 = random.choice(types_depths2[chosen_key])

    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2
