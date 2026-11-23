from numbers import Integral
from textwrap import fill

import matplotlib.pyplot as plt


def plot_dtl_tree(model, feature_names, top_n=10, category_values=None):
    """Draw at most top_n branches, starting at the root."""
    if model.tree_ is None:
        raise ValueError("Fit the model before plotting the tree.")
    if not isinstance(top_n, Integral) or top_n < 0:
        raise ValueError("top_n must be a nonnegative integer.")
    feature_names = list(feature_names)
    if len(feature_names) != model.n_features_in_:
        raise ValueError("Provide one name per training feature.")
    category_values = category_values or {}

    nodes = [model.tree_]
    edges = []
    children = {0: []}
    # Visit the tree one level at a time, with heavier sibling branches first.
    for parent_id, node in enumerate(nodes):
        remaining = top_n - len(edges)
        if remaining == 0:
            break
        branches = sorted(node["children"].items(), key=lambda item: -item[1][1])
        for branch, (child, weight) in branches[:remaining]:
            child_id = len(nodes)
            nodes.append(child)
            children[parent_id].append(child_id)
            children[child_id] = []
            edges.append((parent_id, child_id, branch, weight))

    positions = {}
    leaf_count = _place_nodes(0, children, positions, depth=0, next_x=0)
    max_depth = -min(y for _, y in positions.values())
    fig, ax = plt.subplots(figsize=(max(7, 1.65 * leaf_count), 2.2 + 1.3 * max_depth))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.13)

    for parent_id, child_id, branch, weight in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        ax.plot([x1, x2], [y1, y2], color="#788a9a", linewidth=1.3, zorder=1)

        parent = nodes[parent_id]
        feature = feature_names[parent["feature"]]
        threshold = parent["threshold"]
        if threshold is not None:
            operator = "≤" if branch == "left" else ">"
            value = str(int(threshold)) if threshold == int(threshold) else str(threshold)
            condition = f"{operator} {value}"
        elif feature in category_values:
            condition = category_values[feature][int(branch)]
        else:
            condition = str(branch)

        ax.text(
            x1 + 0.62 * (x2 - x1), y1 + 0.62 * (y2 - y1),
            f"{fill(condition, width=15)}\n{weight:.0%}",
            ha="center", va="center", fontsize=9, color="#334155",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
            zorder=2,
        )

    for node_id, node in enumerate(nodes):
        title = "Daun"
        if node["children"]:
            title = feature_names[node["feature"]].replace("_", " ")
        distribution = "   ".join(
            f"{label}: {probability:.0%}"
            for label, probability in zip(model.classes_, node["probabilities"])
        )
        label = f"{fill(title, width=19)}\n{distribution}"
        hidden = len(node["children"]) - len(children[node_id])
        if hidden:
            label += f"\n+{hidden} cabang lain"

        x, y = positions[node_id]
        ax.text(
            x, y, label, ha="center", va="center", fontsize=10, color="#172c45",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#e8f1fa" if node["children"] else "#f1f5f9",
                "edgecolor": "#4e7096",
            },
            zorder=3,
        )

    ax.set_title(f"Pohon Keputusan C4.5 ({len(edges)} cabang)", fontweight="bold")
    ax.set_xlim(-0.65, leaf_count - 0.35)
    ax.set_ylim(-max_depth - 0.45, 0.4)
    ax.set_axis_off()
    fig.text(
        0.5, 0.04,
        "Persentase di simpul menunjukkan distribusi kelas. "
        "Bobot cabang dihitung terhadap simpul induknya.",
        ha="center", fontsize=9, color="#334155",
    )
    return fig


def _place_nodes(node_id, children, positions, depth, next_x):
    child_ids = children[node_id]
    if not child_ids:
        positions[node_id] = (next_x, -depth)
        return next_x + 1

    # Give each subtree its own horizontal space.
    for child_id in child_ids:
        next_x = _place_nodes(child_id, children, positions, depth + 1, next_x)
    first_x = positions[child_ids[0]][0]
    last_x = positions[child_ids[-1]][0]
    positions[node_id] = ((first_x + last_x) / 2, -depth)
    return next_x
