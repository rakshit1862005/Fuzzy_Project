"""Baseline explainer: a decision tree trained to copy the network.

This is the standard "global surrogate" in explainable AI. Comparing against
it at the same number of rules (tree leaves) answers the question
"why not just use a decision tree?".
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config


def fit_tree_surrogate(X, nn_labels, max_rules, seed=config.SEED) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_leaf_nodes=max(2, max_rules), min_samples_leaf=5, random_state=seed)
    return tree.fit(X, nn_labels)


def tree_complexity(tree: DecisionTreeClassifier) -> dict:
    """Each leaf is one IF-THEN rule whose length is the leaf's depth."""
    t = tree.tree_
    depth = np.zeros(t.node_count, dtype=int)
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                depth[child] = depth[node] + 1
                stack.append(child)
    leaves = t.children_left == -1
    return {
        "rules": int(leaves.sum()),
        "conditions": int(depth[leaves].sum()),
        "avg_rule_length": float(depth[leaves].mean()),
    }


def tree_rules_text(tree, feature_names) -> str:
    return export_text(tree, feature_names=list(feature_names), decimals=1)
