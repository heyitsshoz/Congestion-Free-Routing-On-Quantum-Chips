#!/usr/bin/env python3
"""Shared helpers for the manuscript validation harness.

The utilities in this file are intentionally modest: exact graph coloring for
small graphs, reproducible JSON/CSV output, and directory/bootstrap helpers.
They are used by the benchmark and theorem-validation scripts so both follow
the same bookkeeping and exact combinatorial logic.
"""

from __future__ import annotations

import csv
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx


def ensure_result_dir(script_path: str) -> Path:
    script = Path(script_path).resolve()
    result_dir = script.parent / f"results_{script.stem}"
    result_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = result_dir / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return result_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ordered_nodes(graph: nx.Graph) -> list[Any]:
    return sorted(
        graph.nodes(),
        key=lambda node: (graph.degree(node), node),
        reverse=True,
    )


def clique_number(graph: nx.Graph) -> int:
    return max((len(clique) for clique in nx.find_cliques(graph)), default=0)


def greedy_upper_bound(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return 1 + max(coloring.values(), default=-1)


def is_k_colorable(graph: nx.Graph, k: int) -> bool:
    if k < 0:
        return False
    if graph.number_of_nodes() == 0:
        return True
    if k == 0:
        return False
    nodes = ordered_nodes(graph)
    neighbors = {node: set(graph.neighbors(node)) for node in nodes}
    colors: dict[Any, int] = {}

    def backtrack(index: int) -> bool:
        if index == len(nodes):
            return True
        node = nodes[index]
        forbidden = {colors[other] for other in neighbors[node] if other in colors}
        for color in range(k):
            if color in forbidden:
                continue
            colors[node] = color
            if backtrack(index + 1):
                return True
            del colors[node]
        return False

    return backtrack(0)


def exact_chromatic_number(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0

    lower = max(1, clique_number(graph))
    upper = max(lower, greedy_upper_bound(graph))

    for candidate in range(lower, upper + 1):
        if is_k_colorable(graph, candidate):
            return candidate
    raise RuntimeError("Exact chromatic-number search failed unexpectedly.")


def minimum_rounds_via_dp(graph: nx.Graph, k: int) -> int:
    """Exact minimum number of K-colorable routing rounds for a small graph.

    This is used only on modest route-conflict graphs in the theorem checker,
    where exact subset dynamic programming is practical and provides an
    independent numerical check of the `ceil(chi/K)` formula.
    """

    if graph.number_of_nodes() == 0:
        return 0
    if k <= 0:
        raise ValueError("k must be positive.")

    nodes = list(sorted(graph.nodes()))
    n = len(nodes)
    if n > 18:
        raise ValueError(
            "Subset-DP round search is intentionally limited to n <= 18 vertices."
        )
    index = {node: i for i, node in enumerate(nodes)}
    adjacency_masks = [0] * n
    for u, v in graph.edges():
        i = index[u]
        j = index[v]
        adjacency_masks[i] |= 1 << j
        adjacency_masks[j] |= 1 << i

    @lru_cache(maxsize=None)
    def mask_is_k_colorable(mask: int) -> bool:
        if mask == 0:
            return True
        active = [i for i in range(n) if mask & (1 << i)]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(active)
        for i in active:
            nbr_mask = adjacency_masks[i] & mask
            for j in active:
                if j <= i:
                    continue
                if nbr_mask & (1 << j):
                    subgraph.add_edge(i, j)
        return is_k_colorable(subgraph, k)

    @lru_cache(maxsize=None)
    def best(mask: int) -> int:
        if mask == 0:
            return 0
        answer = math.inf
        submask = mask
        while submask:
            if mask_is_k_colorable(submask):
                answer = min(answer, 1 + best(mask ^ submask))
            submask = (submask - 1) & mask
        if answer is math.inf:
            raise RuntimeError("No feasible partition found; this should be impossible.")
        return int(answer)

    return best((1 << n) - 1)
