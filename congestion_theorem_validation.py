#!/usr/bin/env python3
"""Exact combinatorial validation for the congestion-round claims.

This script does not try to "prove the theorem by numerics." Instead, it runs
independent exact computations on small instances to check that the paper's
graph-theoretic formula matches brute-force round minimization, and that the
hotspot clique examples behave exactly as advertised.

Checks
------
1. Exhaustive labeled-graph validation:
   for every graph on up to N vertices, confirm that
   `min_rounds_K(Gamma) = ceil(chi(Gamma)/K)`.
2. Hardware-grounded hotspot examples:
   build actual shortest-path route sets on star and line topologies whose
   conflict graphs are cliques, then verify the exact-round formula there too.
3. State-count sanity table:
   report the binary encoding's exact dimension `2^(K+1)` and the resulting
   one-qubit infeasibility boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx

from validation_common import (
    clique_number,
    ensure_result_dir,
    exact_chromatic_number,
    minimum_rounds_via_dp,
    write_csv,
    write_json,
)


RESULT_DIR = ensure_result_dir(__file__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("K values must be a nonempty list of positive integers.")
    return values


def labeled_graphs(num_vertices: int):
    edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]
    total = 1 << len(edges)
    for mask in range(total):
        graph = nx.Graph()
        graph.add_nodes_from(range(num_vertices))
        for bit, edge in enumerate(edges):
            if mask & (1 << bit):
                graph.add_edge(*edge)
        yield mask, graph


def exhaustive_graph_check(max_vertices: int, k_values: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    totals_by_n: list[dict[str, Any]] = []

    for num_vertices in range(1, max_vertices + 1):
        graph_count = 0
        chi_histogram: dict[int, int] = {}
        for _, graph in labeled_graphs(num_vertices):
            graph_count += 1
            chi = exact_chromatic_number(graph)
            chi_histogram[chi] = chi_histogram.get(chi, 0) + 1
            for k in k_values:
                exact_rounds = minimum_rounds_via_dp(graph, k)
                predicted = math.ceil(chi / k)
                rows.append(
                    {
                        "num_vertices": num_vertices,
                        "k": k,
                        "chi": chi,
                        "exact_rounds": exact_rounds,
                        "predicted_rounds": predicted,
                        "match": int(exact_rounds == predicted),
                    }
                )
                if exact_rounds != predicted:
                    mismatches.append(
                        {
                            "num_vertices": num_vertices,
                            "k": k,
                            "chi": chi,
                            "exact_rounds": exact_rounds,
                            "predicted_rounds": predicted,
                            "edges": sorted(tuple(sorted(edge)) for edge in graph.edges()),
                        }
                    )
        totals_by_n.append(
            {
                "num_vertices": num_vertices,
                "graph_count": graph_count,
                "chi_histogram": chi_histogram,
            }
        )

    return {
        "rows": rows,
        "mismatches": mismatches,
        "totals_by_n": totals_by_n,
    }


def route_conflict_graph(paths: list[tuple[int, ...]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(paths)))
    path_sets = [set(path) for path in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if path_sets[i] & path_sets[j]:
                graph.add_edge(i, j)
    return graph


def hotspot_examples(max_routes: int, k_values: list[int]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    # Star-topology hotspot: all leaf-to-leaf paths pass through the center.
    for route_count in range(2, max_routes + 1):
        graph = nx.star_graph(2 * route_count)
        paths = []
        for route_index in range(route_count):
            left = 2 * route_index + 1
            right = 2 * route_index + 2
            paths.append((left, 0, right))
        conflict = route_conflict_graph(paths)
        chi = exact_chromatic_number(conflict)
        record = {
            "family": "star_hotspot",
            "route_count": route_count,
            "paths": [list(path) for path in paths],
            "chi": chi,
            "clique_number": clique_number(conflict),
            "rounds": {},
        }
        for k in k_values:
            record["rounds"][f"K={k}"] = {
                "exact": minimum_rounds_via_dp(conflict, k),
                "predicted": math.ceil(chi / k),
            }
        examples.append(record)

    # Line-topology interval hotspot: long paths pairwise overlap on the middle segment.
    for route_count in range(2, max_routes + 1):
        width = 2 * route_count + 2
        graph = nx.path_graph(width)
        paths = []
        for route_index in range(route_count):
            start = route_index
            end = width - 1 - route_index
            paths.append(tuple(range(start, end + 1)))
        conflict = route_conflict_graph(paths)
        chi = exact_chromatic_number(conflict)
        record = {
            "family": "line_overlap",
            "route_count": route_count,
            "paths": [list(path) for path in paths],
            "chi": chi,
            "clique_number": clique_number(conflict),
            "rounds": {},
        }
        for k in k_values:
            record["rounds"][f"K={k}"] = {
                "exact": minimum_rounds_via_dp(conflict, k),
                "predicted": math.ceil(chi / k),
            }
        examples.append(record)

    return examples


def state_count_table(max_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in range(0, max_k + 1):
        d_min = 2 ** (k + 1)
        rows.append(
            {
                "K": k,
                "d_min_exact_overlap": d_min,
                "binary_encoding_states": d_min,
                "single_qubit_feasible": int(2 >= d_min),
            }
        )
    return rows


def render_hotspot_plot(examples: list[dict[str, Any]], k_values: list[int], output_dir: Path) -> Path | None:
    if plt is None or not examples:
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(example["family"], []).append(example)

    fig, axes = plt.subplots(1, len(grouped), figsize=(5.5 * len(grouped), 4.5), constrained_layout=True)
    if len(grouped) == 1:
        axes = [axes]

    for axis, (family, family_examples) in zip(axes, sorted(grouped.items())):
        family_examples = sorted(family_examples, key=lambda item: item["route_count"])
        x = [item["route_count"] for item in family_examples]
        for k in k_values:
            axis.plot(
                x,
                [item["rounds"][f"K={k}"]["exact"] for item in family_examples],
                marker="o",
                linewidth=1.8,
                label=f"exact K={k}",
            )
            axis.plot(
                x,
                [item["rounds"][f"K={k}"]["predicted"] for item in family_examples],
                linestyle="--",
                linewidth=1.2,
                label=f"ceil(chi/{k})",
            )
        axis.set_title(family.replace("_", " "))
        axis.set_xlabel("pairwise-overlapping routes")
        axis.set_ylabel("routing rounds")
        axis.legend(fontsize=8)

    path = output_dir / "hotspot_rounds_validation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=5,
        help="Exhaustively validate all labeled graphs up to this many vertices.",
    )
    parser.add_argument(
        "--k-values",
        default="1,2,3",
        help="Comma-separated bus counts K to validate.",
    )
    parser.add_argument(
        "--max-hotspot-routes",
        type=int,
        default=5,
        help="Maximum number of routes in the explicit hotspot examples.",
    )
    parser.add_argument(
        "--max-statecount-k",
        type=int,
        default=6,
        help="Largest K shown in the state-count table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_DIR,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_k_values(args.k_values)

    exhaustive = exhaustive_graph_check(args.max_vertices, k_values)
    hotspot = hotspot_examples(args.max_hotspot_routes, k_values)
    statecount = state_count_table(args.max_statecount_k)

    exhaustive_csv = args.output_dir / "exhaustive_graph_rounds.csv"
    hotspot_csv = args.output_dir / "hotspot_examples.csv"
    statecount_csv = args.output_dir / "state_count_table.csv"
    json_path = args.output_dir / "congestion_theorem_validation.json"

    hotspot_rows: list[dict[str, Any]] = []
    for example in hotspot:
        row = {
            "family": example["family"],
            "route_count": example["route_count"],
            "chi": example["chi"],
            "clique_number": example["clique_number"],
        }
        for k in k_values:
            row[f"exact_rounds_K{k}"] = example["rounds"][f"K={k}"]["exact"]
            row[f"predicted_rounds_K{k}"] = example["rounds"][f"K={k}"]["predicted"]
        hotspot_rows.append(row)

    plot_path = render_hotspot_plot(hotspot, k_values, args.output_dir)

    write_csv(exhaustive_csv, exhaustive["rows"])
    write_csv(hotspot_csv, hotspot_rows)
    write_csv(statecount_csv, statecount)
    write_json(
        json_path,
        {
            "methodology": {
                "exact_graph_check_scope": f"all labeled graphs on up to {args.max_vertices} vertices",
                "k_values": k_values,
                "important_scope_note": (
                    "This is an exact combinatorial validation of the graph-theoretic formula and "
                    "hotspot examples. It complements, but does not replace, the paper's proofs."
                ),
            },
            "exhaustive_graph_check": exhaustive,
            "hotspot_examples": hotspot,
            "state_count_table": statecount,
            "artifacts": {
                "exhaustive_csv": exhaustive_csv,
                "hotspot_csv": hotspot_csv,
                "statecount_csv": statecount_csv,
                "plot": plot_path,
            },
        },
    )

    mismatches = len(exhaustive["mismatches"])
    print(
        json.dumps(
            {
                "exhaustive_graphs_checked": sum(item["graph_count"] for item in exhaustive["totals_by_n"]),
                "k_values": k_values,
                "mismatches": mismatches,
                "json": str(json_path),
                "plot": str(plot_path) if plot_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

