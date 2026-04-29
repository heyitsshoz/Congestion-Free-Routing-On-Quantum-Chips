#!/usr/bin/env python3

"""What this script does
---------------------
1. Builds fixed, reproducible logical circuits from nontrivial families.
2. Extracts only the two-qubit interaction layers that drive routing demand.
3. Routes each interaction along topology shortest paths using a policy that is
   deterministic or shortest-path sampled, never hand-tuned for the qudit model.
4. Reports exactly the metrics tied to the paper's claims:
   - path lengths,
    - `2L+1` routed primitive counts,
   - `3L` SWAP-transport counts,
   - route-conflict chromatic number `chi`,
   - congestion rounds `ceil(chi/K)` for user-chosen K values.
5. Separately records actual Qiskit qubit-compiler reference stats so the
   logical-route metrics are not mistaken for a full qudit compiler result.

The aim is narrower and cleaner: verify the logical-routing and congestion
claims on real circuit families under a reproducible, bias-resistant method.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap

from validation_common import (
    clique_number,
    ensure_result_dir,
    exact_chromatic_number,
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


@dataclass(frozen=True)
class BenchmarkSpec:
    family: str
    num_qubits: int
    topology: str


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_items(raw)]


def parse_k_values(raw: str) -> list[int]:
    values = sorted(set(parse_int_csv(raw)))
    if not values or any(value <= 0 for value in values):
        raise ValueError("K values must be a nonempty list of positive integers.")
    return values


def build_line_topology(num_qubits: int) -> nx.Graph:
    return nx.path_graph(num_qubits)


def build_ring_topology(num_qubits: int) -> nx.Graph:
    if num_qubits <= 2:
        return nx.path_graph(num_qubits)
    return nx.cycle_graph(num_qubits)


def build_grid_topology(num_qubits: int) -> nx.Graph:
    rows = math.isqrt(num_qubits)
    while rows * rows < num_qubits:
        rows += 1
    cols = math.ceil(num_qubits / rows)
    graph = nx.Graph()
    for node in range(num_qubits):
        graph.add_node(node)
    for node in range(num_qubits):
        row = node // cols
        col = node % cols
        right = row * cols + (col + 1)
        down = (row + 1) * cols + col
        if col + 1 < cols and right < num_qubits:
            graph.add_edge(node, right)
        if row + 1 < rows and down < num_qubits:
            graph.add_edge(node, down)
    return graph


def topology_graph(name: str, num_qubits: int) -> nx.Graph:
    normalized = name.strip().lower()
    if normalized == "line":
        return build_line_topology(num_qubits)
    if normalized == "ring":
        return build_ring_topology(num_qubits)
    if normalized == "grid":
        return build_grid_topology(num_qubits)
    raise ValueError(f"Unsupported topology '{name}'. Use one of: line, ring, grid.")


def topology_coupling_map(graph: nx.Graph) -> CouplingMap:
    directed_edges: list[tuple[int, int]] = []
    for u, v in sorted(graph.edges()):
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    return CouplingMap(directed_edges)


def build_qft_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name=f"qft_{num_qubits}")
    for target in range(num_qubits):
        qc.h(target)
        for control in range(target + 1, num_qubits):
            angle = math.pi / (2 ** (control - target))
            qc.cp(angle, control, target)
    return qc


def build_qaoa_maxcut_circuit(
    num_qubits: int, depth: int, degree: int, seed: int
) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
    if degree >= num_qubits:
        raise ValueError("QAOA regular-graph degree must be strictly smaller than n.")
    if (degree * num_qubits) % 2 != 0:
        raise ValueError("degree * num_qubits must be even for a regular graph.")
    graph = nx.random_regular_graph(degree, num_qubits, seed=seed)
    qc = QuantumCircuit(num_qubits, name=f"qaoa_d{degree}_p{depth}_{num_qubits}")
    qc.h(range(num_qubits))
    for layer in range(depth):
        gamma = 0.7 / (layer + 1)
        beta = 0.35 / (layer + 1)
        for u, v in sorted(graph.edges()):
            qc.rzz(2.0 * gamma, u, v)
        for qubit in range(num_qubits):
            qc.rx(2.0 * beta, qubit)
    return qc, sorted(tuple(sorted(edge)) for edge in graph.edges())


def build_mirror_matching_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name=f"mirror_{num_qubits}")
    for left in range(num_qubits // 2):
        right = num_qubits - 1 - left
        if left >= right:
            break
        qc.cx(left, right)
    return qc


def build_marked_state_phase_oracle(num_qubits: int, marked_state: str) -> QuantumCircuit:
    if len(marked_state) != num_qubits or any(bit not in "01" for bit in marked_state):
        raise ValueError("marked_state must be a bitstring of length num_qubits.")
    qc = QuantumCircuit(num_qubits, name=f"oracle_{marked_state}")
    for qubit, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(qubit)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for qubit, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(qubit)
    return qc


def build_diffusion_operator(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffusion")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_amplitude_amplification_circuit(
    num_qubits: int, iterations: int, marked_state: str | None = None
) -> tuple[QuantumCircuit, str]:
    if num_qubits < 3:
        raise ValueError("Amplitude-amplification benchmark requires at least 3 qubits.")
    if iterations <= 0:
        raise ValueError("Amplitude-amplification iterations must be positive.")
    resolved_marked = marked_state or ("0" * (num_qubits - 1) + "1")
    qc = QuantumCircuit(num_qubits, name=f"amplitude_{num_qubits}")
    qc.h(range(num_qubits))
    oracle = build_marked_state_phase_oracle(num_qubits, resolved_marked)
    diffusion = build_diffusion_operator(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    # Decompose the multi-control structure into ordinary 1q/2q logical gates so
    # the route-demand extractor measures the actual interaction pattern.
    return qc.decompose(reps=3), resolved_marked


def logical_circuit_and_metadata(
    family: str,
    num_qubits: int,
    qaoa_depth: int,
    qaoa_degree: int,
    amplitude_iterations: int,
    seed: int,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    normalized = family.strip().lower()
    if normalized == "qft":
        qc = build_qft_circuit(num_qubits)
        return qc, {"family": "qft"}
    if normalized == "qaoa":
        qc, edges = build_qaoa_maxcut_circuit(num_qubits, qaoa_depth, qaoa_degree, seed)
        return qc, {
            "family": "qaoa",
            "qaoa_depth": qaoa_depth,
            "qaoa_degree": qaoa_degree,
            "qaoa_seed": seed,
            "logical_graph_edges": edges,
        }
    if normalized == "mirror":
        qc = build_mirror_matching_circuit(num_qubits)
        return qc, {"family": "mirror"}
    if normalized in {"amplitude", "amplitude_amplification", "grover"}:
        qc, marked_state = build_amplitude_amplification_circuit(
            num_qubits, amplitude_iterations
        )
        return qc, {
            "family": "amplitude_amplification",
            "iterations": amplitude_iterations,
            "marked_state": marked_state,
        }
    raise ValueError(
        "Unsupported family. Use one of: qft, qaoa, mirror, amplitude."
    )


def extract_two_qubit_layers(qc: QuantumCircuit) -> list[list[dict[str, Any]]]:
    dag = circuit_to_dag(qc)
    layers: list[list[dict[str, Any]]] = []
    for dag_layer in dag.layers():
        ops: list[dict[str, Any]] = []
        for node in dag_layer["graph"].op_nodes():
            if len(node.qargs) != 2:
                continue
            q0 = qc.find_bit(node.qargs[0]).index
            q1 = qc.find_bit(node.qargs[1]).index
            ops.append({"pair": tuple(sorted((q0, q1))), "name": node.op.name})
        if ops:
            layers.append(ops)
    return layers


def sorted_shortest_paths(graph: nx.Graph, source: int, target: int) -> list[tuple[int, ...]]:
    paths = [tuple(path) for path in nx.all_shortest_paths(graph, source, target)]
    return sorted(paths)


def route_conflict_graph(paths: list[tuple[int, ...]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(paths)))
    path_sets = [set(path) for path in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if path_sets[i] & path_sets[j]:
                graph.add_edge(i, j)
    return graph


def sample_layer_metrics(
    interactions: list[dict[str, Any]],
    graph: nx.Graph,
    k_values: list[int],
    seed: int,
    path_samples: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    path_options = [
        sorted_shortest_paths(graph, item["pair"][0], item["pair"][1]) for item in interactions
    ]
    sample_count = 1 if all(len(options) == 1 for options in path_options) else max(1, path_samples)

    samples: list[dict[str, Any]] = []
    for _ in range(sample_count):
        paths: list[tuple[int, ...]] = []
        for options in path_options:
            if len(options) == 1:
                paths.append(options[0])
            else:
                paths.append(rng.choice(options))
        path_lengths = [len(path) - 1 for path in paths]
        conflict = route_conflict_graph(paths)
        chi = exact_chromatic_number(conflict)
        samples.append(
            {
                "paths": [list(path) for path in paths],
                "path_lengths": path_lengths,
                "swap_transport_cnot_eq": sum(3 * length for length in path_lengths),
                "routed_primitives": sum((2 * length) + 1 for length in path_lengths),
                "chi": chi,
                "clique_number": clique_number(conflict),
                "rounds": {f"K={k}": math.ceil(chi / k) for k in k_values},
            }
        )

    mean_path_length = sum(
        sum(sample["path_lengths"]) / max(1, len(sample["path_lengths"])) for sample in samples
    ) / len(samples)
    mean_swap = sum(sample["swap_transport_cnot_eq"] for sample in samples) / len(samples)
    mean_routed = sum(sample["routed_primitives"] for sample in samples) / len(samples)
    mean_chi = sum(sample["chi"] for sample in samples) / len(samples)
    mean_clique = sum(sample["clique_number"] for sample in samples) / len(samples)

    rounds_summary: dict[str, float] = {}
    for k in k_values:
        label = f"K={k}"
        rounds_summary[label] = sum(sample["rounds"][label] for sample in samples) / len(samples)

    return {
        "interaction_count": len(interactions),
        "operations": interactions,
        "sample_count": sample_count,
        "mean_path_length": mean_path_length,
        "max_path_length": max(max(sample["path_lengths"], default=[0]) for sample in samples),
        "mean_swap_transport_cnot_eq": mean_swap,
        "mean_routed_primitives": mean_routed,
        "mean_chi": mean_chi,
        "max_chi": max(sample["chi"] for sample in samples),
        "mean_clique_number": mean_clique,
        "max_clique_number": max(sample["clique_number"] for sample in samples),
        "mean_rounds": rounds_summary,
        "sample_details": samples,
    }


def count_ops_safe(qc: QuantumCircuit, name: str) -> int:
    return int(qc.count_ops().get(name, 0))


def transpile_reference(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    layout_method: str,
    routing_method: str,
    seed: int,
) -> dict[str, Any]:
    tqc = transpile(
        qc,
        basis_gates=["cx", "rz", "sx", "x"],
        coupling_map=coupling_map,
        optimization_level=3,
        layout_method=layout_method,
        routing_method=routing_method,
        seed_transpiler=seed,
    )
    return {
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx_count": count_ops_safe(tqc, "cx"),
        "swap_count": count_ops_safe(tqc, "swap"),
        "two_qubit_ops": int(
            sum(count for name, count in tqc.count_ops().items() if name in {"cx", "swap", "cz", "ecr"})
        ),
    }


def summarize_case(
    spec: BenchmarkSpec,
    qaoa_depth: int,
    qaoa_degree: int,
    amplitude_iterations: int,
    seed: int,
    k_values: list[int],
    path_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    qc, metadata = logical_circuit_and_metadata(
        family=spec.family,
        num_qubits=spec.num_qubits,
        qaoa_depth=qaoa_depth,
        qaoa_degree=qaoa_degree,
        amplitude_iterations=amplitude_iterations,
        seed=seed,
    )
    graph = topology_graph(spec.topology, spec.num_qubits)
    coupling_map = topology_coupling_map(graph)
    layers = extract_two_qubit_layers(qc)
    layer_records: list[dict[str, Any]] = []
    for layer_index, interactions in enumerate(layers):
        layer_records.append(
            {
                "layer_index": layer_index,
                **sample_layer_metrics(
                    interactions=interactions,
                    graph=graph,
                    k_values=k_values,
                    seed=seed + 10_000 * (layer_index + 1),
                    path_samples=path_samples,
                ),
            }
        )

    trivial = transpile_reference(qc, coupling_map, "trivial", "basic", seed)
    sabre = transpile_reference(qc, coupling_map, "sabre", "sabre", seed)

    total_swap_mean = sum(layer["mean_swap_transport_cnot_eq"] for layer in layer_records)
    total_routed_mean = sum(layer["mean_routed_primitives"] for layer in layer_records)
    mean_rounds_by_k = {
        f"K={k}": (
            sum(layer["mean_rounds"][f"K={k}"] for layer in layer_records) / max(1, len(layer_records))
        )
        for k in k_values
    }
    total_rounds_by_k = {
        f"K={k}": sum(layer["mean_rounds"][f"K={k}"] for layer in layer_records) for k in k_values
    }

    row = {
        "family": spec.family,
        "num_qubits": spec.num_qubits,
        "topology": spec.topology,
        "logical_two_qubit_layers": len(layer_records),
        "logical_two_qubit_gates": sum(layer["interaction_count"] for layer in layer_records),
        "mean_path_length": (
            sum(layer["mean_path_length"] for layer in layer_records) / max(1, len(layer_records))
        ),
        "max_path_length": max((layer["max_path_length"] for layer in layer_records), default=0),
        "mean_chi": sum(layer["mean_chi"] for layer in layer_records) / max(1, len(layer_records)),
        "max_chi": max((layer["max_chi"] for layer in layer_records), default=0),
        "mean_clique_number": (
            sum(layer["mean_clique_number"] for layer in layer_records) / max(1, len(layer_records))
        ),
        "max_clique_number": max((layer["max_clique_number"] for layer in layer_records), default=0),
        "mean_swap_transport_cnot_eq": total_swap_mean / max(1, len(layer_records)),
        "total_swap_transport_cnot_eq": total_swap_mean,
        "mean_routed_primitives": total_routed_mean / max(1, len(layer_records)),
        "total_routed_primitives": total_routed_mean,
        "routed_to_swap_transport_ratio": (
            total_routed_mean / total_swap_mean if total_swap_mean else 0.0
        ),
        "qiskit_trivial_depth": trivial["depth"],
        "qiskit_trivial_cx": trivial["cx_count"],
        "qiskit_trivial_swap": trivial["swap_count"],
        "qiskit_trivial_two_qubit_ops": trivial["two_qubit_ops"],
        "qiskit_sabre_depth": sabre["depth"],
        "qiskit_sabre_cx": sabre["cx_count"],
        "qiskit_sabre_swap": sabre["swap_count"],
        "qiskit_sabre_two_qubit_ops": sabre["two_qubit_ops"],
    }
    for k in k_values:
        row[f"mean_rounds_K{k}"] = mean_rounds_by_k[f"K={k}"]
        row[f"total_rounds_K{k}"] = total_rounds_by_k[f"K={k}"]

    detail = {
        "spec": {
            "family": spec.family,
            "num_qubits": spec.num_qubits,
            "topology": spec.topology,
        },
        "metadata": metadata,
        "logical_two_qubit_layers": layer_records,
        "transpile_reference": {"trivial_basic": trivial, "sabre_sabre": sabre},
    }
    return row, detail


def render_summary_plot(rows: list[dict[str, Any]], k_values: list[int], output_dir: Path) -> Path | None:
    if plt is None or not rows:
        return None

    labels = [f"{row['family']}-{row['num_qubits']}-{row['topology']}" for row in rows]
    x_positions = list(range(len(rows)))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 0.9), 9), constrained_layout=True)

    axes[0].bar(
        x_positions,
        [row["routed_to_swap_transport_ratio"] for row in rows],
        color="#4C78A8",
    )
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("routed / SWAP transport")
    axes[0].set_title("Logical transport ratio on fixed shortest-path demand")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    for k in k_values:
        axes[1].plot(
            x_positions,
            [row[f"mean_rounds_K{k}"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=f"K={k}",
        )
    axes[1].bar(
        x_positions,
        [row["mean_chi"] for row in rows],
        alpha=0.18,
        color="#E45756",
        label=r"mean $\chi(\Gamma)$",
    )
    axes[1].set_ylabel("mean layer rounds")
    axes[1].set_title("Congestion-round demand from route-conflict graphs")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].legend()

    path = output_dir / "compiler_benchmark_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        default="qft,qaoa,mirror,amplitude",
        help="Comma-separated benchmark families. Choices: qft,qaoa,mirror,amplitude.",
    )
    parser.add_argument(
        "--sizes",
        default="6,8",
        help="Comma-separated qubit counts to benchmark.",
    )
    parser.add_argument(
        "--topologies",
        default="line,grid",
        help="Comma-separated hardware topologies. Choices: line,ring,grid.",
    )
    parser.add_argument(
        "--k-values",
        default="1,2,3",
        help="Comma-separated bus counts K for congestion-round reporting.",
    )
    parser.add_argument(
        "--qaoa-depth",
        type=int,
        default=2,
        help="Number of QAOA cost/mixer layers for the QAOA benchmark family.",
    )
    parser.add_argument(
        "--qaoa-degree",
        type=int,
        default=3,
        help="Regular-graph degree for the QAOA benchmark family.",
    )
    parser.add_argument(
        "--amplitude-iterations",
        type=int,
        default=1,
        help="Number of Grover-style amplitude-amplification iterations.",
    )
    parser.add_argument(
        "--path-samples",
        type=int,
        default=16,
        help="Number of shortest-path tie-break samples per logical layer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Global seed for QAOA graph generation and shortest-path sampling.",
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

    families = parse_csv_items(args.families)
    sizes = parse_int_csv(args.sizes)
    topologies = parse_csv_items(args.topologies)
    k_values = parse_k_values(args.k_values)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for family in families:
        for num_qubits in sizes:
            for topology in topologies:
                spec = BenchmarkSpec(family=family, num_qubits=num_qubits, topology=topology)
                row, detail = summarize_case(
                    spec=spec,
                    qaoa_depth=args.qaoa_depth,
                    qaoa_degree=args.qaoa_degree,
                    amplitude_iterations=args.amplitude_iterations,
                    seed=args.seed,
                    k_values=k_values,
                    path_samples=args.path_samples,
                )
                rows.append(row)
                details.append(detail)

    rows = sorted(rows, key=lambda item: (item["family"], item["num_qubits"], item["topology"]))

    csv_path = args.output_dir / "compiler_benchmark_summary.csv"
    json_path = args.output_dir / "compiler_benchmark_details.json"
    png_path = render_summary_plot(rows, k_values, args.output_dir)

    methodology = {
        "families": families,
        "sizes": sizes,
        "topologies": topologies,
        "k_values": k_values,
        "path_samples": args.path_samples,
        "seed": args.seed,
        "bias_controls": [
            "Fixed circuit families and sizes.",
            "Fixed topology embeddings.",
            "Shortest-path routing only; no hand-tuned detours.",
            "Uniform shortest-path tie sampling with a published seed.",
            "Qiskit qubit transpilation reported separately as a reference, not repackaged as routed data.",
        ],
        "important_scope_note": (
            "These outputs validate logical transport and congestion metrics implied by the paper. "
            "They do not constitute a full device-level qudit compiler benchmark."
        ),
    }

    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "methodology": methodology,
            "summary_rows": rows,
            "case_details": details,
            "artifacts": {
                "csv": csv_path,
                "json": json_path,
                "plot": png_path,
            },
        },
    )

    summary = {
        "csv": str(csv_path),
        "json": str(json_path),
        "plot": str(png_path) if png_path else None,
        "cases": len(rows),
        "families": families,
        "sizes": sizes,
        "topologies": topologies,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
