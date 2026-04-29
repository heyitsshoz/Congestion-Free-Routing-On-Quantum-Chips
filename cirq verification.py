#!/usr/bin/env python3
"""Ideal Cirq verification suite for the qudit-routing paper.

This script implements three checks that mirror the manuscript:

1. Single-path correctness for the swap-free non-local CNOT protocol.
2. Crossroads overlap verification for two simultaneous routes sharing a center node.
3. Depth scaling against the paper's naive SWAP-routing baseline.

Examples
--------
python3 "cirq verification.py" all --output-dir cirq_results
python3 "cirq verification.py" single-path --path-length 4 --dimension 8 --seed 7
python3 "cirq verification.py" depth-scaling --lengths 2:20 --output-dir cirq_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import cirq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Cirq is required for this script. Install it with `pip install cirq`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def parse_int_list(spec: str) -> list[int]:
    """Parse `2:20` as an inclusive range or `2,4,6` as an explicit list."""
    spec = spec.strip()
    if ":" in spec:
        start_text, end_text = spec.split(":", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError("Range end must be >= range start.")
        return list(range(start, end + 1))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def print_json(payload: object) -> None:
    print(json.dumps(to_jsonable(payload), indent=2))


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def basis_state(dimension: int, level: int) -> np.ndarray:
    state = np.zeros(dimension, dtype=np.complex128)
    state[level] = 1.0
    return state


def plus_state(dimension: int) -> np.ndarray:
    state = np.zeros(dimension, dtype=np.complex128)
    state[0] = 1.0 / math.sqrt(2.0)
    state[1] = 1.0 / math.sqrt(2.0)
    return state


def product_state(local_states: Sequence[np.ndarray]) -> np.ndarray:
    state = local_states[0]
    for local in local_states[1:]:
        state = np.kron(state, local)
    return state


def reshape_statevector(
    statevector: np.ndarray, num_sites: int, dimension: int
) -> np.ndarray:
    return np.asarray(statevector).reshape((dimension,) * num_sites)


def site_probabilities(
    statevector: np.ndarray, site: int, num_sites: int, dimension: int
) -> np.ndarray:
    tensor = reshape_statevector(statevector, num_sites, dimension)
    axes = tuple(index for index in range(num_sites) if index != site)
    return np.sum(np.abs(tensor) ** 2, axis=axes)


def reduced_density_matrix(
    statevector: np.ndarray, keep: Sequence[int], dims: Sequence[int]
) -> np.ndarray:
    keep = tuple(keep)
    total_sites = len(dims)
    tensor = np.asarray(statevector).reshape(tuple(dims))
    trace_out = tuple(index for index in range(total_sites) if index not in keep)
    permutation = keep + trace_out
    tensor = np.transpose(tensor, permutation)
    kept_dim = int(np.prod([dims[index] for index in keep], dtype=int))
    traced_dim = int(np.prod([dims[index] for index in trace_out], dtype=int))
    matrix = tensor.reshape((kept_dim, traced_dim))
    return matrix @ matrix.conj().T


def state_infidelity(actual: np.ndarray, expected: np.ndarray) -> float:
    overlap = np.vdot(expected, actual)
    return float(1.0 - min(1.0, abs(overlap) ** 2))


class ConditionalShiftGate(cirq.Gate):
    """CBL/BCP-style controlled shift on a pair of qudits."""

    def __init__(
        self,
        dimension: int,
        delta: int,
        mode: str,
        inverse: bool = False,
    ) -> None:
        if mode not in {"cbl", "bcp"}:
            raise ValueError("mode must be 'cbl' or 'bcp'")
        self.dimension = dimension
        self.delta = delta
        self.mode = mode
        self.inverse = inverse

    def _qid_shape_(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def _unitary_(self) -> np.ndarray:
        shift = -self.delta if self.inverse else self.delta
        size = self.dimension * self.dimension
        matrix = np.zeros((size, size), dtype=np.complex128)
        for control in range(self.dimension):
            for target in range(self.dimension):
                if self.mode == "cbl":
                    active = control % 2
                else:
                    active = (control // self.delta) % 2
                next_target = (target + active * shift) % self.dimension
                row = control * self.dimension + next_target
                column = control * self.dimension + target
                matrix[row, column] = 1.0
        return matrix

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> tuple[str, str]:
        label = "CBL" if self.mode == "cbl" else "BCP"
        suffix = "-1" if self.inverse else ""
        return (f"{label}{suffix}[{self.delta}]",) * 2


class RoutingControlledXGate(cirq.Gate):
    """Applies X to the least-significant logical bit iff a bus digit is active."""

    def __init__(self, dimension: int, delta: int) -> None:
        self.dimension = dimension
        self.delta = delta

    def _qid_shape_(self) -> tuple[int]:
        return (self.dimension,)

    def _unitary_(self) -> np.ndarray:
        matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for basis_value in range(self.dimension):
            bus_active = (basis_value // self.delta) % 2
            target = basis_value ^ 1 if bus_active else basis_value
            matrix[target, basis_value] = 1.0
        return matrix

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"CXR[{self.delta}]"


def build_single_path_circuit(
    qudits: Sequence[cirq.Qid], bus_index: int
) -> cirq.Circuit:
    dimension = getattr(qudits[0], "dimension", None)
    if dimension is None:
        raise ValueError("Expected qudits with an explicit local dimension.")
    delta = 2**bus_index
    path = list(range(len(qudits)))
    operations: list[cirq.Operation] = []
    operations.append(
        ConditionalShiftGate(dimension, delta, mode="cbl").on(qudits[path[0]], qudits[path[1]])
    )
    for source, target in zip(path[1:-1], path[2:]):
        operations.append(
            ConditionalShiftGate(dimension, delta, mode="bcp").on(qudits[source], qudits[target])
        )
    operations.append(RoutingControlledXGate(dimension, delta).on(qudits[path[-1]]))
    for source, target in reversed(list(zip(path[1:-1], path[2:]))):
        operations.append(
            ConditionalShiftGate(
                dimension, delta, mode="bcp", inverse=True
            ).on(qudits[source], qudits[target])
        )
    operations.append(
        ConditionalShiftGate(dimension, delta, mode="cbl", inverse=True).on(
            qudits[path[0]], qudits[path[1]]
        )
    )
    return cirq.Circuit(operations)


def build_crossroads_circuits(
    dimension: int,
) -> tuple[cirq.Circuit, cirq.Circuit, Sequence[cirq.LineQid]]:
    qudits = cirq.LineQid.range(5, dimension=dimension)
    left, center, right, up, down = qudits
    horizontal_delta = 2
    vertical_delta = 4

    overlap = cirq.Circuit(
        [
            ConditionalShiftGate(dimension, horizontal_delta, mode="cbl").on(left, center),
            ConditionalShiftGate(dimension, vertical_delta, mode="cbl").on(up, center),
        ]
    )

    full = cirq.Circuit(
        [
            ConditionalShiftGate(dimension, horizontal_delta, mode="cbl").on(left, center),
            ConditionalShiftGate(dimension, vertical_delta, mode="cbl").on(up, center),
            ConditionalShiftGate(dimension, horizontal_delta, mode="bcp").on(center, right),
            ConditionalShiftGate(dimension, vertical_delta, mode="bcp").on(center, down),
            RoutingControlledXGate(dimension, horizontal_delta).on(right),
            RoutingControlledXGate(dimension, vertical_delta).on(down),
            ConditionalShiftGate(dimension, vertical_delta, mode="bcp", inverse=True).on(
                center, down
            ),
            ConditionalShiftGate(dimension, horizontal_delta, mode="bcp", inverse=True).on(
                center, right
            ),
            ConditionalShiftGate(dimension, vertical_delta, mode="cbl", inverse=True).on(up, center),
            ConditionalShiftGate(
                dimension, horizontal_delta, mode="cbl", inverse=True
            ).on(left, center),
        ]
    )

    return overlap, full, qudits


def simulate_statevector(
    circuit: cirq.Circuit, qudits: Sequence[cirq.Qid], initial_state: np.ndarray
) -> np.ndarray:
    simulator = cirq.Simulator(dtype=np.complex128)
    result = simulator.simulate(circuit, qubit_order=qudits, initial_state=initial_state)
    return np.asarray(result.final_state_vector)


def expected_single_path_state(
    dimension: int, intermediate_bits: Sequence[int], target_bit: int
) -> np.ndarray:
    source_zero_branch = [basis_state(dimension, 0)]
    source_one_branch = [basis_state(dimension, 1)]
    body = [basis_state(dimension, bit) for bit in intermediate_bits]
    zero_branch = source_zero_branch + body + [basis_state(dimension, target_bit)]
    one_branch = source_one_branch + body + [basis_state(dimension, target_bit ^ 1)]
    return (
        product_state(zero_branch) + product_state(one_branch)
    ) / math.sqrt(2.0)


def expected_crossroads_final_state(
    dimension: int, center_bit: int, right_bit: int, down_bit: int
) -> np.ndarray:
    branches: list[np.ndarray] = []
    amplitude = 0.5
    for horizontal_source in (0, 1):
        for vertical_source in (0, 1):
            branches.append(
                amplitude
                * product_state(
                    [
                        basis_state(dimension, horizontal_source),
                        basis_state(dimension, center_bit),
                        basis_state(dimension, right_bit ^ horizontal_source),
                        basis_state(dimension, vertical_source),
                        basis_state(dimension, down_bit ^ vertical_source),
                    ]
                )
            )
    return sum(branches)


def expected_crossroads_overlap_state(
    dimension: int, center_bit: int, right_bit: int, down_bit: int
) -> np.ndarray:
    branches: list[np.ndarray] = []
    amplitude = 0.5
    for horizontal_source in (0, 1):
        for vertical_source in (0, 1):
            branches.append(
                amplitude
                * product_state(
                    [
                        basis_state(dimension, horizontal_source),
                        basis_state(
                            dimension, center_bit + 2 * horizontal_source + 4 * vertical_source
                        ),
                        basis_state(dimension, right_bit),
                        basis_state(dimension, vertical_source),
                        basis_state(dimension, down_bit),
                    ]
                )
            )
    return sum(branches)


def expected_crossroads_center_distribution(center_bit: int, dimension: int) -> np.ndarray:
    distribution = np.zeros(dimension, dtype=np.float64)
    for horizontal_source in (0, 1):
        for vertical_source in (0, 1):
            level = center_bit + 2 * horizontal_source + 4 * vertical_source
            distribution[level] += 0.25
    return distribution


def single_path_verification(
    path_length: int,
    dimension: int,
    bus_index: int,
    seed: int,
    tolerance: float,
) -> dict[str, object]:
    if path_length < 2:
        raise ValueError("Path length must be at least 2.")

    rng = np.random.default_rng(seed)
    intermediate_bits = rng.integers(0, 2, size=path_length - 1).tolist()
    target_bit = int(rng.integers(0, 2))
    qudits = cirq.LineQid.range(path_length + 1, dimension=dimension)

    initial_state = product_state(
        [plus_state(dimension)]
        + [basis_state(dimension, bit) for bit in intermediate_bits]
        + [basis_state(dimension, target_bit)]
    )
    final_state = simulate_statevector(
        build_single_path_circuit(qudits, bus_index), qudits, initial_state
    )
    expected_state = expected_single_path_state(dimension, intermediate_bits, target_bit)

    dims = [dimension] * (path_length + 1)
    source_target_actual = reduced_density_matrix(final_state, [0, path_length], dims)
    source_target_expected = reduced_density_matrix(expected_state, [0, path_length], dims)

    intermediate_restored = True
    max_intermediate_routing = 0.0
    for site, bit in zip(range(1, path_length), intermediate_bits):
        probabilities = site_probabilities(final_state, site, path_length + 1, dimension)
        intermediate_restored &= abs(probabilities[bit] - 1.0) <= tolerance
        max_intermediate_routing = max(max_intermediate_routing, float(np.sum(probabilities[2:])))

    target_probabilities = site_probabilities(final_state, path_length, path_length + 1, dimension)
    routing_population_total = 0.0
    for site in range(path_length + 1):
        probabilities = site_probabilities(final_state, site, path_length + 1, dimension)
        routing_population_total += float(np.sum(probabilities[2:]))

    result = {
        "path_length": path_length,
        "dimension": dimension,
        "bus_index": bus_index,
        "seed": seed,
        "intermediate_bits": intermediate_bits,
        "target_bit": target_bit,
        "global_infidelity": state_infidelity(final_state, expected_state),
        "source_target_fro_error": float(
            np.linalg.norm(source_target_actual - source_target_expected, ord="fro")
        ),
        "target_logical_population_0": float(target_probabilities[0]),
        "target_logical_population_1": float(target_probabilities[1]),
        "intermediate_restored": intermediate_restored,
        "max_intermediate_routing_population": max_intermediate_routing,
        "total_routing_population": routing_population_total,
        "passed": (
            state_infidelity(final_state, expected_state) <= tolerance
            and np.linalg.norm(source_target_actual - source_target_expected, ord="fro")
            <= tolerance
            and intermediate_restored
            and max_intermediate_routing <= tolerance
            and routing_population_total <= tolerance
        ),
    }
    return result


def crossroads_verification(dimension: int, tolerance: float) -> dict[str, object]:
    if dimension < 8:
        raise ValueError("The crossroads test needs dimension >= 8 for buses 2 and 4.")

    overlap_circuit, full_circuit, qudits = build_crossroads_circuits(dimension)
    center_bit = 1
    right_bit = 0
    down_bit = 1

    initial_state = product_state(
        [
            plus_state(dimension),
            basis_state(dimension, center_bit),
            basis_state(dimension, right_bit),
            plus_state(dimension),
            basis_state(dimension, down_bit),
        ]
    )

    overlap_state = simulate_statevector(overlap_circuit, qudits, initial_state)
    final_state = simulate_statevector(full_circuit, qudits, initial_state)

    overlap_center_distribution = site_probabilities(overlap_state, 1, 5, dimension)
    expected_center_distribution = expected_crossroads_center_distribution(center_bit, dimension)
    expected_overlap_state = expected_crossroads_overlap_state(
        dimension, center_bit, right_bit, down_bit
    )
    final_expected_state = expected_crossroads_final_state(
        dimension, center_bit, right_bit, down_bit
    )

    final_routing_population = 0.0
    for site in range(5):
        final_routing_population += float(np.sum(site_probabilities(final_state, site, 5, dimension)[2:]))

    return {
        "dimension": dimension,
        "center_bit": center_bit,
        "right_bit": right_bit,
        "down_bit": down_bit,
        "center_distribution_l1_error": float(
            np.sum(np.abs(overlap_center_distribution - expected_center_distribution))
        ),
        "overlap_infidelity": state_infidelity(overlap_state, expected_overlap_state),
        "center_distribution": overlap_center_distribution.tolist(),
        "expected_center_distribution": expected_center_distribution.tolist(),
        "final_infidelity": state_infidelity(final_state, final_expected_state),
        "final_total_routing_population": final_routing_population,
        "passed": (
            np.sum(np.abs(overlap_center_distribution - expected_center_distribution))
            <= tolerance
            and state_infidelity(overlap_state, expected_overlap_state) <= tolerance
            and state_infidelity(final_state, final_expected_state) <= tolerance
            and final_routing_population <= tolerance
        ),
    }


def build_swap_routing_baseline_circuit(path_length: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(path_length + 1)
    operations: list[cirq.Operation] = []
    for edge in range(path_length):
        control, target = qubits[edge], qubits[edge + 1]
        operations.extend(
            [
                cirq.CNOT(control, target),
                cirq.CNOT(target, control),
                cirq.CNOT(control, target),
            ]
        )
    return cirq.Circuit(operations)


def depth_scaling_verification(
    lengths: Sequence[int], dimension: int, bus_index: int
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path_length in lengths:
        qudits = cirq.LineQid.range(path_length + 1, dimension=dimension)
        routed_circuit = build_single_path_circuit(qudits, bus_index)
        swap_circuit = build_swap_routing_baseline_circuit(path_length)
        rows.append(
            {
                "path_length": path_length,
                "swap_routing_depth": len(swap_circuit),
                "swap_routing_gate_count": len(tuple(swap_circuit.all_operations())),
                "qudit_protocol_depth": len(routed_circuit),
                "qudit_protocol_gate_count": len(tuple(routed_circuit.all_operations())),
            }
        )

    x = np.array([row["path_length"] for row in rows], dtype=np.float64)
    swap_y = np.array([row["swap_routing_depth"] for row in rows], dtype=np.float64)
    routed_y = np.array([row["qudit_protocol_depth"] for row in rows], dtype=np.float64)
    swap_slope, swap_intercept = np.polyfit(x, swap_y, deg=1)
    routed_slope, routed_intercept = np.polyfit(x, routed_y, deg=1)

    return {
        "rows": rows,
        "swap_depth_fit": {
            "slope": float(swap_slope),
            "intercept": float(swap_intercept),
        },
        "qudit_depth_fit": {
            "slope": float(routed_slope),
            "intercept": float(routed_intercept),
        },
    }


def maybe_plot_depth_scaling(summary: dict[str, object], output_dir: Path) -> None:
    if plt is None:
        return
    rows = summary["rows"]
    lengths = [row["path_length"] for row in rows]
    swap_depths = [row["swap_routing_depth"] for row in rows]
    routed_depths = [row["qudit_protocol_depth"] for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(lengths, swap_depths, marker="o", label="Naive SWAP routing depth")
    plt.plot(lengths, routed_depths, marker="s", label="Qudit routing depth")
    plt.xlabel("Path length L")
    plt.ylabel("Circuit depth (moments)")
    plt.title("Depth scaling: SWAP baseline vs. qudit routing")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "depth_scaling.png", dpi=200)
    plt.close()


def run_single_path(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    result = single_path_verification(
        path_length=args.path_length,
        dimension=args.dimension,
        bus_index=args.bus_index,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    write_json(output_dir / "single_path.json", result)
    return result


def run_crossroads(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    result = crossroads_verification(dimension=args.dimension, tolerance=args.tolerance)
    write_json(output_dir / "crossroads.json", result)
    return result


def run_depth_scaling(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    summary = depth_scaling_verification(
        lengths=parse_int_list(args.lengths),
        dimension=args.dimension,
        bus_index=args.bus_index,
    )
    write_json(output_dir / "depth_scaling.json", summary)
    write_csv(output_dir / "depth_scaling.csv", summary["rows"])
    maybe_plot_depth_scaling(summary, output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cirq_results"),
        help="Directory where JSON/CSV/plot outputs are written.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = {
        "dimension": ("--dimension", dict(type=int, default=8)),
        "bus_index": ("--bus-index", dict(type=int, default=1)),
        "tolerance": ("--tolerance", dict(type=float, default=1e-9)),
    }

    single = subparsers.add_parser("single-path", help="Run the Theorem 4.1 check.")
    single.add_argument("--path-length", type=int, default=4)
    single.add_argument("--seed", type=int, default=7)
    for _, (flag, kwargs) in common.items():
        single.add_argument(flag, **kwargs)

    crossroads = subparsers.add_parser("crossroads", help="Run the overlap/crosstalk check.")
    crossroads.add_argument("--dimension", type=int, default=8)
    crossroads.add_argument("--tolerance", type=float, default=1e-9)

    depth = subparsers.add_parser("depth-scaling", help="Run the depth scaling sweep.")
    depth.add_argument("--lengths", type=str, default="2:20")
    depth.add_argument("--dimension", type=int, default=8)
    depth.add_argument("--bus-index", type=int, default=1)

    all_tests = subparsers.add_parser("all", help="Run every Cirq verification.")
    all_tests.add_argument("--path-length", type=int, default=4)
    all_tests.add_argument("--seed", type=int, default=7)
    all_tests.add_argument("--lengths", type=str, default="2:20")
    for _, (flag, kwargs) in common.items():
        all_tests.add_argument(flag, **kwargs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "single-path":
        result = run_single_path(args, output_dir)
        print_json(result)
        return

    if args.command == "crossroads":
        result = run_crossroads(args, output_dir)
        print_json(result)
        return

    if args.command == "depth-scaling":
        summary = run_depth_scaling(args, output_dir)
        print_json(summary)
        return

    if args.command == "all":
        single_path = run_single_path(args, output_dir)
        crossroads = run_crossroads(args, output_dir)
        depth = run_depth_scaling(args, output_dir)
        combined = {
            "single_path": single_path,
            "crossroads": crossroads,
            "depth_scaling": depth,
        }
        write_json(output_dir / "cirq_verification_summary.json", combined)
        print_json(combined)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
