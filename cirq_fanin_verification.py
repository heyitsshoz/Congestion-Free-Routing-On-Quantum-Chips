#!/usr/bin/env python3
"""Ideal Cirq verification suite for multi-bus Boolean fan-in routing.

This script extends the repository's single-bus checks to the fan-in model. It verifies that several routed controls can arrive on
distinct buses at one target, trigger a Boolean target rule, and then clean up
all routing offsets exactly.

Checks
------
1. Exhaustive basis-state truth-table verification for routed Boolean fan-in.
2. Coherent superposition verification of the full routed unitary.
3. Depth scaling against the manuscript's `2L + D_g` / `3L + D_g` formulas.

Examples
--------
python3 "fanin experiments/cirq_fanin_verification.py" all
python3 "fanin experiments/cirq_fanin_verification.py" basis-check --controls 3 --path-lengths 2,3,2 --boolean threshold:2
python3 "fanin experiments/cirq_fanin_verification.py" coherent-check --controls 3 --boolean truth:00010111 --unitary h
python3 "fanin experiments/cirq_fanin_verification.py" depth-scaling --controls 3 --lengths 1:8
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cirq_results"


@dataclass(frozen=True)
class BooleanRule:
    name: str
    evaluator: Callable[[Sequence[int]], int]


@dataclass(frozen=True)
class RoutingLayout:
    paths: tuple[tuple[int, ...], ...]
    target_site: int
    total_sites: int

    @property
    def source_sites(self) -> tuple[int, ...]:
        return tuple(path[0] for path in self.paths)

    @property
    def payload_sites(self) -> tuple[int, ...]:
        payload: list[int] = []
        for path in self.paths:
            payload.extend(path[1:-1])
        return tuple(payload)


def parse_int_list(spec: str) -> list[int]:
    spec = spec.strip()
    if ":" in spec:
        start_text, end_text = spec.split(":", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError("Range end must be >= range start.")
        return list(range(start, end + 1))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def parse_path_lengths(spec: str, controls: int) -> list[int]:
    values = parse_int_list(spec)
    if not values:
        raise ValueError("Provide at least one path length.")
    if len(values) == 1:
        values *= controls
    if len(values) != controls:
        raise ValueError(
            f"Expected either one path length or {controls} path lengths, got {len(values)}."
        )
    if any(length < 1 for length in values):
        raise ValueError("Every path length must be at least 1.")
    return values


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
    for local_state in local_states[1:]:
        state = np.kron(state, local_state)
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


def state_infidelity(actual: np.ndarray, expected: np.ndarray) -> float:
    overlap = np.vdot(expected, actual)
    return float(1.0 - min(1.0, abs(overlap) ** 2))


def logical_target_state(
    dimension: int, target_bit: int, unitary: np.ndarray, apply_unitary: bool
) -> np.ndarray:
    logical = basis_state(2, target_bit)
    transformed = unitary @ logical if apply_unitary else logical
    state = np.zeros(dimension, dtype=np.complex128)
    state[:2] = transformed
    return state


def minimum_dimension_for_controls(controls: int) -> int:
    return 2 ** (controls + 1)


def parse_unitary(spec: str) -> tuple[np.ndarray, str]:
    normalized = spec.strip().lower()
    if normalized == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), "X"
    if normalized == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128), "Z"
    if normalized == "h":
        return (
            np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0),
            "H",
        )
    if normalized == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128), "S"
    if normalized.startswith("phase:"):
        theta = float(normalized.split(":", maxsplit=1)[1])
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * theta)]], dtype=np.complex128), spec
    raise ValueError(
        "Unsupported unitary. Use one of: x, z, h, s, phase:<theta>."
    )


def parse_boolean_rule(spec: str, controls: int) -> BooleanRule:
    normalized = spec.strip().lower()
    if normalized == "and":
        return BooleanRule("and", lambda bits: int(all(bits)))
    if normalized == "or":
        return BooleanRule("or", lambda bits: int(any(bits)))
    if normalized == "parity":
        return BooleanRule("parity", lambda bits: int(sum(bits) % 2))
    if normalized == "majority":
        threshold = math.ceil(controls / 2)
        return BooleanRule("majority", lambda bits: int(sum(bits) >= threshold))
    if normalized.startswith("threshold:"):
        threshold = int(normalized.split(":", maxsplit=1)[1])
        if threshold < 0 or threshold > controls:
            raise ValueError(f"threshold must lie in [0, {controls}].")
        return BooleanRule(
            f"threshold:{threshold}", lambda bits, threshold=threshold: int(sum(bits) >= threshold)
        )
    if normalized.startswith("bus:"):
        bus = int(normalized.split(":", maxsplit=1)[1])
        if bus < 1 or bus > controls:
            raise ValueError(f"bus index must lie in [1, {controls}].")
        return BooleanRule(f"bus:{bus}", lambda bits, bus=bus: int(bits[bus - 1]))
    if normalized.startswith("truth:"):
        table = normalized.split(":", maxsplit=1)[1].strip()
        if len(table) != 2**controls or any(bit not in "01" for bit in table):
            raise ValueError(
                f"truth table must be a {2**controls}-bit binary string for {controls} controls."
            )

        def evaluate(bits: Sequence[int], table: str = table) -> int:
            index = 0
            for bit in bits:
                index = (index << 1) | int(bit)
            return int(table[index])

        return BooleanRule(f"truth:{table}", evaluate)
    raise ValueError(
        "Unsupported Boolean rule. Use one of: and, or, parity, majority, "
        "threshold:t, bus:k, truth:<bitstring>."
    )


def build_layout(path_lengths: Sequence[int]) -> RoutingLayout:
    if not path_lengths:
        raise ValueError("Provide at least one path length.")

    cursor = 0
    path_list: list[tuple[int, ...]] = []
    for path_length in path_lengths:
        if path_length < 1:
            raise ValueError("Each path length must be at least 1.")
        path_nodes = list(range(cursor, cursor + path_length))
        cursor += path_length
        path_list.append(tuple(path_nodes))

    target_site = cursor
    full_paths = tuple(path + (target_site,) for path in path_list)
    return RoutingLayout(paths=full_paths, target_site=target_site, total_sites=target_site + 1)


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


class BooleanRoutingGate(cirq.Gate):
    """Applies a logical 2x2 unitary on each lifted block selected by a Boolean rule."""

    def __init__(
        self,
        dimension: int,
        bus_indices: Sequence[int],
        boolean_rule: BooleanRule,
        unitary: np.ndarray,
        unitary_label: str,
    ) -> None:
        self.dimension = dimension
        self.bus_indices = tuple(bus_indices)
        self.boolean_rule = boolean_rule
        self.unitary = np.asarray(unitary, dtype=np.complex128)
        self.unitary_label = unitary_label

    def _qid_shape_(self) -> tuple[int]:
        return (self.dimension,)

    def _unitary_(self) -> np.ndarray:
        matrix = np.eye(self.dimension, dtype=np.complex128)
        for base in range(0, self.dimension - 1, 2):
            bits = tuple((base // (2**bus)) % 2 for bus in self.bus_indices)
            block = self.unitary if self.boolean_rule.evaluator(bits) else np.eye(2)
            matrix[base : base + 2, base : base + 2] = block
        return matrix

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"CUg[{self.boolean_rule.name},{self.unitary_label}]"


def append_packed_moments(
    moments: list[cirq.Moment], operations: Sequence[cirq.Operation]
) -> None:
    """Split a logical layer into as few non-overlapping Cirq moments as possible."""
    pending = list(operations)
    while pending:
        used_qudits: set[cirq.Qid] = set()
        moment_ops: list[cirq.Operation] = []
        next_pending: list[cirq.Operation] = []
        for operation in pending:
            operation_qudits = set(operation.qubits)
            if used_qudits.isdisjoint(operation_qudits):
                moment_ops.append(operation)
                used_qudits.update(operation_qudits)
            else:
                next_pending.append(operation)
        moments.append(cirq.Moment(moment_ops))
        pending = next_pending


def build_fanin_circuit(
    qudits: Sequence[cirq.Qid],
    layout: RoutingLayout,
    bus_indices: Sequence[int],
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
    unitary_label: str,
) -> cirq.Circuit:
    dimension = getattr(qudits[0], "dimension", None)
    if dimension is None:
        raise ValueError("Expected qudits with an explicit local dimension.")
    if len(bus_indices) != len(layout.paths):
        raise ValueError("Each path must have a corresponding bus index.")

    moments: list[cirq.Moment] = []

    forward_cbl_ops = [
        ConditionalShiftGate(dimension, 2**bus, mode="cbl").on(qudits[path[0]], qudits[path[1]])
        for path, bus in zip(layout.paths, bus_indices)
    ]
    append_packed_moments(moments, forward_cbl_ops)

    max_path_length = max(len(path) - 1 for path in layout.paths)
    for hop in range(1, max_path_length):
        layer_ops = []
        for path, bus in zip(layout.paths, bus_indices):
            if hop < len(path) - 1:
                layer_ops.append(
                    ConditionalShiftGate(dimension, 2**bus, mode="bcp").on(
                        qudits[path[hop]], qudits[path[hop + 1]]
                    )
                )
        if layer_ops:
            append_packed_moments(moments, layer_ops)

    append_packed_moments(
        moments,
        [
            BooleanRoutingGate(
                dimension=dimension,
                bus_indices=bus_indices,
                boolean_rule=boolean_rule,
                unitary=unitary,
                unitary_label=unitary_label,
            ).on(qudits[layout.target_site])
        ],
    )

    for hop in reversed(range(1, max_path_length)):
        layer_ops = []
        for path, bus in zip(layout.paths, bus_indices):
            if hop < len(path) - 1:
                layer_ops.append(
                    ConditionalShiftGate(dimension, 2**bus, mode="bcp", inverse=True).on(
                        qudits[path[hop]], qudits[path[hop + 1]]
                    )
                )
        if layer_ops:
            append_packed_moments(moments, layer_ops)

    reverse_cbl_ops = [
        ConditionalShiftGate(dimension, 2**bus, mode="cbl", inverse=True).on(
            qudits[path[0]], qudits[path[1]]
        )
        for path, bus in zip(layout.paths, bus_indices)
    ]
    append_packed_moments(moments, reverse_cbl_ops)

    return cirq.Circuit(moments)


def simulate_statevector(
    circuit: cirq.Circuit, qudits: Sequence[cirq.Qid], initial_state: np.ndarray
) -> np.ndarray:
    simulator = cirq.Simulator(dtype=np.complex128)
    result = simulator.simulate(circuit, qubit_order=qudits, initial_state=initial_state)
    return np.asarray(result.final_state_vector)


def build_payload_bits(layout: RoutingLayout, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    return {site: int(rng.integers(0, 2)) for site in layout.payload_sites}


def build_initial_basis_state(
    layout: RoutingLayout,
    dimension: int,
    control_bits: Sequence[int],
    payload_bits: dict[int, int],
    target_bit: int,
) -> np.ndarray:
    site_states: list[np.ndarray] = []
    source_map = {site: bit for site, bit in zip(layout.source_sites, control_bits)}
    for site in range(layout.total_sites):
        if site == layout.target_site:
            site_states.append(basis_state(dimension, target_bit))
        elif site in source_map:
            site_states.append(basis_state(dimension, source_map[site]))
        else:
            site_states.append(basis_state(dimension, payload_bits[site]))
    return product_state(site_states)


def build_expected_basis_state(
    layout: RoutingLayout,
    dimension: int,
    control_bits: Sequence[int],
    payload_bits: dict[int, int],
    target_bit: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
) -> np.ndarray:
    site_states: list[np.ndarray] = []
    source_map = {site: bit for site, bit in zip(layout.source_sites, control_bits)}
    apply_unitary = bool(boolean_rule.evaluator(control_bits))
    for site in range(layout.total_sites):
        if site == layout.target_site:
            site_states.append(
                logical_target_state(dimension, target_bit, unitary, apply_unitary)
            )
        elif site in source_map:
            site_states.append(basis_state(dimension, source_map[site]))
        else:
            site_states.append(basis_state(dimension, payload_bits[site]))
    return product_state(site_states)


def build_expected_coherent_state(
    layout: RoutingLayout,
    dimension: int,
    controls: int,
    payload_bits: dict[int, int],
    target_bit: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
) -> np.ndarray:
    branches: list[np.ndarray] = []
    amplitude = 1.0 / math.sqrt(2**controls)
    for control_bits in itertools.product((0, 1), repeat=controls):
        branches.append(
            amplitude
            * build_expected_basis_state(
                layout=layout,
                dimension=dimension,
                control_bits=control_bits,
                payload_bits=payload_bits,
                target_bit=target_bit,
                boolean_rule=boolean_rule,
                unitary=unitary,
            )
        )
    return sum(branches)


def max_routing_population(
    statevector: np.ndarray, sites: Sequence[int], num_sites: int, dimension: int
) -> float:
    if not sites:
        return 0.0
    return max(float(np.sum(site_probabilities(statevector, site, num_sites, dimension)[2:])) for site in sites)


def basis_truth_table_verification(
    controls: int,
    path_lengths: Sequence[int],
    dimension: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
    unitary_label: str,
    seed: int,
    tolerance: float,
) -> dict[str, object]:
    layout = build_layout(path_lengths)
    payload_bits = build_payload_bits(layout, seed)
    bus_indices = list(range(1, controls + 1))
    qudits = cirq.LineQid.range(layout.total_sites, dimension=dimension)
    circuit = build_fanin_circuit(
        qudits=qudits,
        layout=layout,
        bus_indices=bus_indices,
        boolean_rule=boolean_rule,
        unitary=unitary,
        unitary_label=unitary_label,
    )

    rows: list[dict[str, object]] = []
    worst_infidelity = 0.0
    worst_routing_population = 0.0

    for control_bits in itertools.product((0, 1), repeat=controls):
        for target_bit in (0, 1):
            initial_state = build_initial_basis_state(
                layout=layout,
                dimension=dimension,
                control_bits=control_bits,
                payload_bits=payload_bits,
                target_bit=target_bit,
            )
            final_state = simulate_statevector(circuit, qudits, initial_state)
            expected_state = build_expected_basis_state(
                layout=layout,
                dimension=dimension,
                control_bits=control_bits,
                payload_bits=payload_bits,
                target_bit=target_bit,
                boolean_rule=boolean_rule,
                unitary=unitary,
            )
            infidelity = state_infidelity(final_state, expected_state)
            routing_population = max_routing_population(
                final_state,
                sites=layout.payload_sites + (layout.target_site,),
                num_sites=layout.total_sites,
                dimension=dimension,
            )
            worst_infidelity = max(worst_infidelity, infidelity)
            worst_routing_population = max(worst_routing_population, routing_population)
            rows.append(
                {
                    "controls": "".join(str(bit) for bit in control_bits),
                    "target_bit": target_bit,
                    "g_value": int(boolean_rule.evaluator(control_bits)),
                    "infidelity": infidelity,
                    "max_routing_population": routing_population,
                    "passed": infidelity <= tolerance and routing_population <= tolerance,
                }
            )

    return {
        "check": "basis_truth_table",
        "controls": controls,
        "path_lengths": list(path_lengths),
        "dimension": dimension,
        "boolean_rule": boolean_rule.name,
        "unitary": unitary_label,
        "payload_bits": payload_bits,
        "bus_indices": bus_indices,
        "circuit_depth": len(circuit),
        "total_sites": layout.total_sites,
        "rows": rows,
        "worst_infidelity": worst_infidelity,
        "worst_routing_population": worst_routing_population,
        "passed": all(row["passed"] for row in rows),
    }


def coherent_fanin_verification(
    controls: int,
    path_lengths: Sequence[int],
    dimension: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
    unitary_label: str,
    seed: int,
    tolerance: float,
    target_bit: int,
) -> dict[str, object]:
    layout = build_layout(path_lengths)
    payload_bits = build_payload_bits(layout, seed)
    bus_indices = list(range(1, controls + 1))
    qudits = cirq.LineQid.range(layout.total_sites, dimension=dimension)
    circuit = build_fanin_circuit(
        qudits=qudits,
        layout=layout,
        bus_indices=bus_indices,
        boolean_rule=boolean_rule,
        unitary=unitary,
        unitary_label=unitary_label,
    )

    source_map = set(layout.source_sites)
    site_states: list[np.ndarray] = []
    for site in range(layout.total_sites):
        if site == layout.target_site:
            site_states.append(basis_state(dimension, target_bit))
        elif site in source_map:
            site_states.append(plus_state(dimension))
        else:
            site_states.append(basis_state(dimension, payload_bits[site]))

    initial_state = product_state(site_states)
    final_state = simulate_statevector(circuit, qudits, initial_state)
    expected_state = build_expected_coherent_state(
        layout=layout,
        dimension=dimension,
        controls=controls,
        payload_bits=payload_bits,
        target_bit=target_bit,
        boolean_rule=boolean_rule,
        unitary=unitary,
    )

    target_population = site_probabilities(
        final_state, layout.target_site, layout.total_sites, dimension
    )
    payload_routing_population = max_routing_population(
        final_state,
        sites=layout.payload_sites,
        num_sites=layout.total_sites,
        dimension=dimension,
    )

    return {
        "check": "coherent_fanin",
        "controls": controls,
        "path_lengths": list(path_lengths),
        "dimension": dimension,
        "boolean_rule": boolean_rule.name,
        "unitary": unitary_label,
        "target_bit": target_bit,
        "payload_bits": payload_bits,
        "circuit_depth": len(circuit),
        "global_infidelity": state_infidelity(final_state, expected_state),
        "max_payload_routing_population": payload_routing_population,
        "target_logical_population_0": float(target_population[0]),
        "target_logical_population_1": float(target_population[1]),
        "target_routing_population": float(np.sum(target_population[2:])),
        "passed": (
            state_infidelity(final_state, expected_state) <= tolerance
            and payload_routing_population <= tolerance
            and float(np.sum(target_population[2:])) <= tolerance
        ),
    }


def depth_scaling_verification(
    controls: int,
    lengths: Sequence[int],
    dimension: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
    unitary_label: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path_length in lengths:
        layout = build_layout([path_length] * controls)
        qudits = cirq.LineQid.range(layout.total_sites, dimension=dimension)
        circuit = build_fanin_circuit(
            qudits=qudits,
            layout=layout,
            bus_indices=list(range(1, controls + 1)),
            boolean_rule=boolean_rule,
            unitary=unitary,
            unitary_label=unitary_label,
        )
        rows.append(
            {
                "path_length": path_length,
                "controls": controls,
                "total_sites": layout.total_sites,
                "routed_depth": len(circuit),
                "expected_routed_depth": 2 * path_length + 1,
                "expected_swap_baseline_depth": 3 * path_length + 1,
                "routed_gate_count": len(tuple(circuit.all_operations())),
            }
        )

    x = np.array([row["path_length"] for row in rows], dtype=np.float64)
    routed_y = np.array([row["routed_depth"] for row in rows], dtype=np.float64)
    expected_y = np.array([row["expected_routed_depth"] for row in rows], dtype=np.float64)
    routed_slope, routed_intercept = np.polyfit(x, routed_y, deg=1)
    expected_slope, expected_intercept = np.polyfit(x, expected_y, deg=1)

    return {
        "check": "depth_scaling",
        "controls": controls,
        "dimension": dimension,
        "boolean_rule": boolean_rule.name,
        "unitary": unitary_label,
        "rows": rows,
        "routed_depth_fit": {
            "slope": float(routed_slope),
            "intercept": float(routed_intercept),
        },
        "expected_depth_fit": {
            "slope": float(expected_slope),
            "intercept": float(expected_intercept),
        },
    }


def maybe_plot_depth_scaling(summary: dict[str, object], output_dir: Path) -> None:
    if plt is None:
        return
    rows = summary["rows"]
    lengths = [row["path_length"] for row in rows]
    routed_depths = [row["routed_depth"] for row in rows]
    expected_routed = [row["expected_routed_depth"] for row in rows]
    expected_swap = [row["expected_swap_baseline_depth"] for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(lengths, routed_depths, marker="o", label="Constructed routed depth")
    plt.plot(lengths, expected_routed, marker="s", linestyle="--", label="2L + 1")
    plt.plot(lengths, expected_swap, marker="^", linestyle=":", label="3L + 1")
    plt.xlabel("Common path length L")
    plt.ylabel("Circuit depth (moments)")
    plt.title("Depth scaling for routed Boolean fan-in")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "depth_scaling.png", dpi=200)
    plt.close()


def validate_dimension(dimension: int, controls: int) -> None:
    minimum = minimum_dimension_for_controls(controls)
    if dimension < minimum:
        raise ValueError(
            f"dimension must satisfy d >= 2^(K+1) = {minimum} for {controls} controls."
        )


def run_basis_check(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    validate_dimension(args.dimension, args.controls)
    boolean_rule = parse_boolean_rule(args.boolean, args.controls)
    unitary, unitary_label = parse_unitary(args.unitary)
    result = basis_truth_table_verification(
        controls=args.controls,
        path_lengths=parse_path_lengths(args.path_lengths, args.controls),
        dimension=args.dimension,
        boolean_rule=boolean_rule,
        unitary=unitary,
        unitary_label=unitary_label,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    write_json(output_dir / "basis_check.json", result)
    write_csv(output_dir / "basis_check.csv", result["rows"])
    return result


def run_coherent_check(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    validate_dimension(args.dimension, args.controls)
    boolean_rule = parse_boolean_rule(args.boolean, args.controls)
    unitary, unitary_label = parse_unitary(args.unitary)
    result = coherent_fanin_verification(
        controls=args.controls,
        path_lengths=parse_path_lengths(args.path_lengths, args.controls),
        dimension=args.dimension,
        boolean_rule=boolean_rule,
        unitary=unitary,
        unitary_label=unitary_label,
        seed=args.seed,
        tolerance=args.tolerance,
        target_bit=args.target_bit,
    )
    write_json(output_dir / "coherent_check.json", result)
    return result


def run_depth_scaling(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    validate_dimension(args.dimension, args.controls)
    boolean_rule = parse_boolean_rule(args.boolean, args.controls)
    unitary, unitary_label = parse_unitary(args.unitary)
    summary = depth_scaling_verification(
        controls=args.controls,
        lengths=parse_int_list(args.lengths),
        dimension=args.dimension,
        boolean_rule=boolean_rule,
        unitary=unitary,
        unitary_label=unitary_label,
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON/CSV/plot outputs are written.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = {
        "controls": ("--controls", dict(type=int, default=3)),
        "path_lengths": ("--path-lengths", dict(type=str, default="1")),
        "dimension": ("--dimension", dict(type=int, default=16)),
        "boolean": ("--boolean", dict(type=str, default="and")),
        "unitary": ("--unitary", dict(type=str, default="x")),
        "seed": ("--seed", dict(type=int, default=7)),
        "tolerance": ("--tolerance", dict(type=float, default=1e-9)),
    }

    basis_check = subparsers.add_parser(
        "basis-check", help="Exhaustively verify the routed Boolean truth table."
    )
    for _, (flag, kwargs) in common.items():
        basis_check.add_argument(flag, **kwargs)

    coherent = subparsers.add_parser(
        "coherent-check", help="Verify the full coherent routed fan-in unitary."
    )
    for _, (flag, kwargs) in common.items():
        coherent.add_argument(flag, **kwargs)
    coherent.add_argument("--target-bit", type=int, default=0, choices=[0, 1])

    depth = subparsers.add_parser(
        "depth-scaling", help="Compare constructed depth against 2L+1 and 3L+1."
    )
    depth.add_argument("--controls", type=int, default=3)
    depth.add_argument("--dimension", type=int, default=16)
    depth.add_argument("--boolean", type=str, default="and")
    depth.add_argument("--unitary", type=str, default="x")
    depth.add_argument("--lengths", type=str, default="1:10")

    all_tests = subparsers.add_parser("all", help="Run every ideal fan-in verification.")
    for _, (flag, kwargs) in common.items():
        all_tests.add_argument(flag, **kwargs)
    all_tests.add_argument("--target-bit", type=int, default=0, choices=[0, 1])
    all_tests.add_argument("--lengths", type=str, default="1:10")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "basis-check":
        result = run_basis_check(args, output_dir)
        print_json(result)
        return

    if args.command == "coherent-check":
        result = run_coherent_check(args, output_dir)
        print_json(result)
        return

    if args.command == "depth-scaling":
        result = run_depth_scaling(args, output_dir)
        print_json(result)
        return

    if args.command == "all":
        basis_result = run_basis_check(args, output_dir)
        coherent_result = run_coherent_check(args, output_dir)
        depth_result = run_depth_scaling(args, output_dir)
        combined = {
            "basis_check": basis_result,
            "coherent_check": coherent_result,
            "depth_scaling": depth_result,
        }
        write_json(output_dir / "cirq_fanin_summary.json", combined)
        print_json(combined)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
