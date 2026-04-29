#!/usr/bin/env python3
"""Noisy QuTiP verification suite for multi-bus Boolean fan-in routing.

This script mirrors `cirq_fanin_verification.py`, but runs the routed fan-in
protocol in a multilevel QuTiP model with optional relaxation, dephasing, and
routing leakage. In the ideal limit it should agree with the exact Cirq checks;
with noise turned on it gives a physical stress test of the same construction.

Examples
--------
python3 "fanin experiments/qutip_fanin_verification.py" single-run
python3 "fanin experiments/qutip_fanin_verification.py" single-run --controls 3 --boolean threshold:2 --routing-gate-time 1.0 --target-gate-time 1.0 --t1-levels 80,45,25,14
python3 "fanin experiments/qutip_fanin_verification.py" length-sweep --controls 3 --lengths 1:4 --boolean and --routing-gate-time 0.5 --target-gate-time 0.5
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
from scipy import sparse

try:
    import qutip as qt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "QuTiP is required for this script. Install it with `pip install qutip`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "qutip_results"

TRAJECTORY_DIMENSION_THRESHOLD = 400
MONTE_CARLO_TRAJECTORIES = 100
_TRAJECTORY_SOLVER_OPTIONS = {
    "keep_runs_results": True,
    "store_final_state": True,
    "store_states": False,
    "progress_bar": "",
}
_EXACT_SOLVER_OPTIONS = {
    "store_final_state": True,
    "store_states": False,
    "progress_bar": "",
}
_RNG = np.random.default_rng()
_MC_SEED_SEQUENCE: np.random.SeedSequence | None = None
StateLike = qt.Qobj | list[qt.Qobj]


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


def parse_float_list(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


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


def parse_level_times(spec: str, label: str) -> list[float]:
    values = parse_float_list(spec)
    if not values:
        raise ValueError(f"Provide at least one value for {label}.")
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


def configure_monte_carlo(seed: int | None, trajectories: int) -> None:
    global MONTE_CARLO_TRAJECTORIES, _RNG, _MC_SEED_SEQUENCE
    MONTE_CARLO_TRAJECTORIES = trajectories
    _RNG = np.random.default_rng(seed)
    _MC_SEED_SEQUENCE = np.random.SeedSequence(seed) if seed is not None else None


def next_monte_carlo_seeds(count: int) -> list[int | np.random.SeedSequence] | None:
    if _MC_SEED_SEQUENCE is None:
        return None
    return _MC_SEED_SEQUENCE.spawn(count)


def basis_state(dimension: int, level: int) -> qt.Qobj:
    return qt.basis(dimension, level)


def plus_state(dimension: int) -> qt.Qobj:
    return (qt.basis(dimension, 0) + qt.basis(dimension, 1)).unit()


def projector(level: int, dimension: int) -> qt.Qobj:
    ket = basis_state(dimension, level)
    return (ket * ket.dag()).to("csr")


def product_ket(states: Sequence[qt.Qobj]) -> qt.Qobj:
    return qt.tensor(list(states))


def zero_hamiltonian(dims: Sequence[int]) -> qt.Qobj:
    return qt.qzero(list(dims)).to("csr")


def logical_target_state(
    dimension: int, target_bit: int, unitary: np.ndarray, apply_unitary: bool
) -> qt.Qobj:
    logical = np.zeros(2, dtype=np.complex128)
    logical[target_bit] = 1.0
    transformed = unitary @ logical if apply_unitary else logical
    state = np.zeros(dimension, dtype=np.complex128)
    state[:2] = transformed
    return qt.Qobj(state.reshape((-1, 1)), dims=[[dimension], [1]]).unit()


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


def conditional_shift_matrix(
    dimension: int, delta: int, mode: str, inverse: bool = False
) -> sparse.csr_matrix:
    if mode not in {"cbl", "bcp"}:
        raise ValueError("mode must be 'cbl' or 'bcp'")
    shift = -delta if inverse else delta
    size = dimension * dimension
    matrix = sparse.lil_matrix((size, size), dtype=np.complex128)
    for control in range(dimension):
        for target in range(dimension):
            if mode == "cbl":
                active = control % 2
            else:
                active = (control // delta) % 2
            next_target = (target + active * shift) % dimension
            row = control * dimension + next_target
            column = control * dimension + target
            matrix[row, column] = 1.0
    return matrix.tocsr()


def boolean_routed_unitary_matrix(
    dimension: int,
    bus_indices: Sequence[int],
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
) -> sparse.csr_matrix:
    matrix = sparse.identity(dimension, dtype=np.complex128, format="lil")
    for base in range(0, dimension - 1, 2):
        bits = tuple((base // (2**bus)) % 2 for bus in bus_indices)
        block = unitary if boolean_rule.evaluator(bits) else np.eye(2)
        for row in range(2):
            for column in range(2):
                matrix[base + row, base + column] = block[row, column]
    return matrix.tocsr()


def embed_single_site(operator: qt.Qobj, site: int, dims: Sequence[int]) -> qt.Qobj:
    factors = [qt.qeye(dimension).to("csr") for dimension in dims]
    factors[site] = operator
    return qt.tensor(factors).to("csr")


def embed_two_site(
    local_matrix: sparse.spmatrix, site_a: int, site_b: int, dims: Sequence[int]
) -> qt.Qobj:
    if site_a == site_b:
        raise ValueError("The two embedded sites must be distinct.")
    current_order = [site_a, site_b] + [
        index for index in range(len(dims)) if index not in {site_a, site_b}
    ]
    local = qt.Qobj(
        local_matrix,
        dims=[[dims[site_a], dims[site_b]], [dims[site_a], dims[site_b]]],
    ).to("csr")
    factors = [local] + [qt.qeye(dims[index]).to("csr") for index in current_order[2:]]
    embedded = qt.tensor(factors)
    permutation = [current_order.index(index) for index in range(len(dims))]
    return embedded.permute(permutation).to("csr")


def build_decay_collapse_ops(dims: Sequence[int], t1_levels: Sequence[float]) -> list[qt.Qobj]:
    local_dimension = dims[0]
    collapse_ops: list[qt.Qobj] = []
    for site in range(len(dims)):
        for level in range(1, local_dimension):
            t1 = t1_levels[min(level - 1, len(t1_levels) - 1)]
            if math.isinf(t1) or t1 <= 0.0:
                continue
            local = sparse.lil_matrix((local_dimension, local_dimension), dtype=np.complex128)
            local[level - 1, level] = math.sqrt(1.0 / t1)
            operator = qt.Qobj(
                local.tocsr(), dims=[[local_dimension], [local_dimension]]
            ).to("csr")
            collapse_ops.append(embed_single_site(operator, site, dims))
    return collapse_ops


def build_dephasing_collapse_ops(
    dims: Sequence[int], tphi_levels: Sequence[float]
) -> list[qt.Qobj]:
    local_dimension = dims[0]
    collapse_ops: list[qt.Qobj] = []
    for site in range(len(dims)):
        for level in range(1, local_dimension):
            tphi = tphi_levels[min(level - 1, len(tphi_levels) - 1)]
            if math.isinf(tphi) or tphi <= 0.0:
                continue
            local = sparse.lil_matrix((local_dimension, local_dimension), dtype=np.complex128)
            local[level, level] = math.sqrt(1.0 / tphi)
            operator = qt.Qobj(
                local.tocsr(), dims=[[local_dimension], [local_dimension]]
            ).to("csr")
            collapse_ops.append(embed_single_site(operator, site, dims))
    return collapse_ops


def build_relaxation_and_dephasing_ops(
    dims: Sequence[int],
    t1_levels: Sequence[float],
    tphi_levels: Sequence[float],
) -> list[qt.Qobj]:
    return build_decay_collapse_ops(dims, t1_levels) + build_dephasing_collapse_ops(
        dims, tphi_levels
    )


def build_leakage_kraus(dimension: int, epsilon: float) -> list[qt.Qobj]:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("leakage epsilon must lie in [0, 1].")
    if epsilon <= 0.0:
        return [qt.qeye(dimension).to("csr")]
    keep = sparse.eye(dimension, dtype=np.complex128, format="lil")
    jump = sparse.lil_matrix((dimension, dimension), dtype=np.complex128)
    for level in range(2, dimension - 1):
        keep[level, level] = math.sqrt(1.0 - epsilon)
        jump[level + 1, level] = math.sqrt(epsilon)
    return [
        qt.Qobj(keep.tocsr(), dims=[[dimension], [dimension]]).to("csr"),
        qt.Qobj(jump.tocsr(), dims=[[dimension], [dimension]]).to("csr"),
    ]


def apply_local_channel(
    rho: qt.Qobj,
    site: int,
    local_kraus_ops: Sequence[qt.Qobj],
    dims: Sequence[int],
) -> qt.Qobj:
    updated = 0.0 * rho
    for kraus in local_kraus_ops:
        embedded = embed_single_site(kraus, site, dims)
        updated += embedded * rho * embedded.dag()
    return updated


def apply_local_channel_trajectory(
    psi: qt.Qobj,
    site: int,
    local_kraus_ops: Sequence[qt.Qobj],
    dims: Sequence[int],
) -> qt.Qobj:
    embedded_ops = [embed_single_site(kraus, site, dims) for kraus in local_kraus_ops]
    branches: list[tuple[float, qt.Qobj]] = []
    for embedded in embedded_ops:
        branch = embedded * psi
        probability = float(np.real_if_close(branch.norm() ** 2))
        if probability > 0.0:
            branches.append((probability, branch))
    if not branches:
        return psi
    weights = np.array([probability for probability, _ in branches], dtype=float)
    weights /= weights.sum()
    selected = int(_RNG.choice(len(branches), p=weights))
    return branches[selected][1].unit()


def evolve_trajectory_ensemble(
    trajectories: Sequence[qt.Qobj],
    h0: qt.Qobj,
    collapse_ops: Sequence[qt.Qobj],
    gate_time: float,
) -> list[qt.Qobj]:
    if not trajectories:
        return []
    solver = qt.MCSolver(h0, list(collapse_ops), options=_TRAJECTORY_SOLVER_OPTIONS)
    weighted_ensemble = [(psi, 1.0 / len(trajectories)) for psi in trajectories]
    result = solver.run(
        weighted_ensemble,
        [0.0, gate_time],
        ntraj=[1] * len(trajectories),
        seeds=next_monte_carlo_seeds(len(trajectories)),
    )
    final_states = getattr(result, "runs_final_states", None)
    if final_states is None:
        final_states = getattr(result, "runs_final_state", None)
    if final_states is None:
        raise RuntimeError("QuTiP did not return per-trajectory final states.")
    normalized = []
    for state in final_states:
        normalized.append(state.unit() if state.norm() > 0.0 else state)
    return normalized


def compose_layer_operator(operators: Sequence[qt.Qobj]) -> qt.Qobj:
    if not operators:
        raise ValueError("Need at least one operator to compose a layer.")
    gate = operators[0]
    for operator in operators[1:]:
        gate = operator * gate
    return gate


def apply_leakage_channels(
    state: StateLike,
    leakage_sites: Sequence[int],
    leakage_kraus: Sequence[qt.Qobj] | None,
    dims: Sequence[int],
) -> StateLike:
    if leakage_kraus is None:
        return state
    updated = state
    for site in leakage_sites:
        if isinstance(updated, list):
            updated = [
                apply_local_channel_trajectory(psi, site, leakage_kraus, dims)
                for psi in updated
            ]
        elif updated.isket:
            updated = apply_local_channel_trajectory(updated, site, leakage_kraus, dims)
        else:
            updated = apply_local_channel(updated, site, leakage_kraus, dims)
    return updated


def apply_global_gate_with_noise(
    state: StateLike,
    gate: qt.Qobj,
    gate_time: float,
    h0: qt.Qobj,
    collapse_ops: Sequence[qt.Qobj],
    leakage_sites: Sequence[int] = (),
    leakage_kraus: Sequence[qt.Qobj] | None = None,
) -> StateLike:
    if isinstance(state, list):
        updated: StateLike = [gate * psi for psi in state]
    elif state.isket:
        updated = gate * state
    else:
        updated = gate * state * gate.dag()

    updated = apply_leakage_channels(updated, leakage_sites, leakage_kraus, h0.dims[0])

    if gate_time > 0.0 and collapse_ops:
        dimension = int(np.prod(h0.dims[0], dtype=int))
        if isinstance(updated, list):
            updated = evolve_trajectory_ensemble(updated, h0, collapse_ops, gate_time)
        elif dimension > TRAJECTORY_DIMENSION_THRESHOLD and updated.isket:
            trajectories = [updated.copy() for _ in range(MONTE_CARLO_TRAJECTORIES)]
            updated = evolve_trajectory_ensemble(trajectories, h0, collapse_ops, gate_time)
        else:
            result = qt.mesolve(
                h0,
                updated,
                [0.0, gate_time],
                c_ops=list(collapse_ops),
                options=_EXACT_SOLVER_OPTIONS,
            )
            final_state = getattr(result, "final_state", None)
            updated = final_state if final_state is not None else result.states[-1]
    return updated


def pure_state_overlap(state: StateLike, ideal_ket: qt.Qobj) -> float:
    if isinstance(state, list):
        samples = [float(np.abs(ideal_ket.overlap(psi)) ** 2) for psi in state]
        overlap = float(np.mean(samples)) if samples else 0.0
        return max(0.0, min(1.0, overlap))
    if state.isket:
        overlap = float(np.abs(ideal_ket.overlap(state)) ** 2)
    else:
        amplitude = ideal_ket.dag() * state * ideal_ket
        amplitude_value = amplitude[0, 0] if isinstance(amplitude, qt.Qobj) else amplitude
        overlap = float(np.real_if_close(amplitude_value))
    return max(0.0, min(1.0, overlap))


def routing_population(
    state: StateLike, site: int, dimension: int, dims: Sequence[int]
) -> float:
    local = 0.0 * projector(0, dimension)
    for level in range(2, dimension):
        local += projector(level, dimension)
    observable = embed_single_site(local, site, dims)
    if isinstance(state, list):
        samples = [float(np.real_if_close(qt.expect(observable, psi))) for psi in state]
        population = float(np.mean(samples)) if samples else 0.0
    else:
        population = float(np.real_if_close(qt.expect(observable, state)))
    return max(0.0, min(1.0, population))


def build_payload_bits(layout: RoutingLayout, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    return {site: int(rng.integers(0, 2)) for site in layout.payload_sites}


def build_initial_coherent_state(
    layout: RoutingLayout,
    dimension: int,
    payload_bits: dict[int, int],
    target_bit: int,
) -> qt.Qobj:
    source_map = set(layout.source_sites)
    site_states: list[qt.Qobj] = []
    for site in range(layout.total_sites):
        if site == layout.target_site:
            site_states.append(basis_state(dimension, target_bit))
        elif site in source_map:
            site_states.append(plus_state(dimension))
        else:
            site_states.append(basis_state(dimension, payload_bits[site]))
    return product_ket(site_states)


def build_expected_coherent_ket(
    layout: RoutingLayout,
    dimension: int,
    controls: int,
    payload_bits: dict[int, int],
    target_bit: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
) -> qt.Qobj:
    branches: list[qt.Qobj] = []
    amplitude = 1.0 / math.sqrt(2**controls)
    source_map_sites = layout.source_sites
    for control_bits in itertools.product((0, 1), repeat=controls):
        source_map = {site: bit for site, bit in zip(source_map_sites, control_bits)}
        apply_unitary = bool(boolean_rule.evaluator(control_bits))
        site_states: list[qt.Qobj] = []
        for site in range(layout.total_sites):
            if site == layout.target_site:
                site_states.append(
                    logical_target_state(dimension, target_bit, unitary, apply_unitary)
                )
            elif site in source_map:
                site_states.append(basis_state(dimension, source_map[site]))
            else:
                site_states.append(basis_state(dimension, payload_bits[site]))
        branches.append(amplitude * product_ket(site_states))

    total = 0.0 * branches[0]
    for branch in branches:
        total += branch
    return total.unit()


def simulate_routed_fanin_protocol(
    controls: int,
    path_lengths: Sequence[int],
    dimension: int,
    boolean_rule: BooleanRule,
    unitary: np.ndarray,
    routing_gate_time: float,
    target_gate_time: float,
    t1_levels: Sequence[float],
    tphi_levels: Sequence[float],
    leakage_epsilon: float,
    seed: int,
    target_bit: int,
) -> tuple[StateLike, qt.Qobj, RoutingLayout, dict[int, int]]:
    layout = build_layout(path_lengths)
    payload_bits = build_payload_bits(layout, seed)
    bus_indices = list(range(1, controls + 1))
    dims = [dimension] * layout.total_sites
    h0 = zero_hamiltonian(dims)
    collapse_ops = build_relaxation_and_dephasing_ops(dims, t1_levels, tphi_levels)
    leakage_kraus = build_leakage_kraus(dimension, leakage_epsilon)
    state: StateLike = build_initial_coherent_state(
        layout=layout,
        dimension=dimension,
        payload_bits=payload_bits,
        target_bit=target_bit,
    )

    forward_ops = [
        embed_two_site(
            conditional_shift_matrix(dimension, 2**bus, "cbl"),
            path[0],
            path[1],
            dims,
        )
        for path, bus in zip(layout.paths, bus_indices)
    ]
    state = apply_global_gate_with_noise(
        state=state,
        gate=compose_layer_operator(forward_ops),
        gate_time=routing_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
        leakage_sites=[path[1] for path in layout.paths],
        leakage_kraus=leakage_kraus,
    )

    max_path_length = max(len(path) - 1 for path in layout.paths)
    for hop in range(1, max_path_length):
        layer_ops = []
        leakage_sites: list[int] = []
        for path, bus in zip(layout.paths, bus_indices):
            if hop < len(path) - 1:
                layer_ops.append(
                    embed_two_site(
                        conditional_shift_matrix(dimension, 2**bus, "bcp"),
                        path[hop],
                        path[hop + 1],
                        dims,
                    )
                )
                leakage_sites.append(path[hop + 1])
        if layer_ops:
            state = apply_global_gate_with_noise(
                state=state,
                gate=compose_layer_operator(layer_ops),
                gate_time=routing_gate_time,
                h0=h0,
                collapse_ops=collapse_ops,
                leakage_sites=leakage_sites,
                leakage_kraus=leakage_kraus,
            )

    target_gate = embed_single_site(
        qt.Qobj(
            boolean_routed_unitary_matrix(dimension, bus_indices, boolean_rule, unitary),
            dims=[[dimension], [dimension]],
        ).to("csr"),
        layout.target_site,
        dims,
    )
    state = apply_global_gate_with_noise(
        state=state,
        gate=target_gate,
        gate_time=target_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
    )

    for hop in reversed(range(1, max_path_length)):
        layer_ops = []
        leakage_sites = []
        for path, bus in zip(layout.paths, bus_indices):
            if hop < len(path) - 1:
                layer_ops.append(
                    embed_two_site(
                        conditional_shift_matrix(dimension, 2**bus, "bcp", inverse=True),
                        path[hop],
                        path[hop + 1],
                        dims,
                    )
                )
                leakage_sites.append(path[hop + 1])
        if layer_ops:
            state = apply_global_gate_with_noise(
                state=state,
                gate=compose_layer_operator(layer_ops),
                gate_time=routing_gate_time,
                h0=h0,
                collapse_ops=collapse_ops,
                leakage_sites=leakage_sites,
                leakage_kraus=leakage_kraus,
            )

    reverse_ops = [
        embed_two_site(
            conditional_shift_matrix(dimension, 2**bus, "cbl", inverse=True),
            path[0],
            path[1],
            dims,
        )
        for path, bus in zip(layout.paths, bus_indices)
    ]
    state = apply_global_gate_with_noise(
        state=state,
        gate=compose_layer_operator(reverse_ops),
        gate_time=routing_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
        leakage_sites=[path[1] for path in layout.paths],
        leakage_kraus=leakage_kraus,
    )

    ideal_ket = build_expected_coherent_ket(
        layout=layout,
        dimension=dimension,
        controls=controls,
        payload_bits=payload_bits,
        target_bit=target_bit,
        boolean_rule=boolean_rule,
        unitary=unitary,
    )
    return state, ideal_ket, layout, payload_bits


def single_run_verification(args: argparse.Namespace) -> dict[str, object]:
    boolean_rule = parse_boolean_rule(args.boolean, args.controls)
    unitary, unitary_label = parse_unitary(args.unitary)
    path_lengths = parse_path_lengths(args.path_lengths, args.controls)
    t1_levels = parse_level_times(args.t1_levels, "T1")
    tphi_levels = parse_level_times(args.tphi_levels, "Tphi")

    state, ideal_ket, layout, payload_bits = simulate_routed_fanin_protocol(
        controls=args.controls,
        path_lengths=path_lengths,
        dimension=args.dimension,
        boolean_rule=boolean_rule,
        unitary=unitary,
        routing_gate_time=args.routing_gate_time,
        target_gate_time=args.target_gate_time,
        t1_levels=t1_levels,
        tphi_levels=tphi_levels,
        leakage_epsilon=args.leakage_epsilon,
        seed=args.seed,
        target_bit=args.target_bit,
    )

    dims = [args.dimension] * layout.total_sites
    payload_populations = [
        routing_population(state, site=site, dimension=args.dimension, dims=dims)
        for site in layout.payload_sites
    ]
    summary = {
        "check": "single_run",
        "controls": args.controls,
        "path_lengths": path_lengths,
        "dimension": args.dimension,
        "boolean_rule": boolean_rule.name,
        "unitary": unitary_label,
        "target_bit": args.target_bit,
        "payload_bits": payload_bits,
        "routing_gate_time": args.routing_gate_time,
        "target_gate_time": args.target_gate_time,
        "t1_levels": t1_levels,
        "tphi_levels": tphi_levels,
        "leakage_epsilon": args.leakage_epsilon,
        "monte_carlo_trajectories": MONTE_CARLO_TRAJECTORIES,
        "seed": args.seed,
        "fidelity_to_ideal_fanin": pure_state_overlap(state, ideal_ket),
        "max_payload_routing_population": max(payload_populations, default=0.0),
        "target_routing_population": routing_population(
            state, site=layout.target_site, dimension=args.dimension, dims=dims
        ),
    }
    summary["matches_ideal_unitary"] = (
        abs(1.0 - summary["fidelity_to_ideal_fanin"]) <= args.tolerance
        and summary["max_payload_routing_population"] <= args.tolerance
        and summary["target_routing_population"] <= args.tolerance
    )
    return summary


def length_sweep_verification(args: argparse.Namespace) -> dict[str, object]:
    boolean_rule = parse_boolean_rule(args.boolean, args.controls)
    unitary, unitary_label = parse_unitary(args.unitary)
    lengths = parse_int_list(args.lengths)
    t1_levels = parse_level_times(args.t1_levels, "T1")
    tphi_levels = parse_level_times(args.tphi_levels, "Tphi")
    rows: list[dict[str, object]] = []

    for path_length in lengths:
        state, ideal_ket, layout, _ = simulate_routed_fanin_protocol(
            controls=args.controls,
            path_lengths=[path_length] * args.controls,
            dimension=args.dimension,
            boolean_rule=boolean_rule,
            unitary=unitary,
            routing_gate_time=args.routing_gate_time,
            target_gate_time=args.target_gate_time,
            t1_levels=t1_levels,
            tphi_levels=tphi_levels,
            leakage_epsilon=args.leakage_epsilon,
            seed=args.seed + path_length,
            target_bit=args.target_bit,
        )
        dims = [args.dimension] * layout.total_sites
        payload_population = max(
            [
                routing_population(state, site=site, dimension=args.dimension, dims=dims)
                for site in layout.payload_sites
            ],
            default=0.0,
        )
        rows.append(
            {
                "path_length": path_length,
                "fidelity_to_ideal_fanin": pure_state_overlap(state, ideal_ket),
                "max_payload_routing_population": payload_population,
                "target_routing_population": routing_population(
                    state, site=layout.target_site, dimension=args.dimension, dims=dims
                ),
            }
        )

    return {
        "check": "length_sweep",
        "controls": args.controls,
        "dimension": args.dimension,
        "boolean_rule": boolean_rule.name,
        "unitary": unitary_label,
        "target_bit": args.target_bit,
        "routing_gate_time": args.routing_gate_time,
        "target_gate_time": args.target_gate_time,
        "t1_levels": t1_levels,
        "tphi_levels": tphi_levels,
        "leakage_epsilon": args.leakage_epsilon,
        "monte_carlo_trajectories": MONTE_CARLO_TRAJECTORIES,
        "seed": args.seed,
        "rows": rows,
    }


def maybe_plot_length_sweep(summary: dict[str, object], output_dir: Path) -> None:
    if plt is None:
        return
    rows = summary["rows"]
    lengths = [row["path_length"] for row in rows]
    fidelities = [row["fidelity_to_ideal_fanin"] for row in rows]
    residuals = [row["max_payload_routing_population"] for row in rows]
    t1_levels = summary.get("t1_levels", [])
    tphi_levels = summary.get("tphi_levels", [])
    leakage_epsilon = float(summary.get("leakage_epsilon", 0.0))
    is_ideal_limit = (
        leakage_epsilon == 0.0
        and all(level == float("inf") for level in t1_levels)
        and all(level == float("inf") for level in tphi_levels)
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(lengths, fidelities, marker="o", label="Fidelity to ideal fan-in")
    plt.plot(lengths, residuals, marker="s", label="Max payload routing population")
    plt.xlabel("Common path length L")
    plt.ylabel("Metric value")
    plt.title(
        "Ideal routed fan-in vs. path length"
        if is_ideal_limit
        else "Noisy routed fan-in vs. path length"
    )
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "length_sweep.png", dpi=200)
    plt.close()


def validate_dimension(dimension: int, controls: int) -> None:
    minimum = minimum_dimension_for_controls(controls)
    if dimension < minimum:
        raise ValueError(
            f"dimension must satisfy d >= 2^(K+1) = {minimum} for {controls} controls."
        )


def run_single(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    validate_dimension(args.dimension, args.controls)
    configure_monte_carlo(args.seed, args.trajectories)
    summary = single_run_verification(args)
    write_json(output_dir / "single_run.json", summary)
    return summary


def run_length_sweep(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    validate_dimension(args.dimension, args.controls)
    configure_monte_carlo(args.seed, args.trajectories)
    summary = length_sweep_verification(args)
    write_json(output_dir / "length_sweep.json", summary)
    write_csv(output_dir / "length_sweep.csv", summary["rows"])
    maybe_plot_length_sweep(summary, output_dir)
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
        "controls": ("--controls", dict(type=int, default=2)),
        "dimension": ("--dimension", dict(type=int, default=8)),
        "boolean": ("--boolean", dict(type=str, default="and")),
        "unitary": ("--unitary", dict(type=str, default="x")),
        "target_bit": ("--target-bit", dict(type=int, choices=[0, 1], default=0)),
        "seed": ("--seed", dict(type=int, default=7)),
        "trajectories": ("--trajectories", dict(type=int, default=100)),
        "routing_gate_time": ("--routing-gate-time", dict(type=float, default=0.0)),
        "target_gate_time": ("--target-gate-time", dict(type=float, default=0.0)),
        "t1_levels": ("--t1-levels", dict(type=str, default="inf")),
        "tphi_levels": ("--tphi-levels", dict(type=str, default="inf")),
        "leakage_epsilon": ("--leakage-epsilon", dict(type=float, default=0.0)),
        "tolerance": ("--tolerance", dict(type=float, default=1e-8)),
    }

    single = subparsers.add_parser(
        "single-run",
        help="Run one coherent routed fan-in experiment and compare it to the ideal fan-in unitary.",
    )
    single.add_argument("--path-lengths", type=str, default="1")
    for _, (flag, kwargs) in common.items():
        single.add_argument(flag, **kwargs)

    sweep = subparsers.add_parser(
        "length-sweep",
        help="Sweep a common path length across all routed controls.",
    )
    sweep.add_argument("--lengths", type=str, default="1:2")
    for _, (flag, kwargs) in common.items():
        sweep.add_argument(flag, **kwargs)

    all_tests = subparsers.add_parser("all", help="Run single-run plus length-sweep.")
    all_tests.add_argument("--path-lengths", type=str, default="1")
    all_tests.add_argument("--lengths", type=str, default="1:2")
    for _, (flag, kwargs) in common.items():
        all_tests.add_argument(flag, **kwargs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "single-run":
        result = run_single(args, output_dir)
        print_json(result)
        return

    if args.command == "length-sweep":
        result = run_length_sweep(args, output_dir)
        print_json(result)
        return

    if args.command == "all":
        single_run = run_single(args, output_dir)
        length_sweep = run_length_sweep(args, output_dir)
        combined = {
            "single_run": single_run,
            "length_sweep": length_sweep,
        }
        write_json(output_dir / "qutip_fanin_summary.json", combined)
        print_json(combined)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
