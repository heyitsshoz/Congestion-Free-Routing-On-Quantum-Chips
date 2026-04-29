#!/usr/bin/env python3
"""Noisy QuTiP verification suite for the qudit-routing paper.

This script targets the paper's physical-validation claims:

1. Fidelity vs. distance under level-dependent T1 decay.
2. Leakage-threshold sweeps for imperfect CBL/BCP routing channels.
3. Residual routing-subspace population after noisy cleanup.
4. Optional pure dephasing (Tphi / T2) for stricter transmon-style noise studies.

Notes
-----
- Exact Lindblad evolution grows exponentially with path length. Start small locally
  and move the longer sweeps to HPC.
- The SWAP baseline below uses a conventional move-control-and-restore circuit, while
  the ideal-depth comparison remains in the Cirq script.

Examples
--------
python3 "qutip verification.py" distance-sweep --lengths 2:6 --output-dir qutip_results
python3 "qutip verification.py" leakage-sweep --length 4 --epsilons 0,0.002,0.005,0.01
python3 "qutip verification.py" all --lengths 2:5 --output-dir qutip_results
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
    import qutip as qt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "QuTiP is required for this script. Install it with `pip install qutip`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


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


def parse_level_times(spec: str, label: str, min_values: int = 1) -> list[float]:
    values = parse_float_list(spec)
    if len(values) < min_values:
        raise ValueError(
            f"Provide at least {min_values} value(s) for {label} level times."
        )
    return values


def parse_t1_levels(spec: str) -> list[float]:
    return parse_level_times(spec, "T1", min_values=2)


def parse_tphi_levels(spec: str) -> list[float]:
    return parse_level_times(spec, "Tphi", min_values=1)


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


def basis_state(dimension: int, level: int) -> qt.Qobj:
    return qt.basis(dimension, level)


def plus_state(dimension: int) -> qt.Qobj:
    return (qt.basis(dimension, 0) + qt.basis(dimension, 1)).unit()


def projector(level: int, dimension: int) -> qt.Qobj:
    ket = basis_state(dimension, level)
    return ket * ket.dag()


def product_ket(states: Sequence[qt.Qobj]) -> qt.Qobj:
    return qt.tensor(list(states))


def zero_hamiltonian(dims: Sequence[int]) -> qt.Qobj:
    dimension = int(np.prod(dims, dtype=int))
    return qt.Qobj(
        np.zeros((dimension, dimension), dtype=np.complex128),
        dims=[list(dims), list(dims)],
    )


def conditional_shift_matrix(
    dimension: int, delta: int, mode: str, inverse: bool = False
) -> np.ndarray:
    if mode not in {"cbl", "bcp"}:
        raise ValueError("mode must be 'cbl' or 'bcp'")
    shift = -delta if inverse else delta
    size = dimension * dimension
    matrix = np.zeros((size, size), dtype=np.complex128)
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
    return matrix


def routing_controlled_x_matrix(dimension: int, delta: int) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis_value in range(dimension):
        target = basis_value ^ 1 if (basis_value // delta) % 2 else basis_value
        matrix[target, basis_value] = 1.0
    return matrix


def cnot_matrix() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    transitions = {0: 0, 1: 1, 2: 3, 3: 2}
    for column, row in transitions.items():
        matrix[row, column] = 1.0
    return matrix


def swap_matrix() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    transitions = {0: 0, 1: 2, 2: 1, 3: 3}
    for column, row in transitions.items():
        matrix[row, column] = 1.0
    return matrix


def embed_single_site(
    operator: qt.Qobj, site: int, dims: Sequence[int]
) -> qt.Qobj:
    factors = [qt.qeye(dimension) for dimension in dims]
    factors[site] = operator
    return qt.tensor(factors)


def embed_two_site(
    local_matrix: np.ndarray, site_a: int, site_b: int, dims: Sequence[int]
) -> qt.Qobj:
    if site_a == site_b:
        raise ValueError("The two embedded sites must be distinct.")
    current_order = [site_a, site_b] + [
        index for index in range(len(dims)) if index not in {site_a, site_b}
    ]
    local = qt.Qobj(
        local_matrix,
        dims=[[dims[site_a], dims[site_b]], [dims[site_a], dims[site_b]]],
    )
    factors = [local] + [qt.qeye(dims[index]) for index in current_order[2:]]
    embedded = qt.tensor(factors)
    permutation = [current_order.index(index) for index in range(len(dims))]
    return embedded.permute(permutation)


def build_decay_collapse_ops(dims: Sequence[int], t1_levels: Sequence[float]) -> list[qt.Qobj]:
    local_dimension = dims[0]
    collapse_ops: list[qt.Qobj] = []
    for site in range(len(dims)):
        for level in range(1, local_dimension):
            t1 = t1_levels[min(level - 1, len(t1_levels) - 1)]
            if math.isinf(t1) or t1 <= 0.0:
                continue
            local = np.zeros((local_dimension, local_dimension), dtype=np.complex128)
            local[level - 1, level] = math.sqrt(1.0 / t1)
            operator = qt.Qobj(local, dims=[[local_dimension], [local_dimension]])
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
            local = np.zeros((local_dimension, local_dimension), dtype=np.complex128)
            local[level, level] = math.sqrt(1.0 / tphi)
            operator = qt.Qobj(local, dims=[[local_dimension], [local_dimension]])
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
        return [qt.qeye(dimension)]
    keep = np.eye(dimension, dtype=np.complex128)
    jump = np.zeros((dimension, dimension), dtype=np.complex128)
    for level in range(2, dimension - 1):
        keep[level, level] = math.sqrt(1.0 - epsilon)
        jump[level + 1, level] = math.sqrt(epsilon)
    return [
        qt.Qobj(keep, dims=[[dimension], [dimension]]),
        qt.Qobj(jump, dims=[[dimension], [dimension]]),
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


def apply_gate_with_noise(
    rho: qt.Qobj,
    gate: qt.Qobj,
    gate_time: float,
    h0: qt.Qobj,
    collapse_ops: Sequence[qt.Qobj],
    leakage_site: int | None = None,
    leakage_kraus: Sequence[qt.Qobj] | None = None,
) -> qt.Qobj:
    rho = gate * rho * gate.dag()
    if leakage_site is not None and leakage_kraus is not None:
        rho = apply_local_channel(rho, leakage_site, leakage_kraus, h0.dims[0])
    if gate_time > 0.0 and collapse_ops:
        evolution = qt.mesolve(h0, rho, [0.0, gate_time], c_ops=list(collapse_ops), e_ops=[])
        rho = evolution.states[-1]
    return rho


def ideal_remote_cnot_ket(dimension: int, path_length: int) -> qt.Qobj:
    nodes = [basis_state(dimension, 0) for _ in range(path_length + 1)]
    zero_branch = nodes.copy()
    one_branch = nodes.copy()
    zero_branch[0] = basis_state(dimension, 0)
    one_branch[0] = basis_state(dimension, 1)
    zero_branch[-1] = basis_state(dimension, 0)
    one_branch[-1] = basis_state(dimension, 1)
    return (product_ket(zero_branch) + product_ket(one_branch)).unit()


def ideal_remote_cnot_qubit_ket(path_length: int) -> qt.Qobj:
    return ideal_remote_cnot_ket(dimension=2, path_length=path_length)


def pure_state_overlap(rho: qt.Qobj, ideal_ket: qt.Qobj) -> float:
    value = (ideal_ket.dag() * rho * ideal_ket).full()[0, 0]
    overlap = float(np.real_if_close(value))
    return max(0.0, min(1.0, overlap))


def routing_population(rho: qt.Qobj, site: int, dimension: int, dims: Sequence[int]) -> float:
    local = 0.0 * projector(0, dimension)
    for level in range(2, dimension):
        local += projector(level, dimension)
    observable = embed_single_site(local, site, dims)
    population = float(np.real_if_close(qt.expect(observable, rho)))
    return max(0.0, min(1.0, population))


def simulate_routed_protocol(
    path_length: int,
    dimension: int,
    bus_index: int,
    routing_gate_time: float,
    target_gate_time: float,
    t1_levels: Sequence[float],
    tphi_levels: Sequence[float],
    leakage_epsilon: float,
) -> qt.Qobj:
    if path_length < 2:
        raise ValueError("Path length must be at least 2.")
    dims = [dimension] * (path_length + 1)
    h0 = zero_hamiltonian(dims)
    collapse_ops = build_relaxation_and_dephasing_ops(dims, t1_levels, tphi_levels)
    leakage_kraus = build_leakage_kraus(dimension, leakage_epsilon)
    delta = 2**bus_index

    initial = product_ket(
        [plus_state(dimension)]
        + [basis_state(dimension, 0) for _ in range(path_length)]
    )
    rho = qt.ket2dm(initial)

    gate = embed_two_site(conditional_shift_matrix(dimension, delta, "cbl"), 0, 1, dims)
    rho = apply_gate_with_noise(
        rho,
        gate,
        gate_time=routing_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
        leakage_site=1,
        leakage_kraus=leakage_kraus,
    )

    for site in range(1, path_length):
        gate = embed_two_site(
            conditional_shift_matrix(dimension, delta, "bcp"), site, site + 1, dims
        )
        rho = apply_gate_with_noise(
            rho,
            gate,
            gate_time=routing_gate_time,
            h0=h0,
            collapse_ops=collapse_ops,
            leakage_site=site + 1,
            leakage_kraus=leakage_kraus,
        )

    gate = embed_single_site(
        qt.Qobj(
            routing_controlled_x_matrix(dimension, delta),
            dims=[[dimension], [dimension]],
        ),
        path_length,
        dims,
    )
    rho = apply_gate_with_noise(
        rho,
        gate,
        gate_time=target_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
    )

    for site in range(path_length - 1, 0, -1):
        gate = embed_two_site(
            conditional_shift_matrix(dimension, delta, "bcp", inverse=True),
            site,
            site + 1,
            dims,
        )
        rho = apply_gate_with_noise(
            rho,
            gate,
            gate_time=routing_gate_time,
            h0=h0,
            collapse_ops=collapse_ops,
            leakage_site=site + 1,
            leakage_kraus=leakage_kraus,
        )

    gate = embed_two_site(
        conditional_shift_matrix(dimension, delta, "cbl", inverse=True), 0, 1, dims
    )
    rho = apply_gate_with_noise(
        rho,
        gate,
        gate_time=routing_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
        leakage_site=1,
        leakage_kraus=leakage_kraus,
    )
    return rho


def simulate_swap_baseline(
    path_length: int,
    swap_gate_time: float,
    cnot_gate_time: float,
    qubit_t1: float,
    qubit_tphi: float,
) -> qt.Qobj:
    dims = [2] * (path_length + 1)
    h0 = zero_hamiltonian(dims)
    collapse_ops = build_relaxation_and_dephasing_ops(dims, [qubit_t1], [qubit_tphi])
    initial = product_ket([plus_state(2)] + [basis_state(2, 0) for _ in range(path_length)])
    rho = qt.ket2dm(initial)

    for site in range(max(path_length - 1, 0)):
        gate = embed_two_site(swap_matrix(), site, site + 1, dims)
        rho = apply_gate_with_noise(
            rho,
            gate,
            gate_time=swap_gate_time,
            h0=h0,
            collapse_ops=collapse_ops,
        )

    control_site = max(path_length - 1, 0)
    target_site = path_length
    gate = embed_two_site(cnot_matrix(), control_site, target_site, dims)
    rho = apply_gate_with_noise(
        rho,
        gate,
        gate_time=cnot_gate_time,
        h0=h0,
        collapse_ops=collapse_ops,
    )

    for site in range(max(path_length - 2, -1), -1, -1):
        gate = embed_two_site(swap_matrix(), site, site + 1, dims)
        rho = apply_gate_with_noise(
            rho,
            gate,
            gate_time=swap_gate_time,
            h0=h0,
            collapse_ops=collapse_ops,
        )

    return rho


def run_distance_sweep(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    lengths = parse_int_list(args.lengths)
    t1_levels = parse_t1_levels(args.t1_levels)
    tphi_levels = parse_tphi_levels(args.tphi_levels)
    rows: list[dict[str, object]] = []

    for path_length in lengths:
        routed_rho = simulate_routed_protocol(
            path_length=path_length,
            dimension=args.dimension,
            bus_index=args.bus_index,
            routing_gate_time=args.routing_gate_time,
            target_gate_time=args.target_gate_time,
            t1_levels=t1_levels,
            tphi_levels=tphi_levels,
            leakage_epsilon=args.leakage_epsilon,
        )
        swap_rho = simulate_swap_baseline(
            path_length=path_length,
            swap_gate_time=args.swap_gate_time,
            cnot_gate_time=args.cnot_gate_time,
            qubit_t1=args.qubit_t1,
            qubit_tphi=args.qubit_tphi,
        )
        routed_fidelity = pure_state_overlap(
            routed_rho, ideal_remote_cnot_ket(args.dimension, path_length)
        )
        swap_fidelity = pure_state_overlap(swap_rho, ideal_remote_cnot_qubit_ket(path_length))
        rows.append(
            {
                "path_length": path_length,
                "routed_fidelity": routed_fidelity,
                "swap_fidelity": swap_fidelity,
                "fidelity_advantage": routed_fidelity - swap_fidelity,
            }
        )

    summary = {
        "rows": rows,
        "dimension": args.dimension,
        "bus_index": args.bus_index,
        "t1_levels": t1_levels,
        "tphi_levels": tphi_levels,
        "routing_gate_time": args.routing_gate_time,
        "target_gate_time": args.target_gate_time,
        "swap_gate_time": args.swap_gate_time,
        "cnot_gate_time": args.cnot_gate_time,
        "qubit_t1": args.qubit_t1,
        "qubit_tphi": args.qubit_tphi,
        "leakage_epsilon": args.leakage_epsilon,
    }
    write_json(output_dir / "distance_sweep.json", summary)
    write_csv(output_dir / "distance_sweep.csv", rows)

    if plt is not None:
        plt.figure(figsize=(7, 4.5))
        plt.plot(
            [row["path_length"] for row in rows],
            [row["routed_fidelity"] for row in rows],
            marker="o",
            label="Qudit routed protocol",
        )
        plt.plot(
            [row["path_length"] for row in rows],
            [row["swap_fidelity"] for row in rows],
            marker="s",
            label="SWAP baseline",
        )
        plt.xlabel("Path length L")
        plt.ylabel("State fidelity")
        plt.title("Fidelity vs. distance under decoherence")
        plt.ylim(0.0, 1.02)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "distance_sweep.png", dpi=200)
        plt.close()

    return summary


def run_leakage_sweep(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    epsilons = parse_float_list(args.epsilons)
    t1_levels = parse_t1_levels(args.t1_levels)
    tphi_levels = parse_tphi_levels(args.tphi_levels)
    swap_rho = simulate_swap_baseline(
        path_length=args.length,
        swap_gate_time=args.swap_gate_time,
        cnot_gate_time=args.cnot_gate_time,
        qubit_t1=args.qubit_t1,
        qubit_tphi=args.qubit_tphi,
    )
    swap_fidelity = pure_state_overlap(swap_rho, ideal_remote_cnot_qubit_ket(args.length))

    rows: list[dict[str, object]] = []
    leakage_threshold_estimate = None
    for epsilon in epsilons:
        routed_rho = simulate_routed_protocol(
            path_length=args.length,
            dimension=args.dimension,
            bus_index=args.bus_index,
            routing_gate_time=args.routing_gate_time,
            target_gate_time=args.target_gate_time,
            t1_levels=t1_levels,
            tphi_levels=tphi_levels,
            leakage_epsilon=epsilon,
        )
        routed_fidelity = pure_state_overlap(
            routed_rho, ideal_remote_cnot_ket(args.dimension, args.length)
        )
        residuals = [
            routing_population(
                routed_rho, site=site, dimension=args.dimension, dims=[args.dimension] * (args.length + 1)
            )
            for site in range(1, args.length)
        ]
        outperforms_swap = routed_fidelity >= swap_fidelity
        if outperforms_swap:
            leakage_threshold_estimate = epsilon
        rows.append(
            {
                "leakage_epsilon": epsilon,
                "routed_fidelity": routed_fidelity,
                "swap_fidelity": swap_fidelity,
                "fidelity_advantage": routed_fidelity - swap_fidelity,
                "max_intermediate_routing_population": max(residuals) if residuals else 0.0,
                "outperforms_swap": outperforms_swap,
            }
        )

    summary = {
        "length": args.length,
        "dimension": args.dimension,
        "bus_index": args.bus_index,
        "t1_levels": t1_levels,
        "tphi_levels": tphi_levels,
        "swap_fidelity": swap_fidelity,
        "qubit_t1": args.qubit_t1,
        "qubit_tphi": args.qubit_tphi,
        "leakage_threshold_estimate": leakage_threshold_estimate,
        "rows": rows,
    }
    write_json(output_dir / "leakage_sweep.json", summary)
    write_csv(output_dir / "leakage_sweep.csv", rows)

    if plt is not None:
        plt.figure(figsize=(7, 4.5))
        plt.plot(
            [row["leakage_epsilon"] for row in rows],
            [row["routed_fidelity"] for row in rows],
            marker="o",
            label="Qudit routed protocol",
        )
        plt.axhline(
            y=swap_fidelity,
            color="tab:red",
            linestyle="--",
            label="SWAP baseline",
        )
        plt.xlabel("Per-hop leakage epsilon")
        plt.ylabel("State fidelity")
        plt.title("Leakage threshold sweep")
        plt.ylim(0.0, 1.02)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "leakage_sweep.png", dpi=200)
        plt.close()

    return summary


def run_residual_excitation(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    t1_levels = parse_t1_levels(args.t1_levels)
    tphi_levels = parse_tphi_levels(args.tphi_levels)
    routed_rho = simulate_routed_protocol(
        path_length=args.length,
        dimension=args.dimension,
        bus_index=args.bus_index,
        routing_gate_time=args.routing_gate_time,
        target_gate_time=args.target_gate_time,
        t1_levels=t1_levels,
        tphi_levels=tphi_levels,
        leakage_epsilon=args.leakage_epsilon,
    )
    dims = [args.dimension] * (args.length + 1)
    intermediate_rows = []
    for site in range(1, args.length):
        intermediate_rows.append(
            {
                "site": site,
                "routing_population": routing_population(
                    routed_rho, site=site, dimension=args.dimension, dims=dims
                ),
            }
        )

    summary = {
        "length": args.length,
        "dimension": args.dimension,
        "bus_index": args.bus_index,
        "leakage_epsilon": args.leakage_epsilon,
        "t1_levels": t1_levels,
        "tphi_levels": tphi_levels,
        "intermediate_rows": intermediate_rows,
        "max_intermediate_routing_population": max(
            [row["routing_population"] for row in intermediate_rows],
            default=0.0,
        ),
    }
    write_json(output_dir / "residual_excitation.json", summary)
    write_csv(output_dir / "residual_excitation.csv", intermediate_rows)

    if plt is not None and intermediate_rows:
        plt.figure(figsize=(7, 4.5))
        plt.bar(
            [row["site"] for row in intermediate_rows],
            [row["routing_population"] for row in intermediate_rows],
            color="tab:blue",
        )
        plt.xlabel("Intermediate site")
        plt.ylabel("Routing-subspace population")
        plt.title("Residual excitation after noisy cleanup")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "residual_excitation.png", dpi=200)
        plt.close()

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qutip_results"),
        help="Directory where JSON/CSV/plot outputs are written.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dimension", type=int, default=8)
        subparser.add_argument("--bus-index", type=int, default=1)
        subparser.add_argument(
            "--t1-levels",
            type=str,
            default="80,45,25,14,8,5,3",
            help="Comma-separated T1 values for levels |1>, |2>, ... in the qudit ladder.",
        )
        subparser.add_argument(
            "--tphi-levels",
            type=str,
            default="inf,inf,inf,inf,inf,inf,inf",
            help=(
                "Comma-separated pure-dephasing times for levels |1>, |2>, ... . "
                "Use `inf` to disable dephasing for a level."
            ),
        )
        subparser.add_argument(
            "--routing-gate-time",
            type=float,
            default=1.0,
            help="Compiled duration assigned to each CBL/BCP primitive.",
        )
        subparser.add_argument(
            "--target-gate-time",
            type=float,
            default=1.0,
            help="Duration assigned to the routed target interaction.",
        )
        subparser.add_argument(
            "--swap-gate-time",
            type=float,
            default=3.0,
            help="Duration assigned to each SWAP in the baseline model.",
        )
        subparser.add_argument(
            "--cnot-gate-time",
            type=float,
            default=1.0,
            help="Duration assigned to the baseline nearest-neighbor CNOT.",
        )
        subparser.add_argument("--qubit-t1", type=float, default=80.0)
        subparser.add_argument(
            "--qubit-tphi",
            type=float,
            default=math.inf,
            help="Pure-dephasing time used in the qubit SWAP baseline.",
        )

    distance = subparsers.add_parser(
        "distance-sweep",
        help="Compare routed fidelity and SWAP fidelity over path length.",
    )
    add_common_options(distance)
    distance.add_argument("--lengths", type=str, default="2:5")
    distance.add_argument("--leakage-epsilon", type=float, default=0.0)

    leakage = subparsers.add_parser(
        "leakage-sweep",
        help="Find the per-hop leakage level where the routed protocol stops beating SWAPs.",
    )
    add_common_options(leakage)
    leakage.add_argument("--length", type=int, default=4)
    leakage.add_argument("--epsilons", type=str, default="0,0.002,0.005,0.01,0.02")

    residual = subparsers.add_parser(
        "residual",
        help="Measure residual routing-subspace population after cleanup.",
    )
    add_common_options(residual)
    residual.add_argument("--length", type=int, default=4)
    residual.add_argument("--leakage-epsilon", type=float, default=0.01)

    all_tests = subparsers.add_parser("all", help="Run every QuTiP verification.")
    add_common_options(all_tests)
    all_tests.add_argument("--lengths", type=str, default="2:5")
    all_tests.add_argument("--length", type=int, default=4)
    all_tests.add_argument("--epsilons", type=str, default="0,0.002,0.005,0.01,0.02")
    all_tests.add_argument("--leakage-epsilon", type=float, default=0.01)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "distance-sweep":
        summary = run_distance_sweep(args, output_dir)
        print_json(summary)
        return

    if args.command == "leakage-sweep":
        summary = run_leakage_sweep(args, output_dir)
        print_json(summary)
        return

    if args.command == "residual":
        summary = run_residual_excitation(args, output_dir)
        print_json(summary)
        return

    if args.command == "all":
        distance = run_distance_sweep(args, output_dir)
        leakage = run_leakage_sweep(args, output_dir)
        residual = run_residual_excitation(args, output_dir)
        combined = {
            "distance_sweep": distance,
            "leakage_sweep": leakage,
            "residual_excitation": residual,
        }
        write_json(output_dir / "qutip_verification_summary.json", combined)
        print_json(combined)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
