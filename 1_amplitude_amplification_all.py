#!/usr/bin/env python3

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
try:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex_mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

@dataclass(frozen=True)
class verification_common__RunContext:
    script_path: Path
    result_dir: Path

def verification_common__setup_run_context(script_file: str) -> verification_common__RunContext:
    script_path = Path(script_file).resolve()
    result_dir = script_path.parent / f'[RESULT]{script_path.stem}'
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(result_dir / '.mplconfig'))
    return verification_common__RunContext(script_path=script_path, result_dir=result_dir)

def verification_common__parse_int_list(spec: str) -> list[int]:
    text = spec.strip()
    if ':' in text:
        start_text, end_text = text.split(':', maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError('Range end must be >= range start.')
        return list(range(start, end + 1))
    return [int(item.strip()) for item in text.split(',') if item.strip()]

def verification_common__parse_float_list(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(',') if item.strip()]

def verification_common__write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(verification_common__to_jsonable(payload), indent=2), encoding='utf-8')

def verification_common__write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def verification_common__to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): verification_common__to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [verification_common__to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def verification_common__statevector_fidelity(left: Sequence[complex], right: Sequence[complex]) -> float:
    lhs = np.asarray(left, dtype=np.complex128)
    rhs = np.asarray(right, dtype=np.complex128)
    overlap = np.vdot(lhs, rhs)
    return float(np.clip(abs(overlap) ** 2, 0.0, 1.0))

def verification_common__maybe_save_figure(fig: Any, path: Path, *, dpi: int=220) -> None:
    if plt is None:
        return
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def verification_common__probability_mass(statevector: Sequence[complex], good_indices: Sequence[int]) -> float:
    amplitudes = np.asarray(statevector, dtype=np.complex128)
    probs = np.abs(amplitudes) ** 2
    return float(np.sum(probs[list(good_indices)]))

def verification_common__format_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]

def verification_common__launched_without_cli() -> bool:
    try:
        return len(sys.argv) == 1 and sys.stdin.isatty()
    except Exception:
        return False

def verification_common__prompt_int(message: str, default: int) -> int:
    raw = input(f'{message} [{default}]: ').strip()
    return default if not raw else int(raw)

def verification_common__configure_native_thread_limits(limit: int | None) -> None:
    if limit is None:
        return
    resolved_limit = int(limit)
    if resolved_limit <= 0:
        return
    value = str(resolved_limit)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = value

def verification_common__resolve_parallelism(workers: int | None, native_threads_per_worker: int | None) -> tuple[int, int | None]:
    resolved_workers = max(1, int(workers or 1))
    resolved_native_threads = native_threads_per_worker
    if resolved_native_threads is not None and int(resolved_native_threads) <= 0:
        resolved_native_threads = None
    if resolved_native_threads is None and resolved_workers > 1:
        resolved_native_threads = 1
    if resolved_native_threads is not None:
        resolved_native_threads = max(1, int(resolved_native_threads))
    return (resolved_workers, resolved_native_threads)

def verification_common__parallel_worker_init(native_threads_per_worker: int | None) -> None:
    verification_common__configure_native_thread_limits(native_threads_per_worker)

def verification_common__parallel_map(worker: Callable[[Any], Any], tasks: Sequence[Any], *, workers: int, native_threads_per_worker: int | None) -> list[Any]:
    task_list = list(tasks)
    if not task_list:
        return []
    verification_common__configure_native_thread_limits(native_threads_per_worker)
    if workers <= 1 or len(task_list) <= 1:
        return [worker(task) for task in task_list]
    ctx = mp.get_context("spawn")
    timeout_env = os.environ.get("VERIFY_PARALLEL_TIMEOUT_SEC")
    timeout_seconds = float(timeout_env) if timeout_env else None
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=verification_common__parallel_worker_init, initargs=(native_threads_per_worker,), mp_context=ctx) as executor:
        futures = [executor.submit(worker, task) for task in task_list]
        results: list[Any] = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout_seconds):
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(f'Worker task failed: {exc}') from exc
            results.append(future.result())
        if len(results) != len(futures):
            raise RuntimeError('Parallel execution did not complete all tasks.')
        return results


# Inlined from: qubit_reference.py

import math
import random
from typing import Sequence
import numpy as np
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)

def qubit_reference__rx(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array([[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]], dtype=np.complex128)

def qubit_reference__rz(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array([[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128)

def qubit_reference__phase(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)

def qubit_reference__uniform_superposition(n_qubits: int) -> np.ndarray:
    state = np.ones(2 ** n_qubits, dtype=np.complex128)
    return state / np.linalg.norm(state)

def qubit_reference__normalize_state(state: Sequence[complex]) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError('Cannot normalize the zero vector.')
    return vector / norm

def qubit_reference__random_statevector(n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=2 ** n_qubits)
    imag = rng.normal(size=2 ** n_qubits)
    return qubit_reference__normalize_state(real + 1j * imag)

def qubit_reference__index_from_bits(bits: Sequence[int]) -> int:
    index = 0
    for bit in bits:
        index = index << 1 | int(bit)
    return index

def qubit_reference__bits_from_index(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> shift & 1 for shift in reversed(range(width))))

def qubit_reference__apply_single_qubit_gate(statevector: Sequence[complex], gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = np.asarray(statevector, dtype=np.complex128).reshape((2,) * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return np.asarray(tensor, dtype=np.complex128).reshape(-1)

def qubit_reference__apply_swap(statevector: Sequence[complex], left: int, right: int, n_qubits: int) -> np.ndarray:
    if left == right:
        return np.asarray(statevector, dtype=np.complex128)
    tensor = np.asarray(statevector, dtype=np.complex128).reshape((2,) * n_qubits)
    tensor = np.swapaxes(tensor, left, right)
    return np.asarray(tensor, dtype=np.complex128).reshape(-1)

def qubit_reference__apply_controlled_x(statevector: Sequence[complex], control: int, target: int, n_qubits: int) -> np.ndarray:
    result = np.zeros(2 ** n_qubits, dtype=np.complex128)
    amplitudes = np.asarray(statevector, dtype=np.complex128)
    for index, amplitude in enumerate(amplitudes):
        bits = list(qubit_reference__bits_from_index(index, n_qubits))
        if bits[control] == 1:
            bits[target] ^= 1
        result[qubit_reference__index_from_bits(bits)] += amplitude
    return result

def qubit_reference__apply_controlled_phase(statevector: Sequence[complex], control: int, target: int, theta: float, n_qubits: int) -> np.ndarray:
    result = np.asarray(statevector, dtype=np.complex128).copy()
    phase_value = np.exp(1j * theta)
    for index in range(result.size):
        bits = qubit_reference__bits_from_index(index, n_qubits)
        if bits[control] == 1 and bits[target] == 1:
            result[index] *= phase_value
    return result

def qubit_reference__apply_marked_phase(statevector: Sequence[complex], marked_bits: Sequence[int]) -> np.ndarray:
    result = np.asarray(statevector, dtype=np.complex128).copy()
    result[qubit_reference__index_from_bits(marked_bits)] *= -1.0
    return result

def qubit_reference__grover_statevector(n_qubits: int, marked_bits: Sequence[int], iterations: int) -> np.ndarray:
    state = qubit_reference__uniform_superposition(n_qubits)
    uniform = state.copy()
    for _ in range(iterations):
        state = qubit_reference__apply_marked_phase(state, marked_bits)
        reflected = 2.0 * np.vdot(uniform, state) * uniform - state
        state = np.asarray(reflected, dtype=np.complex128)
    return state

def qubit_reference__grover_success_formula(n_qubits: int, marked_count: int, iteration: int) -> float:
    theta = math.asin(math.sqrt(marked_count / 2 ** n_qubits))
    return float(math.sin((2 * iteration + 1) * theta) ** 2)

def qubit_reference__qft_statevector(input_state: Sequence[complex], n_qubits: int, *, with_swaps: bool=False) -> np.ndarray:
    state = qubit_reference__normalize_state(input_state)
    for target in range(n_qubits):
        for control in range(target + 1, n_qubits):
            theta = math.pi / 2 ** (control - target)
            state = qubit_reference__apply_controlled_phase(state, control, target, theta, n_qubits)
        state = qubit_reference__apply_single_qubit_gate(state, H, target, n_qubits)
    if with_swaps:
        for index in range(n_qubits // 2):
            state = qubit_reference__apply_swap(state, index, n_qubits - 1 - index, n_qubits)
    return state

def qubit_reference__build_graph(n_qubits: int, graph_type: str, *, seed: int, edge_probability: float) -> list[tuple[int, int]]:
    normalized = graph_type.strip().lower()
    if normalized == 'path':
        return [(index, index + 1) for index in range(n_qubits - 1)]
    if normalized == 'cycle':
        if n_qubits < 3:
            return qubit_reference__build_graph(n_qubits, 'path', seed=seed, edge_probability=edge_probability)
        return [(index, (index + 1) % n_qubits) for index in range(n_qubits)]
    if normalized == 'complete':
        return [(left, right) for left in range(n_qubits) for right in range(left + 1, n_qubits)]
    if normalized == 'random':
        rng = random.Random(seed)
        edges: list[tuple[int, int]] = []
        for left in range(n_qubits):
            for right in range(left + 1, n_qubits):
                if rng.random() <= edge_probability:
                    edges.append((left, right))
        if not edges:
            return qubit_reference__build_graph(n_qubits, 'path', seed=seed, edge_probability=edge_probability)
        return edges
    raise ValueError(f'Unsupported graph type: {graph_type}')

def qubit_reference__qaoa_statevector(n_qubits: int, edges: Sequence[tuple[int, int]], gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    state = qubit_reference__uniform_superposition(n_qubits)
    for gamma, beta in zip(gammas, betas):
        for left, right in edges:
            state = qubit_reference__apply_controlled_x(state, left, right, n_qubits)
            state = qubit_reference__apply_single_qubit_gate(state, qubit_reference__rz(-float(gamma)), right, n_qubits)
            state = qubit_reference__apply_controlled_x(state, left, right, n_qubits)
        for qubit in range(n_qubits):
            state = qubit_reference__apply_single_qubit_gate(state, qubit_reference__rx(2.0 * float(beta)), qubit, n_qubits)
    return state

def qubit_reference__maxcut_value(index: int, edges: Sequence[tuple[int, int]]) -> int:
    if not edges:
        return 0
    n_qubits = max((max(left, right) for left, right in edges)) + 1
    bits = qubit_reference__bits_from_index(index, n_qubits)
    return qubit_reference__maxcut_value_bits(bits, edges)

def qubit_reference__maxcut_value_bits(bits: Sequence[int], edges: Sequence[tuple[int, int]]) -> int:
    return int(sum((int(bits[left] != bits[right]) for left, right in edges)))

def qubit_reference__maxcut_expectation(statevector: Sequence[complex], edges: Sequence[tuple[int, int]]) -> float:
    amplitudes = np.asarray(statevector, dtype=np.complex128)
    probs = np.abs(amplitudes) ** 2
    n_qubits = int(round(math.log2(len(amplitudes))))
    total = 0.0
    for index, probability in enumerate(probs):
        total += float(probability) * qubit_reference__maxcut_value_bits(qubit_reference__bits_from_index(index, n_qubits), edges)
    return total

def qubit_reference__classical_maxcut_optimum(n_qubits: int, edges: Sequence[tuple[int, int]]) -> int:
    return max((qubit_reference__maxcut_value_bits(qubit_reference__bits_from_index(index, n_qubits), edges) for index in range(2 ** n_qubits)))


# Inlined from: verification_aer.py

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
except Exception as exc:
    raise SystemExit('Qiskit is required for these verification scripts.') from exc
try:
    from qiskit_aer import AerSimulator
except Exception as exc:
    raise SystemExit('Qiskit Aer is required for the Aer CPU/GPU verification scripts.') from exc

@dataclass(frozen=True)
class verification_aer__CircuitStats:
    logical_depth: int
    logical_size: int
    logical_two_qubit_count: int
    span_total: int
    span_max: int
    span_mean: float
    max_gate_width: int
    max_fan_in: int
    max_qubit_load: int
    weighted_qubit_load_max: int
    target_load_max: int
    multi_qubit_gate_count: int

def verification_aer__build_coupling_map(topology: str, num_qubits: int) -> CouplingMap | None:
    name = topology.strip().lower()
    if name == 'alltoall':
        return None
    edges: list[tuple[int, int]] = []
    if name == 'line':
        for left in range(num_qubits - 1):
            edges.append((left, left + 1))
            edges.append((left + 1, left))
        return CouplingMap(edges)
    if name == 'ring':
        if num_qubits < 3:
            return verification_aer__build_coupling_map('line', num_qubits)
        for left in range(num_qubits):
            right = (left + 1) % num_qubits
            edges.append((left, right))
            edges.append((right, left))
        return CouplingMap(edges)
    if name == 'grid':
        if num_qubits <= 3:
            return verification_aer__build_coupling_map('line', num_qubits)
        rows = int(math.floor(math.sqrt(num_qubits)))
        cols = int(math.ceil(num_qubits / max(rows, 1)))
        for index in range(num_qubits):
            row = index // cols
            col = index % cols
            if col + 1 < cols:
                neighbor = index + 1
                if neighbor < num_qubits and neighbor // cols == row:
                    edges.append((index, neighbor))
                    edges.append((neighbor, index))
            if row + 1 < rows:
                neighbor = index + cols
                if neighbor < num_qubits:
                    edges.append((index, neighbor))
                    edges.append((neighbor, index))
        if not edges:
            return verification_aer__build_coupling_map('line', num_qubits)
        return CouplingMap(edges)
    raise ValueError(f'Unsupported topology: {topology}')

def verification_aer__parse_topologies(spec: str) -> list[str]:
    names = [item.strip().lower() for item in spec.split(',') if item.strip()]
    if not names:
        raise ValueError('Provide at least one topology.')
    return names

def verification_aer__two_qubit_gate_spans(circuit: QuantumCircuit) -> tuple[int, int, float, int]:
    qubit_to_index = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    spans: list[int] = []
    count = 0
    for instruction in circuit.data:
        qargs = instruction.qubits
        if len(qargs) != 2:
            continue
        idx0 = qubit_to_index[qargs[0]]
        idx1 = qubit_to_index[qargs[1]]
        spans.append(abs(idx1 - idx0))
        count += 1
    if not spans:
        return (0, 0, 0.0, 0)
    return (int(sum(spans)), int(max(spans)), float(np.mean(spans)), count)

def verification_aer__logical_circuit_stats(circuit: QuantumCircuit) -> verification_aer__CircuitStats:
    span_total, span_max, span_mean, two_qubit_count = verification_aer__two_qubit_gate_spans(circuit)
    qubit_to_index = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    qubit_load = [0 for _ in circuit.qubits]
    weighted_qubit_load = [0 for _ in circuit.qubits]
    target_load = [0 for _ in circuit.qubits]
    max_gate_width = 1
    max_fan_in = 0
    multi_qubit_gate_count = 0
    for instruction in circuit.data:
        qargs = instruction.qubits
        width = len(qargs)
        if width <= 0:
            continue
        max_gate_width = max(max_gate_width, width)
        if width >= 2:
            max_fan_in = max(max_fan_in, width - 1)
            if width >= 3:
                multi_qubit_gate_count += 1
            indices = [qubit_to_index[qubit] for qubit in qargs]
            span = max(indices) - min(indices)
            for index in indices:
                qubit_load[index] += 1
                weighted_qubit_load[index] += max(1, span)
            target_load[indices[-1]] += 1
    return verification_aer__CircuitStats(logical_depth=int(circuit.depth() or 0), logical_size=int(circuit.size()), logical_two_qubit_count=int(two_qubit_count), span_total=span_total, span_max=span_max, span_mean=span_mean, max_gate_width=max_gate_width, max_fan_in=max_fan_in, max_qubit_load=max(qubit_load, default=0), weighted_qubit_load_max=max(weighted_qubit_load, default=0), target_load_max=max(target_load, default=0), multi_qubit_gate_count=multi_qubit_gate_count)

def verification_aer__transpile_with_metrics(circuit: QuantumCircuit, *, topology: str, optimization_level: int, seed: int) -> tuple[QuantumCircuit, dict[str, Any]]:
    coupling_map = verification_aer__build_coupling_map(topology, circuit.num_qubits)
    kwargs: dict[str, Any] = {'optimization_level': optimization_level, 'seed_transpiler': seed, 'initial_layout': list(range(circuit.num_qubits))}
    if coupling_map is not None:
        kwargs['coupling_map'] = coupling_map
        kwargs['routing_method'] = 'sabre'
        kwargs['layout_method'] = 'sabre'
    transpiled_circuit = transpile(circuit, **kwargs)
    counts = transpiled_circuit.count_ops()
    return (transpiled_circuit, {'topology': topology, 'transpiled_depth': int(transpiled_circuit.depth() or 0), 'transpiled_size': int(transpiled_circuit.size()), 'cx_count': int(counts.get('cx', 0)), 'swap_count': int(counts.get('swap', 0)), 'op_counts': {str(key): int(val) for key, val in counts.items()}})

def verification_aer__simulate_statevector(circuit: QuantumCircuit, *, device: str, seed: int) -> np.ndarray:
    probe = circuit.copy()
    probe.save_statevector(label='final_statevector')
    backend_kwargs: dict[str, Any] = {'method': 'statevector', 'seed_simulator': seed}
    if device.upper() == 'GPU':
        backend_kwargs['device'] = 'GPU'
    backend = AerSimulator(**backend_kwargs)
    result = backend.run(probe, shots=1).result()
    state = result.data(0)['final_statevector']
    return np.asarray(state, dtype=np.complex128)


# Inlined lazily from: qudit_cirq_verification.py

_qudit_cirq_verification_ready = False

def _init_qudit_cirq_verification() -> None:
    global _qudit_cirq_verification_ready
    if _qudit_cirq_verification_ready:
        return
    global math, dataclass, Callable, Sequence, np, qudit_cirq_verification__BooleanRule, qudit_cirq_verification__RoutingLayout, qudit_cirq_verification__CircuitMetrics, qudit_cirq_verification__and_rule, qudit_cirq_verification__minimum_dimension_for_controls, qudit_cirq_verification__basis_state, qudit_cirq_verification__product_state
    global qudit_cirq_verification__logical_block_operator, qudit_cirq_verification__x_matrix, qudit_cirq_verification__z_matrix, qudit_cirq_verification__h_matrix, qudit_cirq_verification__phase_matrix, qudit_cirq_verification__rz_matrix, qudit_cirq_verification__rx_matrix, qudit_cirq_verification__LogicalQuditGate, qudit_cirq_verification__ConditionalShiftGate, qudit_cirq_verification__CombinedShiftGate, qudit_cirq_verification__BooleanRoutingGate, qudit_cirq_verification__RoutedSingleBusGate
    global qudit_cirq_verification__MultiBusPhaseGate, qudit_cirq_verification__append_packed_moments, qudit_cirq_verification__build_layout, qudit_cirq_verification__build_fanin_circuit, qudit_cirq_verification__build_fanin_circuit_serialized, qudit_cirq_verification__build_line_path, qudit_cirq_verification__build_single_bus_circuit, qudit_cirq_verification__simulate_circuit, qudit_cirq_verification__circuit_metrics, qudit_cirq_verification__flat_index, qudit_cirq_verification__embed_qubit_state, qudit_cirq_verification__extract_clean_logical_state
    global qudit_cirq_verification__site_probabilities, qudit_cirq_verification__max_routing_population
    import math
    from dataclasses import dataclass
    from typing import Callable, Sequence
    import numpy as np
    try:
        import cirq
    except ImportError as exc:
        raise SystemExit('Cirq is required for the qudit verification scripts. Install it with `pip install cirq`.') from exc

    @dataclass(frozen=True)
    class qudit_cirq_verification__BooleanRule:
        name: str
        evaluator: Callable[[Sequence[int]], int]

    @dataclass(frozen=True)
    class qudit_cirq_verification__RoutingLayout:
        paths: tuple[tuple[int, ...], ...]
        target_site: int
        total_sites: int

        @property
        def source_sites(self) -> tuple[int, ...]:
            return tuple((path[0] for path in self.paths))

        @property
        def payload_sites(self) -> tuple[int, ...]:
            payload: list[int] = []
            for path in self.paths:
                payload.extend(path[1:-1])
            return tuple(payload)

    @dataclass(frozen=True)
    class qudit_cirq_verification__CircuitMetrics:
        moment_count: int
        operation_count: int
        one_qudit_gate_count: int
        two_qudit_gate_count: int
        max_gate_width: int

    def qudit_cirq_verification__and_rule(controls: int) -> qudit_cirq_verification__BooleanRule:
        return qudit_cirq_verification__BooleanRule(f'and:{controls}', lambda bits: int(all(bits)))

    def qudit_cirq_verification__minimum_dimension_for_controls(controls: int) -> int:
        return 2 if controls <= 0 else 2 ** (controls + 1)

    def qudit_cirq_verification__basis_state(dimension: int, level: int) -> np.ndarray:
        state = np.zeros(dimension, dtype=np.complex128)
        state[level] = 1.0
        return state

    def qudit_cirq_verification__product_state(local_states: Sequence[np.ndarray]) -> np.ndarray:
        state = np.asarray(local_states[0], dtype=np.complex128)
        for local_state in local_states[1:]:
            state = np.kron(state, np.asarray(local_state, dtype=np.complex128))
        return state

    def qudit_cirq_verification__logical_block_operator(dimension: int, unitary: np.ndarray) -> np.ndarray:
        matrix = np.eye(dimension, dtype=np.complex128)
        matrix[:2, :2] = np.asarray(unitary, dtype=np.complex128)
        return matrix

    def qudit_cirq_verification__x_matrix() -> np.ndarray:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def qudit_cirq_verification__z_matrix() -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def qudit_cirq_verification__h_matrix() -> np.ndarray:
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)

    def qudit_cirq_verification__phase_matrix(theta: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)

    def qudit_cirq_verification__rz_matrix(theta: float) -> np.ndarray:
        half = theta / 2.0
        return np.array([[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128)

    def qudit_cirq_verification__rx_matrix(theta: float) -> np.ndarray:
        half = theta / 2.0
        return np.array([[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]], dtype=np.complex128)

    class qudit_cirq_verification__LogicalQuditGate(cirq.Gate):

        def __init__(self, dimension: int, unitary: np.ndarray, label: str) -> None:
            self.dimension = dimension
            self.unitary = np.asarray(unitary, dtype=np.complex128)
            self.label = label

        def _qid_shape_(self) -> tuple[int]:
            return (self.dimension,)

        def _unitary_(self) -> np.ndarray:
            return qudit_cirq_verification__logical_block_operator(self.dimension, self.unitary)

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
            return self.label

    class qudit_cirq_verification__ConditionalShiftGate(cirq.Gate):

        def __init__(self, dimension: int, delta: int, mode: str, inverse: bool=False) -> None:
            if mode not in {'cbl', 'bcp'}:
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
                    active = control % 2 if self.mode == 'cbl' else control // self.delta % 2
                    next_target = (target + active * shift) % self.dimension
                    row = control * self.dimension + next_target
                    column = control * self.dimension + target
                    matrix[row, column] = 1.0
            return matrix

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> tuple[str, str]:
            label = 'CBL' if self.mode == 'cbl' else 'BCP'
            suffix = '-1' if self.inverse else ''
            return (f'{label}{suffix}[{self.delta}]',) * 2

    class qudit_cirq_verification__CombinedShiftGate(cirq.Gate):

        def __init__(self, dimension: int, bus_specs: Sequence[tuple[int, str]], inverse: bool=False) -> None:
            self.dimension = dimension
            self.bus_specs = tuple(((int(bus_index), str(mode)) for bus_index, mode in bus_specs))
            for _, mode in self.bus_specs:
                if mode not in {'cbl', 'bcp'}:
                    raise ValueError("mode must be 'cbl' or 'bcp'")
            self.inverse = inverse

        def _qid_shape_(self) -> tuple[int, int]:
            return (self.dimension, self.dimension)

        def _unitary_(self) -> np.ndarray:
            size = self.dimension * self.dimension
            matrix = np.zeros((size, size), dtype=np.complex128)
            for control in range(self.dimension):
                for target in range(self.dimension):
                    total_shift = 0
                    for bus_index, mode in self.bus_specs:
                        delta = 2 ** bus_index
                        if mode == 'cbl':
                            active = control % 2
                        else:
                            active = control // delta % 2
                        total_shift += active * delta
                    if self.inverse:
                        total_shift *= -1
                    next_target = (target + total_shift) % self.dimension
                    row = control * self.dimension + next_target
                    column = control * self.dimension + target
                    matrix[row, column] = 1.0
            return matrix

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> tuple[str, str]:
            body = ','.join((f'{mode}:{bus}' for bus, mode in self.bus_specs))
            suffix = '-1' if self.inverse else ''
            return (f'SHIFT{suffix}[{body}]',) * 2

    class qudit_cirq_verification__BooleanRoutingGate(cirq.Gate):

        def __init__(self, dimension: int, bus_indices: Sequence[int], boolean_rule: qudit_cirq_verification__BooleanRule, unitary: np.ndarray, label: str) -> None:
            self.dimension = dimension
            self.bus_indices = tuple(bus_indices)
            self.boolean_rule = boolean_rule
            self.unitary = np.asarray(unitary, dtype=np.complex128)
            self.label = label

        def _qid_shape_(self) -> tuple[int]:
            return (self.dimension,)

        def _unitary_(self) -> np.ndarray:
            matrix = np.eye(self.dimension, dtype=np.complex128)
            for base in range(0, self.dimension - 1, 2):
                bits = tuple((base // 2 ** bus % 2 for bus in self.bus_indices))
                block = self.unitary if self.boolean_rule.evaluator(bits) else np.eye(2)
                matrix[base:base + 2, base:base + 2] = block
            return matrix

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
            return f'{self.label}[{self.boolean_rule.name}]'

    class qudit_cirq_verification__RoutedSingleBusGate(cirq.Gate):

        def __init__(self, dimension: int, bus_index: int, unitary: np.ndarray, label: str) -> None:
            self.dimension = dimension
            self.bus_index = bus_index
            self.delta = 2 ** bus_index
            self.unitary = np.asarray(unitary, dtype=np.complex128)
            self.label = label

        def _qid_shape_(self) -> tuple[int]:
            return (self.dimension,)

        def _unitary_(self) -> np.ndarray:
            matrix = np.eye(self.dimension, dtype=np.complex128)
            for base in range(0, self.dimension - 1, 2):
                active = base // self.delta % 2
                block = self.unitary if active else np.eye(2)
                matrix[base:base + 2, base:base + 2] = block
            return matrix

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
            return f'{self.label}[{self.delta}]'

    class qudit_cirq_verification__MultiBusPhaseGate(cirq.Gate):

        def __init__(self, dimension: int, bus_indices: Sequence[int], phase_angles: Sequence[float], label: str='CPMUX') -> None:
            if len(bus_indices) != len(phase_angles):
                raise ValueError('Each bus index must have a phase angle.')
            self.dimension = dimension
            self.bus_indices = tuple((int(bus_index) for bus_index in bus_indices))
            self.phase_angles = tuple((float(angle) for angle in phase_angles))
            self.label = label

        def _qid_shape_(self) -> tuple[int]:
            return (self.dimension,)

        def _unitary_(self) -> np.ndarray:
            matrix = np.eye(self.dimension, dtype=np.complex128)
            for base in range(0, self.dimension - 1, 2):
                angle = 0.0
                for bus_index, theta in zip(self.bus_indices, self.phase_angles):
                    angle += theta * (base // 2 ** bus_index % 2)
                matrix[base + 1, base + 1] = np.exp(1j * angle)
            return matrix

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
            return self.label

    def qudit_cirq_verification__append_packed_moments(moments: list[cirq.Moment], operations: Sequence[cirq.Operation]) -> None:
        pending = list(operations)
        while pending:
            used: set[cirq.Qid] = set()
            layer: list[cirq.Operation] = []
            remainder: list[cirq.Operation] = []
            for operation in pending:
                qudits = set(operation.qubits)
                if used.isdisjoint(qudits):
                    layer.append(operation)
                    used.update(qudits)
                else:
                    remainder.append(operation)
            moments.append(cirq.Moment(layer))
            pending = remainder

    def qudit_cirq_verification__build_layout(path_lengths: Sequence[int]) -> qudit_cirq_verification__RoutingLayout:
        if not path_lengths:
            raise ValueError('Provide at least one path length.')
        cursor = 0
        paths: list[tuple[int, ...]] = []
        for path_length in path_lengths:
            nodes = list(range(cursor, cursor + path_length))
            cursor += path_length
            paths.append(tuple(nodes))
        target_site = cursor
        return qudit_cirq_verification__RoutingLayout(paths=tuple((path + (target_site,) for path in paths)), target_site=target_site, total_sites=target_site + 1)

    def qudit_cirq_verification__build_fanin_circuit(qudits: Sequence[cirq.Qid], layout: qudit_cirq_verification__RoutingLayout, bus_indices: Sequence[int], boolean_rule: qudit_cirq_verification__BooleanRule, unitary: np.ndarray, *, label: str) -> cirq.Circuit:
        dimension = getattr(qudits[0], 'dimension', None)
        if dimension is None:
            raise ValueError('Expected explicit qudit dimensions.')
        moments: list[cirq.Moment] = []
        forward_cbl = [qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, 'cbl').on(qudits[path[0]], qudits[path[1]]) for path, bus in zip(layout.paths, bus_indices)]
        qudit_cirq_verification__append_packed_moments(moments, forward_cbl)
        max_path_length = max((len(path) - 1 for path in layout.paths))
        for hop in range(1, max_path_length):
            layer = []
            for path, bus in zip(layout.paths, bus_indices):
                if hop < len(path) - 1:
                    layer.append(qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, 'bcp').on(qudits[path[hop]], qudits[path[hop + 1]]))
            if layer:
                qudit_cirq_verification__append_packed_moments(moments, layer)
        qudit_cirq_verification__append_packed_moments(moments, [qudit_cirq_verification__BooleanRoutingGate(dimension, bus_indices, boolean_rule, unitary, label).on(qudits[layout.target_site])])
        for hop in reversed(range(1, max_path_length)):
            layer = []
            for path, bus in zip(layout.paths, bus_indices):
                if hop < len(path) - 1:
                    layer.append(qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, 'bcp', inverse=True).on(qudits[path[hop]], qudits[path[hop + 1]]))
            if layer:
                qudit_cirq_verification__append_packed_moments(moments, layer)
        reverse_cbl = [qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, 'cbl', inverse=True).on(qudits[path[0]], qudits[path[1]]) for path, bus in zip(layout.paths, bus_indices)]
        qudit_cirq_verification__append_packed_moments(moments, reverse_cbl)
        return cirq.Circuit(moments)

    def qudit_cirq_verification__build_fanin_circuit_serialized(qudits: Sequence[cirq.Qid], layout: qudit_cirq_verification__RoutingLayout, bus_indices: Sequence[int], boolean_rule: qudit_cirq_verification__BooleanRule, unitary: np.ndarray, *, label: str) -> cirq.Circuit:
        dimension = getattr(qudits[0], 'dimension', None)
        if dimension is None:
            raise ValueError('Expected explicit qudit dimensions.')
        forward_specs: list[tuple[str, int, int, int]] = []
        for path, bus in zip(layout.paths, bus_indices):
            forward_specs.append(('cbl', bus, path[0], path[1]))
            for source, target in zip(path[1:-1], path[2:]):
                forward_specs.append(('bcp', bus, source, target))
        operations: list[cirq.Operation] = []
        for mode, bus, source, target in forward_specs:
            operations.append(qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, mode).on(qudits[source], qudits[target]))
        operations.append(qudit_cirq_verification__BooleanRoutingGate(dimension, bus_indices, boolean_rule, unitary, label).on(qudits[layout.target_site]))
        for mode, bus, source, target in reversed(forward_specs):
            operations.append(qudit_cirq_verification__ConditionalShiftGate(dimension, 2 ** bus, mode, inverse=True).on(qudits[source], qudits[target]))
        return cirq.Circuit((cirq.Moment([operation]) for operation in operations))

    def qudit_cirq_verification__build_line_path(start: int, target: int) -> list[int]:
        step = 1 if target >= start else -1
        return list(range(start, target + step, step))

    def qudit_cirq_verification__build_single_bus_circuit(qudits: Sequence[cirq.Qid], path: Sequence[int], *, bus_index: int, unitary: np.ndarray, label: str) -> cirq.Circuit:
        dimension = getattr(qudits[0], 'dimension', None)
        if dimension is None:
            raise ValueError('Expected explicit qudit dimensions.')
        delta = 2 ** bus_index
        operations: list[cirq.Operation] = [qudit_cirq_verification__ConditionalShiftGate(dimension, delta, 'cbl').on(qudits[path[0]], qudits[path[1]])]
        for source, target in zip(path[1:-1], path[2:]):
            operations.append(qudit_cirq_verification__ConditionalShiftGate(dimension, delta, 'bcp').on(qudits[source], qudits[target]))
        operations.append(qudit_cirq_verification__RoutedSingleBusGate(dimension, bus_index, unitary, label).on(qudits[path[-1]]))
        for source, target in reversed(list(zip(path[1:-1], path[2:]))):
            operations.append(qudit_cirq_verification__ConditionalShiftGate(dimension, delta, 'bcp', inverse=True).on(qudits[source], qudits[target]))
        operations.append(qudit_cirq_verification__ConditionalShiftGate(dimension, delta, 'cbl', inverse=True).on(qudits[path[0]], qudits[path[1]]))
        return cirq.Circuit(operations)

    def qudit_cirq_verification__simulate_circuit(circuit: cirq.Circuit, qudits: Sequence[cirq.Qid], initial_state: np.ndarray) -> np.ndarray:
        simulator = cirq.Simulator(dtype=np.complex128)
        result = simulator.simulate(circuit, qubit_order=qudits, initial_state=initial_state)
        return np.asarray(result.final_state_vector)

    def qudit_cirq_verification__circuit_metrics(circuit: cirq.Circuit) -> qudit_cirq_verification__CircuitMetrics:
        operations = tuple(circuit.all_operations())
        return qudit_cirq_verification__CircuitMetrics(moment_count=len(circuit), operation_count=len(operations), one_qudit_gate_count=sum((1 for op in operations if len(op.qubits) == 1)), two_qudit_gate_count=sum((1 for op in operations if len(op.qubits) == 2)), max_gate_width=max((len(op.qubits) for op in operations), default=0))

    def qudit_cirq_verification__flat_index(levels: Sequence[int], dimension: int) -> int:
        index = 0
        for level in levels:
            index = index * dimension + int(level)
        return index

    def qudit_cirq_verification__embed_qubit_state(qubit_state: Sequence[complex], *, dimension: int, total_sites: int, logical_sites: Sequence[int], fixed_levels: dict[int, int] | None=None) -> np.ndarray:
        amplitudes = np.asarray(qubit_state, dtype=np.complex128)
        logical_sites = tuple(logical_sites)
        fixed_levels = dict(fixed_levels or {})
        state = np.zeros(dimension ** total_sites, dtype=np.complex128)
        for basis_index, amplitude in enumerate(amplitudes):
            levels = [fixed_levels.get(site, 0) for site in range(total_sites)]
            bitstring = format(basis_index, f'0{len(logical_sites)}b')
            for site, bit in zip(logical_sites, bitstring):
                levels[site] = int(bit)
            state[qudit_cirq_verification__flat_index(levels, dimension)] = amplitude
        return state

    def qudit_cirq_verification__extract_clean_logical_state(statevector: Sequence[complex], *, dimension: int, total_sites: int, logical_sites: Sequence[int], fixed_levels: dict[int, int] | None=None) -> np.ndarray:
        amplitudes = np.asarray(statevector, dtype=np.complex128)
        logical_sites = tuple(logical_sites)
        fixed_levels = dict(fixed_levels or {})
        extracted = np.zeros(2 ** len(logical_sites), dtype=np.complex128)
        for basis_index in range(extracted.size):
            levels = [fixed_levels.get(site, 0) for site in range(total_sites)]
            bitstring = format(basis_index, f'0{len(logical_sites)}b')
            for site, bit in zip(logical_sites, bitstring):
                levels[site] = int(bit)
            extracted[basis_index] = amplitudes[qudit_cirq_verification__flat_index(levels, dimension)]
        return extracted

    def qudit_cirq_verification__site_probabilities(statevector: Sequence[complex], site: int, *, total_sites: int, dimension: int) -> np.ndarray:
        tensor = np.asarray(statevector, dtype=np.complex128).reshape((dimension,) * total_sites)
        axes = tuple((index for index in range(total_sites) if index != site))
        return np.sum(np.abs(tensor) ** 2, axis=axes)

    def qudit_cirq_verification__max_routing_population(statevector: Sequence[complex], *, total_sites: int, dimension: int) -> float:
        worst = 0.0
        for site in range(total_sites):
            probabilities = qudit_cirq_verification__site_probabilities(statevector, site, total_sites=total_sites, dimension=dimension)
            worst = max(worst, float(np.sum(probabilities[2:])))
        return worst
    _qudit_cirq_verification_ready = True


# Inlined lazily from: qudit_qutip_verification.py

_qudit_qutip_verification_ready = False

def _init_qudit_qutip_verification() -> None:
    global _qudit_qutip_verification_ready
    if _qudit_qutip_verification_ready:
        return
    global math, dataclass, Callable, Sequence, np, sparse, qt, StateLike, TRAJECTORY_DIMENSION_THRESHOLD, DEFAULT_MONTE_CARLO_TRAJECTORIES, _RNG, _MC_SEED_SEQUENCE, _TRAJECTORY_SOLVER_OPTIONS
    global _EXACT_SOLVER_OPTIONS, qudit_qutip_verification__BooleanRule, qudit_qutip_verification__RoutingLayout, qudit_qutip_verification__and_rule, qudit_qutip_verification__minimum_dimension_for_controls, qudit_qutip_verification__configure_monte_carlo, qudit_qutip_verification__next_monte_carlo_seeds, qudit_qutip_verification__basis_state, qudit_qutip_verification__plus_state, qudit_qutip_verification__product_ket, qudit_qutip_verification__zero_hamiltonian, qudit_qutip_verification__logical_block_matrix
    global qudit_qutip_verification__x_matrix, qudit_qutip_verification__z_matrix, qudit_qutip_verification__h_matrix, qudit_qutip_verification__phase_matrix, qudit_qutip_verification__rz_matrix, qudit_qutip_verification__rx_matrix, qudit_qutip_verification__build_layout, qudit_qutip_verification__build_line_path, qudit_qutip_verification__conditional_shift_matrix, qudit_qutip_verification__combined_shift_matrix, qudit_qutip_verification__boolean_routed_unitary_matrix, qudit_qutip_verification__single_bus_controlled_unitary_matrix
    global qudit_qutip_verification__multi_bus_phase_matrix, qudit_qutip_verification__embed_single_site, qudit_qutip_verification__embed_two_site, qudit_qutip_verification__build_decay_collapse_ops, qudit_qutip_verification__build_dephasing_collapse_ops, qudit_qutip_verification__build_relaxation_and_dephasing_ops, qudit_qutip_verification__build_leakage_kraus, qudit_qutip_verification__apply_local_channel, qudit_qutip_verification__apply_local_channel_trajectory, qudit_qutip_verification__evolve_trajectory_ensemble, qudit_qutip_verification__compose_layer_operator, qudit_qutip_verification__apply_leakage_channels
    global qudit_qutip_verification__apply_global_gate_with_noise, qudit_qutip_verification__pure_state_overlap, qudit_qutip_verification__routing_population, qudit_qutip_verification__flat_index, qudit_qutip_verification__embed_qubit_state_as_ket, qudit_qutip_verification__logical_subspace_projector, qudit_qutip_verification__logical_subspace_population, qudit_qutip_verification__apply_logical_unitary, qudit_qutip_verification__apply_routed_single_bus_unitary, qudit_qutip_verification__apply_routed_fanin_unitary, qudit_qutip_verification__apply_routed_fanin_unitary_serialized
    import math
    from dataclasses import dataclass
    from typing import Callable, Sequence
    import numpy as np
    from scipy import sparse
    try:
        import qutip as qt
    except ImportError as exc:
        raise SystemExit('QuTiP is required for the qudit verification scripts. Install it with `pip install qutip`.') from exc
    StateLike = qt.Qobj | list[qt.Qobj]
    TRAJECTORY_DIMENSION_THRESHOLD = 400
    DEFAULT_MONTE_CARLO_TRAJECTORIES = 64
    _RNG = np.random.default_rng()
    _MC_SEED_SEQUENCE = None
    _TRAJECTORY_SOLVER_OPTIONS = {'keep_runs_results': True, 'store_final_state': True, 'store_states': False, 'progress_bar': ''}
    _EXACT_SOLVER_OPTIONS = {'store_final_state': True, 'store_states': False, 'progress_bar': ''}

    @dataclass(frozen=True)
    class qudit_qutip_verification__BooleanRule:
        name: str
        evaluator: Callable[[Sequence[int]], int]

    @dataclass(frozen=True)
    class qudit_qutip_verification__RoutingLayout:
        paths: tuple[tuple[int, ...], ...]
        target_site: int
        total_sites: int

        @property
        def source_sites(self) -> tuple[int, ...]:
            return tuple((path[0] for path in self.paths))

        @property
        def payload_sites(self) -> tuple[int, ...]:
            payload: list[int] = []
            for path in self.paths:
                payload.extend(path[1:-1])
            return tuple(payload)

    def qudit_qutip_verification__and_rule(controls: int) -> qudit_qutip_verification__BooleanRule:
        return qudit_qutip_verification__BooleanRule(f'and:{controls}', lambda bits: int(all(bits)))

    def qudit_qutip_verification__minimum_dimension_for_controls(controls: int) -> int:
        return 2 if controls <= 0 else 2 ** (controls + 1)

    def qudit_qutip_verification__configure_monte_carlo(seed: int | None, trajectories: int) -> None:
        global _RNG, _MC_SEED_SEQUENCE
        _RNG = np.random.default_rng(seed)
        _MC_SEED_SEQUENCE = np.random.SeedSequence(seed) if seed is not None else None
        globals()['DEFAULT_MONTE_CARLO_TRAJECTORIES'] = max(1, int(trajectories))

    def qudit_qutip_verification__next_monte_carlo_seeds(count: int) -> list[int | np.random.SeedSequence] | None:
        if _MC_SEED_SEQUENCE is None:
            return None
        return _MC_SEED_SEQUENCE.spawn(count)

    def qudit_qutip_verification__basis_state(dimension: int, level: int) -> qt.Qobj:
        return qt.basis(dimension, level)

    def qudit_qutip_verification__plus_state(dimension: int) -> qt.Qobj:
        return (qt.basis(dimension, 0) + qt.basis(dimension, 1)).unit()

    def qudit_qutip_verification__product_ket(states: Sequence[qt.Qobj]) -> qt.Qobj:
        return qt.tensor(list(states))

    def qudit_qutip_verification__zero_hamiltonian(dims: Sequence[int]) -> qt.Qobj:
        return qt.qzero(list(dims)).to('csr')

    def qudit_qutip_verification__logical_block_matrix(dimension: int, unitary: np.ndarray) -> sparse.csr_matrix:
        matrix = sparse.identity(dimension, dtype=np.complex128, format='lil')
        matrix[:2, :2] = np.asarray(unitary, dtype=np.complex128)
        return matrix.tocsr()

    def qudit_qutip_verification__x_matrix() -> np.ndarray:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def qudit_qutip_verification__z_matrix() -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def qudit_qutip_verification__h_matrix() -> np.ndarray:
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)

    def qudit_qutip_verification__phase_matrix(theta: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)

    def qudit_qutip_verification__rz_matrix(theta: float) -> np.ndarray:
        half = theta / 2.0
        return np.array([[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128)

    def qudit_qutip_verification__rx_matrix(theta: float) -> np.ndarray:
        half = theta / 2.0
        return np.array([[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]], dtype=np.complex128)

    def qudit_qutip_verification__build_layout(path_lengths: Sequence[int]) -> qudit_qutip_verification__RoutingLayout:
        cursor = 0
        paths: list[tuple[int, ...]] = []
        for path_length in path_lengths:
            nodes = list(range(cursor, cursor + path_length))
            cursor += path_length
            paths.append(tuple(nodes))
        target_site = cursor
        return qudit_qutip_verification__RoutingLayout(paths=tuple((path + (target_site,) for path in paths)), target_site=target_site, total_sites=target_site + 1)

    def qudit_qutip_verification__build_line_path(start: int, target: int) -> list[int]:
        step = 1 if target >= start else -1
        return list(range(start, target + step, step))

    def qudit_qutip_verification__conditional_shift_matrix(dimension: int, delta: int, mode: str, inverse: bool=False) -> sparse.csr_matrix:
        if mode not in {'cbl', 'bcp'}:
            raise ValueError("mode must be 'cbl' or 'bcp'")
        shift = -delta if inverse else delta
        size = dimension * dimension
        matrix = sparse.lil_matrix((size, size), dtype=np.complex128)
        for control in range(dimension):
            for target in range(dimension):
                active = control % 2 if mode == 'cbl' else control // delta % 2
                next_target = (target + active * shift) % dimension
                row = control * dimension + next_target
                column = control * dimension + target
                matrix[row, column] = 1.0
        return matrix.tocsr()

    def qudit_qutip_verification__combined_shift_matrix(dimension: int, bus_specs: Sequence[tuple[int, str]], *, inverse: bool=False) -> sparse.csr_matrix:
        size = dimension * dimension
        matrix = sparse.lil_matrix((size, size), dtype=np.complex128)
        normalized = tuple(((int(bus_index), str(mode)) for bus_index, mode in bus_specs))
        for _, mode in normalized:
            if mode not in {'cbl', 'bcp'}:
                raise ValueError("mode must be 'cbl' or 'bcp'")
        for control in range(dimension):
            for target in range(dimension):
                total_shift = 0
                for bus_index, mode in normalized:
                    delta = 2 ** bus_index
                    if mode == 'cbl':
                        active = control % 2
                    else:
                        active = control // delta % 2
                    total_shift += active * delta
                if inverse:
                    total_shift *= -1
                next_target = (target + total_shift) % dimension
                row = control * dimension + next_target
                column = control * dimension + target
                matrix[row, column] = 1.0
        return matrix.tocsr()

    def qudit_qutip_verification__boolean_routed_unitary_matrix(dimension: int, bus_indices: Sequence[int], boolean_rule: qudit_qutip_verification__BooleanRule, unitary: np.ndarray) -> sparse.csr_matrix:
        matrix = sparse.identity(dimension, dtype=np.complex128, format='lil')
        for base in range(0, dimension - 1, 2):
            bits = tuple((base // 2 ** bus % 2 for bus in bus_indices))
            block = unitary if boolean_rule.evaluator(bits) else np.eye(2)
            matrix[base:base + 2, base:base + 2] = block
        return matrix.tocsr()

    def qudit_qutip_verification__single_bus_controlled_unitary_matrix(dimension: int, bus_index: int, unitary: np.ndarray) -> sparse.csr_matrix:
        return qudit_qutip_verification__boolean_routed_unitary_matrix(dimension, [bus_index], qudit_qutip_verification__BooleanRule(f'bus:{bus_index}', lambda bits: int(bits[0])), unitary)

    def qudit_qutip_verification__multi_bus_phase_matrix(dimension: int, bus_indices: Sequence[int], phase_angles: Sequence[float]) -> sparse.csr_matrix:
        if len(bus_indices) != len(phase_angles):
            raise ValueError('Each bus index must have a phase angle.')
        matrix = sparse.identity(dimension, dtype=np.complex128, format='lil')
        for base in range(0, dimension - 1, 2):
            angle = 0.0
            for bus_index, theta in zip(bus_indices, phase_angles):
                angle += float(theta) * (base // 2 ** int(bus_index) % 2)
            matrix[base + 1, base + 1] = np.exp(1j * angle)
        return matrix.tocsr()

    def qudit_qutip_verification__embed_single_site(operator: qt.Qobj, site: int, dims: Sequence[int]) -> qt.Qobj:
        factors = [qt.qeye(dimension).to('csr') for dimension in dims]
        factors[site] = operator
        return qt.tensor(factors).to('csr')

    def qudit_qutip_verification__embed_two_site(local_matrix: sparse.spmatrix, site_a: int, site_b: int, dims: Sequence[int]) -> qt.Qobj:
        current_order = [site_a, site_b] + [index for index in range(len(dims)) if index not in {site_a, site_b}]
        local = qt.Qobj(local_matrix, dims=[[dims[site_a], dims[site_b]], [dims[site_a], dims[site_b]]]).to('csr')
        factors = [local] + [qt.qeye(dims[index]).to('csr') for index in current_order[2:]]
        embedded = qt.tensor(factors)
        permutation = [current_order.index(index) for index in range(len(dims))]
        return embedded.permute(permutation).to('csr')

    def qudit_qutip_verification__build_decay_collapse_ops(dims: Sequence[int], t1_levels: Sequence[float]) -> list[qt.Qobj]:
        local_dimension = dims[0]
        collapse_ops: list[qt.Qobj] = []
        for site in range(len(dims)):
            for level in range(1, local_dimension):
                t1 = t1_levels[min(level - 1, len(t1_levels) - 1)]
                if math.isinf(t1) or t1 <= 0.0:
                    continue
                local = sparse.lil_matrix((local_dimension, local_dimension), dtype=np.complex128)
                local[level - 1, level] = math.sqrt(1.0 / t1)
                collapse_ops.append(qudit_qutip_verification__embed_single_site(qt.Qobj(local.tocsr(), dims=[[local_dimension], [local_dimension]]).to('csr'), site, dims))
        return collapse_ops

    def qudit_qutip_verification__build_dephasing_collapse_ops(dims: Sequence[int], tphi_levels: Sequence[float]) -> list[qt.Qobj]:
        local_dimension = dims[0]
        collapse_ops: list[qt.Qobj] = []
        for site in range(len(dims)):
            for level in range(1, local_dimension):
                tphi = tphi_levels[min(level - 1, len(tphi_levels) - 1)]
                if math.isinf(tphi) or tphi <= 0.0:
                    continue
                local = sparse.lil_matrix((local_dimension, local_dimension), dtype=np.complex128)
                local[level, level] = math.sqrt(1.0 / tphi)
                collapse_ops.append(qudit_qutip_verification__embed_single_site(qt.Qobj(local.tocsr(), dims=[[local_dimension], [local_dimension]]).to('csr'), site, dims))
        return collapse_ops

    def qudit_qutip_verification__build_relaxation_and_dephasing_ops(dims: Sequence[int], t1_levels: Sequence[float], tphi_levels: Sequence[float]) -> list[qt.Qobj]:
        return qudit_qutip_verification__build_decay_collapse_ops(dims, t1_levels) + qudit_qutip_verification__build_dephasing_collapse_ops(dims, tphi_levels)

    def qudit_qutip_verification__build_leakage_kraus(dimension: int, epsilon: float) -> list[qt.Qobj] | None:
        if epsilon <= 0.0:
            return None
        keep = sparse.eye(dimension, dtype=np.complex128, format='lil')
        jump = sparse.lil_matrix((dimension, dimension), dtype=np.complex128)
        for level in range(2, dimension - 1):
            keep[level, level] = math.sqrt(1.0 - epsilon)
            jump[level + 1, level] = math.sqrt(epsilon)
        return [qt.Qobj(keep.tocsr(), dims=[[dimension], [dimension]]).to('csr'), qt.Qobj(jump.tocsr(), dims=[[dimension], [dimension]]).to('csr')]

    def qudit_qutip_verification__apply_local_channel(rho: qt.Qobj, site: int, local_kraus_ops: Sequence[qt.Qobj], dims: Sequence[int]) -> qt.Qobj:
        updated = 0.0 * rho
        for kraus in local_kraus_ops:
            embedded = qudit_qutip_verification__embed_single_site(kraus, site, dims)
            updated += embedded * rho * embedded.dag()
        return updated

    def qudit_qutip_verification__apply_local_channel_trajectory(psi: qt.Qobj, site: int, local_kraus_ops: Sequence[qt.Qobj], dims: Sequence[int]) -> qt.Qobj:
        embedded_ops = [qudit_qutip_verification__embed_single_site(kraus, site, dims) for kraus in local_kraus_ops]
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

    def qudit_qutip_verification__evolve_trajectory_ensemble(trajectories: Sequence[qt.Qobj], h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj], gate_time: float) -> list[qt.Qobj]:
        solver = qt.MCSolver(h0, list(collapse_ops), options=_TRAJECTORY_SOLVER_OPTIONS)
        weighted = [(psi, 1.0 / len(trajectories)) for psi in trajectories]
        result = solver.run(weighted, [0.0, gate_time], ntraj=[1] * len(trajectories), seeds=qudit_qutip_verification__next_monte_carlo_seeds(len(trajectories)))
        final_states = getattr(result, 'runs_final_states', None) or getattr(result, 'runs_final_state', None)
        if final_states is None:
            raise RuntimeError('QuTiP did not return per-trajectory final states.')
        return [state.unit() if state.norm() > 0.0 else state for state in final_states]

    def qudit_qutip_verification__compose_layer_operator(operators: Sequence[qt.Qobj]) -> qt.Qobj:
        gate = operators[0]
        for operator in operators[1:]:
            gate = operator * gate
        return gate

    def qudit_qutip_verification__apply_leakage_channels(state: StateLike, leakage_sites: Sequence[int], leakage_kraus: Sequence[qt.Qobj] | None, dims: Sequence[int]) -> StateLike:
        if leakage_kraus is None:
            return state
        updated = state
        for site in leakage_sites:
            if isinstance(updated, list):
                updated = [qudit_qutip_verification__apply_local_channel_trajectory(psi, site, leakage_kraus, dims) for psi in updated]
            elif updated.isket:
                updated = qudit_qutip_verification__apply_local_channel_trajectory(updated, site, leakage_kraus, dims)
            else:
                updated = qudit_qutip_verification__apply_local_channel(updated, site, leakage_kraus, dims)
        return updated

    def qudit_qutip_verification__apply_global_gate_with_noise(state: StateLike, gate: qt.Qobj, gate_time: float, h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj], *, leakage_sites: Sequence[int]=(), leakage_kraus: Sequence[qt.Qobj] | None=None) -> StateLike:
        if isinstance(state, list):
            updated: StateLike = [gate * psi for psi in state]
        elif state.isket:
            updated = gate * state
        else:
            updated = gate * state * gate.dag()
        updated = qudit_qutip_verification__apply_leakage_channels(updated, leakage_sites, leakage_kraus, h0.dims[0])
        if gate_time > 0.0 and collapse_ops:
            dimension = int(np.prod(h0.dims[0], dtype=int))
            if isinstance(updated, list):
                updated = qudit_qutip_verification__evolve_trajectory_ensemble(updated, h0, collapse_ops, gate_time)
            elif dimension > TRAJECTORY_DIMENSION_THRESHOLD and updated.isket:
                trajectories = [updated.copy() for _ in range(DEFAULT_MONTE_CARLO_TRAJECTORIES)]
                updated = qudit_qutip_verification__evolve_trajectory_ensemble(trajectories, h0, collapse_ops, gate_time)
            else:
                result = qt.mesolve(h0, updated, [0.0, gate_time], c_ops=list(collapse_ops), options=_EXACT_SOLVER_OPTIONS)
                final_state = getattr(result, 'final_state', None)
                updated = final_state if final_state is not None else result.states[-1]
        return updated

    def qudit_qutip_verification__pure_state_overlap(state: StateLike, ideal_ket: qt.Qobj) -> float:
        if isinstance(state, list):
            values = [float(np.abs(ideal_ket.overlap(psi)) ** 2) for psi in state]
            overlap = float(np.mean(values)) if values else 0.0
            return max(0.0, min(1.0, overlap))
        if state.isket:
            overlap = float(np.abs(ideal_ket.overlap(state)) ** 2)
        else:
            amplitude = ideal_ket.dag() * state * ideal_ket
            amplitude_value = amplitude[0, 0] if isinstance(amplitude, qt.Qobj) else amplitude
            overlap = float(np.real_if_close(amplitude_value))
        return max(0.0, min(1.0, overlap))

    def qudit_qutip_verification__routing_population(state: StateLike, site: int, dimension: int, dims: Sequence[int]) -> float:
        local = 0.0 * qt.basis(dimension, 0).proj().to('csr')
        for level in range(2, dimension):
            local += qt.basis(dimension, level).proj().to('csr')
        observable = qudit_qutip_verification__embed_single_site(local, site, dims)
        if isinstance(state, list):
            samples = [float(np.real_if_close(qt.expect(observable, psi))) for psi in state]
            population = float(np.mean(samples)) if samples else 0.0
        else:
            population = float(np.real_if_close(qt.expect(observable, state)))
        return max(0.0, min(1.0, population))

    def qudit_qutip_verification__flat_index(levels: Sequence[int], dimension: int) -> int:
        index = 0
        for level in levels:
            index = index * dimension + int(level)
        return index

    def qudit_qutip_verification__embed_qubit_state_as_ket(qubit_state: Sequence[complex], *, dimension: int, total_sites: int, logical_sites: Sequence[int], fixed_levels: dict[int, int] | None=None) -> qt.Qobj:
        amplitudes = np.asarray(qubit_state, dtype=np.complex128)
        fixed_levels = dict(fixed_levels or {})
        logical_sites = tuple(logical_sites)
        vector = np.zeros(dimension ** total_sites, dtype=np.complex128)
        for basis_index, amplitude in enumerate(amplitudes):
            levels = [fixed_levels.get(site, 0) for site in range(total_sites)]
            bitstring = format(basis_index, f'0{len(logical_sites)}b')
            for site, bit in zip(logical_sites, bitstring):
                levels[site] = int(bit)
            vector[qudit_qutip_verification__flat_index(levels, dimension)] = amplitude
        dims = [dimension] * total_sites
        return qt.Qobj(vector.reshape((-1, 1)), dims=[dims, [1] * total_sites]).unit()

    def qudit_qutip_verification__logical_subspace_projector(*, dimension: int, total_sites: int, logical_sites: Sequence[int], fixed_levels: dict[int, int] | None=None) -> qt.Qobj:
        fixed_levels = dict(fixed_levels or {})
        logical_sites = tuple(logical_sites)
        diagonal = sparse.lil_matrix((dimension ** total_sites, dimension ** total_sites), dtype=np.complex128)
        for basis_index in range(2 ** len(logical_sites)):
            levels = [fixed_levels.get(site, 0) for site in range(total_sites)]
            bitstring = format(basis_index, f'0{len(logical_sites)}b')
            for site, bit in zip(logical_sites, bitstring):
                levels[site] = int(bit)
            flat = qudit_qutip_verification__flat_index(levels, dimension)
            diagonal[flat, flat] = 1.0
        dims = [dimension] * total_sites
        return qt.Qobj(diagonal.tocsr(), dims=[dims, dims]).to('csr')

    def qudit_qutip_verification__logical_subspace_population(state: StateLike, *, dimension: int, total_sites: int, logical_sites: Sequence[int], fixed_levels: dict[int, int] | None=None) -> float:
        projector = qudit_qutip_verification__logical_subspace_projector(dimension=dimension, total_sites=total_sites, logical_sites=logical_sites, fixed_levels=fixed_levels)
        if isinstance(state, list):
            values = [float(np.real_if_close(qt.expect(projector, psi))) for psi in state]
            population = float(np.mean(values)) if values else 0.0
        else:
            population = float(np.real_if_close(qt.expect(projector, state)))
        return max(0.0, min(1.0, population))

    def qudit_qutip_verification__apply_logical_unitary(state: StateLike, *, site: int, dimension: int, dims: Sequence[int], unitary: np.ndarray, gate_time: float, h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj]) -> StateLike:
        gate = qudit_qutip_verification__embed_single_site(qt.Qobj(qudit_qutip_verification__logical_block_matrix(dimension, unitary), dims=[[dimension], [dimension]]).to('csr'), site, dims)
        return qudit_qutip_verification__apply_global_gate_with_noise(state, gate, gate_time, h0, collapse_ops)

    def qudit_qutip_verification__apply_routed_single_bus_unitary(state: StateLike, *, path: Sequence[int], dimension: int, dims: Sequence[int], unitary: np.ndarray, routing_gate_time: float, target_gate_time: float, h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj], leakage_kraus: Sequence[qt.Qobj] | None, bus_index: int=1) -> StateLike:
        delta = 2 ** bus_index
        forward = qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, delta, 'cbl'), path[0], path[1], dims)
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, forward, routing_gate_time, h0, collapse_ops, leakage_sites=[path[1]], leakage_kraus=leakage_kraus)
        for source, target in zip(path[1:-1], path[2:]):
            state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, delta, 'bcp'), source, target, dims), routing_gate_time, h0, collapse_ops, leakage_sites=[target], leakage_kraus=leakage_kraus)
        target_gate = qudit_qutip_verification__embed_single_site(qt.Qobj(qudit_qutip_verification__single_bus_controlled_unitary_matrix(dimension, bus_index, unitary), dims=[[dimension], [dimension]]).to('csr'), path[-1], dims)
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, target_gate, target_gate_time, h0, collapse_ops)
        for source, target in reversed(list(zip(path[1:-1], path[2:]))):
            state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, delta, 'bcp', inverse=True), source, target, dims), routing_gate_time, h0, collapse_ops, leakage_sites=[target], leakage_kraus=leakage_kraus)
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, delta, 'cbl', inverse=True), path[0], path[1], dims), routing_gate_time, h0, collapse_ops, leakage_sites=[path[1]], leakage_kraus=leakage_kraus)
        return state

    def qudit_qutip_verification__apply_routed_fanin_unitary(state: StateLike, *, layout: qudit_qutip_verification__RoutingLayout, dimension: int, dims: Sequence[int], bus_indices: Sequence[int], boolean_rule: qudit_qutip_verification__BooleanRule, unitary: np.ndarray, routing_gate_time: float, target_gate_time: float, h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj], leakage_kraus: Sequence[qt.Qobj] | None) -> StateLike:
        forward = [qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, 'cbl'), path[0], path[1], dims) for path, bus in zip(layout.paths, bus_indices)]
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(forward), routing_gate_time, h0, collapse_ops, leakage_sites=[path[1] for path in layout.paths], leakage_kraus=leakage_kraus)
        max_path_length = max((len(path) - 1 for path in layout.paths))
        for hop in range(1, max_path_length):
            layer = []
            leakage_sites: list[int] = []
            for path, bus in zip(layout.paths, bus_indices):
                if hop < len(path) - 1:
                    layer.append(qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, 'bcp'), path[hop], path[hop + 1], dims))
                    leakage_sites.append(path[hop + 1])
            if layer:
                state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(layer), routing_gate_time, h0, collapse_ops, leakage_sites=leakage_sites, leakage_kraus=leakage_kraus)
        target_gate = qudit_qutip_verification__embed_single_site(qt.Qobj(qudit_qutip_verification__boolean_routed_unitary_matrix(dimension, bus_indices, boolean_rule, unitary), dims=[[dimension], [dimension]]).to('csr'), layout.target_site, dims)
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, target_gate, target_gate_time, h0, collapse_ops)
        for hop in reversed(range(1, max_path_length)):
            layer = []
            leakage_sites = []
            for path, bus in zip(layout.paths, bus_indices):
                if hop < len(path) - 1:
                    layer.append(qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, 'bcp', inverse=True), path[hop], path[hop + 1], dims))
                    leakage_sites.append(path[hop + 1])
            if layer:
                state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(layer), routing_gate_time, h0, collapse_ops, leakage_sites=leakage_sites, leakage_kraus=leakage_kraus)
        reverse = [qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, 'cbl', inverse=True), path[0], path[1], dims) for path, bus in zip(layout.paths, bus_indices)]
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(reverse), routing_gate_time, h0, collapse_ops, leakage_sites=[path[1] for path in layout.paths], leakage_kraus=leakage_kraus)
        return state

    def qudit_qutip_verification__apply_routed_fanin_unitary_serialized(state: StateLike, *, layout: qudit_qutip_verification__RoutingLayout, dimension: int, dims: Sequence[int], bus_indices: Sequence[int], boolean_rule: qudit_qutip_verification__BooleanRule, unitary: np.ndarray, routing_gate_time: float, target_gate_time: float, h0: qt.Qobj, collapse_ops: Sequence[qt.Qobj], leakage_kraus: Sequence[qt.Qobj] | None) -> StateLike:
        forward_specs: list[tuple[str, int, int, int]] = []
        for path, bus in zip(layout.paths, bus_indices):
            forward_specs.append(('cbl', bus, path[0], path[1]))
            for source, target in zip(path[1:-1], path[2:]):
                forward_specs.append(('bcp', bus, source, target))
        for mode, bus, source, target in forward_specs:
            state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, mode), source, target, dims), routing_gate_time, h0, collapse_ops, leakage_sites=[target], leakage_kraus=leakage_kraus)
        target_gate = qudit_qutip_verification__embed_single_site(qt.Qobj(qudit_qutip_verification__boolean_routed_unitary_matrix(dimension, bus_indices, boolean_rule, unitary), dims=[[dimension], [dimension]]).to('csr'), layout.target_site, dims)
        state = qudit_qutip_verification__apply_global_gate_with_noise(state, target_gate, target_gate_time, h0, collapse_ops)
        for mode, bus, source, target in reversed(forward_specs):
            state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__embed_two_site(qudit_qutip_verification__conditional_shift_matrix(dimension, 2 ** bus, mode, inverse=True), source, target, dims), routing_gate_time, h0, collapse_ops, leakage_sites=[target], leakage_kraus=leakage_kraus)
        return state
    _qudit_qutip_verification_ready = True


# Inlined from: amplitude_amplification_verification.py

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def amplitude_amplification_verification__parse_good_indices(spec: str, n_qubits: int) -> list[int]:
    values = verification_common__parse_int_list(spec)
    if not values:
        raise ValueError('Provide at least one good index.')
    limit = 2 ** n_qubits
    for value in values:
        if value < 0 or value >= limit:
            raise ValueError(f'good index {value} lies outside [0, {limit - 1}]')
    return sorted(set(values))

def amplitude_amplification_verification__grover_success_formula(n_qubits: int, marked_count: int, iteration: int) -> float:
    total_states = 2 ** n_qubits
    theta = math.asin(math.sqrt(marked_count / total_states))
    return float(math.sin((2 * iteration + 1) * theta) ** 2)

def amplitude_amplification_verification__optimal_grover_iteration(n_qubits: int, marked_count: int) -> int:
    theta = math.asin(math.sqrt(marked_count / 2 ** n_qubits))
    if theta == 0.0:
        return 0
    return int(max(0, math.floor(math.pi / (4.0 * theta) - 0.5)))

def amplitude_amplification_verification__build_oracle(n_qubits: int, good_indices: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name='oracle')
    for index in good_indices:
        pattern = format(index, f'0{n_qubits}b')[::-1]
        for qubit, bit in enumerate(pattern):
            if bit == '0':
                qc.x(qubit)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for qubit, bit in enumerate(pattern):
            if bit == '0':
                qc.x(qubit)
    return qc

def amplitude_amplification_verification__build_diffusion(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name='diffusion')
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc

def amplitude_amplification_verification__build_grover_circuit(n_qubits: int, good_indices: Sequence[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    if iterations <= 0:
        return qc
    oracle = amplitude_amplification_verification__build_oracle(n_qubits, good_indices)
    diffusion = amplitude_amplification_verification__build_diffusion(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    return qc

def amplitude_amplification_verification__add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--n-qubits', type=int, default=6, help='Number of search qubits.')
    parser.add_argument('--good-indices', default='1', help="Marked basis indices, e.g. '1' or '1,3'.")
    parser.add_argument('--max-iterations', type=int, default=None, help='Maximum Grover iteration count. Defaults to 2*k_opt+2.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workers', type=int, default=1, help='Worker processes for independent experiment points.')
    parser.add_argument('--native-threads-per-worker', type=int, default=None, help='Caps BLAS/OpenMP threads inside each worker. Defaults to 1 when workers > 1.')

def amplitude_amplification_verification__evaluate_theory_task(task: dict[str, object]) -> dict[str, object]:
    n_qubits = int(task['n_qubits'])
    good_indices = [int(value) for value in task['good_indices']]
    marked_count = len(good_indices)
    iteration = int(task['iteration'])
    exact_mode = bool(task['exact_mode'])
    circuit = amplitude_amplification_verification__build_grover_circuit(n_qubits, good_indices, iteration)
    theory_success = amplitude_amplification_verification__grover_success_formula(n_qubits, marked_count, iteration)
    exact_success = theory_success
    if exact_mode:
        state = Statevector.from_instruction(circuit)
        exact_success = verification_common__probability_mass(state.data, good_indices)
    logical_stats = verification_aer__logical_circuit_stats(circuit)
    return {'iteration': iteration, 'evaluation_mode': 'exact' if exact_mode else 'closed_form_only', 'theory_success_probability': theory_success, 'exact_success_probability': exact_success, 'logical_depth': logical_stats.logical_depth, 'logical_size': logical_stats.logical_size, 'logical_two_qubit_count': logical_stats.logical_two_qubit_count, 'interaction_span_total': logical_stats.span_total, 'interaction_span_max': logical_stats.span_max, 'interaction_span_mean': logical_stats.span_mean, 'max_gate_width': logical_stats.max_gate_width, 'max_fan_in': logical_stats.max_fan_in, 'max_qubit_load': logical_stats.max_qubit_load, 'weighted_qubit_load_max': logical_stats.weighted_qubit_load_max, 'target_load_max': logical_stats.target_load_max, 'multi_qubit_gate_count': logical_stats.multi_qubit_gate_count}

def amplitude_amplification_verification__evaluate_aer_task(task: dict[str, object]) -> list[dict[str, object]]:
    n_qubits = int(task['n_qubits'])
    good_indices = [int(value) for value in task['good_indices']]
    iteration = int(task['iteration'])
    topologies = [str(topology) for topology in task['topologies']]
    optimization_level = int(task['optimization_level'])
    seed = int(task['seed'])
    device = str(task['device'])
    exact_mode = bool(task['exact_mode'])
    marked_count = len(good_indices)
    logical_circuit = amplitude_amplification_verification__build_grover_circuit(n_qubits, good_indices, iteration)
    logical_stats = verification_aer__logical_circuit_stats(logical_circuit)
    theory_success = amplitude_amplification_verification__grover_success_formula(n_qubits, marked_count, iteration)
    exact_success = theory_success
    logical_state = None
    if exact_mode:
        logical_state = Statevector.from_instruction(logical_circuit).data
        exact_success = verification_common__probability_mass(logical_state, good_indices)
    rows: list[dict[str, object]] = []
    for topology in topologies:
        transpiled_circuit, metrics = verification_aer__transpile_with_metrics(logical_circuit, topology=topology, optimization_level=optimization_level, seed=seed)
        fidelity = None
        if exact_mode and logical_state is not None and (int(metrics['swap_count']) == 0):
            transpiled_state = verification_aer__simulate_statevector(transpiled_circuit, device=device, seed=seed)
            fidelity = verification_common__statevector_fidelity(logical_state, transpiled_state)
        rows.append({'iteration': iteration, 'topology': topology, 'evaluation_mode': 'exact' if exact_mode else 'metrics_only', 'theory_success_probability': theory_success, 'exact_success_probability': exact_success, 'state_fidelity': fidelity, 'logical_depth': logical_stats.logical_depth, 'logical_size': logical_stats.logical_size, 'logical_two_qubit_count': logical_stats.logical_two_qubit_count, 'interaction_span_total': logical_stats.span_total, 'interaction_span_max': logical_stats.span_max, 'interaction_span_mean': logical_stats.span_mean, 'max_gate_width': logical_stats.max_gate_width, 'max_fan_in': logical_stats.max_fan_in, 'max_qubit_load': logical_stats.max_qubit_load, 'weighted_qubit_load_max': logical_stats.weighted_qubit_load_max, 'target_load_max': logical_stats.target_load_max, 'multi_qubit_gate_count': logical_stats.multi_qubit_gate_count, **metrics})
    return rows

def amplitude_amplification_verification__make_theory_plot(rows: Sequence[dict[str, object]], figure_path: Path) -> None:
    if plt is None or not rows:
        return
    iterations = [int(row['iteration']) for row in rows]
    theory = [float(row['theory_success_probability']) for row in rows]
    exact = [float(row['exact_success_probability']) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, theory, marker='o', label='closed-form')
    ax.plot(iterations, exact, marker='s', linestyle='--', label='exact statevector')
    ax.set_xlabel('Grover iterations k')
    ax.set_ylabel('Success probability')
    ax.set_title('Amplitude Amplification Verification')
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_verification__make_aer_plot(rows: Sequence[dict[str, object]], figure_path: Path) -> None:
    if plt is None or not rows:
        return
    topologies = sorted({str(row['topology']) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    reference_rows = [row for row in rows if str(row['topology']) == topologies[0]]
    iterations = [int(row['iteration']) for row in reference_rows]
    theory = [float(row['theory_success_probability']) for row in reference_rows]
    success = [float(row['exact_success_probability']) for row in reference_rows]
    axes[0].plot(iterations, theory, marker='o', label='closed-form')
    axes[0].plot(iterations, success, marker='s', linestyle='--', label='logical exact')
    axes[0].set_xlabel('Grover iterations k')
    axes[0].set_ylabel('Success probability')
    axes[0].set_title('Success curve')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for topology in topologies:
        topo_rows = [row for row in rows if str(row['topology']) == topology]
        axes[1].plot([int(row['iteration']) for row in topo_rows], [float(row['transpiled_depth']) for row in topo_rows], marker='o', label=topology)
    axes[1].set_xlabel('Grover iterations k')
    axes[1].set_ylabel('Transpiled depth')
    axes[1].set_title('Routing overhead proxy')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_verification__main_theory(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='Theory-side amplitude amplification verification.')
    amplitude_amplification_verification__add_shared_arguments(parser)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.n_qubits = verification_common__prompt_int('How many qubits do you want to test?', args.n_qubits)
    ctx = verification_common__setup_run_context(script_file)
    good_indices = amplitude_amplification_verification__parse_good_indices(args.good_indices, args.n_qubits)
    marked_count = len(good_indices)
    k_opt = amplitude_amplification_verification__optimal_grover_iteration(args.n_qubits, marked_count)
    max_iterations = args.max_iterations if args.max_iterations is not None else 2 * k_opt + 2
    exact_limit = 18
    exact_mode = args.n_qubits <= exact_limit
    if not exact_mode:
        print(f'[theory-aa] n={args.n_qubits} is above the exact-state limit ({exact_limit}). Using closed-form success plus logical congestion/fan-in metrics.')
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'n_qubits': int(args.n_qubits), 'good_indices': list(good_indices), 'iteration': iteration, 'exact_mode': exact_mode} for iteration in range(max_iterations + 1)]
    print(f'[aa_theory] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = verification_common__parallel_map(amplitude_amplification_verification__evaluate_theory_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker)
    print(f'[aa_theory] parallel section complete rows={len(rows)}')
    verification_common__write_csv(ctx.result_dir / 'amplitude_amplification_theory.csv', rows)
    amplitude_amplification_verification__make_theory_plot(rows, ctx.result_dir / 'amplitude_amplification_theory.png')
    summary = {'suite': 'amplitude_amplification_theory', 'n_qubits': args.n_qubits, 'good_indices': good_indices, 'focus': 'congestion_and_fan_in', 'optimal_iteration': k_opt, 'max_iterations': max_iterations, 'evaluation_mode': 'exact' if exact_mode else 'closed_form_only', 'peak_exact_success': max((float(row['exact_success_probability']) for row in rows)), 'rows': rows}
    verification_common__write_json(ctx.result_dir / 'amplitude_amplification_theory.json', summary)
    print(f'Saved results to: {ctx.result_dir}')

def amplitude_amplification_verification__main_aer(script_file: str, *, device: str) -> None:
    parser = argparse.ArgumentParser(description=f'Aer-{device.lower()} amplitude amplification verification.')
    amplitude_amplification_verification__add_shared_arguments(parser)
    parser.add_argument('--topologies', default='alltoall,line,ring,grid', help='Comma-separated topologies: alltoall,line,ring,grid')
    parser.add_argument('--optimization-level', type=int, default=2)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.n_qubits = verification_common__prompt_int('How many qubits do you want to test?', args.n_qubits)
    ctx = verification_common__setup_run_context(script_file)
    good_indices = amplitude_amplification_verification__parse_good_indices(args.good_indices, args.n_qubits)
    marked_count = len(good_indices)
    k_opt = amplitude_amplification_verification__optimal_grover_iteration(args.n_qubits, marked_count)
    max_iterations = args.max_iterations if args.max_iterations is not None else 2 * k_opt + 2
    topologies = verification_aer__parse_topologies(args.topologies)
    simulation_limit = 20
    exact_mode = args.n_qubits <= simulation_limit
    if not exact_mode:
        print(f'[aer-aa] n={args.n_qubits} is above the Aer exact-state limit ({simulation_limit}). Running transpilation, congestion, and fan-in metrics only.')
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'n_qubits': int(args.n_qubits), 'good_indices': list(good_indices), 'iteration': iteration, 'topologies': list(topologies), 'optimization_level': int(args.optimization_level), 'seed': int(args.seed), 'device': device, 'exact_mode': exact_mode} for iteration in range(max_iterations + 1)]
    print(f'[aa_aer] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(amplitude_amplification_verification__evaluate_aer_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[aa_aer] parallel section complete rows={len(rows)}')
    stem = f'amplitude_amplification_aer_{device.lower()}'
    verification_common__write_csv(ctx.result_dir / f'{stem}.csv', rows)
    amplitude_amplification_verification__make_aer_plot(rows, ctx.result_dir / f'{stem}.png')
    summary = {'suite': stem, 'backend_device': device, 'n_qubits': args.n_qubits, 'good_indices': good_indices, 'focus': 'congestion_and_fan_in', 'evaluation_mode': 'exact' if exact_mode else 'metrics_only', 'optimal_iteration': k_opt, 'topologies': topologies, 'rows': rows}
    verification_common__write_json(ctx.result_dir / f'{stem}.json', summary)
    print(f'Saved results to: {ctx.result_dir}')


# Inlined from: amplitude_amplification_qudit_verification.py

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
except Exception:
    QuantumCircuit = None
    transpile = None
    CouplingMap = None
FAMILY_LABELS = {'grover': 'Grover / Trivial AA Instance', 'dqaa': 'DQAA / Canonical AA Form', 'fpaa': 'FPAA / Canonicalized Through DQAA', 'oaa': 'OAA / Canonicalized Through DQAA', 'cqaa': 'CQAA / Canonicalized Through DQAA', 'foaa': 'FOAA / Canonicalized Through DQAA', 'vtaa': 'VTAA / Canonicalized Through DQAA', 'qsvt': 'QSVT / Canonicalized Through DQAA'}
FAMILY_CANONICAL = {'grover': 'grover', 'dqaa': 'dqaa', 'fpaa': 'dqaa', 'oaa': 'dqaa', 'cqaa': 'dqaa', 'foaa': 'dqaa', 'vtaa': 'dqaa', 'qsvt': 'dqaa'}
FAMILY_ALIASES = {'aa': 'grover', 'basic_aa': 'grover', 'basic-aa': 'grover', 'foqa': 'foaa'}
CORE_FAMILY_ORDER = ('grover', 'dqaa')
FAMILY_ORDER = ('grover', 'dqaa', 'fpaa', 'oaa', 'cqaa', 'foaa', 'vtaa', 'qsvt')
MODE_LABELS = {'swap_baseline': 'SWAP Baseline', 'routed_serialized': 'Routed Serialized', 'routed_parallel': 'Routed Parallel'}

def amplitude_amplification_qudit_verification__parse_float_csv(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(',') if item.strip()]

def amplitude_amplification_qudit_verification__parse_path_lengths(spec: str, controls: int) -> list[int]:
    values = verification_common__parse_int_list(spec)
    if not values:
        raise ValueError('Provide at least one path length.')
    if len(values) == 1:
        values *= controls
    if len(values) != controls:
        raise ValueError(f'Expected either one path length or {controls} path lengths, got {len(values)}.')
    if any((length < 1 for length in values)):
        raise ValueError('Every path length must be at least 1.')
    return values

def amplitude_amplification_qudit_verification__parse_marked_bits(spec: str, n_qubits: int) -> tuple[int, ...]:
    text = spec.strip().replace(' ', '')
    if len(text) != n_qubits or any((bit not in '01' for bit in text)):
        raise ValueError(f'Marked bitstring must be a {n_qubits}-bit binary string.')
    return tuple((int(bit) for bit in text))

def amplitude_amplification_qudit_verification__resolve_marked_argument(marked: str, n_qubits: int) -> str:
    return format(1, f'0{n_qubits}b') if marked == '0001' and n_qubits != 4 else marked

def amplitude_amplification_qudit_verification__build_qiskit_grover_circuit(n_qubits: int, marked_bits: Sequence[int], iterations: int) -> QuantumCircuit:
    if QuantumCircuit is None:
        raise RuntimeError('Qiskit is not available.')
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    if iterations <= 0:
        return qc
    little_endian_bits = tuple(reversed(tuple((int(bit) for bit in marked_bits))))

    def append_marked_controlled_phase(target: QuantumCircuit) -> None:
        for qubit, bit in enumerate(little_endian_bits):
            if bit == 0:
                target.x(qubit)
        if n_qubits == 1:
            target.z(0)
        else:
            target.h(n_qubits - 1)
            target.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            target.h(n_qubits - 1)
        for qubit, bit in enumerate(little_endian_bits):
            if bit == 0:
                target.x(qubit)

    def append_diffusion(target: QuantumCircuit) -> None:
        target.h(range(n_qubits))
        target.x(range(n_qubits))
        if n_qubits == 1:
            target.z(0)
        else:
            target.h(n_qubits - 1)
            target.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            target.h(n_qubits - 1)
        target.x(range(n_qubits))
        target.h(range(n_qubits))
    for _ in range(iterations):
        append_marked_controlled_phase(qc)
        append_diffusion(qc)
    return qc

def amplitude_amplification_qudit_verification__line_coupling_map(num_qubits: int) -> CouplingMap:
    if CouplingMap is None:
        raise RuntimeError('Qiskit CouplingMap is not available.')
    edges: list[tuple[int, int]] = []
    for left in range(num_qubits - 1):
        edges.append((left, left + 1))
        edges.append((left + 1, left))
    return CouplingMap(edges)

def amplitude_amplification_qudit_verification__baseline_swap_metrics(n_qubits: int, marked_bits: Sequence[int], iteration: int, *, seed: int, optimization_level: int) -> dict[str, object] | None:
    if QuantumCircuit is None or transpile is None or CouplingMap is None:
        return None
    logical = amplitude_amplification_qudit_verification__build_qiskit_grover_circuit(n_qubits, marked_bits, iteration)
    transpiled = transpile(logical, coupling_map=amplitude_amplification_qudit_verification__line_coupling_map(n_qubits), routing_method='sabre', layout_method='sabre', optimization_level=optimization_level, seed_transpiler=seed, initial_layout=list(range(n_qubits)))
    counts = transpiled.count_ops()
    return {'transpiled_depth': int(transpiled.depth() or 0), 'transpiled_size': int(transpiled.size()), 'swap_count': int(counts.get('swap', 0)), 'cx_count': int(counts.get('cx', 0)), 'op_counts': {str(key): int(value) for key, value in counts.items()}}

def amplitude_amplification_qudit_verification__theoretical_layout_metrics(n_qubits: int, path_lengths: Sequence[int]) -> dict[str, object]:
    controls = max(0, n_qubits - 1)
    total_path_length = int(sum(path_lengths)) if path_lengths else 0
    max_path_length = int(max(path_lengths)) if path_lengths else 0
    total_sites = int(total_path_length + 1) if controls > 0 else 1
    active_paths_per_hop = [sum((1 for length in path_lengths if length > hop)) for hop in range(max_path_length)]
    target_hits_per_hop = [sum((1 for length in path_lengths if length == hop + 1)) for hop in range(max_path_length)]
    routing_shift_ops_per_fanin = 2 * total_path_length if controls > 0 else 0
    fanin_block_depth_bound = 2 * max_path_length + 1 if controls > 0 else 1
    serialized_fanin_block_depth_bound = 2 * total_path_length + 1 if controls > 0 else 1
    minimum_parallel_dimension = max(2, 2 ** (controls + 1)) if controls > 0 else 2
    return {'logical_qubits': n_qubits, 'controls': controls, 'fan_in_width': controls, 'total_sites': total_sites, 'path_lengths': list(path_lengths), 'total_path_length': total_path_length, 'max_path_length': max_path_length, 'routing_shift_ops_per_fanin': routing_shift_ops_per_fanin, 'fanin_block_depth_bound': fanin_block_depth_bound, 'serialized_fanin_block_depth_bound': serialized_fanin_block_depth_bound, 'routing_shift_ops_per_iteration': 2 * routing_shift_ops_per_fanin, 'iteration_depth_bound': 2 * fanin_block_depth_bound, 'serialized_iteration_depth_bound': 2 * serialized_fanin_block_depth_bound, 'fanin_target_concurrency': controls, 'required_buses_max': controls, 'minimum_parallel_dimension': int(minimum_parallel_dimension), 'minimum_parallel_dimension_log2': float(math.log2(minimum_parallel_dimension)), 'max_parallel_edges_per_layer': int(max(active_paths_per_hop, default=0)), 'edge_congestion_max': int(max(active_paths_per_hop, default=0)), 'target_edge_conflict_max': int(max(target_hits_per_hop, default=0)), 'swap_proxy_depth_per_fanin': 4 * total_path_length + 1 if controls > 0 else 1, 'swap_proxy_depth_per_iteration': 2 * (4 * total_path_length + 1) if controls > 0 else 2, 'swap_proxy_swaps_per_fanin': 2 * total_path_length if controls > 0 else 0, 'swap_proxy_swaps_per_iteration': 4 * total_path_length if controls > 0 else 0}

def amplitude_amplification_qudit_verification__parse_family_list(spec: str) -> list[str]:
    entries = [chunk.strip().lower() for chunk in spec.split(',') if chunk.strip()]
    if not entries or entries == ['all']:
        return list(FAMILY_ORDER)
    resolved: list[str] = []
    for entry in entries:
        family = FAMILY_ALIASES.get(entry, entry)
        if family not in FAMILY_LABELS:
            raise ValueError(f"Unsupported family '{entry}'. Use one of: {', '.join(FAMILY_ORDER)} or all.")
        if family not in resolved:
            resolved.append(family)
    return resolved

def amplitude_amplification_qudit_verification__initial_success_probability(n_qubits: int, marked_count: int=1) -> float:
    return float(marked_count / 2 ** n_qubits)

def amplitude_amplification_qudit_verification__fpaa_passband_edge(length_l: int, delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError('delta must lie in (0, 1).')
    gamma_inv = math.cosh(math.acosh(1.0 / delta) / float(length_l))
    return float(1.0 - gamma_inv ** (-2))

def amplitude_amplification_qudit_verification__normalize_positive_weights(weights: Sequence[float]) -> list[float]:
    values = [max(0.0, float(weight)) for weight in weights]
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError('At least one positive weight is required.')
    return [value / total for value in values]

def amplitude_amplification_qudit_verification__select_local_path_lengths(path_lengths: Sequence[int], controls: int) -> list[int]:
    if controls <= 0:
        return []
    if not path_lengths:
        return [1] * controls
    if len(path_lengths) >= controls:
        return list(path_lengths[:controls])
    return list(path_lengths) + [path_lengths[-1]] * (controls - len(path_lengths))

def amplitude_amplification_qudit_verification__select_suffix_path_lengths(path_lengths: Sequence[int], controls: int) -> list[int]:
    if controls <= 0:
        return []
    if not path_lengths:
        return [1] * controls
    if len(path_lengths) >= controls:
        return list(path_lengths[-controls:])
    return [path_lengths[0]] * (controls - len(path_lengths)) + list(path_lengths)

def amplitude_amplification_qudit_verification__estimate_total_gate_time(*, routing_shift_ops: float, target_applications: float, local_gate_applications: float, routing_gate_time: float, target_gate_time: float, local_gate_time: float) -> float:
    return float(routing_shift_ops) * float(routing_gate_time) + float(target_applications) * float(target_gate_time) + float(local_gate_applications) * float(local_gate_time)

def amplitude_amplification_qudit_verification__proxy_mode_metrics(*, layout_metrics: dict[str, object], representative_rounds: float, mode: str, parallel_lanes: int) -> dict[str, float | int | None]:
    lanes = max(1, int(parallel_lanes))
    if mode == 'swap_baseline':
        return {'estimated_total_routing_shifts': float(layout_metrics['swap_proxy_swaps_per_iteration']) * float(representative_rounds), 'estimated_wall_clock_depth': float(layout_metrics['swap_proxy_depth_per_iteration']) * float(representative_rounds) / lanes, 'estimated_swap_count': float(layout_metrics['swap_proxy_swaps_per_iteration']) * float(representative_rounds)}
    if mode == 'routed_serialized':
        return {'estimated_total_routing_shifts': float(layout_metrics['routing_shift_ops_per_iteration']) * float(representative_rounds), 'estimated_wall_clock_depth': float(layout_metrics['serialized_iteration_depth_bound']) * float(representative_rounds) / lanes, 'estimated_swap_count': 0.0}
    return {'estimated_total_routing_shifts': float(layout_metrics['routing_shift_ops_per_iteration']) * float(representative_rounds), 'estimated_wall_clock_depth': float(layout_metrics['iteration_depth_bound']) * float(representative_rounds) / lanes, 'estimated_swap_count': 0.0}

def amplitude_amplification_qudit_verification__dqaa_local_problem(*, global_n: int, partition_bits: int, marked_bits: Sequence[int], path_lengths: Sequence[int]) -> dict[str, object]:
    if partition_bits <= 0 or partition_bits >= global_n:
        raise ValueError('partition_bits must satisfy 1 <= partition_bits < global_n.')
    local_n = global_n - partition_bits
    local_marked_bits = tuple((int(bit) for bit in marked_bits[partition_bits:]))
    local_controls = max(0, local_n - 1)
    local_path_lengths = amplitude_amplification_qudit_verification__select_suffix_path_lengths(path_lengths, local_controls)
    return {'partition_bits': int(partition_bits), 'parallel_lanes': int(2 ** partition_bits), 'local_logical_qubits': int(local_n), 'local_marked_bits': local_marked_bits, 'local_path_lengths': local_path_lengths, 'advantaged_prefix': ''.join((str(int(bit)) for bit in marked_bits[:partition_bits]))}

def amplitude_amplification_qudit_verification__rewrite_dqaa_metrics(local_metrics: dict[str, object], *, global_n: int, partition_bits: int) -> dict[str, object]:
    rewritten = dict(local_metrics)
    rewritten['logical_qubits'] = int(global_n)
    rewritten['partition_bits'] = int(partition_bits)
    rewritten['local_logical_qubits'] = int(local_metrics['logical_qubits'])
    rewritten['parallel_lanes'] = int(2 ** partition_bits)
    return rewritten

def amplitude_amplification_qudit_verification__requested_dqaa_families(families: Sequence[str]) -> list[str]:
    return [family for family in families if FAMILY_CANONICAL.get(family) == 'dqaa']

def amplitude_amplification_qudit_verification__canonical_family_note(family: str) -> str:
    if family == 'dqaa':
        return 'Canonical DQAA exact routed circuit.'
    return f'Exact canonical DQAA realization used to evaluate the {family.upper()} family at the same routing rigor.'

def amplitude_amplification_qudit_verification__amplitude_row(**updates: object) -> dict[str, object]:
    row = {'family': None, 'family_label': None, 'canonical_family': None, 'canonical_family_label': None, 'mode': None, 'mode_label': None, 'evidence_tier': None, 'family_mode': None, 'family_parameter_name': None, 'family_parameter': None, 'family_note': None, 'iteration': None, 'backend': None, 'evaluation_mode': None, 'dimension': None, 'required_dimension_min': None, 'dimension_condition_met': None, 'theory_success_probability': None, 'routed_success_probability': None, 'logical_state_fidelity': None, 'logical_subspace_probability': None, 'max_routing_population': None, 'moment_count': None, 'operation_count': None, 'one_qudit_gate_count': None, 'two_qudit_gate_count': None, 'max_gate_width': None, 'logical_qubits': None, 'controls': None, 'fan_in_width': None, 'total_sites': None, 'path_lengths': None, 'total_path_length': None, 'max_path_length': None, 'routing_shift_ops_per_fanin': None, 'fanin_block_depth_bound': None, 'serialized_fanin_block_depth_bound': None, 'routing_shift_ops_per_iteration': None, 'iteration_depth_bound': None, 'serialized_iteration_depth_bound': None, 'fanin_target_concurrency': None, 'required_buses_max': None, 'minimum_parallel_dimension': None, 'minimum_parallel_dimension_log2': None, 'max_parallel_edges_per_layer': None, 'edge_congestion_max': None, 'target_edge_conflict_max': None, 'swap_baseline_depth': None, 'swap_baseline_size': None, 'swap_baseline_swap_count': None, 'swap_baseline_cx_count': None, 'swap_baseline_available': None, 'representative_rounds': None, 'oracle_calls': None, 'state_reflections': None, 'block_encoding_calls': None, 'phase_count': None, 'ancilla_qubits': None, 'parallel_lanes': None, 'partition_bits': None, 'local_logical_qubits': None, 'weighted_branch_time': None, 'worst_case_branch_time': None, 'passband_edge': None, 'passband_condition_met': None, 'success_lower_bound': None, 'estimated_total_routing_shifts': None, 'estimated_wall_clock_depth': None, 'estimated_swap_count': None, 'estimated_total_gate_time': None, 'routing_gate_time': None, 'target_gate_time': None, 'local_gate_time': None, 'leakage_epsilon': None}
    row.update(updates)
    return row

def amplitude_amplification_qudit_verification__depth_metric(row: dict[str, object]) -> float:
    if str(row.get('mode') or '') == 'swap_baseline':
        return amplitude_amplification_qudit_verification__to_float(row.get('swap_baseline_depth') or row.get('estimated_wall_clock_depth'), default=0.0)
    return amplitude_amplification_qudit_verification__to_float(row.get('moment_count') or row.get('estimated_wall_clock_depth'), default=0.0)

def amplitude_amplification_qudit_verification__fanin_bound(row: dict[str, object]) -> float:
    if str(row.get('mode') or '') in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    return max(1.0, amplitude_amplification_qudit_verification__to_float(row.get('fanin_target_concurrency'), default=1.0))

def amplitude_amplification_qudit_verification__bus_bound(row: dict[str, object]) -> float:
    if str(row.get('mode') or '') in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    return max(1.0, amplitude_amplification_qudit_verification__to_float(row.get('required_buses_max'), default=1.0))

def amplitude_amplification_qudit_verification__dimension_bound(row: dict[str, object]) -> float:
    mode = str(row.get('mode') or '')
    if mode in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    configured_dimension = row.get('dimension')
    if configured_dimension in (None, '', 'None'):
        return 1.0
    configured_value = max(2.0, amplitude_amplification_qudit_verification__to_float(configured_dimension, default=2.0))
    fanin_bound = amplitude_amplification_qudit_verification__fanin_bound(row)
    return float(min(fanin_bound, max(1.0, math.floor(math.log2(configured_value)) - 1.0)))

def amplitude_amplification_qudit_verification__predicted_concurrency(row: dict[str, object]) -> tuple[float, float, float, float, str]:
    c_fanin = amplitude_amplification_qudit_verification__fanin_bound(row)
    c_bus = amplitude_amplification_qudit_verification__bus_bound(row)
    c_dimension = amplitude_amplification_qudit_verification__dimension_bound(row)
    c_pred = float(min(c_fanin, c_bus, c_dimension))
    if str(row.get('mode') or '') in {'swap_baseline', 'routed_serialized'}:
        limiting_factor = 'fanin'
    elif c_pred == c_dimension and c_dimension <= min(c_fanin, c_bus):
        limiting_factor = 'dimension'
    elif c_pred == c_fanin and c_fanin <= c_bus:
        limiting_factor = 'fanin'
    else:
        limiting_factor = 'bus'
    return (c_fanin, c_bus, c_dimension, c_pred, limiting_factor)

def amplitude_amplification_qudit_verification__actual_concurrency(row: dict[str, object]) -> float:
    if str(row.get('mode') or '') in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    return max(1.0, amplitude_amplification_qudit_verification__to_float(row.get('max_parallel_edges_per_layer') or row.get('fanin_target_concurrency'), default=1.0))

def amplitude_amplification_qudit_verification__annotate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        c_fanin, c_bus, c_dimension, c_pred, limiting_factor = amplitude_amplification_qudit_verification__predicted_concurrency(enriched)
        logical_fidelity = enriched.get('logical_state_fidelity')
        required_dimension_min = amplitude_amplification_qudit_verification__to_float(enriched.get('required_dimension_min'), default=0.0)
        configured_dimension = amplitude_amplification_qudit_verification__to_float(enriched.get('dimension'), default=0.0)
        dimension_ratio = None if required_dimension_min <= 0.0 else float(configured_dimension / required_dimension_min)
        prediction_efficiency = float(amplitude_amplification_qudit_verification__actual_concurrency(enriched) / max(c_pred, 1.0))
        enriched.update({
            'algorithm_family': 'aa',
            'depth_metric': amplitude_amplification_qudit_verification__depth_metric(enriched),
            'c_fanin': c_fanin,
            'c_bus': c_bus,
            'c_dimension': c_dimension,
            'c_pred': c_pred,
            'c_actual': amplitude_amplification_qudit_verification__actual_concurrency(enriched),
            'prediction_efficiency': prediction_efficiency,
            'prediction_error_abs': float(abs(amplitude_amplification_qudit_verification__actual_concurrency(enriched) - c_pred)),
            'prediction_tight_10pct': bool(prediction_efficiency >= 0.9),
            'limiting_factor': limiting_factor,
            'dimension_ratio': dimension_ratio,
            'resource_satisfaction_ratio': dimension_ratio,
            'fidelity_error': None if logical_fidelity in (None, '', 'None') else float(max(0.0, 1.0 - float(logical_fidelity))),
            'success_metric': amplitude_amplification_qudit_verification__to_float(enriched.get('routed_success_probability') if enriched.get('routed_success_probability') not in (None, '', 'None') else enriched.get('theory_success_probability'), default=0.0),
            'failure_flag': bool('limited' in str(enriched.get('evaluation_mode') or '')),
        })
        annotated.append(enriched)
    return annotated

def amplitude_amplification_qudit_verification__theory_mapping_payload() -> dict[str, object]:
    return {
        'suite': 'amplitude_amplification',
        'equations': {
            'concurrency_bound': 'C_pred = min(C_fanin, C_bus, C_dimension)',
            'fanin_bound': 'C_fanin = fanin_target_concurrency',
            'bus_bound': 'C_bus = required_buses_max',
            'dimension_bound': 'C_dimension = min(C_fanin, floor(log2(dimension)) - 1)',
            'depth_proxy': 'Depth_proxy ~ representative_rounds / C_pred',
            'exact_parallel_condition': 'dimension >= required_dimension_min',
        },
        'mapping': {
            'fanin': 'fanin_target_concurrency',
            'buses': 'required_buses_max',
            'dimension': 'dimension, dimension_condition_met, required_dimension_min',
            'observed_parallelism': 'max_parallel_edges_per_layer',
            'depth': 'depth_metric',
            'success_probability': 'routed_success_probability',
            'fidelity': 'logical_state_fidelity',
        },
    }

def amplitude_amplification_qudit_verification__representative_rows(rows: Sequence[dict[str, object]], *, family: str, mode: str) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows if str(row.get('family')) == family and str(row.get('mode')) == mode]
    if not candidates:
        candidates = [dict(row) for row in rows if str(row.get('canonical_family')) == family and str(row.get('mode')) == mode]
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in candidates:
        key = (
            str(row.get('backend') or ''),
            str(row.get('family') or row.get('canonical_family') or ''),
            str(row.get('family_parameter') or row.get('iteration') or ''),
        )
        score = (
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            amplitude_amplification_qudit_verification__to_float(row.get('success_metric'), default=0.0),
            -amplitude_amplification_qudit_verification__depth_metric(row),
            amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0),
        )
        previous = grouped.get(key)
        if previous is None or score > previous.get('_score', (-1.0, -1.0, 0.0, 0.0)):
            row['_score'] = score
            grouped[key] = row
    result = []
    for row in grouped.values():
        row.pop('_score', None)
        result.append(row)
    return result

def amplitude_amplification_qudit_verification__make_exact_validation_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    exact_rows = [row for row in rows if str(row.get('evaluation_mode') or '') == 'exact' and str(row.get('family') or '') in {'grover', 'dqaa'}]
    if plt is None or not exact_rows:
        return
    modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    families = ['grover', 'dqaa']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(modes))
    for family in families:
        mode_rows = []
        for mode in modes:
            candidates = [row for row in exact_rows if str(row.get('family')) == family and str(row.get('mode')) == mode]
            mode_rows.append(max(candidates, key=lambda row: (amplitude_amplification_qudit_verification__to_float(row.get('success_metric'), default=0.0), -amplitude_amplification_qudit_verification__depth_metric(row))) if candidates else None)
        axes[0].plot(x, [amplitude_amplification_qudit_verification__to_float(row.get('logical_state_fidelity') or row.get('success_metric'), default=0.0) if row is not None else 0.0 for row in mode_rows], marker='o', label=family)
        axes[1].plot(x, [amplitude_amplification_qudit_verification__depth_metric(row) if row is not None else 0.0 for row in mode_rows], marker='o', label=family)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODE_LABELS[mode] for mode in modes], rotation=15)
    axes[0].set_ylabel('Fidelity / success')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODE_LABELS[mode] for mode in modes], rotation=15)
    axes[1].set_ylabel('Depth / moments')
    axes[1].set_title('Depth by mode')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_dimension_sweep_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    sweep_rows = [row for row in rows if row.get('dimension') not in (None, '', 'None')]
    if plt is None or len({int(float(row['dimension'])) for row in sweep_rows}) <= 1:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for family in ['grover', 'dqaa']:
        parallel_rows = sorted([row for row in sweep_rows if str(row.get('mode')) == 'routed_parallel' and (str(row.get('family')) == family or str(row.get('canonical_family')) == family)], key=lambda row: int(float(row['dimension'])))
        if not parallel_rows:
            continue
        axes[0].plot([int(float(row['dimension'])) for row in parallel_rows], [amplitude_amplification_qudit_verification__depth_metric(row) for row in parallel_rows], marker='o', label=family)
        axes[1].plot([int(float(row['dimension'])) for row in parallel_rows], [amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0) for row in parallel_rows], marker='o', label=family)
    axes[0].set_xlabel('Configured dimension d')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('Configured dimension d')
    axes[1].set_ylabel('Observed concurrency')
    axes[1].set_title('Concurrency vs dimension')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_fanin_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    fanin_rows = [row for row in rows if str(row.get('canonical_family') or row.get('family')) == 'dqaa' and str(row.get('mode')) == 'routed_parallel']
    if plt is None or not fanin_rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for family in sorted({str(row.get('family') or '') for row in fanin_rows}):
        family_rows = sorted([row for row in fanin_rows if str(row.get('family')) == family], key=lambda row: amplitude_amplification_qudit_verification__to_float(row.get('fanin_target_concurrency'), default=0.0))
        if not family_rows:
            continue
        ax.plot([amplitude_amplification_qudit_verification__to_float(row.get('fanin_target_concurrency'), default=0.0) for row in family_rows], [amplitude_amplification_qudit_verification__depth_metric(row) for row in family_rows], marker='o', label=family)
    ax.set_xlabel('Fan-in target concurrency')
    ax.set_ylabel('Depth / moments')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_prediction_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    plot_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline']
    if plt is None or not plot_rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for family in ['grover', 'dqaa']:
        family_rows = [row for row in plot_rows if str(row.get('family')) == family or str(row.get('canonical_family')) == family]
        if not family_rows:
            continue
        ax.scatter([amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0) for row in family_rows], [amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0) for row in family_rows], label=family, alpha=0.8)
    max_axis = max([1.0] + [float(max(amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0), amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0))) for row in plot_rows])
    ax.plot([0.0, max_axis], [0.0, max_axis], linestyle='--', color='black', linewidth=1.0)
    ax.set_xlabel('Predicted concurrency C_pred')
    ax.set_ylabel('Measured concurrency C_actual')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_failure_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    failure_rows = [row for row in rows if bool(row.get('failure_flag')) or str(row.get('mode')) == 'routed_parallel']
    if plt is None or not failure_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    exact_rows = [row for row in failure_rows if row.get('dimension') not in (None, '', 'None')]
    axes[0].scatter([amplitude_amplification_qudit_verification__to_float(row.get('dimension'), default=0.0) for row in exact_rows], [amplitude_amplification_qudit_verification__depth_metric(row) for row in exact_rows], alpha=0.8)
    fidelity_rows = [row for row in exact_rows if row.get('logical_state_fidelity') not in (None, '', 'None')]
    if fidelity_rows:
        axes[1].scatter([amplitude_amplification_qudit_verification__to_float(row.get('dimension'), default=0.0) for row in fidelity_rows], [amplitude_amplification_qudit_verification__to_float(row.get('logical_state_fidelity'), default=0.0) for row in fidelity_rows], alpha=0.8)
    axes[0].set_xlabel('Configured dimension d')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Configured dimension d')
    axes[1].set_ylabel('Logical fidelity')
    axes[1].set_title('Failure / breakdown regime')
    axes[1].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_noise_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    noisy_rows = [row for row in rows if str(row.get('backend')) == 'qutip' and ((amplitude_amplification_qudit_verification__to_float(row.get('routing_gate_time'), default=0.0) > 0.0) or (amplitude_amplification_qudit_verification__to_float(row.get('target_gate_time'), default=0.0) > 0.0) or (amplitude_amplification_qudit_verification__to_float(row.get('local_gate_time'), default=0.0) > 0.0) or (amplitude_amplification_qudit_verification__to_float(row.get('leakage_epsilon'), default=0.0) > 0.0)) and row.get('logical_state_fidelity') not in (None, '', 'None')]
    if plt is None or not noisy_rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for mode in ['routed_serialized', 'routed_parallel']:
        mode_rows = [row for row in noisy_rows if str(row.get('mode')) == mode]
        if not mode_rows:
            continue
        ax.scatter([amplitude_amplification_qudit_verification__depth_metric(row) for row in mode_rows], [amplitude_amplification_qudit_verification__to_float(row.get('logical_state_fidelity'), default=0.0) for row in mode_rows], label=MODE_LABELS[mode], alpha=0.8)
    ax.set_xlabel('Depth / moments')
    ax.set_ylabel('Logical fidelity')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__build_seed_values(base_seed: int, seed_count: int) -> list[int]:
    if seed_count <= 0:
        raise ValueError('seed_count must be positive.')
    return [int(base_seed + 9973 * index) for index in range(seed_count)]

def amplitude_amplification_qudit_verification__fit_power_law(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, object]:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if float(x) > 0.0 and float(y) > 0.0]
    if len(pairs) < 2:
        return {'point_count': len(pairs), 'slope': None, 'intercept': None, 'r_squared': None, 'big_o': None}
    log_x = np.log(np.asarray([pair[0] for pair in pairs], dtype=float))
    log_y = np.log(np.asarray([pair[1] for pair in pairs], dtype=float))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    predicted = slope * log_x + intercept
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 if total <= 0.0 else float(max(0.0, 1.0 - residual / total))
    return {'point_count': len(pairs), 'slope': float(slope), 'intercept': float(intercept), 'r_squared': r_squared, 'big_o': f'O(n^{float(slope):.2f})'}

def amplitude_amplification_qudit_verification__seed_representative_rows(rows: Sequence[dict[str, object]], *, family: str, mode: str) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows if str(row.get('mode')) == mode and (str(row.get('family')) == family or str(row.get('canonical_family')) == family)]
    grouped: dict[int, dict[str, object]] = {}
    for row in candidates:
        key = int(row.get('run_seed') or 0)
        score = (
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            amplitude_amplification_qudit_verification__to_float(row.get('success_metric'), default=0.0),
            -amplitude_amplification_qudit_verification__depth_metric(row),
            amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0),
        )
        previous = grouped.get(key)
        if previous is None or score > previous.get('_score', (-1.0, -1.0, 0.0, 0.0)):
            row['_score'] = score
            grouped[key] = row
    result: list[dict[str, object]] = []
    for row in grouped.values():
        row.pop('_score', None)
        result.append(row)
    return sorted(result, key=lambda entry: int(entry.get('run_seed') or 0))

def amplitude_amplification_qudit_verification__seed_statistics(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    statistics: list[dict[str, object]] = []
    for family in ['grover', 'dqaa']:
        for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
            representative = amplitude_amplification_qudit_verification__seed_representative_rows(rows, family=family, mode=mode)
            if not representative:
                continue
            depth_values = np.asarray([amplitude_amplification_qudit_verification__depth_metric(row) for row in representative], dtype=float)
            success_values = np.asarray([amplitude_amplification_qudit_verification__to_float(row.get('success_metric'), default=0.0) for row in representative], dtype=float)
            statistics.append({
                'family': family,
                'mode': mode,
                'mode_label': MODE_LABELS[mode],
                'sample_count': int(len(representative)),
                'depth_mean': float(np.mean(depth_values)),
                'depth_std': float(np.std(depth_values)),
                'success_mean': float(np.mean(success_values)),
                'success_std': float(np.std(success_values)),
            })
    return statistics

def amplitude_amplification_qudit_verification__make_seed_variability_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    seed_values = sorted({int(row.get('run_seed') or 0) for row in rows})
    if plt is None or len(seed_values) <= 1:
        return
    statistics = amplitude_amplification_qudit_verification__seed_statistics(rows)
    if not statistics:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    families = ['grover', 'dqaa']
    x = np.arange(3)
    modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    width = 0.35
    for index, family in enumerate(families):
        family_rows = {str(row['mode']): row for row in statistics if str(row['family']) == family}
        if not family_rows:
            continue
        axes[index].bar(x, [float(family_rows.get(mode, {}).get('depth_mean', 0.0)) for mode in modes], yerr=[float(family_rows.get(mode, {}).get('depth_std', 0.0)) for mode in modes], width=width, capsize=4, color=['tab:gray', 'tab:orange', 'tab:blue'])
        axes[index].set_xticks(x)
        axes[index].set_xticklabels([MODE_LABELS[mode] for mode in modes], rotation=15)
        axes[index].set_ylabel('Depth / moments')
        axes[index].set_title(f'{title} ({family})')
        axes[index].grid(alpha=0.3, axis='y')
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_efficiency_plot(rows: Sequence[dict[str, object]], scaling_rows: Sequence[dict[str, object]] | None, figure_path: Path, title: str) -> None:
    exact_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline' and row.get('prediction_efficiency') not in (None, '', 'None')]
    scaling_rows = list(scaling_rows or [])
    if plt is None or (not exact_rows and not scaling_rows):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    scaling_candidates = [row for row in scaling_rows if str(row.get('mode') or '') != 'swap_baseline']
    for family in ['grover', 'dqaa']:
        family_scaling = [row for row in scaling_candidates if str(row.get('family')) == family or str(row.get('canonical_family')) == family]
        if family_scaling:
            family_scaling = sorted(family_scaling, key=lambda row: amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits') or row.get('family_parameter'), default=0))
            axes[0].plot([amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits') or row.get('family_parameter'), default=0) for row in family_scaling], [amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) for row in family_scaling], marker='o', label=family)
        family_exact = [row for row in exact_rows if str(row.get('family')) == family or str(row.get('canonical_family')) == family]
        if family_exact:
            dim_rows = [row for row in family_exact if row.get('dimension') not in (None, '', 'None')]
            if dim_rows:
                axes[1].scatter([amplitude_amplification_qudit_verification__to_float(row.get('dimension'), default=0.0) for row in dim_rows], [amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) for row in dim_rows], label=family, alpha=0.8)
            ratio_rows = [row for row in family_exact if row.get('resource_satisfaction_ratio') not in (None, '', 'None')]
            if ratio_rows:
                axes[2].scatter([amplitude_amplification_qudit_verification__to_float(row.get('resource_satisfaction_ratio'), default=0.0) for row in ratio_rows], [amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) for row in ratio_rows], label=family, alpha=0.8)
    axes[0].axhline(1.0, linestyle='--', color='black', linewidth=1.0)
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Efficiency = C_actual / C_pred')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].axhline(1.0, linestyle='--', color='black', linewidth=1.0)
    axes[1].set_xlabel('Configured dimension d')
    axes[1].set_ylabel('Efficiency')
    axes[1].set_title('Efficiency vs dimension')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[2].axhline(1.0, linestyle='--', color='black', linewidth=1.0)
    axes[2].set_xlabel('Violation severity = d / d_req')
    axes[2].set_ylabel('Efficiency')
    axes[2].set_title('Efficiency vs violation severity')
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_phase_diagram(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    phase_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel' and row.get('dimension') not in (None, '', 'None')]
    if plt is None or not phase_rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    factor_colors = {'fanin': 'tab:orange', 'bus': 'tab:blue', 'dimension': 'tab:red'}
    family_markers = {'grover': 'o', 'dqaa': 's'}
    for family in ['grover', 'dqaa']:
        family_rows = [row for row in phase_rows if str(row.get('family')) == family or str(row.get('canonical_family')) == family]
        for factor in ['fanin', 'bus', 'dimension']:
            factor_rows = [row for row in family_rows if str(row.get('limiting_factor')) == factor]
            if not factor_rows:
                continue
            ax.scatter([amplitude_amplification_qudit_verification__to_float(row.get('dimension'), default=0.0) for row in factor_rows], [amplitude_amplification_qudit_verification__to_float(row.get('fanin_target_concurrency') or row.get('logical_qubits'), default=0.0) for row in factor_rows], color=factor_colors[factor], marker=family_markers[family], alpha=0.85, label=f'{family} / {factor}')
    ax.set_xlabel('Configured dimension d')
    ax.set_ylabel('Fan-in demand / logical concurrency target')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small', ncol=2)
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__failure_summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    ideal_lookup: dict[tuple[str, str, str, str, int, int], dict[str, object]] = {}
    for row in [row for row in rows if str(row.get('mode')) == 'routed_parallel']:
        key = (
            str(row.get('family') or row.get('canonical_family') or ''),
            str(row.get('backend') or ''),
            str(row.get('family_parameter_name') or ''),
            str(row.get('family_parameter') or ''),
            amplitude_amplification_qudit_verification__to_int(row.get('iteration'), default=0),
            amplitude_amplification_qudit_verification__to_int(row.get('local_logical_qubits') or row.get('logical_qubits'), default=0),
        )
        score = (
            1 if amplitude_amplification_qudit_verification__to_float(row.get('resource_satisfaction_ratio'), default=0.0) >= 1.0 else 0,
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0),
            amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0),
            -amplitude_amplification_qudit_verification__depth_metric(row),
        )
        previous = ideal_lookup.get(key)
        if previous is None or score > previous.get('_score', (-1, -1, -1.0, -1.0, 0.0)):
            ideal_lookup[key] = dict(row, _score=score)
    failure_rows: list[dict[str, object]] = []
    for row in [row for row in rows if str(row.get('mode')) == 'routed_parallel' and row.get('resource_satisfaction_ratio') not in (None, '', 'None')]:
        key = (
            str(row.get('family') or row.get('canonical_family') or ''),
            str(row.get('backend') or ''),
            str(row.get('family_parameter_name') or ''),
            str(row.get('family_parameter') or ''),
            amplitude_amplification_qudit_verification__to_int(row.get('iteration'), default=0),
            amplitude_amplification_qudit_verification__to_int(row.get('local_logical_qubits') or row.get('logical_qubits'), default=0),
        )
        ideal = ideal_lookup.get(key)
        if ideal is None:
            continue
        ideal_depth = max(amplitude_amplification_qudit_verification__depth_metric(ideal), 1.0)
        failure_rows.append({
            'family': row.get('family') or row.get('canonical_family'),
            'backend': row.get('backend'),
            'resource_satisfaction_ratio': amplitude_amplification_qudit_verification__to_float(row.get('resource_satisfaction_ratio'), default=0.0),
            'depth_blowup_ratio': float(amplitude_amplification_qudit_verification__depth_metric(row) / ideal_depth),
            'concurrency_loss_ratio': amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0),
            'limiting_factor': row.get('limiting_factor'),
        })
    return failure_rows

def amplitude_amplification_qudit_verification__make_quantitative_failure_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    failure_rows = amplitude_amplification_qudit_verification__failure_summary_rows(rows)
    if plt is None or not failure_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    color_lookup = {'fanin': 'tab:orange', 'bus': 'tab:blue', 'dimension': 'tab:red'}
    axes[0].scatter([float(row['resource_satisfaction_ratio']) for row in failure_rows], [float(row['depth_blowup_ratio']) for row in failure_rows], c=[color_lookup.get(str(row.get('limiting_factor') or ''), 'tab:gray') for row in failure_rows], alpha=0.85)
    axes[1].scatter([float(row['resource_satisfaction_ratio']) for row in failure_rows], [float(row['concurrency_loss_ratio']) for row in failure_rows], c=[color_lookup.get(str(row.get('limiting_factor') or ''), 'tab:gray') for row in failure_rows], alpha=0.85)
    axes[0].set_xlabel('Violation severity = d / d_req')
    axes[0].set_ylabel('Depth actual / depth ideal')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Violation severity = d / d_req')
    axes[1].set_ylabel('Concurrency efficiency = C_actual / C_pred')
    axes[1].set_title('Quantitative degradation')
    axes[1].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_constraint_distribution_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    distribution_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel']
    if plt is None or not distribution_rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    factors = ['fanin', 'bus', 'dimension']
    families = ['grover', 'dqaa']
    x = np.arange(len(families))
    width = 0.22
    for offset_index, factor in enumerate(factors):
        heights = []
        for family in families:
            family_rows = [row for row in distribution_rows if str(row.get('family')) == family or str(row.get('canonical_family')) == family]
            count = sum((1 for row in family_rows if str(row.get('limiting_factor')) == factor))
            heights.append(0.0 if not family_rows else float(count / len(family_rows)))
        ax.bar(x + (offset_index - 1) * width, heights, width=width, label=factor)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel('Fraction of runs')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__write_claim_summary(rows: Sequence[dict[str, object]], result_dir: Path, *, suite_stem: str, title_prefix: str, scaling_rows: Sequence[dict[str, object]] | None=None) -> None:
    modeled_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline']
    tight_fraction = 0.0 if not modeled_rows else float(sum((1 for row in modeled_rows if bool(row.get('prediction_tight_10pct')))) / len(modeled_rows))
    mean_efficiency = 0.0 if not modeled_rows else float(np.mean([amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) for row in modeled_rows]))
    routed_parallel_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel']
    dominance_counts = {factor: sum((1 for row in routed_parallel_rows if str(row.get('limiting_factor')) == factor)) for factor in ['fanin', 'bus', 'dimension']}
    scaling_laws = amplitude_amplification_qudit_verification__scaling_law_summary(list(scaling_rows or [])) if scaling_rows else []
    scaling_lookup = {(str(row.get('family') or ''), str(row.get('mode') or '')): row for row in scaling_laws}
    grover_serialized = scaling_lookup.get(('grover', 'routed_serialized'), {})
    grover_parallel = scaling_lookup.get(('grover', 'routed_parallel'), {})
    class_change = 'undetermined'
    if grover_serialized.get('slope') is not None and grover_parallel.get('slope') is not None:
        class_change = 'class-improving' if float(grover_parallel['slope']) + 0.2 < float(grover_serialized['slope']) else 'primarily constant-factor'
    noise_rows = [row for row in rows if str(row.get('backend')) == 'qutip' and row.get('logical_state_fidelity') not in (None, '', 'None')]
    parallel_noise = [row for row in noise_rows if str(row.get('mode')) == 'routed_parallel']
    serialized_noise = [row for row in noise_rows if str(row.get('mode')) == 'routed_serialized']
    lines = [
        f'{title_prefix} claim summary',
        f'Modeled runs: {len(modeled_rows)}',
        f'Tight prediction fraction (efficiency >= 0.9): {tight_fraction:.3f}',
        f'Mean concurrency efficiency C_actual/C_pred: {mean_efficiency:.3f}',
        f'Dominant constraint counts: fanin={dominance_counts["fanin"]}, bus={dominance_counts["bus"]}, dimension={dominance_counts["dimension"]}',
        f'Scaling interpretation (Grover proxy): serialized={grover_serialized.get("big_o")}, parallel={grover_parallel.get("big_o")}, verdict={class_change}',
    ]
    if parallel_noise and serialized_noise:
        lines.append(f'Noise-depth punchline: mean parallel depth={np.mean([amplitude_amplification_qudit_verification__depth_metric(row) for row in parallel_noise]):.3f}, mean serialized depth={np.mean([amplitude_amplification_qudit_verification__depth_metric(row) for row in serialized_noise]):.3f}, mean parallel fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("logical_state_fidelity"), default=0.0) for row in parallel_noise]):.3f}, mean serialized fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("logical_state_fidelity"), default=0.0) for row in serialized_noise]):.3f}')
    edge_rows = sorted(amplitude_amplification_qudit_verification__failure_summary_rows(rows), key=lambda row: float(row.get('resource_satisfaction_ratio') or 1.0))
    if edge_rows:
        worst = edge_rows[0]
        lines.append(f'Worst stress case: family={worst["family"]}, violation={float(worst["resource_satisfaction_ratio"]):.3f}, depth_blowup={float(worst["depth_blowup_ratio"]):.3f}, efficiency={float(worst["concurrency_loss_ratio"]):.3f}, limiting={worst["limiting_factor"]}')
    (result_dir / f'{suite_stem}_claim_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def amplitude_amplification_qudit_verification__write_analysis_bundle(rows: Sequence[dict[str, object]], result_dir: Path, *, suite_stem: str, title_prefix: str, scaling_rows: Sequence[dict[str, object]] | None=None) -> None:
    mapping = amplitude_amplification_qudit_verification__theory_mapping_payload()
    seed_statistics = amplitude_amplification_qudit_verification__seed_statistics(rows)
    exact_rows = [
        row for row in rows
        if str(row.get('evaluation_mode') or '') == 'exact'
        and bool(row.get('dimension_condition_met', True))
    ]
    verification_common__write_json(result_dir / 'theory_to_observable.json', mapping)
    verification_common__write_csv(result_dir / f'{suite_stem}_theory_observables.csv', rows)
    verification_common__write_json(result_dir / f'{suite_stem}_theory_observables.json', {'suite': suite_stem, 'mapping': mapping, 'rows': rows})
    verification_common__write_csv(result_dir / f'{suite_stem}_theory_observables_exact.csv', exact_rows)
    verification_common__write_json(result_dir / f'{suite_stem}_theory_observables_exact.json', {'suite': suite_stem, 'subset': 'exact', 'mapping': mapping, 'rows': exact_rows})
    verification_common__write_csv(result_dir / f'{suite_stem}_theory_observables_mixed.csv', rows)
    verification_common__write_json(result_dir / f'{suite_stem}_theory_observables_mixed.json', {'suite': suite_stem, 'subset': 'mixed', 'mapping': mapping, 'rows': rows})
    verification_common__write_csv(result_dir / f'{suite_stem}_seed_statistics.csv', seed_statistics)
    verification_common__write_json(result_dir / f'{suite_stem}_seed_statistics.json', {'suite': suite_stem, 'rows': seed_statistics})
    amplitude_amplification_qudit_verification__make_exact_validation_plot(rows, result_dir / f'{suite_stem}_exact_validation.png', f'{title_prefix} exact validation')
    amplitude_amplification_qudit_verification__make_dimension_sweep_plot(rows, result_dir / f'{suite_stem}_dimension_sweep.png', f'{title_prefix} dimension sweep')
    amplitude_amplification_qudit_verification__make_fanin_plot(rows, result_dir / f'{suite_stem}_fanin_sweep.png', f'{title_prefix} fan-in sweep')
    amplitude_amplification_qudit_verification__make_seed_variability_plot(rows, result_dir / f'{suite_stem}_seed_variability.png', f'{title_prefix} seed variability')
    amplitude_amplification_qudit_verification__make_efficiency_plot(rows, scaling_rows, result_dir / f'{suite_stem}_efficiency.png', f'{title_prefix} efficiency')
    amplitude_amplification_qudit_verification__make_phase_diagram(rows, result_dir / f'{suite_stem}_phase_diagram.png', f'{title_prefix} dominant-constraint phase diagram')
    amplitude_amplification_qudit_verification__make_quantitative_failure_plot(rows, result_dir / f'{suite_stem}_quantitative_failure.png', f'{title_prefix} failure severity')
    amplitude_amplification_qudit_verification__make_constraint_distribution_plot(rows, result_dir / f'{suite_stem}_constraint_distribution.png', f'{title_prefix} dominant constraint distribution')
    amplitude_amplification_qudit_verification__make_prediction_plot(rows, result_dir / f'{suite_stem}_prediction_vs_measurement.png', f'{title_prefix} prediction')
    amplitude_amplification_qudit_verification__make_failure_plot(rows, result_dir / f'{suite_stem}_failure_regime.png', f'{title_prefix} failure regime')
    amplitude_amplification_qudit_verification__make_noise_plot(rows, result_dir / f'{suite_stem}_noise_validation.png', f'{title_prefix} noisy validation')
    if scaling_rows:
        scaling_laws = amplitude_amplification_qudit_verification__scaling_law_summary(scaling_rows)
        verification_common__write_csv(result_dir / f'{suite_stem}_scaling.csv', list(scaling_rows))
        verification_common__write_json(result_dir / f'{suite_stem}_scaling.json', {'suite': suite_stem, 'rows': list(scaling_rows)})
        verification_common__write_csv(result_dir / f'{suite_stem}_scaling_laws.csv', scaling_laws)
        verification_common__write_json(result_dir / f'{suite_stem}_scaling_laws.json', {'suite': suite_stem, 'rows': scaling_laws})
        amplitude_amplification_qudit_verification__make_scaling_plot(scaling_rows, result_dir / f'{suite_stem}_scaling.png', f'{title_prefix} scaling')
    amplitude_amplification_qudit_verification__write_claim_summary(rows, result_dir, suite_stem=suite_stem, title_prefix=title_prefix, scaling_rows=scaling_rows)

def amplitude_amplification_qudit_verification__build_proxy_rows(*, families: Sequence[str], backend: str, n_qubits: int, path_lengths: Sequence[int], configured_dimension: int | None, max_iterations: int, routing_gate_time: float=0.0, target_gate_time: float=0.0, local_gate_time: float=0.0, leakage_epsilon: float=0.0, fpaa_lengths: Sequence[int]=(), fpaa_delta: float=0.2, oaa_rounds: int=3, dqaa_partition_bits: Sequence[int]=(), cqaa_rounds: int=3, foaa_lengths: Sequence[int]=(), vtaa_branch_times: Sequence[float]=(), vtaa_branch_weights: Sequence[float]=(), vtaa_branch_successes: Sequence[float]=(), qsvt_degrees: Sequence[int]=()) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    full_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(n_qubits, path_lengths)
    base_success = amplitude_amplification_qudit_verification__initial_success_probability(n_qubits)

    def add_common_proxy(*, family: str, parameter_name: str, parameter_value: object, layout_metrics: dict[str, object], representative_rounds: float, parallel_lanes: int, family_mode: str, family_note: str, oracle_calls: float | None=None, state_reflections: float | None=None, block_encoding_calls: float | None=None, phase_count: float | None=None, ancilla_qubits: int | None=None, partition_bits: int | None=None, local_logical_qubits: int | None=None, weighted_branch_time: float | None=None, worst_case_branch_time: float | None=None, passband_edge: float | None=None, passband_condition_met: bool | None=None, success_lower_bound: float | None=None) -> None:
        local_applications = float(layout_metrics['logical_qubits']) * max(1.0, float(representative_rounds))
        target_applications = max(1.0, float(representative_rounds))
        required_dimension = int(layout_metrics['minimum_parallel_dimension'])
        for mode in ('swap_baseline', 'routed_serialized', 'routed_parallel'):
            mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=layout_metrics, representative_rounds=representative_rounds, mode=mode, parallel_lanes=parallel_lanes)
            is_swap_mode = mode == 'swap_baseline'
            effective_routing_shift_ops = float(mode_metrics['estimated_total_routing_shifts'] or 0.0)
            rows.append(amplitude_amplification_qudit_verification__amplitude_row(family=family, family_label=FAMILY_LABELS[family], canonical_family=FAMILY_CANONICAL.get(family, family), canonical_family_label=FAMILY_LABELS[FAMILY_CANONICAL.get(family, family)], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='structural_proxy', family_mode=family_mode, family_parameter_name=parameter_name, family_parameter=parameter_value, family_note=family_note, backend=backend, evaluation_mode='reference_proxy' if is_swap_mode else 'proxy_only', dimension=configured_dimension if not is_swap_mode else None, required_dimension_min=None if is_swap_mode else required_dimension, dimension_condition_met=None if is_swap_mode or configured_dimension is None else bool(configured_dimension >= required_dimension), logical_qubits=layout_metrics['logical_qubits'], controls=layout_metrics['controls'], fan_in_width=layout_metrics['fan_in_width'], total_sites=layout_metrics['total_sites'], path_lengths=layout_metrics['path_lengths'], total_path_length=layout_metrics['total_path_length'], max_path_length=layout_metrics['max_path_length'], routing_shift_ops_per_fanin=layout_metrics['routing_shift_ops_per_fanin'], fanin_block_depth_bound=layout_metrics['fanin_block_depth_bound'], serialized_fanin_block_depth_bound=layout_metrics['serialized_fanin_block_depth_bound'], routing_shift_ops_per_iteration=layout_metrics['routing_shift_ops_per_iteration'], iteration_depth_bound=layout_metrics['iteration_depth_bound'], serialized_iteration_depth_bound=layout_metrics['serialized_iteration_depth_bound'], fanin_target_concurrency=layout_metrics['fanin_target_concurrency'], required_buses_max=layout_metrics['required_buses_max'], minimum_parallel_dimension=layout_metrics['minimum_parallel_dimension'], minimum_parallel_dimension_log2=layout_metrics['minimum_parallel_dimension_log2'], max_parallel_edges_per_layer=layout_metrics['max_parallel_edges_per_layer'], edge_congestion_max=layout_metrics['edge_congestion_max'], target_edge_conflict_max=layout_metrics['target_edge_conflict_max'], representative_rounds=representative_rounds, oracle_calls=oracle_calls, state_reflections=state_reflections, block_encoding_calls=block_encoding_calls, phase_count=phase_count, ancilla_qubits=ancilla_qubits, parallel_lanes=parallel_lanes, partition_bits=partition_bits, local_logical_qubits=local_logical_qubits, weighted_branch_time=weighted_branch_time, worst_case_branch_time=worst_case_branch_time, passband_edge=passband_edge, passband_condition_met=passband_condition_met, success_lower_bound=success_lower_bound, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], estimated_total_gate_time=amplitude_amplification_qudit_verification__estimate_total_gate_time(routing_shift_ops=effective_routing_shift_ops, target_applications=target_applications, local_gate_applications=local_applications, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon, swap_baseline_available=False))
    if 'fpaa' in families:
        for length_l in fpaa_lengths:
            rounds = max(1, 2 * (int(length_l) - 1))
            edge = amplitude_amplification_qudit_verification__fpaa_passband_edge(int(length_l), fpaa_delta)
            add_common_proxy(family='fpaa', parameter_name='L', parameter_value=int(length_l), layout_metrics=full_metrics, representative_rounds=rounds, parallel_lanes=1, family_mode='fixed_point_schedule_proxy', family_note='Mapped from the fixed-point schedule family in the theory/transpile folders.', oracle_calls=int(length_l) - 1, state_reflections=int(length_l) - 1, passband_edge=edge, passband_condition_met=base_success >= edge, success_lower_bound=max(0.0, 1.0 - fpaa_delta ** 2) if base_success >= edge else None)
    if 'oaa' in families:
        add_common_proxy(family='oaa', parameter_name='rounds', parameter_value=int(oaa_rounds), layout_metrics=full_metrics, representative_rounds=max(1, 2 * int(oaa_rounds) + 1), parallel_lanes=1, family_mode='oblivious_block_encoding_proxy', family_note='Tracks repeated block-encoding use plus clean-projector reflections.', block_encoding_calls=2 * int(oaa_rounds) + 1, ancilla_qubits=1)
    if 'dqaa' in families:
        for partition_bits in dqaa_partition_bits:
            local_qubits = max(1, n_qubits - int(partition_bits))
            local_controls = max(0, local_qubits - 1)
            local_path_lengths = amplitude_amplification_qudit_verification__select_local_path_lengths(path_lengths, local_controls)
            local_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(local_qubits, local_path_lengths)
            local_rounds = max(1, 2 * amplitude_amplification_qudit_verification__optimal_iteration(local_qubits))
            partitions = 2 ** int(partition_bits)
            add_common_proxy(family='dqaa', parameter_name='partition_bits', parameter_value=int(partition_bits), layout_metrics=local_metrics, representative_rounds=local_rounds, parallel_lanes=partitions, family_mode='distributed_parallel_proxy', family_note='Uses partition parallelism as the main architecture advantage signal.', oracle_calls=local_rounds / 2.0, state_reflections=local_rounds / 2.0, partition_bits=int(partition_bits), local_logical_qubits=local_qubits, success_lower_bound=base_success)
    if 'cqaa' in families:
        add_common_proxy(family='cqaa', parameter_name='rounds', parameter_value=int(cqaa_rounds), layout_metrics=full_metrics, representative_rounds=max(1, int(cqaa_rounds)), parallel_lanes=1, family_mode='controlled_amplification_proxy', family_note='Adds a control layer over the amplified search operator; useful for routed-control stress.', oracle_calls=int(cqaa_rounds), ancilla_qubits=1)
    if 'foaa' in families:
        for length_l in foaa_lengths:
            add_common_proxy(family='foaa', parameter_name='L', parameter_value=int(length_l), layout_metrics=full_metrics, representative_rounds=max(1, 2 * (int(length_l) - 1)), parallel_lanes=1, family_mode='fixed_point_oblivious_proxy', family_note='Combines oblivious block-encoding structure with fixed-point damping.', oracle_calls=int(length_l) - 1, block_encoding_calls=2 * (int(length_l) - 1) + 1, ancilla_qubits=1, passband_edge=amplitude_amplification_qudit_verification__fpaa_passband_edge(int(length_l), fpaa_delta))
    if 'vtaa' in families:
        weights = amplitude_amplification_qudit_verification__normalize_positive_weights(vtaa_branch_weights)
        if not len(vtaa_branch_times) == len(weights) == len(vtaa_branch_successes):
            raise ValueError('VTAA branch times, weights, and successes must have the same length.')
        weighted_time = float(sum((weight * time for weight, time in zip(weights, vtaa_branch_times))))
        worst_time = float(max(vtaa_branch_times))
        success = float(sum((weight * probability for weight, probability in zip(weights, vtaa_branch_successes))))
        add_common_proxy(family='vtaa', parameter_name='branch_count', parameter_value=len(weights), layout_metrics=full_metrics, representative_rounds=weighted_time, parallel_lanes=len(weights), family_mode='variable_time_proxy', family_note='Reports weighted branch-time routing pressure and worst-case wall-clock depth.', oracle_calls=weighted_time, weighted_branch_time=weighted_time, worst_case_branch_time=worst_time, success_lower_bound=success)
    if 'qsvt' in families:
        for degree in qsvt_degrees:
            add_common_proxy(family='qsvt', parameter_name='degree', parameter_value=int(degree), layout_metrics=full_metrics, representative_rounds=max(1, int(degree)), parallel_lanes=1, family_mode='phase_sequence_proxy', family_note='Uses phase-sequence length as the dominant routed block-encoding load.', block_encoding_calls=int(degree), phase_count=int(degree) + 1, ancilla_qubits=1)
    return rows

def amplitude_amplification_qudit_verification__build_grover_scaling_proxy_rows(*, backend: str, scaling_qubits: Sequence[int], path_lengths: Sequence[int], routing_gate_time: float, target_gate_time: float, local_gate_time: float, leakage_epsilon: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_qubits in scaling_qubits:
        controls = max(0, int(n_qubits) - 1)
        scaled_path_lengths = amplitude_amplification_qudit_verification__select_local_path_lengths(path_lengths, controls)
        layout_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(int(n_qubits), scaled_path_lengths)
        iteration = amplitude_amplification_qudit_verification__optimal_iteration(int(n_qubits))
        routing_rounds = max(1, 2 * iteration)
        configured_dimension = int(layout_metrics['minimum_parallel_dimension'])
        for mode in ('swap_baseline', 'routed_serialized', 'routed_parallel'):
            mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=layout_metrics, representative_rounds=routing_rounds, mode=mode, parallel_lanes=1)
            is_swap_mode = mode == 'swap_baseline'
            rows.append(amplitude_amplification_qudit_verification__amplitude_row(family='grover', family_label=FAMILY_LABELS['grover'], canonical_family='grover', canonical_family_label=FAMILY_LABELS['grover'], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='structural_proxy', family_mode='grover_scaling_proxy', family_parameter_name='n_qubits', family_parameter=int(n_qubits), family_note='Large-n Grover scaling proxy at the approximate optimal iteration.', iteration=iteration, backend=backend, evaluation_mode='reference_proxy' if is_swap_mode else 'proxy_only', dimension=None if is_swap_mode else configured_dimension, required_dimension_min=None if is_swap_mode else configured_dimension, dimension_condition_met=None if is_swap_mode else True, theory_success_probability=qubit_reference__grover_success_formula(int(n_qubits), 1, iteration), logical_qubits=layout_metrics['logical_qubits'], controls=layout_metrics['controls'], fan_in_width=layout_metrics['fan_in_width'], total_sites=layout_metrics['total_sites'], path_lengths=layout_metrics['path_lengths'], total_path_length=layout_metrics['total_path_length'], max_path_length=layout_metrics['max_path_length'], routing_shift_ops_per_fanin=layout_metrics['routing_shift_ops_per_fanin'], fanin_block_depth_bound=layout_metrics['fanin_block_depth_bound'], serialized_fanin_block_depth_bound=layout_metrics['serialized_fanin_block_depth_bound'], routing_shift_ops_per_iteration=layout_metrics['routing_shift_ops_per_iteration'], iteration_depth_bound=layout_metrics['iteration_depth_bound'], serialized_iteration_depth_bound=layout_metrics['serialized_iteration_depth_bound'], fanin_target_concurrency=layout_metrics['fanin_target_concurrency'], required_buses_max=layout_metrics['required_buses_max'], minimum_parallel_dimension=layout_metrics['minimum_parallel_dimension'], minimum_parallel_dimension_log2=layout_metrics['minimum_parallel_dimension_log2'], max_parallel_edges_per_layer=layout_metrics['max_parallel_edges_per_layer'], edge_congestion_max=layout_metrics['edge_congestion_max'], target_edge_conflict_max=layout_metrics['target_edge_conflict_max'], representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, parallel_lanes=1, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], estimated_total_gate_time=amplitude_amplification_qudit_verification__estimate_total_gate_time(routing_shift_ops=float(mode_metrics['estimated_total_routing_shifts'] or 0.0), target_applications=max(1, routing_rounds), local_gate_applications=float(layout_metrics['logical_qubits']) * max(1.0, float(routing_rounds)), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon, swap_baseline_available=False))
    return rows

def amplitude_amplification_qudit_verification__build_scaling_proxy_rows(*, families: Sequence[str], backend: str, scaling_qubits: Sequence[int], path_lengths: Sequence[int], routing_gate_time: float=0.0, target_gate_time: float=0.0, local_gate_time: float=0.0, leakage_epsilon: float=0.0, fpaa_lengths: Sequence[int]=(), fpaa_delta: float=0.2, oaa_rounds: int=3, dqaa_partition_bits: Sequence[int]=(), cqaa_rounds: int=3, foaa_lengths: Sequence[int]=(), vtaa_branch_times: Sequence[float]=(), vtaa_branch_weights: Sequence[float]=(), vtaa_branch_successes: Sequence[float]=(), qsvt_degrees: Sequence[int]=()) -> list[dict[str, object]]:
    scaling_rows: list[dict[str, object]] = []
    if 'grover' in families:
        scaling_rows.extend(amplitude_amplification_qudit_verification__build_grover_scaling_proxy_rows(backend=backend, scaling_qubits=scaling_qubits, path_lengths=path_lengths, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon))
    non_grover_families = [family for family in families if family != 'grover']
    for n_qubits in scaling_qubits:
        controls = max(0, int(n_qubits) - 1)
        scaled_path_lengths = amplitude_amplification_qudit_verification__select_local_path_lengths(path_lengths, controls)
        required_dimension = int(amplitude_amplification_qudit_verification__theoretical_layout_metrics(int(n_qubits), scaled_path_lengths)['minimum_parallel_dimension'])
        valid_partition_bits = [int(value) for value in dqaa_partition_bits if 0 < int(value) < int(n_qubits)]
        scaling_rows.extend(amplitude_amplification_qudit_verification__build_proxy_rows(families=non_grover_families, backend=backend, n_qubits=int(n_qubits), path_lengths=scaled_path_lengths, configured_dimension=required_dimension, max_iterations=2 * amplitude_amplification_qudit_verification__optimal_iteration(int(n_qubits)) + 2, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon, fpaa_lengths=fpaa_lengths, fpaa_delta=fpaa_delta, oaa_rounds=oaa_rounds, dqaa_partition_bits=valid_partition_bits, cqaa_rounds=cqaa_rounds, foaa_lengths=foaa_lengths, vtaa_branch_times=vtaa_branch_times, vtaa_branch_weights=vtaa_branch_weights, vtaa_branch_successes=vtaa_branch_successes, qsvt_degrees=qsvt_degrees))
    return amplitude_amplification_qudit_verification__annotate_rows(scaling_rows)

def amplitude_amplification_qudit_verification__make_scaling_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    families = [family for family in ['grover', 'dqaa'] if any((str(row.get('family')) == family or str(row.get('canonical_family')) == family for row in rows))]
    if not families:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {'grover': '-', 'dqaa': '--'}
    colors = {'swap_baseline': 'tab:gray', 'routed_serialized': 'tab:orange', 'routed_parallel': 'tab:blue'}
    for family in families:
        serialized_lookup = {int(row.get('logical_qubits') or row.get('family_parameter') or 0): row for row in amplitude_amplification_qudit_verification__representative_rows(rows, family=family, mode='routed_serialized')}
        parallel_lookup = {int(row.get('logical_qubits') or row.get('family_parameter') or 0): row for row in amplitude_amplification_qudit_verification__representative_rows(rows, family=family, mode='routed_parallel')}
        for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
            mode_rows = sorted(amplitude_amplification_qudit_verification__representative_rows(rows, family=family, mode=mode), key=lambda row: amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits') or row.get('family_parameter'), default=0))
            if not mode_rows:
                continue
            axes[0].plot([amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits') or row.get('family_parameter'), default=0) for row in mode_rows], [amplitude_amplification_qudit_verification__depth_metric(row) for row in mode_rows], marker='o', color=colors[mode], linestyle=styles[family], label=f'{family} {MODE_LABELS[mode]}')
        gap_points: list[tuple[int, float]] = []
        for n_qubits, serialized_row in serialized_lookup.items():
            parallel_row = parallel_lookup.get(n_qubits)
            if parallel_row is None:
                continue
            serialized_depth = max(amplitude_amplification_qudit_verification__depth_metric(serialized_row), 1.0)
            parallel_depth = amplitude_amplification_qudit_verification__depth_metric(parallel_row)
            gap_points.append((n_qubits, float((serialized_depth - parallel_depth) / serialized_depth)))
        if gap_points:
            gap_points.sort()
            axes[1].plot([point[0] for point in gap_points], [point[1] for point in gap_points], marker='o', linestyle=styles[family], label=family)
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize='small', ncol=2)
    axes[1].set_xlabel('Logical qubits')
    axes[1].set_ylabel('(serialized - parallel) / serialized')
    axes[1].set_title('Parallel advantage gap')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__scaling_law_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    families = sorted({str(row.get('family') or row.get('canonical_family') or '') for row in rows if str(row.get('family') or row.get('canonical_family') or '')})
    summary_rows: list[dict[str, object]] = []
    for family in families:
        for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
            mode_rows = sorted(amplitude_amplification_qudit_verification__representative_rows(rows, family=family, mode=mode), key=lambda row: amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits') or row.get('family_parameter'), default=0))
            fit = amplitude_amplification_qudit_verification__fit_power_law([amplitude_amplification_qudit_verification__to_float(row.get('logical_qubits') or row.get('family_parameter'), default=0.0) for row in mode_rows], [amplitude_amplification_qudit_verification__depth_metric(row) for row in mode_rows])
            summary_rows.append({'family': family, 'mode': mode, 'mode_label': MODE_LABELS[mode], **fit})
    return summary_rows

def amplitude_amplification_qudit_verification__exact_state_limit(total_dimension: int, *, backend: str) -> bool:
    if backend == 'cirq':
        return total_dimension <= 262144
    return total_dimension <= 4096

def amplitude_amplification_qudit_verification__make_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    grover_rows = [row for row in rows if row.get('family') == 'grover']
    if plt is None or not grover_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    reference_rows = sorted(amplitude_amplification_qudit_verification__representative_rows(grover_rows, family='grover', mode='swap_baseline'), key=lambda row: int(row['iteration']))
    if reference_rows:
        axes[0].plot([int(row['iteration']) for row in reference_rows], [float(row['theory_success_probability']) for row in reference_rows], marker='o', label='closed-form')
    for mode in ('routed_serialized', 'routed_parallel'):
        mode_rows = sorted(amplitude_amplification_qudit_verification__representative_rows(grover_rows, family='grover', mode=mode), key=lambda row: int(row['iteration']))
        routed_pairs = [(int(row['iteration']), float(row['routed_success_probability'])) for row in mode_rows if row.get('routed_success_probability') is not None]
        if routed_pairs:
            axes[0].plot([item[0] for item in routed_pairs], [item[1] for item in routed_pairs], marker='s' if mode == 'routed_serialized' else '^', linestyle='--', label=MODE_LABELS[mode])
    axes[0].set_xlabel('Grover iterations')
    axes[0].set_ylabel('Success probability')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    for mode in ('swap_baseline', 'routed_serialized', 'routed_parallel'):
        mode_rows = sorted(amplitude_amplification_qudit_verification__representative_rows(grover_rows, family='grover', mode=mode), key=lambda row: int(row['iteration']))
        if not mode_rows:
            continue
        depth_values = []
        for row in mode_rows:
            if mode == 'swap_baseline':
                depth_values.append(float(row['swap_baseline_depth'] or row['estimated_wall_clock_depth'] or 0.0))
            else:
                depth_values.append(float(row['moment_count'] or row['estimated_wall_clock_depth'] or 0.0))
        axes[1].plot([int(row['iteration']) for row in mode_rows], depth_values, marker='o', label=MODE_LABELS[mode])
    axes[1].set_xlabel('Grover iterations')
    axes[1].set_ylabel('Depth / moments')
    axes[1].set_title('Routing mode depth')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_family_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    summary_rows: list[dict[str, object]] = []
    seen = set()
    for row in rows:
        label = f'{row['family']}:{row.get('mode')}' if row.get('family_parameter') in (None, '') else f'{row['family']}:{row['family_parameter_name']}={row['family_parameter']}:{row.get('mode')}'
        key = (row.get('family'), row.get('family_parameter_name'), row.get('family_parameter'), row.get('iteration'), row.get('mode'))
        if row.get('family') == 'grover':
            continue
        if key in seen:
            continue
        seen.add(key)
        summary_rows.append(dict(row, _label=label))
    for mode in ('swap_baseline', 'routed_serialized', 'routed_parallel'):
        grover_rows = [row for row in rows if row.get('family') == 'grover' and row.get('mode') == mode and (row.get('iteration') is not None)]
        if grover_rows:
            final_grover = max(grover_rows, key=lambda row: int(row['iteration']))
            summary_rows.insert(0, dict(final_grover, _label=f'grover:k={final_grover['iteration']}:{mode}'))
    if not summary_rows:
        return
    labels = [str(row['_label']) for row in summary_rows]
    routing = [float(row['estimated_wall_clock_depth'] or row['swap_baseline_depth'] or 0.0) for row in summary_rows]
    concurrency = [float(row['required_buses_max'] or row['parallel_lanes'] or 1.0) for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, routing, color='tab:blue')
    axes[0].set_ylabel('Estimated depth')
    axes[0].set_title(title)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, axis='y')
    axes[1].bar(labels, concurrency, color='tab:orange')
    axes[1].set_ylabel('Buses / concurrency')
    axes[1].set_title('Routing width')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, axis='y')
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def amplitude_amplification_qudit_verification__to_float(value: object, default: float=0.0) -> float:
    if value in (None, '', 'None'):
        return default
    return float(value)

def amplitude_amplification_qudit_verification__to_int(value: object, default: int=0) -> int:
    if value in (None, '', 'None'):
        return default
    return int(float(value))

def amplitude_amplification_qudit_verification__select_representative_row(rows: Sequence[dict[str, object]], *, family: str, mode: str) -> dict[str, object] | None:
    candidates = [row for row in rows if row.get('family') == family and row.get('mode') == mode]
    if not candidates:
        candidates = [row for row in rows if row.get('canonical_family') == family and row.get('mode') == mode]
    if not candidates:
        return None

    def key(row: dict[str, object]) -> tuple[float, float, float, float]:
        evaluation = str(row.get('evaluation_mode') or '')
        exact_rank = 2.0 if evaluation == 'exact' else 1.0 if 'reference' in evaluation else 0.0
        success = amplitude_amplification_qudit_verification__to_float(row.get('theory_success_probability'), default=0.0)
        negative_depth = -amplitude_amplification_qudit_verification__to_float(row.get('moment_count') or row.get('swap_baseline_depth') or row.get('estimated_wall_clock_depth'), default=0.0)
        concurrency = amplitude_amplification_qudit_verification__to_float(row.get('parallel_lanes') or row.get('required_buses_max'), default=1.0)
        return (exact_rank, success, negative_depth, concurrency)
    return max(candidates, key=key)

def amplitude_amplification_qudit_verification__select_qft_row(rows: Sequence[dict[str, str]], *, n_qubits: int, mode: str) -> dict[str, str] | None:
    candidates = [row for row in rows if amplitude_amplification_qudit_verification__to_int(row.get('n_qubits')) == n_qubits and row.get('mode') == mode]
    if not candidates:
        return None

    def key(row: dict[str, str]) -> tuple[float, float]:
        evaluation = str(row.get('evaluation_mode') or '')
        exact_rank = 2.0 if evaluation == 'exact' else 1.0 if 'reference' in evaluation else 0.0
        negative_depth = -amplitude_amplification_qudit_verification__to_float(row.get('moment_count') or row.get('swap_baseline_depth'), default=0.0)
        return (exact_rank, negative_depth)
    return max(candidates, key=key)

def amplitude_amplification_qudit_verification__make_core_comparison_plot(*, amplitude_rows: Sequence[dict[str, object]], qft_rows: Sequence[dict[str, str]], n_qubits: int, figure_path: Path, title: str) -> None:
    if plt is None or not amplitude_rows or (not qft_rows):
        return
    modes = ('swap_baseline', 'routed_serialized', 'routed_parallel')
    algorithms = ('QFT', 'Grover', 'DQAA')
    depth_lookup: dict[tuple[str, str], float] = {}
    concurrency_lookup: dict[str, tuple[float, float]] = {}
    for mode in modes:
        qft_row = amplitude_amplification_qudit_verification__select_qft_row(qft_rows, n_qubits=n_qubits, mode=mode)
        if qft_row is not None:
            depth_lookup['QFT', mode] = amplitude_amplification_qudit_verification__to_float(qft_row.get('swap_baseline_depth') if mode == 'swap_baseline' else qft_row.get('moment_count'), default=0.0)
            if mode == 'routed_parallel':
                concurrency_lookup['QFT'] = (amplitude_amplification_qudit_verification__to_float(qft_row.get('max_target_concurrency'), default=0.0), amplitude_amplification_qudit_verification__to_float(qft_row.get('minimum_parallel_dimension_log2'), default=0.0))
        grover_row = amplitude_amplification_qudit_verification__select_representative_row(amplitude_rows, family='grover', mode=mode)
        if grover_row is not None:
            depth_lookup['Grover', mode] = amplitude_amplification_qudit_verification__to_float(grover_row.get('swap_baseline_depth') if mode == 'swap_baseline' else grover_row.get('moment_count'), default=amplitude_amplification_qudit_verification__to_float(grover_row.get('estimated_wall_clock_depth'), default=0.0))
            if mode == 'routed_parallel':
                concurrency_lookup['Grover'] = (amplitude_amplification_qudit_verification__to_float(grover_row.get('fanin_target_concurrency'), default=0.0), amplitude_amplification_qudit_verification__to_float(grover_row.get('minimum_parallel_dimension_log2'), default=0.0))
        dqaa_row = amplitude_amplification_qudit_verification__select_representative_row(amplitude_rows, family='dqaa', mode=mode)
        if dqaa_row is not None:
            depth_lookup['DQAA', mode] = amplitude_amplification_qudit_verification__to_float(dqaa_row.get('swap_baseline_depth') if mode == 'swap_baseline' else dqaa_row.get('moment_count'), default=amplitude_amplification_qudit_verification__to_float(dqaa_row.get('estimated_wall_clock_depth'), default=0.0))
            if mode == 'routed_parallel':
                concurrency_lookup['DQAA'] = (amplitude_amplification_qudit_verification__to_float(dqaa_row.get('fanin_target_concurrency'), default=0.0), amplitude_amplification_qudit_verification__to_float(dqaa_row.get('minimum_parallel_dimension_log2'), default=0.0))
    if not depth_lookup:
        return
    x = np.arange(len(algorithms))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for offset_index, mode in enumerate(modes):
        offsets = (offset_index - 1) * width
        axes[0].bar(x + offsets, [depth_lookup.get((algorithm, mode), 0.0) for algorithm in algorithms], width=width, label=MODE_LABELS[mode])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms)
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].legend()
    concurrency = [concurrency_lookup.get(algorithm, (0.0, 0.0))[0] for algorithm in algorithms]
    dimension_bits = [concurrency_lookup.get(algorithm, (0.0, 0.0))[1] for algorithm in algorithms]
    axes[1].bar(x - width / 2.0, concurrency, width=width, label='Concurrency / fan-in')
    axes[1].bar(x + width / 2.0, dimension_bits, width=width, label='log2(d_min)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms)
    axes[1].set_ylabel('Concurrency / dimension bits')
    axes[1].set_title('Theory to experiment bridge')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__representative_qft_rows(qft_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str], dict[str, str]] = {}
    for row in qft_rows:
        n_qubits = amplitude_amplification_qudit_verification__to_int(row.get('n_qubits'))
        mode = str(row.get('mode') or '')
        if n_qubits <= 0 or mode not in MODE_LABELS:
            continue
        evaluation = str(row.get('evaluation_mode') or '')
        exact_rank = 2.0 if evaluation == 'exact' else 1.0 if 'reference' in evaluation else 0.0
        negative_depth = -amplitude_amplification_qudit_verification__to_float(row.get('depth_metric') or row.get('moment_count') or row.get('swap_baseline_depth') or row.get('estimated_wall_clock_depth'), default=0.0)
        concurrency = amplitude_amplification_qudit_verification__to_float(row.get('c_pred') or row.get('required_buses_max') or row.get('max_target_concurrency'), default=1.0)
        candidate_score = (exact_rank, concurrency, negative_depth)
        key = (n_qubits, mode)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = dict(row, _score=str(candidate_score))
            continue
        previous_score = (
            2.0 if str(previous.get('evaluation_mode') or '') == 'exact' else 1.0 if 'reference' in str(previous.get('evaluation_mode') or '') else 0.0,
            amplitude_amplification_qudit_verification__to_float(previous.get('c_pred') or previous.get('required_buses_max') or previous.get('max_target_concurrency'), default=1.0),
            -amplitude_amplification_qudit_verification__to_float(previous.get('depth_metric') or previous.get('moment_count') or previous.get('swap_baseline_depth') or previous.get('estimated_wall_clock_depth'), default=0.0),
        )
        if candidate_score > previous_score:
            grouped[key] = dict(row, _score=str(candidate_score))
    result: list[dict[str, str]] = []
    for row in grouped.values():
        row.pop('_score', None)
        result.append(row)
    return result

def amplitude_amplification_qudit_verification__build_unified_paper_rows(*, amplitude_rows: Sequence[dict[str, object]], qft_rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    unified_rows: list[dict[str, object]] = []
    for row in amplitude_amplification_qudit_verification__representative_qft_rows(qft_rows):
        mode = str(row.get('mode') or '')
        unified_rows.append({
            'algorithm': 'QFT',
            'algorithm_family': 'qft',
            'backend': row.get('backend'),
            'logical_qubits': amplitude_amplification_qudit_verification__to_int(row.get('n_qubits')),
            'mode': mode,
            'mode_label': MODE_LABELS.get(mode, mode),
            'depth_metric': amplitude_amplification_qudit_verification__to_float(row.get('depth_metric') or row.get('moment_count') or row.get('swap_baseline_depth') or row.get('estimated_wall_clock_depth'), default=0.0),
            'c_pred': amplitude_amplification_qudit_verification__to_float(row.get('c_pred') or row.get('required_buses_max') or row.get('max_target_concurrency'), default=1.0),
            'c_actual': amplitude_amplification_qudit_verification__to_float(row.get('c_actual') or row.get('max_parallel_edges_per_layer') or row.get('max_target_concurrency'), default=1.0),
            'prediction_efficiency': amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency') or 0.0, default=0.0),
            'limiting_factor': str(row.get('limiting_factor') or 'bus'),
        })
    for family in ('grover', 'dqaa'):
        for mode in ('swap_baseline', 'routed_serialized', 'routed_parallel'):
            row = amplitude_amplification_qudit_verification__select_representative_row(amplitude_rows, family=family, mode=mode)
            if row is None:
                continue
            unified_rows.append({
                'algorithm': 'Grover' if family == 'grover' else 'DQAA',
                'algorithm_family': family,
                'backend': row.get('backend'),
                'logical_qubits': amplitude_amplification_qudit_verification__to_int(row.get('logical_qubits')),
                'mode': mode,
                'mode_label': MODE_LABELS.get(mode, mode),
                'depth_metric': amplitude_amplification_qudit_verification__depth_metric(row),
                'c_pred': amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=1.0),
                'c_actual': amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=1.0),
                'prediction_efficiency': amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0),
                'limiting_factor': str(row.get('limiting_factor') or 'fanin'),
            })
    return unified_rows

def amplitude_amplification_qudit_verification__make_unified_paper_figure(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    modes = ('swap_baseline', 'routed_serialized', 'routed_parallel')
    mode_colors = {'swap_baseline': 'tab:gray', 'routed_serialized': 'tab:orange', 'routed_parallel': 'tab:blue'}
    algorithm_styles = {'QFT': '-', 'Grover': '--', 'DQAA': ':'}
    algorithm_markers = {'QFT': 'o', 'Grover': 's', 'DQAA': '^'}
    for algorithm in ('QFT', 'Grover', 'DQAA'):
        for mode in modes:
            mode_rows = sorted([row for row in rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) == mode], key=lambda row: int(row.get('logical_qubits') or 0))
            if not mode_rows:
                continue
            axes[0].plot([int(row.get('logical_qubits') or 0) for row in mode_rows], [amplitude_amplification_qudit_verification__to_float(row.get('depth_metric'), default=0.0) for row in mode_rows], color=mode_colors[mode], linestyle=algorithm_styles[algorithm], marker=algorithm_markers[algorithm], label=f'{algorithm} {MODE_LABELS[mode]}')
    for algorithm in ('QFT', 'Grover', 'DQAA'):
        algorithm_rows = [row for row in rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) != 'swap_baseline']
        if not algorithm_rows:
            continue
        axes[1].scatter([amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0) for row in algorithm_rows], [amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0) for row in algorithm_rows], label=algorithm, marker=algorithm_markers[algorithm], alpha=0.85)
    prediction_rows = [row for row in rows if str(row.get('mode')) != 'swap_baseline']
    max_axis = max([1.0] + [max(amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0), amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0)) for row in prediction_rows])
    axes[1].plot([0.0, max_axis], [0.0, max_axis], linestyle='--', color='black', linewidth=1.0)
    factor_to_y = {'fanin': 0.0, 'bus': 1.0, 'dimension': 2.0}
    for algorithm in ('QFT', 'Grover', 'DQAA'):
        dominance_rows = [row for row in rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) == 'routed_parallel']
        if not dominance_rows:
            continue
        axes[2].scatter([int(row.get('logical_qubits') or 0) for row in dominance_rows], [factor_to_y.get(str(row.get('limiting_factor') or ''), 2.0) for row in dominance_rows], label=algorithm, marker=algorithm_markers[algorithm], alpha=0.85)
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize='small', ncol=2)
    axes[1].set_xlabel('Predicted concurrency')
    axes[1].set_ylabel('Measured concurrency')
    axes[1].set_title('Prediction vs measurement')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[2].set_xlabel('Logical qubits')
    axes[2].set_yticks([0.0, 1.0, 2.0])
    axes[2].set_yticklabels(['fanin', 'bus', 'dimension'])
    axes[2].set_ylabel('Limiting factor')
    axes[2].set_title('Constraint dominance')
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__build_mechanism_ablation_rows(*, amplitude_rows: Sequence[dict[str, object]], qft_rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    qft_serialized = amplitude_amplification_qudit_verification__select_qft_row(qft_rows, n_qubits=amplitude_amplification_qudit_verification__to_int(next((row.get('logical_qubits') for row in amplitude_rows if row.get('logical_qubits') not in (None, '', 'None')), 0)), mode='routed_serialized')
    qft_parallel = amplitude_amplification_qudit_verification__select_qft_row(qft_rows, n_qubits=amplitude_amplification_qudit_verification__to_int(next((row.get('logical_qubits') for row in amplitude_rows if row.get('logical_qubits') not in (None, '', 'None')), 0)), mode='routed_parallel')
    dqaa_serialized = amplitude_amplification_qudit_verification__select_representative_row(amplitude_rows, family='dqaa', mode='routed_serialized')
    dqaa_parallel = amplitude_amplification_qudit_verification__select_representative_row(amplitude_rows, family='dqaa', mode='routed_parallel')
    if qft_serialized is None or qft_parallel is None or dqaa_serialized is None or dqaa_parallel is None:
        return []
    aa_ratio = amplitude_amplification_qudit_verification__depth_metric(dqaa_parallel) / max(amplitude_amplification_qudit_verification__depth_metric(dqaa_serialized), 1.0)
    qft_ratio = amplitude_amplification_qudit_verification__to_float(qft_parallel.get('depth_metric') or qft_parallel.get('moment_count'), default=0.0) / max(amplitude_amplification_qudit_verification__to_float(qft_serialized.get('depth_metric') or qft_serialized.get('moment_count'), default=1.0), 1.0)
    aa_concurrency = amplitude_amplification_qudit_verification__to_float(dqaa_parallel.get('c_actual'), default=1.0)
    qft_concurrency = amplitude_amplification_qudit_verification__to_float(qft_parallel.get('c_actual') or qft_parallel.get('max_parallel_edges_per_layer') or qft_parallel.get('max_target_concurrency'), default=1.0)
    return [
        {'configuration': 'both_off', 'config_label': 'Fan-in off / buses off', 'source_algorithm': 'serialized baseline', 'normalized_depth': 1.0, 'observed_concurrency': 1.0},
        {'configuration': 'fanin_only', 'config_label': 'Fan-in on / buses off', 'source_algorithm': 'DQAA', 'normalized_depth': float(aa_ratio), 'observed_concurrency': float(aa_concurrency)},
        {'configuration': 'bus_only', 'config_label': 'Fan-in off / buses on', 'source_algorithm': 'QFT', 'normalized_depth': float(qft_ratio), 'observed_concurrency': float(qft_concurrency)},
        {'configuration': 'both_on', 'config_label': 'Fan-in on / buses on', 'source_algorithm': 'AA+QFT composite', 'normalized_depth': float(np.mean([aa_ratio, qft_ratio])), 'observed_concurrency': float(np.mean([aa_concurrency, qft_concurrency]))},
    ]

def amplitude_amplification_qudit_verification__make_mechanism_ablation_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    labels = [str(row.get('config_label') or row.get('configuration') or '') for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(labels, [amplitude_amplification_qudit_verification__to_float(row.get('normalized_depth'), default=0.0) for row in rows], color=['tab:gray', 'tab:orange', 'tab:blue', 'tab:green'])
    axes[0].set_ylabel('Normalized depth')
    axes[0].set_title(title)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(alpha=0.3, axis='y')
    axes[1].bar(labels, [amplitude_amplification_qudit_verification__to_float(row.get('observed_concurrency'), default=0.0) for row in rows], color=['tab:gray', 'tab:orange', 'tab:blue', 'tab:green'])
    axes[1].set_ylabel('Observed concurrency')
    axes[1].set_title('Mechanism activation summary')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].grid(alpha=0.3, axis='y')
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_killer_unified_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    factor_colors = {'fanin': 'tab:orange', 'bus': 'tab:blue', 'dimension': 'tab:red'}
    algorithm_markers = {'QFT': 'o', 'Grover': 's', 'DQAA': '^'}
    for algorithm in ['QFT', 'Grover', 'DQAA']:
        algorithm_rows = [row for row in rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) != 'swap_baseline']
        for factor in ['fanin', 'bus', 'dimension']:
            factor_rows = [row for row in algorithm_rows if str(row.get('limiting_factor')) == factor]
            if not factor_rows:
                continue
            ax.scatter([amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0) for row in factor_rows], [amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0) for row in factor_rows], color=factor_colors[factor], marker=algorithm_markers[algorithm], alpha=0.85, label=f'{algorithm} / {factor}')
    max_axis = max([1.0] + [max(amplitude_amplification_qudit_verification__to_float(row.get('c_pred'), default=0.0), amplitude_amplification_qudit_verification__to_float(row.get('c_actual'), default=0.0)) for row in rows if str(row.get('mode')) != 'swap_baseline'])
    ax.plot([0.0, max_axis], [0.0, max_axis], linestyle='--', color='black', linewidth=1.0)
    ax.set_xlabel('Predicted concurrency C_pred')
    ax.set_ylabel('Measured concurrency C_actual')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small', ncol=2)
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__make_algorithm_constraint_distribution_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    algorithms = ['QFT', 'Grover', 'DQAA']
    factors = ['fanin', 'bus', 'dimension']
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(algorithms))
    width = 0.22
    for offset_index, factor in enumerate(factors):
        heights = []
        for algorithm in algorithms:
            algorithm_rows = [row for row in rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) == 'routed_parallel']
            count = sum((1 for row in algorithm_rows if str(row.get('limiting_factor')) == factor))
            heights.append(0.0 if not algorithm_rows else float(count / len(algorithm_rows)))
        ax.bar(x + (offset_index - 1) * width, heights, width=width, label=factor)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel('Fraction of routed-parallel runs')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def amplitude_amplification_qudit_verification__write_paper_claim_summary(*, amplitude_rows: Sequence[dict[str, object]], qft_rows: Sequence[dict[str, str]], unified_rows: Sequence[dict[str, object]], scaling_rows: Sequence[dict[str, object]] | None, result_dir: Path, backend_label: str) -> None:
    modeled_rows = [row for row in unified_rows if str(row.get('mode')) != 'swap_baseline']
    tight_fraction = 0.0 if not modeled_rows else float(sum((1 for row in modeled_rows if amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) >= 0.9)) / len(modeled_rows))
    dominance_lines = []
    for algorithm in ['QFT', 'Grover', 'DQAA']:
        algorithm_rows = [row for row in unified_rows if str(row.get('algorithm')) == algorithm and str(row.get('mode')) == 'routed_parallel']
        if not algorithm_rows:
            continue
        counts = {factor: sum((1 for row in algorithm_rows if str(row.get('limiting_factor')) == factor)) for factor in ['fanin', 'bus', 'dimension']}
        total = len(algorithm_rows)
        dominance_lines.append(f'{algorithm}: fanin={counts["fanin"]/total:.3f}, bus={counts["bus"]/total:.3f}, dimension={counts["dimension"]/total:.3f}')
    qft_mode_rows = {}
    for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
        candidates = [row for row in qft_rows if str(row.get('mode') or '') == mode]
        n_values = sorted({amplitude_amplification_qudit_verification__to_int(row.get('n_qubits')) for row in candidates})
        if n_values:
            fit = amplitude_amplification_qudit_verification__fit_power_law(n_values, [max(1.0, amplitude_amplification_qudit_verification__to_float(amplitude_amplification_qudit_verification__select_qft_row(qft_rows, n_qubits=n_value, mode=mode).get('depth_metric') if amplitude_amplification_qudit_verification__select_qft_row(qft_rows, n_qubits=n_value, mode=mode) is not None else 0.0, default=1.0)) for n_value in n_values])
            qft_mode_rows[mode] = fit
    aa_scaling = amplitude_amplification_qudit_verification__scaling_law_summary(list(scaling_rows or [])) if scaling_rows else []
    grover_serialized = next((row for row in aa_scaling if str(row.get('family')) == 'grover' and str(row.get('mode')) == 'routed_serialized'), None)
    grover_parallel = next((row for row in aa_scaling if str(row.get('family')) == 'grover' and str(row.get('mode')) == 'routed_parallel'), None)
    lines = [
        f'Paper claim summary ({backend_label})',
        f'Total modeled runs: {len(modeled_rows)}',
        f'Prediction tightness fraction (C_actual/C_pred >= 0.9): {tight_fraction:.3f}',
        'Dominant-constraint fractions by algorithm:',
        *dominance_lines,
        f'QFT scaling: swap={qft_mode_rows.get("swap_baseline", {}).get("big_o")}, serialized={qft_mode_rows.get("routed_serialized", {}).get("big_o")}, parallel={qft_mode_rows.get("routed_parallel", {}).get("big_o")}',
        f'AA scaling (Grover proxy): serialized={None if grover_serialized is None else grover_serialized.get("big_o")}, parallel={None if grover_parallel is None else grover_parallel.get("big_o")}',
    ]
    qft_noise_parallel = [row for row in qft_rows if str(row.get('backend') or '') == 'qutip' and str(row.get('mode') or '') == 'routed_parallel' and row.get('random_state_fidelity_mean') not in (None, '', 'None')]
    qft_noise_serialized = [row for row in qft_rows if str(row.get('backend') or '') == 'qutip' and str(row.get('mode') or '') == 'routed_serialized' and row.get('random_state_fidelity_mean') not in (None, '', 'None')]
    aa_noise_parallel = [row for row in amplitude_rows if str(row.get('backend') or '') == 'qutip' and str(row.get('mode') or '') == 'routed_parallel' and row.get('logical_state_fidelity') not in (None, '', 'None')]
    aa_noise_serialized = [row for row in amplitude_rows if str(row.get('backend') or '') == 'qutip' and str(row.get('mode') or '') == 'routed_serialized' and row.get('logical_state_fidelity') not in (None, '', 'None')]
    if qft_noise_parallel and qft_noise_serialized:
        lines.append(f'QFT noise punchline: parallel depth={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("depth_metric") or row.get("moment_count"), default=0.0) for row in qft_noise_parallel]):.3f} vs serialized depth={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("depth_metric") or row.get("moment_count"), default=0.0) for row in qft_noise_serialized]):.3f}; parallel fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("random_state_fidelity_mean"), default=0.0) for row in qft_noise_parallel]):.3f} vs serialized fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("random_state_fidelity_mean"), default=0.0) for row in qft_noise_serialized]):.3f}')
    if aa_noise_parallel and aa_noise_serialized:
        lines.append(f'AA noise punchline: parallel depth={np.mean([amplitude_amplification_qudit_verification__depth_metric(row) for row in aa_noise_parallel]):.3f} vs serialized depth={np.mean([amplitude_amplification_qudit_verification__depth_metric(row) for row in aa_noise_serialized]):.3f}; parallel fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("logical_state_fidelity"), default=0.0) for row in aa_noise_parallel]):.3f} vs serialized fidelity={np.mean([amplitude_amplification_qudit_verification__to_float(row.get("logical_state_fidelity"), default=0.0) for row in aa_noise_serialized]):.3f}')
    worst_rows = sorted([row for row in modeled_rows if amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=0.0) > 0.0], key=lambda row: amplitude_amplification_qudit_verification__to_float(row.get('prediction_efficiency'), default=1.0))
    if worst_rows:
        worst = worst_rows[0]
        lines.append(f'Worst modeled efficiency case: algorithm={worst.get("algorithm")}, mode={worst.get("mode")}, C_pred={amplitude_amplification_qudit_verification__to_float(worst.get("c_pred"), default=0.0):.3f}, C_actual={amplitude_amplification_qudit_verification__to_float(worst.get("c_actual"), default=0.0):.3f}, efficiency={amplitude_amplification_qudit_verification__to_float(worst.get("prediction_efficiency"), default=0.0):.3f}, limiting={worst.get("limiting_factor")}')
    (result_dir / 'paper_claim_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def amplitude_amplification_qudit_verification__write_unified_paper_artifacts(*, amplitude_rows: Sequence[dict[str, object]], qft_rows: Sequence[dict[str, str]], result_dir: Path, backend_label: str, scaling_rows: Sequence[dict[str, object]] | None=None) -> None:
    unified_rows = amplitude_amplification_qudit_verification__build_unified_paper_rows(amplitude_rows=amplitude_rows, qft_rows=qft_rows)
    if not unified_rows:
        return
    mechanism_rows = amplitude_amplification_qudit_verification__build_mechanism_ablation_rows(amplitude_rows=amplitude_rows, qft_rows=qft_rows)
    verification_common__write_csv(result_dir / 'paper_unified_concurrency.csv', unified_rows)
    verification_common__write_json(result_dir / 'paper_unified_concurrency.json', {'backend': backend_label, 'rows': unified_rows})
    amplitude_amplification_qudit_verification__make_unified_paper_figure(unified_rows, result_dir / 'paper_unified_figure.png', f'Unified concurrency theory validation ({backend_label})')
    amplitude_amplification_qudit_verification__make_killer_unified_plot(unified_rows, result_dir / 'paper_unified_killer_figure.png', f'Unified concurrency killer figure ({backend_label})')
    amplitude_amplification_qudit_verification__make_algorithm_constraint_distribution_plot(unified_rows, result_dir / 'paper_constraint_distribution.png', f'Constraint distribution by algorithm ({backend_label})')
    if mechanism_rows:
        verification_common__write_csv(result_dir / 'paper_mechanism_ablation.csv', mechanism_rows)
        verification_common__write_json(result_dir / 'paper_mechanism_ablation.json', {'backend': backend_label, 'rows': mechanism_rows})
        amplitude_amplification_qudit_verification__make_mechanism_ablation_plot(mechanism_rows, result_dir / 'paper_mechanism_ablation.png', f'Mechanism ablation ({backend_label})')
    amplitude_amplification_qudit_verification__write_paper_claim_summary(amplitude_rows=amplitude_rows, qft_rows=qft_rows, unified_rows=unified_rows, scaling_rows=scaling_rows, result_dir=result_dir, backend_label=backend_label)

def amplitude_amplification_qudit_verification__add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--n-qubits', type=int, default=4, help='Logical Grover qubits.')
    parser.add_argument('--marked', default='0001', help='Marked bitstring in big-endian order.')
    parser.add_argument('--families', default='grover,dqaa', help='Paper-core default: grover,dqaa. Use all to expand to grover,dqaa,fpaa,oaa,cqaa,foaa,vtaa,qsvt. Non-Grover legacy families run through the exact canonical DQAA backbone.')
    parser.add_argument('--path-lengths', default='1', help='One value or one per control path.')
    parser.add_argument('--max-iterations', type=int, default=None, help='Defaults to 2*k_opt+2 for the single-marked Grover case.')
    parser.add_argument('--dimension', type=int, default=None, help='Local qudit dimension. Defaults to the theory minimum for the fan-in width.')
    parser.add_argument('--dimension-sweep', default='', help='Optional comma-separated dimension sweep, e.g. 2,4,8,16,32,64.')
    parser.add_argument('--fpaa-lengths', default='3,5,7', help='Odd FPAA schedule lengths.')
    parser.add_argument('--fpaa-delta', type=float, default=0.2)
    parser.add_argument('--oaa-rounds', type=int, default=3)
    parser.add_argument('--dqaa-partition-bits', default='1,2', help='Prefix partition bits for DQAA.')
    parser.add_argument('--cqaa-rounds', type=int, default=3)
    parser.add_argument('--foaa-lengths', default='3,5,7', help='Odd FOAA schedule lengths.')
    parser.add_argument('--vtaa-branch-times', default='1,2,4')
    parser.add_argument('--vtaa-branch-weights', default='0.5,0.3,0.2')
    parser.add_argument('--vtaa-branch-successes', default='0.2,0.45,0.8')
    parser.add_argument('--qsvt-degrees', default='3,5,7,9', help='Odd QSVT phase-sequence degrees.')
    parser.add_argument('--optimization-level', type=int, default=2)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--seed-count', type=int, default=1, help='Repeat exact routed runs with deterministic seed offsets for mean/std error bars.')
    parser.add_argument('--scaling-qubits', default='6,10,16,24', help='Comma-separated AA scaling qubit sizes for proxy-law extraction, e.g. 6,10,16,24.')
    parser.add_argument('--workers', type=int, default=1, help='Worker processes for independent qudit configurations.')
    parser.add_argument('--native-threads-per-worker', type=int, default=None, help='Caps BLAS/OpenMP threads inside each worker. Defaults to 1 when workers > 1.')

def amplitude_amplification_qudit_verification__optimal_iteration(n_qubits: int) -> int:
    theta = math.asin(1.0 / math.sqrt(2 ** n_qubits))
    return int(max(0, math.floor(math.pi / (4.0 * theta) - 0.5)))

def amplitude_amplification_qudit_verification__parse_positive_int_values(spec: str, *, odd_only: bool, label: str) -> list[int]:
    values = [int(value) for value in verification_common__parse_int_list(spec)]
    if not values:
        raise ValueError(f'Provide at least one value for {label}.')
    if any((value <= 0 for value in values)):
        raise ValueError(f'All {label} values must be positive.')
    if odd_only and any((value % 2 == 0 for value in values)):
        raise ValueError(f'All {label} values must be odd.')
    return values

def amplitude_amplification_qudit_verification__build_cirq_circuit(*, n_qubits: int, marked_bits: Sequence[int], path_lengths: Sequence[int], dimension: int, iterations: int, mode: str):
    _init_qudit_cirq_verification()
    import cirq
    if mode not in {'routed_serialized', 'routed_parallel'}:
        raise ValueError(f'Unsupported amplitude mode: {mode}')
    controls = max(0, n_qubits - 1)
    required_dimension = qudit_cirq_verification__minimum_dimension_for_controls(controls)
    circuit_dimension = max(int(dimension), required_dimension)
    fanin_builder = qudit_cirq_verification__build_fanin_circuit_serialized if mode == 'routed_serialized' else qudit_cirq_verification__build_fanin_circuit
    if n_qubits == 1:
        qudits = cirq.LineQid.range(1, dimension=circuit_dimension)
        logical_sites = [0]
        layout = None
    else:
        layout = qudit_cirq_verification__build_layout(path_lengths)
        qudits = cirq.LineQid.range(layout.total_sites, dimension=circuit_dimension)
        logical_sites = list(layout.source_sites) + [layout.target_site]
    x_gate = qudit_cirq_verification__LogicalQuditGate(circuit_dimension, qudit_cirq_verification__x_matrix(), 'X')
    h_gate = qudit_cirq_verification__LogicalQuditGate(circuit_dimension, qudit_cirq_verification__h_matrix(), 'H')
    z_gate = qudit_cirq_verification__LogicalQuditGate(circuit_dimension, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128), 'Z')
    circuit = cirq.Circuit()
    for site in logical_sites:
        circuit.append(h_gate.on(qudits[site]))

    def apply_marked_controlled_phase() -> None:
        nonlocal circuit
        if n_qubits == 1:
            if marked_bits[0] == 0:
                circuit.append(x_gate.on(qudits[0]))
            circuit.append(z_gate.on(qudits[0]))
            if marked_bits[0] == 0:
                circuit.append(x_gate.on(qudits[0]))
            return
        source_sites = list(layout.source_sites)
        target_site = layout.target_site
        for site, bit in zip(source_sites, marked_bits[:-1]):
            if bit == 0:
                circuit.append(x_gate.on(qudits[site]))
        if marked_bits[-1] == 0:
            circuit.append(x_gate.on(qudits[target_site]))
        circuit.append(h_gate.on(qudits[target_site]))
        circuit += fanin_builder(qudits, layout, list(range(1, controls + 1)), qudit_cirq_verification__and_rule(controls), qudit_cirq_verification__x_matrix(), label='MCX')
        circuit.append(h_gate.on(qudits[target_site]))
        if marked_bits[-1] == 0:
            circuit.append(x_gate.on(qudits[target_site]))
        for site, bit in reversed(list(zip(source_sites, marked_bits[:-1]))):
            if bit == 0:
                circuit.append(x_gate.on(qudits[site]))

    def apply_diffusion() -> None:
        nonlocal circuit
        if n_qubits == 1:
            circuit.append(h_gate.on(qudits[0]))
            circuit.append(x_gate.on(qudits[0]))
            circuit.append(z_gate.on(qudits[0]))
            circuit.append(x_gate.on(qudits[0]))
            circuit.append(h_gate.on(qudits[0]))
            return
        for site in logical_sites:
            circuit.append(h_gate.on(qudits[site]))
            circuit.append(x_gate.on(qudits[site]))
        circuit.append(h_gate.on(qudits[layout.target_site]))
        circuit += fanin_builder(qudits, layout, list(range(1, controls + 1)), qudit_cirq_verification__and_rule(controls), qudit_cirq_verification__x_matrix(), label='MCX')
        circuit.append(h_gate.on(qudits[layout.target_site]))
        for site in logical_sites:
            circuit.append(x_gate.on(qudits[site]))
            circuit.append(h_gate.on(qudits[site]))
    for _ in range(iterations):
        apply_marked_controlled_phase()
        apply_diffusion()
    layout_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(n_qubits, path_lengths)
    total_sites = int(layout_metrics['total_sites'])
    total_dimension = circuit_dimension ** total_sites
    exact_mode = dimension >= required_dimension and amplitude_amplification_qudit_verification__exact_state_limit(total_dimension, backend='cirq')
    result = {'circuit': circuit, 'metrics': qudit_cirq_verification__circuit_metrics(circuit), 'layout_metrics': layout_metrics, 'required_dimension_min': required_dimension, 'dimension_condition_met': dimension >= required_dimension, 'configured_dimension': int(dimension), 'circuit_dimension': circuit_dimension, 'evaluation_mode': 'exact' if exact_mode else 'metrics_only_dimension_limited' if dimension < required_dimension else 'metrics_only_state_limited'}
    if exact_mode:
        initial = qudit_cirq_verification__embed_qubit_state(np.array([1.0], dtype=np.complex128), dimension=circuit_dimension, total_sites=total_sites, logical_sites=[])
        initial[0] = 1.0
        final_state = qudit_cirq_verification__simulate_circuit(circuit, qudits, initial)
        logical_state = qudit_cirq_verification__extract_clean_logical_state(final_state, dimension=circuit_dimension, total_sites=total_sites, logical_sites=logical_sites)
        ideal_state = qubit_reference__grover_statevector(n_qubits, marked_bits, iterations)
        logical_probability = float(np.linalg.norm(logical_state) ** 2)
        routed_success = float(np.abs(logical_state[qubit_reference__index_from_bits(marked_bits)]) ** 2)
        normalized_logical = logical_state / np.linalg.norm(logical_state) if np.linalg.norm(logical_state) > 0.0 else logical_state
        result.update({'logical_state_fidelity': verification_common__statevector_fidelity(normalized_logical, ideal_state) if np.linalg.norm(logical_state) > 0.0 else 0.0, 'logical_subspace_probability': logical_probability, 'routed_success_probability': routed_success, 'max_routing_population': qudit_cirq_verification__max_routing_population(final_state, total_sites=total_sites, dimension=circuit_dimension)})
    return result

def amplitude_amplification_qudit_verification__build_qutip_run(*, n_qubits: int, marked_bits: Sequence[int], path_lengths: Sequence[int], dimension: int, iterations: int, mode: str, routing_gate_time: float, target_gate_time: float, local_gate_time: float, t1_levels: Sequence[float], tphi_levels: Sequence[float], leakage_epsilon: float, monte_carlo_trajectories: int, seed: int):
    _init_qudit_qutip_verification()
    if mode not in {'routed_serialized', 'routed_parallel'}:
        raise ValueError(f'Unsupported amplitude mode: {mode}')
    controls = max(0, n_qubits - 1)
    required_dimension = qudit_qutip_verification__minimum_dimension_for_controls(controls)
    circuit_dimension = max(int(dimension), required_dimension)
    routed_fanin = qudit_qutip_verification__apply_routed_fanin_unitary_serialized if mode == 'routed_serialized' else qudit_qutip_verification__apply_routed_fanin_unitary
    layout_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(n_qubits, path_lengths)
    total_sites = int(layout_metrics['total_sites'])
    total_dimension = circuit_dimension ** total_sites
    exact_mode = dimension >= required_dimension and amplitude_amplification_qudit_verification__exact_state_limit(total_dimension, backend='qutip')
    if not exact_mode:
        return {'layout_metrics': layout_metrics, 'required_dimension_min': required_dimension, 'dimension_condition_met': dimension >= required_dimension, 'configured_dimension': int(dimension), 'circuit_dimension': circuit_dimension, 'evaluation_mode': 'metrics_only_dimension_limited' if dimension < required_dimension else 'metrics_only_state_limited'}
    if n_qubits == 1:
        layout = None
        logical_sites = [0]
    else:
        layout = qudit_qutip_verification__build_layout(path_lengths)
        logical_sites = list(layout.source_sites) + [layout.target_site]
    dims = [circuit_dimension] * total_sites
    h0 = qudit_qutip_verification__zero_hamiltonian(dims)
    qudit_qutip_verification__configure_monte_carlo(seed, monte_carlo_trajectories)
    collapse_ops = qudit_qutip_verification__build_relaxation_and_dephasing_ops(dims, t1_levels, tphi_levels)
    leakage_kraus = qudit_qutip_verification__build_leakage_kraus(circuit_dimension, leakage_epsilon)
    computational_zero = np.zeros(2 ** n_qubits, dtype=np.complex128)
    computational_zero[0] = 1.0
    initial = qudit_qutip_verification__embed_qubit_state_as_ket(computational_zero, dimension=circuit_dimension, total_sites=total_sites, logical_sites=logical_sites)
    state = initial
    for site in logical_sites:
        state = qudit_qutip_verification__apply_logical_unitary(state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)

    def apply_marked_controlled_phase(current_state):
        if n_qubits == 1:
            if marked_bits[0] == 0:
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=0, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=0, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__z_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
            if marked_bits[0] == 0:
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=0, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
            return current_state
        for site, bit in zip(layout.source_sites, marked_bits[:-1]):
            if bit == 0:
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        if marked_bits[-1] == 0:
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        current_state = routed_fanin(current_state, layout=layout, dimension=circuit_dimension, dims=dims, bus_indices=list(range(1, controls + 1)), boolean_rule=qudit_qutip_verification__and_rule(controls), unitary=qudit_qutip_verification__x_matrix(), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, h0=h0, collapse_ops=collapse_ops, leakage_kraus=leakage_kraus)
        current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        if marked_bits[-1] == 0:
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        for site, bit in reversed(list(zip(layout.source_sites, marked_bits[:-1]))):
            if bit == 0:
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        return current_state

    def apply_diffusion(current_state):
        if n_qubits == 1:
            for unitary in (qudit_qutip_verification__h_matrix(), qudit_qutip_verification__x_matrix(), qudit_qutip_verification__z_matrix(), qudit_qutip_verification__x_matrix(), qudit_qutip_verification__h_matrix()):
                current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=0, dimension=circuit_dimension, dims=dims, unitary=unitary, gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
            return current_state
        for site in logical_sites:
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        current_state = routed_fanin(current_state, layout=layout, dimension=circuit_dimension, dims=dims, bus_indices=list(range(1, controls + 1)), boolean_rule=qudit_qutip_verification__and_rule(controls), unitary=qudit_qutip_verification__x_matrix(), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, h0=h0, collapse_ops=collapse_ops, leakage_kraus=leakage_kraus)
        current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=layout.target_site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        for site in logical_sites:
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__x_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
            current_state = qudit_qutip_verification__apply_logical_unitary(current_state, site=site, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        return current_state
    for _ in range(iterations):
        state = apply_marked_controlled_phase(state)
        state = apply_diffusion(state)
    ideal = qudit_qutip_verification__embed_qubit_state_as_ket(qubit_reference__grover_statevector(n_qubits, marked_bits, iterations), dimension=circuit_dimension, total_sites=total_sites, logical_sites=logical_sites)
    marked_basis = np.zeros(2 ** n_qubits, dtype=np.complex128)
    marked_basis[qubit_reference__index_from_bits(marked_bits)] = 1.0
    marked_ket = qudit_qutip_verification__embed_qubit_state_as_ket(marked_basis, dimension=circuit_dimension, total_sites=total_sites, logical_sites=logical_sites)
    return {'layout_metrics': layout_metrics, 'evaluation_mode': 'exact', 'required_dimension_min': required_dimension, 'dimension_condition_met': dimension >= required_dimension, 'configured_dimension': int(dimension), 'circuit_dimension': circuit_dimension, 'logical_state_fidelity': qudit_qutip_verification__pure_state_overlap(state, ideal), 'logical_subspace_probability': qudit_qutip_verification__logical_subspace_population(state, dimension=circuit_dimension, total_sites=total_sites, logical_sites=logical_sites), 'routed_success_probability': qudit_qutip_verification__pure_state_overlap(state, marked_ket), 'max_routing_population': max((qudit_qutip_verification__routing_population(state, site, circuit_dimension, dims) for site in range(total_sites)))}

def amplitude_amplification_qudit_verification__evaluate_cirq_exact_task(task: dict[str, object]) -> list[dict[str, object]]:
    task_type = str(task['task_type'])
    run_seed = int(task['run_seed'])
    configured_dimension = int(task['configured_dimension'])
    iteration = int(task['iteration'])
    global_n = int(task['global_n'])
    marked_bits = tuple((int(bit) for bit in task['marked_bits']))
    path_lengths = [int(length) for length in task['path_lengths']]
    optimization_level = int(task['optimization_level'])
    rows: list[dict[str, object]] = []
    if task_type == 'grover':
        routing_rounds = 2 * iteration
        layout_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(global_n, path_lengths)
        baseline = amplitude_amplification_qudit_verification__baseline_swap_metrics(global_n, marked_bits, iteration, seed=run_seed + iteration, optimization_level=optimization_level)
        rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family='grover', family_label=FAMILY_LABELS['grover'], canonical_family='grover', canonical_family_label=FAMILY_LABELS['grover'], mode='swap_baseline', mode_label=MODE_LABELS['swap_baseline'], evidence_tier='qubit_reference', family_mode='swap_baseline_reference', family_parameter_name='iteration', family_parameter=iteration, family_note='Nearest-neighbor qubit SWAP baseline for the same Grover iteration count.', iteration=iteration, backend='cirq', evaluation_mode='reference_metrics_only' if baseline is not None else 'reference_unavailable', dimension=None, theory_success_probability=qubit_reference__grover_success_formula(global_n, 1, iteration), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, parallel_lanes=1, estimated_wall_clock_depth=None if baseline is None else float(baseline['transpiled_depth']), estimated_swap_count=None if baseline is None else float(baseline['swap_count']), swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **layout_metrics))
        for mode in ('routed_serialized', 'routed_parallel'):
            result = amplitude_amplification_qudit_verification__build_cirq_circuit(n_qubits=global_n, marked_bits=marked_bits, path_lengths=path_lengths, dimension=configured_dimension, iterations=iteration, mode=mode)
            circuit_stats = result['metrics']
            mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=result['layout_metrics'], representative_rounds=routing_rounds, mode=mode, parallel_lanes=1)
            rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family='grover', family_label=FAMILY_LABELS['grover'], canonical_family='grover', canonical_family_label=FAMILY_LABELS['grover'], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='exact_routed', family_mode='exact_routed', family_parameter_name='iteration', family_parameter=iteration, family_note='Exact routed Grover / basic amplitude amplification.', iteration=iteration, backend='cirq', evaluation_mode=result['evaluation_mode'], dimension=configured_dimension, required_dimension_min=result['required_dimension_min'], dimension_condition_met=result['dimension_condition_met'], theory_success_probability=qubit_reference__grover_success_formula(global_n, 1, iteration), routed_success_probability=result.get('routed_success_probability'), logical_state_fidelity=result.get('logical_state_fidelity'), logical_subspace_probability=result.get('logical_subspace_probability'), max_routing_population=result.get('max_routing_population'), moment_count=circuit_stats.moment_count, operation_count=circuit_stats.operation_count, one_qudit_gate_count=circuit_stats.one_qudit_gate_count, two_qudit_gate_count=circuit_stats.two_qudit_gate_count, max_gate_width=circuit_stats.max_gate_width, representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, parallel_lanes=1, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **result['layout_metrics']))
        return rows
    partition_bits = int(task['partition_bits'])
    family_name = str(task['family_name'])
    dqaa_problem = amplitude_amplification_qudit_verification__dqaa_local_problem(global_n=global_n, partition_bits=partition_bits, marked_bits=marked_bits, path_lengths=path_lengths)
    local_n = int(dqaa_problem['local_logical_qubits'])
    local_marked_bits = tuple((int(bit) for bit in dqaa_problem['local_marked_bits']))
    local_path_lengths = list(dqaa_problem['local_path_lengths'])
    routing_rounds = 2 * iteration
    baseline = amplitude_amplification_qudit_verification__baseline_swap_metrics(local_n, local_marked_bits, iteration, seed=run_seed + 100 * partition_bits + iteration, optimization_level=optimization_level)
    baseline_family_mode = 'dqaa_swap_baseline' if family_name == 'dqaa' else 'canonicalized_dqaa_swap_baseline'
    baseline_family_note = 'Canonical DQAA node-local baseline.' if family_name == 'dqaa' else amplitude_amplification_qudit_verification__canonical_family_note(family_name)
    rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family=family_name, family_label=FAMILY_LABELS[family_name], canonical_family='dqaa', canonical_family_label=FAMILY_LABELS['dqaa'], mode='swap_baseline', mode_label=MODE_LABELS['swap_baseline'], evidence_tier='qubit_reference', family_mode=baseline_family_mode, family_parameter_name='partition_bits', family_parameter=partition_bits, family_note=baseline_family_note, iteration=iteration, backend='cirq', evaluation_mode='reference_metrics_only' if baseline is not None else 'reference_unavailable', dimension=None, theory_success_probability=qubit_reference__grover_success_formula(local_n, 1, iteration), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, estimated_wall_clock_depth=None if baseline is None else float(baseline['transpiled_depth']), estimated_swap_count=None if baseline is None else float(baseline['swap_count']), swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **amplitude_amplification_qudit_verification__rewrite_dqaa_metrics(amplitude_amplification_qudit_verification__theoretical_layout_metrics(local_n, local_path_lengths), global_n=global_n, partition_bits=partition_bits)))
    for mode in ('routed_serialized', 'routed_parallel'):
        result = amplitude_amplification_qudit_verification__build_cirq_circuit(n_qubits=local_n, marked_bits=local_marked_bits, path_lengths=local_path_lengths, dimension=configured_dimension, iterations=iteration, mode=mode)
        circuit_stats = result['metrics']
        mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=result['layout_metrics'], representative_rounds=routing_rounds, mode=mode, parallel_lanes=1)
        rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family=family_name, family_label=FAMILY_LABELS[family_name], canonical_family='dqaa', canonical_family_label=FAMILY_LABELS['dqaa'], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='exact_routed', family_mode='exact_dqaa_routed' if family_name == 'dqaa' else 'exact_dqaa_canonicalized', family_parameter_name='partition_bits', family_parameter=partition_bits, family_note=amplitude_amplification_qudit_verification__canonical_family_note(family_name), iteration=iteration, backend='cirq', evaluation_mode=result['evaluation_mode'], dimension=configured_dimension, required_dimension_min=result['required_dimension_min'], dimension_condition_met=result['dimension_condition_met'], theory_success_probability=qubit_reference__grover_success_formula(local_n, 1, iteration), routed_success_probability=result.get('routed_success_probability'), logical_state_fidelity=result.get('logical_state_fidelity'), logical_subspace_probability=result.get('logical_subspace_probability'), max_routing_population=result.get('max_routing_population'), moment_count=circuit_stats.moment_count, operation_count=circuit_stats.operation_count, one_qudit_gate_count=circuit_stats.one_qudit_gate_count, two_qudit_gate_count=circuit_stats.two_qudit_gate_count, max_gate_width=circuit_stats.max_gate_width, representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **amplitude_amplification_qudit_verification__rewrite_dqaa_metrics(result['layout_metrics'], global_n=global_n, partition_bits=partition_bits)))
    return rows

def amplitude_amplification_qudit_verification__evaluate_qutip_exact_task(task: dict[str, object]) -> list[dict[str, object]]:
    task_type = str(task['task_type'])
    run_seed = int(task['run_seed'])
    configured_dimension = int(task['configured_dimension'])
    iteration = int(task['iteration'])
    global_n = int(task['global_n'])
    marked_bits = tuple((int(bit) for bit in task['marked_bits']))
    path_lengths = [int(length) for length in task['path_lengths']]
    optimization_level = int(task['optimization_level'])
    routing_gate_time = float(task['routing_gate_time'])
    target_gate_time = float(task['target_gate_time'])
    local_gate_time = float(task['local_gate_time'])
    t1_levels = [float(value) for value in task['t1_levels']]
    tphi_levels = [float(value) for value in task['tphi_levels']]
    leakage_epsilon = float(task['leakage_epsilon'])
    monte_carlo_trajectories = int(task['monte_carlo_trajectories'])
    rows: list[dict[str, object]] = []
    if task_type == 'grover':
        routing_rounds = 2 * iteration
        layout_metrics = amplitude_amplification_qudit_verification__theoretical_layout_metrics(global_n, path_lengths)
        baseline = amplitude_amplification_qudit_verification__baseline_swap_metrics(global_n, marked_bits, iteration, seed=run_seed + iteration, optimization_level=optimization_level)
        rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family='grover', family_label=FAMILY_LABELS['grover'], canonical_family='grover', canonical_family_label=FAMILY_LABELS['grover'], mode='swap_baseline', mode_label=MODE_LABELS['swap_baseline'], evidence_tier='qubit_reference', family_mode='swap_baseline_reference', family_parameter_name='iteration', family_parameter=iteration, family_note='Nearest-neighbor qubit SWAP baseline for the same Grover iteration count.', iteration=iteration, backend='qutip', evaluation_mode='reference_metrics_only' if baseline is not None else 'reference_unavailable', dimension=None, theory_success_probability=qubit_reference__grover_success_formula(global_n, 1, iteration), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, parallel_lanes=1, estimated_wall_clock_depth=None if baseline is None else float(baseline['transpiled_depth']), estimated_swap_count=None if baseline is None else float(baseline['swap_count']), swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **layout_metrics))
        for mode in ('routed_serialized', 'routed_parallel'):
            result = amplitude_amplification_qudit_verification__build_qutip_run(n_qubits=global_n, marked_bits=marked_bits, path_lengths=path_lengths, dimension=configured_dimension, iterations=iteration, mode=mode, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories, seed=run_seed + iteration)
            mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=result['layout_metrics'], representative_rounds=routing_rounds, mode=mode, parallel_lanes=1)
            rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family='grover', family_label=FAMILY_LABELS['grover'], canonical_family='grover', canonical_family_label=FAMILY_LABELS['grover'], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='noisy_routed_exact', family_mode='noisy_routed_exact', family_parameter_name='iteration', family_parameter=iteration, family_note='Exact routed Grover / basic amplitude amplification with QuTiP noise.', iteration=iteration, backend='qutip', evaluation_mode=result['evaluation_mode'], dimension=configured_dimension, required_dimension_min=result['required_dimension_min'], dimension_condition_met=result['dimension_condition_met'], theory_success_probability=qubit_reference__grover_success_formula(global_n, 1, iteration), routed_success_probability=result.get('routed_success_probability'), logical_state_fidelity=result.get('logical_state_fidelity'), logical_subspace_probability=result.get('logical_subspace_probability'), max_routing_population=result.get('max_routing_population'), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, parallel_lanes=1, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], estimated_total_gate_time=amplitude_amplification_qudit_verification__estimate_total_gate_time(routing_shift_ops=float(mode_metrics['estimated_total_routing_shifts'] or 0.0), target_applications=max(1, routing_rounds), local_gate_applications=global_n + routing_rounds * 4 * global_n, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon, swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **result['layout_metrics']))
        return rows
    partition_bits = int(task['partition_bits'])
    family_name = str(task['family_name'])
    dqaa_problem = amplitude_amplification_qudit_verification__dqaa_local_problem(global_n=global_n, partition_bits=partition_bits, marked_bits=marked_bits, path_lengths=path_lengths)
    local_n = int(dqaa_problem['local_logical_qubits'])
    local_marked_bits = tuple((int(bit) for bit in dqaa_problem['local_marked_bits']))
    local_path_lengths = list(dqaa_problem['local_path_lengths'])
    routing_rounds = 2 * iteration
    baseline = amplitude_amplification_qudit_verification__baseline_swap_metrics(local_n, local_marked_bits, iteration, seed=run_seed + 100 * partition_bits + iteration, optimization_level=optimization_level)
    baseline_family_mode = 'dqaa_swap_baseline' if family_name == 'dqaa' else 'canonicalized_dqaa_swap_baseline'
    baseline_family_note = 'Canonical DQAA node-local baseline.' if family_name == 'dqaa' else amplitude_amplification_qudit_verification__canonical_family_note(family_name)
    rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family=family_name, family_label=FAMILY_LABELS[family_name], canonical_family='dqaa', canonical_family_label=FAMILY_LABELS['dqaa'], mode='swap_baseline', mode_label=MODE_LABELS['swap_baseline'], evidence_tier='qubit_reference', family_mode=baseline_family_mode, family_parameter_name='partition_bits', family_parameter=partition_bits, family_note=baseline_family_note, iteration=iteration, backend='qutip', evaluation_mode='reference_metrics_only' if baseline is not None else 'reference_unavailable', dimension=None, theory_success_probability=qubit_reference__grover_success_formula(local_n, 1, iteration), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, estimated_wall_clock_depth=None if baseline is None else float(baseline['transpiled_depth']), estimated_swap_count=None if baseline is None else float(baseline['swap_count']), swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **amplitude_amplification_qudit_verification__rewrite_dqaa_metrics(amplitude_amplification_qudit_verification__theoretical_layout_metrics(local_n, local_path_lengths), global_n=global_n, partition_bits=partition_bits)))
    for mode in ('routed_serialized', 'routed_parallel'):
        result = amplitude_amplification_qudit_verification__build_qutip_run(n_qubits=local_n, marked_bits=local_marked_bits, path_lengths=local_path_lengths, dimension=configured_dimension, iterations=iteration, mode=mode, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories, seed=run_seed + 100 * partition_bits + iteration)
        mode_metrics = amplitude_amplification_qudit_verification__proxy_mode_metrics(layout_metrics=result['layout_metrics'], representative_rounds=routing_rounds, mode=mode, parallel_lanes=1)
        rows.append(amplitude_amplification_qudit_verification__amplitude_row(run_seed=run_seed, family=family_name, family_label=FAMILY_LABELS[family_name], canonical_family='dqaa', canonical_family_label=FAMILY_LABELS['dqaa'], mode=mode, mode_label=MODE_LABELS[mode], evidence_tier='noisy_routed_exact', family_mode='exact_dqaa_routed' if family_name == 'dqaa' else 'exact_dqaa_canonicalized', family_parameter_name='partition_bits', family_parameter=partition_bits, family_note=amplitude_amplification_qudit_verification__canonical_family_note(family_name), iteration=iteration, backend='qutip', evaluation_mode=result['evaluation_mode'], dimension=configured_dimension, required_dimension_min=result['required_dimension_min'], dimension_condition_met=result['dimension_condition_met'], theory_success_probability=qubit_reference__grover_success_formula(local_n, 1, iteration), routed_success_probability=result.get('routed_success_probability'), logical_state_fidelity=result.get('logical_state_fidelity'), logical_subspace_probability=result.get('logical_subspace_probability'), max_routing_population=result.get('max_routing_population'), representative_rounds=routing_rounds, oracle_calls=iteration, state_reflections=iteration, estimated_total_routing_shifts=mode_metrics['estimated_total_routing_shifts'], estimated_wall_clock_depth=mode_metrics['estimated_wall_clock_depth'], estimated_swap_count=mode_metrics['estimated_swap_count'], estimated_total_gate_time=amplitude_amplification_qudit_verification__estimate_total_gate_time(routing_shift_ops=float(mode_metrics['estimated_total_routing_shifts'] or 0.0), target_applications=max(1, routing_rounds), local_gate_applications=local_n + routing_rounds * 4 * local_n, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, leakage_epsilon=leakage_epsilon, swap_baseline_depth=None if baseline is None else baseline['transpiled_depth'], swap_baseline_size=None if baseline is None else baseline['transpiled_size'], swap_baseline_swap_count=None if baseline is None else baseline['swap_count'], swap_baseline_cx_count=None if baseline is None else baseline['cx_count'], swap_baseline_available=baseline is not None, **amplitude_amplification_qudit_verification__rewrite_dqaa_metrics(result['layout_metrics'], global_n=global_n, partition_bits=partition_bits)))
    return rows

def amplitude_amplification_qudit_verification__main_cirq(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='Cirq qudit Grover/amplitude-amplification verification.')
    amplitude_amplification_qudit_verification__add_shared_arguments(parser)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.n_qubits = verification_common__prompt_int('How many logical qubits do you want to test?', args.n_qubits)
    args.marked = amplitude_amplification_qudit_verification__resolve_marked_argument(args.marked, args.n_qubits)
    ctx = verification_common__setup_run_context(script_file)
    families = amplitude_amplification_qudit_verification__parse_family_list(args.families)
    marked_bits = amplitude_amplification_qudit_verification__parse_marked_bits(args.marked, args.n_qubits)
    controls = max(0, args.n_qubits - 1)
    path_lengths = amplitude_amplification_qudit_verification__parse_path_lengths(args.path_lengths, controls) if controls > 0 else []
    default_dimension = args.dimension or max(2, 2 ** args.n_qubits)
    dimension_values = amplitude_amplification_qudit_verification__parse_positive_int_values(args.dimension_sweep, odd_only=False, label='dimension sweep') if args.dimension_sweep.strip() else [int(default_dimension)]
    scaling_qubits = amplitude_amplification_qudit_verification__parse_positive_int_values(args.scaling_qubits, odd_only=False, label='scaling qubits')
    seed_values = amplitude_amplification_qudit_verification__build_seed_values(args.seed, args.seed_count)
    k_opt = amplitude_amplification_qudit_verification__optimal_iteration(args.n_qubits)
    max_iterations = args.max_iterations if args.max_iterations is not None else 2 * k_opt + 2
    fpaa_lengths = amplitude_amplification_qudit_verification__parse_positive_int_values(args.fpaa_lengths, odd_only=True, label='FPAA lengths')
    dqaa_partition_bits = amplitude_amplification_qudit_verification__parse_positive_int_values(args.dqaa_partition_bits, odd_only=False, label='DQAA partition bits') if args.n_qubits > 1 else []
    dqaa_partition_bits = [value for value in dqaa_partition_bits if value < args.n_qubits]
    foaa_lengths = amplitude_amplification_qudit_verification__parse_positive_int_values(args.foaa_lengths, odd_only=True, label='FOAA lengths')
    qsvt_degrees = amplitude_amplification_qudit_verification__parse_positive_int_values(args.qsvt_degrees, odd_only=True, label='QSVT degrees')
    vtaa_branch_times = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_times)
    vtaa_branch_weights = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_weights)
    vtaa_branch_successes = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_successes)
    dqaa_families = amplitude_amplification_qudit_verification__requested_dqaa_families(families)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks: list[dict[str, object]] = []
    for run_seed in seed_values:
        for configured_dimension in dimension_values:
            if 'grover' in families:
                for iteration in range(max_iterations + 1):
                    tasks.append({'task_type': 'grover', 'run_seed': int(run_seed), 'configured_dimension': int(configured_dimension), 'iteration': int(iteration), 'global_n': int(args.n_qubits), 'marked_bits': list(marked_bits), 'path_lengths': list(path_lengths), 'optimization_level': int(args.optimization_level)})
            if dqaa_families:
                for partition_bits in dqaa_partition_bits:
                    local_n = int(amplitude_amplification_qudit_verification__dqaa_local_problem(global_n=args.n_qubits, partition_bits=partition_bits, marked_bits=marked_bits, path_lengths=path_lengths)['local_logical_qubits'])
                    dqaa_max_iterations = min(max_iterations, 2 * amplitude_amplification_qudit_verification__optimal_iteration(local_n) + 2)
                    for family_name in dqaa_families:
                        for iteration in range(dqaa_max_iterations + 1):
                            tasks.append({'task_type': 'dqaa', 'family_name': family_name, 'partition_bits': int(partition_bits), 'run_seed': int(run_seed), 'configured_dimension': int(configured_dimension), 'iteration': int(iteration), 'global_n': int(args.n_qubits), 'marked_bits': list(marked_bits), 'path_lengths': list(path_lengths), 'optimization_level': int(args.optimization_level)})
    print(f'[aa_cirq] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(amplitude_amplification_qudit_verification__evaluate_cirq_exact_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[aa_cirq] parallel section complete rows={len(rows)}')
    rows = amplitude_amplification_qudit_verification__annotate_rows(rows)
    scaling_rows = amplitude_amplification_qudit_verification__build_scaling_proxy_rows(families=families, backend='cirq', scaling_qubits=scaling_qubits, path_lengths=path_lengths, fpaa_lengths=fpaa_lengths, fpaa_delta=args.fpaa_delta, oaa_rounds=args.oaa_rounds, dqaa_partition_bits=dqaa_partition_bits, cqaa_rounds=args.cqaa_rounds, foaa_lengths=foaa_lengths, vtaa_branch_times=vtaa_branch_times, vtaa_branch_weights=vtaa_branch_weights, vtaa_branch_successes=vtaa_branch_successes, qsvt_degrees=qsvt_degrees)
    verification_common__write_csv(ctx.result_dir / 'amplitude_amplification_cirq.csv', rows)
    amplitude_amplification_qudit_verification__make_plot(rows, ctx.result_dir / 'amplitude_amplification_cirq.png', 'Grover routing modes on qudits (Cirq)')
    amplitude_amplification_qudit_verification__make_family_plot(rows, ctx.result_dir / 'amplitude_amplification_cirq_family_summary.png', 'Grover and DQAA routing load (Cirq)')
    amplitude_amplification_qudit_verification__write_analysis_bundle(rows, ctx.result_dir, suite_stem='amplitude_amplification_cirq', title_prefix='Amplitude amplification (Cirq)', scaling_rows=scaling_rows)
    qft_rows = amplitude_amplification_qudit_verification__read_csv_rows(ctx.script_path.parent / '[RESULT]2_qft_cirq' / 'qft_cirq.csv')
    amplitude_amplification_qudit_verification__make_core_comparison_plot(amplitude_rows=rows, qft_rows=qft_rows, n_qubits=args.n_qubits, figure_path=ctx.result_dir / 'paper_core_comparison_cirq.png', title='Core PRX comparison at fixed n (Cirq)')
    amplitude_amplification_qudit_verification__write_unified_paper_artifacts(amplitude_rows=rows, qft_rows=qft_rows, result_dir=ctx.result_dir, backend_label='Cirq', scaling_rows=scaling_rows)
    verification_common__write_json(ctx.result_dir / 'amplitude_amplification_cirq.json', {'suite': 'amplitude_amplification_cirq', 'focus': 'grover_and_dqaa_core_validation', 'families': families, 'marked_bits': list(marked_bits), 'rows': rows})
    print(f'Saved results to: {ctx.result_dir}')

def amplitude_amplification_qudit_verification__main_qutip(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='QuTiP qudit Grover/amplitude-amplification verification.')
    amplitude_amplification_qudit_verification__add_shared_arguments(parser)
    parser.add_argument('--routing-gate-time', type=float, default=0.0)
    parser.add_argument('--target-gate-time', type=float, default=0.0)
    parser.add_argument('--local-gate-time', type=float, default=0.0)
    parser.add_argument('--t1-levels', default='inf')
    parser.add_argument('--tphi-levels', default='inf')
    parser.add_argument('--leakage-epsilon', type=float, default=0.0)
    parser.add_argument('--monte-carlo-trajectories', type=int, default=64)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.n_qubits = verification_common__prompt_int('How many logical qubits do you want to test?', args.n_qubits)
    args.marked = amplitude_amplification_qudit_verification__resolve_marked_argument(args.marked, args.n_qubits)
    ctx = verification_common__setup_run_context(script_file)
    families = amplitude_amplification_qudit_verification__parse_family_list(args.families)
    marked_bits = amplitude_amplification_qudit_verification__parse_marked_bits(args.marked, args.n_qubits)
    controls = max(0, args.n_qubits - 1)
    path_lengths = amplitude_amplification_qudit_verification__parse_path_lengths(args.path_lengths, controls) if controls > 0 else []
    default_dimension = args.dimension or max(2, 2 ** args.n_qubits)
    dimension_values = amplitude_amplification_qudit_verification__parse_positive_int_values(args.dimension_sweep, odd_only=False, label='dimension sweep') if args.dimension_sweep.strip() else [int(default_dimension)]
    scaling_qubits = amplitude_amplification_qudit_verification__parse_positive_int_values(args.scaling_qubits, odd_only=False, label='scaling qubits')
    seed_values = amplitude_amplification_qudit_verification__build_seed_values(args.seed, args.seed_count)
    k_opt = amplitude_amplification_qudit_verification__optimal_iteration(args.n_qubits)
    max_iterations = args.max_iterations if args.max_iterations is not None else 2 * k_opt + 2
    t1_levels = [math.inf] if args.t1_levels.strip().lower() == 'inf' else verification_common__parse_float_list(args.t1_levels)
    tphi_levels = [math.inf] if args.tphi_levels.strip().lower() == 'inf' else verification_common__parse_float_list(args.tphi_levels)
    fpaa_lengths = amplitude_amplification_qudit_verification__parse_positive_int_values(args.fpaa_lengths, odd_only=True, label='FPAA lengths')
    dqaa_partition_bits = amplitude_amplification_qudit_verification__parse_positive_int_values(args.dqaa_partition_bits, odd_only=False, label='DQAA partition bits') if args.n_qubits > 1 else []
    dqaa_partition_bits = [value for value in dqaa_partition_bits if value < args.n_qubits]
    foaa_lengths = amplitude_amplification_qudit_verification__parse_positive_int_values(args.foaa_lengths, odd_only=True, label='FOAA lengths')
    qsvt_degrees = amplitude_amplification_qudit_verification__parse_positive_int_values(args.qsvt_degrees, odd_only=True, label='QSVT degrees')
    vtaa_branch_times = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_times)
    vtaa_branch_weights = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_weights)
    vtaa_branch_successes = amplitude_amplification_qudit_verification__parse_float_csv(args.vtaa_branch_successes)
    dqaa_families = amplitude_amplification_qudit_verification__requested_dqaa_families(families)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks: list[dict[str, object]] = []
    for run_seed in seed_values:
        for configured_dimension in dimension_values:
            if 'grover' in families:
                for iteration in range(max_iterations + 1):
                    tasks.append({'task_type': 'grover', 'run_seed': int(run_seed), 'configured_dimension': int(configured_dimension), 'iteration': int(iteration), 'global_n': int(args.n_qubits), 'marked_bits': list(marked_bits), 'path_lengths': list(path_lengths), 'optimization_level': int(args.optimization_level), 'routing_gate_time': float(args.routing_gate_time), 'target_gate_time': float(args.target_gate_time), 'local_gate_time': float(args.local_gate_time), 't1_levels': [float(value) for value in t1_levels], 'tphi_levels': [float(value) for value in tphi_levels], 'leakage_epsilon': float(args.leakage_epsilon), 'monte_carlo_trajectories': int(args.monte_carlo_trajectories)})
            if dqaa_families:
                for partition_bits in dqaa_partition_bits:
                    local_n = int(amplitude_amplification_qudit_verification__dqaa_local_problem(global_n=args.n_qubits, partition_bits=partition_bits, marked_bits=marked_bits, path_lengths=path_lengths)['local_logical_qubits'])
                    dqaa_max_iterations = min(max_iterations, 2 * amplitude_amplification_qudit_verification__optimal_iteration(local_n) + 2)
                    for family_name in dqaa_families:
                        for iteration in range(dqaa_max_iterations + 1):
                            tasks.append({'task_type': 'dqaa', 'family_name': family_name, 'partition_bits': int(partition_bits), 'run_seed': int(run_seed), 'configured_dimension': int(configured_dimension), 'iteration': int(iteration), 'global_n': int(args.n_qubits), 'marked_bits': list(marked_bits), 'path_lengths': list(path_lengths), 'optimization_level': int(args.optimization_level), 'routing_gate_time': float(args.routing_gate_time), 'target_gate_time': float(args.target_gate_time), 'local_gate_time': float(args.local_gate_time), 't1_levels': [float(value) for value in t1_levels], 'tphi_levels': [float(value) for value in tphi_levels], 'leakage_epsilon': float(args.leakage_epsilon), 'monte_carlo_trajectories': int(args.monte_carlo_trajectories)})
    print(f'[aa_qutip] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(amplitude_amplification_qudit_verification__evaluate_qutip_exact_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[aa_qutip] parallel section complete rows={len(rows)}')
    rows = amplitude_amplification_qudit_verification__annotate_rows(rows)
    scaling_rows = amplitude_amplification_qudit_verification__build_scaling_proxy_rows(families=families, backend='qutip', scaling_qubits=scaling_qubits, path_lengths=path_lengths, routing_gate_time=args.routing_gate_time, target_gate_time=args.target_gate_time, local_gate_time=args.local_gate_time, leakage_epsilon=args.leakage_epsilon, fpaa_lengths=fpaa_lengths, fpaa_delta=args.fpaa_delta, oaa_rounds=args.oaa_rounds, dqaa_partition_bits=dqaa_partition_bits, cqaa_rounds=args.cqaa_rounds, foaa_lengths=foaa_lengths, vtaa_branch_times=vtaa_branch_times, vtaa_branch_weights=vtaa_branch_weights, vtaa_branch_successes=vtaa_branch_successes, qsvt_degrees=qsvt_degrees)
    verification_common__write_csv(ctx.result_dir / 'amplitude_amplification_qutip.csv', rows)
    amplitude_amplification_qudit_verification__make_plot(rows, ctx.result_dir / 'amplitude_amplification_qutip.png', 'Grover routing modes on qudits (QuTiP)')
    amplitude_amplification_qudit_verification__make_family_plot(rows, ctx.result_dir / 'amplitude_amplification_qutip_family_summary.png', 'Grover and DQAA routing load (QuTiP)')
    amplitude_amplification_qudit_verification__write_analysis_bundle(rows, ctx.result_dir, suite_stem='amplitude_amplification_qutip', title_prefix='Amplitude amplification (QuTiP)', scaling_rows=scaling_rows)
    qft_rows = amplitude_amplification_qudit_verification__read_csv_rows(ctx.script_path.parent / '[RESULT]2_qft_qutip' / 'qft_qutip.csv')
    amplitude_amplification_qudit_verification__make_core_comparison_plot(amplitude_rows=rows, qft_rows=qft_rows, n_qubits=args.n_qubits, figure_path=ctx.result_dir / 'paper_core_comparison_qutip.png', title='Core PRX comparison at fixed n (QuTiP)')
    amplitude_amplification_qudit_verification__write_unified_paper_artifacts(amplitude_rows=rows, qft_rows=qft_rows, result_dir=ctx.result_dir, backend_label='QuTiP', scaling_rows=scaling_rows)
    verification_common__write_json(ctx.result_dir / 'amplitude_amplification_qutip.json', {'suite': 'amplitude_amplification_qutip', 'focus': 'grover_and_dqaa_core_validation', 'families': families, 'marked_bits': list(marked_bits), 'rows': rows})
    print(f'Saved results to: {ctx.result_dir}')


# Inlined from: merged_suite_runner.py

import csv
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

@dataclass(frozen=True)
class merged_suite_runner__SuiteStep:
    name: str
    label: str
    script_name: str
    func: Callable[[str], None]
    argv: tuple[str, ...]

@contextmanager
def merged_suite_runner__patched_argv(argv: Sequence[str]):
    previous = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = previous

def merged_suite_runner__parse_include_list(spec: str, *, valid: Iterable[str]) -> list[str]:
    valid_order = [item.strip() for item in valid if item.strip()]
    valid_set = set(valid_order)
    entries = [item.strip() for item in spec.split(',') if item.strip()]
    if not entries or entries == ['all']:
        return list(valid_order)
    unknown = [item for item in entries if item not in valid_set]
    if unknown:
        raise ValueError(f'Unsupported include target(s): {', '.join(unknown)}. Valid choices: {', '.join(sorted(valid_set))}, all.')
    ordered: list[str] = []
    for entry in entries:
        if entry not in ordered:
            ordered.append(entry)
    return ordered

def merged_suite_runner__summary_fieldnames(rows: Sequence[dict[str, object]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ordered:
                ordered.append(key)
    return ordered

def merged_suite_runner__run_suite_steps(*, merged_script_file: str, suite_name: str, steps: Sequence[merged_suite_runner__SuiteStep], stop_on_error: bool) -> Path:
    ctx = verification_common__setup_run_context(merged_script_file)
    parent_dir = Path(merged_script_file).resolve().parent
    rows: list[dict[str, object]] = []
    started_at = time.time()
    for step in steps:
        script_path = parent_dir / step.script_name
        step_started_at = time.time()
        print(f'[{suite_name}] starting {step.name} -> {step.script_name}')
        try:
            with merged_suite_runner__patched_argv([str(script_path), *step.argv]):
                step.func(str(script_path))
            rows.append({'step': step.name, 'label': step.label, 'script': step.script_name, 'status': 'ok', 'duration_seconds': round(time.time() - step_started_at, 3), 'result_dir': str(parent_dir / f'[RESULT]{script_path.stem}'), 'argv': list(step.argv), 'error': None})
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            rows.append({'step': step.name, 'label': step.label, 'script': step.script_name, 'status': 'failed', 'duration_seconds': round(time.time() - step_started_at, 3), 'result_dir': str(parent_dir / f'[RESULT]{script_path.stem}'), 'argv': list(step.argv), 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'[{suite_name}] failed {step.name}: {exc}')
            if stop_on_error:
                break
    summary = {'suite': suite_name, 'merged_script': Path(merged_script_file).name, 'duration_seconds': round(time.time() - started_at, 3), 'steps': rows}
    verification_common__write_json(ctx.result_dir / f'{suite_name}_summary.json', summary)
    csv_path = ctx.result_dir / f'{suite_name}_summary.csv'
    if rows:
        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_suite_runner__summary_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
    print(f'[{suite_name}] merged summary saved to: {ctx.result_dir}')
    return ctx.result_dir


# Launcher

import argparse

main_cirq = amplitude_amplification_qudit_verification__main_cirq
main_qutip = amplitude_amplification_qudit_verification__main_qutip
main_aer = amplitude_amplification_verification__main_aer
main_theory = amplitude_amplification_verification__main_theory
SuiteStep = merged_suite_runner__SuiteStep
parse_include_list = merged_suite_runner__parse_include_list
run_suite_steps = merged_suite_runner__run_suite_steps


def parse_marked_index_list(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def resolve_marked_and_good_indices(*, n_qubits: int, marked: str | None, good_indices: str | None) -> tuple[str, str]:
    if marked is None and good_indices is None:
        marked = format(1, f"0{n_qubits}b")
        good_indices = "1"
    elif marked is not None and good_indices is None:
        marked = marked.strip().replace(" ", "")
        if len(marked) != n_qubits or any(bit not in "01" for bit in marked):
            raise ValueError(f"--marked must be a {n_qubits}-bit binary string.")
        good_indices = str(int(marked, 2))
    elif marked is None and good_indices is not None:
        indices = parse_marked_index_list(good_indices)
        if len(indices) != 1:
            raise ValueError(
                "Provide exactly one good index when --marked is omitted. "
                "The qudit AA path currently requires a single marked bitstring."
            )
        marked = format(indices[0], f"0{n_qubits}b")
        good_indices = ",".join(str(index) for index in indices)
    else:
        marked = str(marked).strip().replace(" ", "")
        if len(marked) != n_qubits or any(bit not in "01" for bit in marked):
            raise ValueError(f"--marked must be a {n_qubits}-bit binary string.")
        good_indices = ",".join(str(index) for index in parse_marked_index_list(str(good_indices)))
        if not good_indices:
            good_indices = str(int(marked, 2))
    return marked, good_indices


def add_optional_flag(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full amplitude-amplification validation stack in one command.")
    parser.add_argument("--include", default="all", help="Comma-separated backends: theory,aer_cpu,aer_gpu,cirq,qutip or all.")
    parser.add_argument("--n-qubits", type=int, default=4, help="Shared logical qubit count.")
    parser.add_argument("--marked", default=None, help="Single marked basis state in big-endian binary form for the qudit runs.")
    parser.add_argument("--good-indices", default=None, help="Marked basis indices for the qubit baseline runs. If omitted, it is derived from --marked.")
    parser.add_argument("--families", default="all", help="AA families for the qudit runs.")
    parser.add_argument("--path-lengths", default="1", help="One routing length or one per control path.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None, help="Configured local qudit dimension.")
    parser.add_argument("--dimension-sweep", default="", help="Optional comma-separated dimension sweep for qudit backends.")
    parser.add_argument("--dqaa-partition-bits", default="1,2")
    parser.add_argument("--topologies", default="alltoall,line,ring,grid")
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--seed-count", type=int, default=1, help="Repeat qudit runs with deterministic seed offsets for error bars.")
    parser.add_argument("--scaling-qubits", default="6,10,16,24", help="Proxy AA scaling qubit sizes for large-n law extraction.")
    parser.add_argument("--routing-gate-time", type=float, default=0.0)
    parser.add_argument("--target-gate-time", type=float, default=0.0)
    parser.add_argument("--local-gate-time", type=float, default=0.0)
    parser.add_argument("--t1-levels", default="inf")
    parser.add_argument("--tphi-levels", default="inf")
    parser.add_argument("--leakage-epsilon", type=float, default=0.0)
    parser.add_argument("--monte-carlo-trajectories", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for independent experiment points.")
    parser.add_argument("--native-threads-per-worker", type=int, default=None, help="Caps BLAS/OpenMP threads inside each worker. Defaults to 1 when workers > 1.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort after the first backend failure instead of continuing.")
    args = parser.parse_args()
    include = parse_include_list(args.include, valid=("theory", "aer_cpu", "aer_gpu", "cirq", "qutip"))
    marked, good_indices = resolve_marked_and_good_indices(n_qubits=args.n_qubits, marked=args.marked, good_indices=args.good_indices)
    theory_argv = ["--n-qubits", str(args.n_qubits), "--good-indices", good_indices, "--seed", str(args.seed), "--workers", str(args.workers)]
    if args.native_threads_per_worker is not None:
        theory_argv.extend(["--native-threads-per-worker", str(args.native_threads_per_worker)])
    add_optional_flag(theory_argv, "--max-iterations", args.max_iterations)
    aer_argv = ["--n-qubits", str(args.n_qubits), "--good-indices", good_indices, "--seed", str(args.seed), "--topologies", args.topologies, "--optimization-level", str(args.optimization_level), "--workers", str(args.workers)]
    if args.native_threads_per_worker is not None:
        aer_argv.extend(["--native-threads-per-worker", str(args.native_threads_per_worker)])
    add_optional_flag(aer_argv, "--max-iterations", args.max_iterations)
    qudit_common_argv = ["--n-qubits", str(args.n_qubits), "--marked", marked, "--families", args.families, "--path-lengths", args.path_lengths, "--dqaa-partition-bits", args.dqaa_partition_bits, "--seed", str(args.seed), "--seed-count", str(args.seed_count), "--scaling-qubits", args.scaling_qubits, "--optimization-level", str(args.optimization_level), "--workers", str(args.workers)]
    if args.native_threads_per_worker is not None:
        qudit_common_argv.extend(["--native-threads-per-worker", str(args.native_threads_per_worker)])
    add_optional_flag(qudit_common_argv, "--max-iterations", args.max_iterations)
    add_optional_flag(qudit_common_argv, "--dimension", args.dimension)
    if args.dimension_sweep.strip():
        qudit_common_argv.extend(["--dimension-sweep", args.dimension_sweep])
    qutip_argv = list(qudit_common_argv)
    qutip_argv.extend(["--routing-gate-time", str(args.routing_gate_time), "--target-gate-time", str(args.target_gate_time), "--local-gate-time", str(args.local_gate_time), "--t1-levels", args.t1_levels, "--tphi-levels", args.tphi_levels, "--leakage-epsilon", str(args.leakage_epsilon), "--monte-carlo-trajectories", str(args.monte_carlo_trajectories)])
    step_map = {
        "theory": SuiteStep(name="theory", label="Qubit theory baseline", script_name="1_amplitude_amplification_theory.py", func=main_theory, argv=tuple(theory_argv)),
        "aer_cpu": SuiteStep(name="aer_cpu", label="Qubit Aer CPU baseline", script_name="1_amplitude_amplification_aer_cpu.py", func=lambda script_file: main_aer(script_file, device="CPU"), argv=tuple(aer_argv)),
        "aer_gpu": SuiteStep(name="aer_gpu", label="Qubit Aer GPU baseline", script_name="1_amplitude_amplification_aer_gpu.py", func=lambda script_file: main_aer(script_file, device="GPU"), argv=tuple(aer_argv)),
        "cirq": SuiteStep(name="cirq", label="Ideal qudit routed validation", script_name="1_amplitude_amplification_cirq.py", func=main_cirq, argv=tuple(qudit_common_argv)),
        "qutip": SuiteStep(name="qutip", label="Noisy qudit routed validation", script_name="1_amplitude_amplification_qutip.py", func=main_qutip, argv=tuple(qutip_argv)),
    }
    ordered_steps = [step_map[name] for name in include]
    result_dir = run_suite_steps(merged_script_file=__file__, suite_name="amplitude_amplification_all", steps=ordered_steps, stop_on_error=args.stop_on_error)
    print("Merged amplitude run includes:")
    for step in ordered_steps:
        print(f"  - {step.name}: {step.label}")
    print(f"Merged amplitude summary: {result_dir}")


if __name__ == "__main__":
    main()
