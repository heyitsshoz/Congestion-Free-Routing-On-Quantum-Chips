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


# Inlined from: qft_verification.py

import argparse
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector

def qft_verification__build_qft_circuit(n_qubits: int, *, with_swaps: bool=False) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for target in range(n_qubits):
        for control in range(target + 1, n_qubits):
            qc.cp(np.pi / 2 ** (control - target), control, target)
        qc.h(target)
    if with_swaps:
        for index in range(n_qubits // 2):
            qc.swap(index, n_qubits - 1 - index)
    return qc

def qft_verification__add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--min-qubits', type=int, default=2)
    parser.add_argument('--max-qubits', type=int, default=7)
    parser.add_argument('--random-samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--workers', type=int, default=1, help='Worker processes for independent experiment points.')
    parser.add_argument('--native-threads-per-worker', type=int, default=None, help='Caps BLAS/OpenMP threads inside each worker. Defaults to 1 when workers > 1.')

def qft_verification__evaluate_theory_task(task: dict[str, object]) -> dict[str, object]:
    n_qubits = int(task['n_qubits'])
    sample_seeds = [int(seed) for seed in task.get('sample_seeds', ())]
    circuit = qft_verification__build_qft_circuit(n_qubits)
    inverse = circuit.inverse()
    stats = verification_aer__logical_circuit_stats(circuit)
    exact_mode = n_qubits <= 12
    fidelities: list[float] = []
    if exact_mode:
        for sample_seed in sample_seeds:
            state = random_statevector(2 ** n_qubits, seed=sample_seed)
            reconstructed = state.evolve(circuit).evolve(inverse)
            fidelities.append(verification_common__statevector_fidelity(state.data, reconstructed.data))
    return {'n_qubits': n_qubits, 'evaluation_mode': 'exact' if exact_mode else 'metrics_only', 'random_state_fidelity_mean': float(np.mean(fidelities)) if fidelities else None, 'random_state_fidelity_min': float(np.min(fidelities)) if fidelities else None, 'logical_depth': stats.logical_depth, 'logical_size': stats.logical_size, 'logical_two_qubit_count': stats.logical_two_qubit_count, 'interaction_span_total': stats.span_total, 'interaction_span_max': stats.span_max, 'interaction_span_mean': stats.span_mean, 'max_gate_width': stats.max_gate_width, 'max_fan_in': stats.max_fan_in, 'max_qubit_load': stats.max_qubit_load, 'weighted_qubit_load_max': stats.weighted_qubit_load_max, 'target_load_max': stats.target_load_max, 'multi_qubit_gate_count': stats.multi_qubit_gate_count}

def qft_verification__evaluate_aer_task(task: dict[str, object]) -> list[dict[str, object]]:
    n_qubits = int(task['n_qubits'])
    topologies = [str(topology) for topology in task['topologies']]
    device = str(task['device'])
    optimization_level = int(task['optimization_level'])
    seed = int(task['seed'])
    sample_seeds = [int(sample_seed) for sample_seed in task.get('sample_seeds', ())]
    logical = qft_verification__build_qft_circuit(n_qubits)
    logical_stats = verification_aer__logical_circuit_stats(logical)
    exact_mode = n_qubits <= 20
    fidelities_by_topology = {topology: [] for topology in topologies}
    transpiled_by_topology = {topology: verification_aer__transpile_with_metrics(logical, topology=topology, optimization_level=optimization_level, seed=seed) for topology in topologies}
    if exact_mode:
        for sample_seed in sample_seeds:
            random_input = random_statevector(2 ** n_qubits, seed=sample_seed)
            prep = QuantumCircuit(n_qubits)
            prep.initialize(random_input.data, range(n_qubits))
            reference_state = verification_aer__simulate_statevector(prep.compose(logical), device=device, seed=seed)
            for topology in topologies:
                transpiled_circuit, metrics = transpiled_by_topology[topology]
                if int(metrics['swap_count']) != 0:
                    continue
                test_state = verification_aer__simulate_statevector(prep.compose(transpiled_circuit), device=device, seed=seed)
                fidelities_by_topology[topology].append(verification_common__statevector_fidelity(reference_state, test_state))
    rows: list[dict[str, object]] = []
    for topology in topologies:
        _, metrics = transpiled_by_topology[topology]
        topology_fidelities = fidelities_by_topology[topology]
        rows.append({'n_qubits': n_qubits, 'topology': topology, 'evaluation_mode': 'exact' if exact_mode else 'metrics_only', 'random_state_fidelity_mean': float(np.mean(topology_fidelities)) if topology_fidelities else None, 'random_state_fidelity_min': float(np.min(topology_fidelities)) if topology_fidelities else None, 'logical_depth': logical_stats.logical_depth, 'logical_size': logical_stats.logical_size, 'logical_two_qubit_count': logical_stats.logical_two_qubit_count, 'interaction_span_total': logical_stats.span_total, 'interaction_span_max': logical_stats.span_max, 'interaction_span_mean': logical_stats.span_mean, 'max_gate_width': logical_stats.max_gate_width, 'max_fan_in': logical_stats.max_fan_in, 'max_qubit_load': logical_stats.max_qubit_load, 'weighted_qubit_load_max': logical_stats.weighted_qubit_load_max, 'target_load_max': logical_stats.target_load_max, 'multi_qubit_gate_count': logical_stats.multi_qubit_gate_count, **metrics})
    return rows

def qft_verification__make_theory_plot(rows: Sequence[dict[str, object]], figure_path: Path) -> None:
    if plt is None or not rows:
        return
    n_values = [int(row['n_qubits']) for row in rows]
    two_qubit_counts = [float(row['logical_two_qubit_count']) for row in rows]
    span_totals = [float(row['interaction_span_total']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(n_values, two_qubit_counts, marker='o')
    axes[0].set_xlabel('Qubits n')
    axes[0].set_ylabel('Logical two-qubit gates')
    axes[0].set_title('QFT entangling count')
    axes[0].grid(alpha=0.3)
    axes[1].plot(n_values, span_totals, marker='o', color='tab:red')
    axes[1].set_xlabel('Qubits n')
    axes[1].set_ylabel('Total interaction span')
    axes[1].set_title('Routing-pressure proxy')
    axes[1].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def qft_verification__make_aer_plot(rows: Sequence[dict[str, object]], figure_path: Path) -> None:
    if plt is None or not rows:
        return
    topologies = sorted({str(row['topology']) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for topology in topologies:
        topo_rows = [row for row in rows if str(row['topology']) == topology]
        axes[0].plot([int(row['n_qubits']) for row in topo_rows], [float(row['transpiled_depth']) for row in topo_rows], marker='o', label=topology)
        axes[1].plot([int(row['n_qubits']) for row in topo_rows], [float(row['swap_count']) for row in topo_rows], marker='o', label=topology)
    axes[0].set_xlabel('Qubits n')
    axes[0].set_ylabel('Transpiled depth')
    axes[0].set_title('QFT depth after routing')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('Qubits n')
    axes[1].set_ylabel('Inserted swaps')
    axes[1].set_title('SWAP overhead proxy')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_verification__main_theory(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='Theory-side QFT verification.')
    qft_verification__add_shared_arguments(parser)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.max_qubits = verification_common__prompt_int('Up to how many qubits do you want to test?', args.max_qubits)
    ctx = verification_common__setup_run_context(script_file)
    rng = np.random.default_rng(args.seed)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'n_qubits': n_qubits, 'sample_seeds': [int(rng.integers(0, 1000000)) for _ in range(args.random_samples)]} for n_qubits in range(args.min_qubits, args.max_qubits + 1)]
    print(f'[qft_theory] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = verification_common__parallel_map(qft_verification__evaluate_theory_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker)
    print(f'[qft_theory] parallel section complete rows={len(rows)}')
    verification_common__write_csv(ctx.result_dir / 'qft_theory.csv', rows)
    qft_verification__make_theory_plot(rows, ctx.result_dir / 'qft_theory.png')
    verification_common__write_json(ctx.result_dir / 'qft_theory.json', {'suite': 'qft_theory', 'focus': 'congestion_and_fan_in', 'rows': rows})
    print(f'Saved results to: {ctx.result_dir}')

def qft_verification__main_aer(script_file: str, *, device: str) -> None:
    parser = argparse.ArgumentParser(description=f'Aer-{device.lower()} QFT verification.')
    qft_verification__add_shared_arguments(parser)
    parser.add_argument('--topologies', default='alltoall,line,ring,grid', help='Comma-separated topologies: alltoall,line,ring,grid')
    parser.add_argument('--optimization-level', type=int, default=2)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.max_qubits = verification_common__prompt_int('Up to how many qubits do you want to test?', args.max_qubits)
    ctx = verification_common__setup_run_context(script_file)
    topologies = verification_aer__parse_topologies(args.topologies)
    rng = np.random.default_rng(args.seed)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'n_qubits': n_qubits, 'topologies': list(topologies), 'device': device, 'optimization_level': args.optimization_level, 'seed': args.seed, 'sample_seeds': [int(rng.integers(0, 1000000)) for _ in range(args.random_samples)]} for n_qubits in range(args.min_qubits, args.max_qubits + 1)]
    print(f'[qft_aer] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(qft_verification__evaluate_aer_task, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[qft_aer] parallel section complete rows={len(rows)}')
    stem = f'qft_aer_{device.lower()}'
    verification_common__write_csv(ctx.result_dir / f'{stem}.csv', rows)
    qft_verification__make_aer_plot(rows, ctx.result_dir / f'{stem}.png')
    verification_common__write_json(ctx.result_dir / f'{stem}.json', {'suite': stem, 'focus': 'congestion_and_fan_in', 'rows': rows})
    print(f'Saved results to: {ctx.result_dir}')


# Inlined from: qft_qudit_verification.py

import argparse
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
MODE_LABELS = {'swap_baseline': 'SWAP Baseline', 'routed_serialized': 'Routed Serialized', 'routed_parallel': 'Routed Parallel'}

def qft_qudit_verification__build_qiskit_qft_circuit(n_qubits: int) -> QuantumCircuit:
    if QuantumCircuit is None:
        raise RuntimeError('Qiskit is not available.')
    qc = QuantumCircuit(n_qubits)
    for target in range(n_qubits):
        for control in range(target + 1, n_qubits):
            qc.cp(np.pi / 2 ** (control - target), control, target)
        qc.h(target)
    return qc

def qft_qudit_verification__line_coupling_map(num_qubits: int) -> CouplingMap:
    if CouplingMap is None:
        raise RuntimeError('Qiskit CouplingMap is not available.')
    edges: list[tuple[int, int]] = []
    for left in range(num_qubits - 1):
        edges.append((left, left + 1))
        edges.append((left + 1, left))
    return CouplingMap(edges)

def qft_qudit_verification__baseline_swap_metrics(n_qubits: int, *, seed: int, optimization_level: int) -> dict[str, object] | None:
    if QuantumCircuit is None or transpile is None or CouplingMap is None:
        return None
    logical = qft_qudit_verification__build_qiskit_qft_circuit(n_qubits)
    transpiled = transpile(logical, coupling_map=qft_qudit_verification__line_coupling_map(n_qubits), routing_method='sabre', layout_method='sabre', optimization_level=optimization_level, seed_transpiler=seed, initial_layout=list(range(n_qubits)))
    counts = transpiled.count_ops()
    return {'transpiled_depth': int(transpiled.depth() or 0), 'transpiled_size': int(transpiled.size()), 'swap_count': int(counts.get('swap', 0)), 'cx_count': int(counts.get('cx', 0)), 'op_counts': {str(key): int(val) for key, val in counts.items()}}

def qft_qudit_verification__qft_structural_metrics(n_qubits: int) -> dict[str, object]:
    pair_spans = [control - target for target in range(n_qubits) for control in range(target + 1, n_qubits)]
    target_concurrency = [n_qubits - 1 - target for target in range(n_qubits)]
    edge_congestion = [boundary * (n_qubits - boundary) for boundary in range(1, n_qubits)]
    minimum_parallel_dimension = 2 ** n_qubits
    return {'logical_qubits': n_qubits, 'total_sites': n_qubits, 'controlled_phase_count': n_qubits * (n_qubits - 1) // 2, 'interaction_span_total': int(sum(pair_spans)), 'interaction_span_max': int(max(pair_spans, default=0)), 'edge_congestion_max': int(max(edge_congestion, default=0)), 'edge_congestion_mean': float(np.mean(edge_congestion)) if edge_congestion else 0.0, 'max_target_concurrency': int(max(target_concurrency, default=0)), 'required_buses_max': int(max(target_concurrency, default=0)), 'minimum_parallel_dimension': int(minimum_parallel_dimension), 'minimum_parallel_dimension_log2': float(math.log2(minimum_parallel_dimension)), 'serialized_interaction_rounds': int(n_qubits * (n_qubits - 1) // 2), 'parallel_target_rounds_total': int(sum(target_concurrency))}

def qft_qudit_verification__parallel_schedule_segment(target_site: int, control_sites: Sequence[int]) -> dict[str, object]:
    controls = list(control_sites)
    phase_angles = [math.pi / 2 ** (control - target_site) for control in controls]
    if not controls:
        return {'target_site': target_site, 'control_sites': [], 'bus_indices': [], 'phase_angles': [], 'layers': [], 'max_parallel_edges_per_layer': 0, 'max_parallel_buses_per_layer': 0}
    bus_indices = list(range(1, len(controls) + 1))
    source_to_bus = {source: bus for source, bus in zip(controls, bus_indices)}
    farthest_source = max(controls)
    transit: dict[int, list[int]] = {}
    injected_sources: set[int] = set()
    layers: list[list[dict[str, object]]] = []
    for layer_index in range(farthest_source - target_site):
        active_parity = (farthest_source - layer_index) % 2
        next_transit = {site: list(buses) for site, buses in transit.items()}
        layer_ops: list[dict[str, object]] = []
        for source in range(target_site + 1, farthest_source + 1):
            if source % 2 != active_parity:
                continue
            current_specs: list[tuple[int, str]] = []
            if source in transit:
                current_specs.extend(((bus, 'bcp') for bus in sorted(transit[source])))
                next_transit.pop(source, None)
            if source in source_to_bus and source not in injected_sources:
                current_specs.append((source_to_bus[source], 'cbl'))
                injected_sources.add(source)
            if not current_specs:
                continue
            destination = source - 1
            next_transit.setdefault(destination, [])
            next_transit[destination].extend((bus for bus, _ in current_specs))
            next_transit[destination] = sorted(set(next_transit[destination]))
            layer_ops.append({'source_site': source, 'target_site': destination, 'bus_specs': tuple(sorted(current_specs, key=lambda item: item[0]))})
        layers.append(layer_ops)
        transit = {site: buses for site, buses in next_transit.items() if buses}
    max_parallel_edges = max((len(layer) for layer in layers), default=0)
    max_parallel_buses = max((sum((len(op['bus_specs']) for op in layer)) for layer in layers), default=0)
    return {'target_site': target_site, 'control_sites': controls, 'bus_indices': bus_indices, 'phase_angles': phase_angles, 'layers': layers, 'max_parallel_edges_per_layer': int(max_parallel_edges), 'max_parallel_buses_per_layer': int(max_parallel_buses)}

def qft_qudit_verification__parallel_target_schedule(target_site: int, n_qubits: int, *, max_bus_capacity: int | None=None) -> dict[str, object]:
    control_sites = list(range(target_site + 1, n_qubits))
    if not control_sites:
        empty_segment = qft_qudit_verification__parallel_schedule_segment(target_site, [])
        return {'target_site': target_site, 'control_sites': [], 'segments': [empty_segment], 'max_parallel_edges_per_layer': 0, 'max_parallel_buses_per_layer': 0, 'actual_target_concurrency': 0}
    if max_bus_capacity is None or max_bus_capacity >= len(control_sites):
        segment = qft_qudit_verification__parallel_schedule_segment(target_site, control_sites)
        return {'target_site': target_site, 'control_sites': control_sites, 'segments': [segment], 'max_parallel_edges_per_layer': int(segment['max_parallel_edges_per_layer']), 'max_parallel_buses_per_layer': int(segment['max_parallel_buses_per_layer']), 'actual_target_concurrency': int(len(control_sites))}
    segments = [qft_qudit_verification__parallel_schedule_segment(target_site, control_sites[start:start + max_bus_capacity]) for start in range(0, len(control_sites), max_bus_capacity)]
    return {'target_site': target_site, 'control_sites': control_sites, 'segments': segments, 'max_parallel_edges_per_layer': max((int(segment['max_parallel_edges_per_layer']) for segment in segments), default=0), 'max_parallel_buses_per_layer': max((int(segment['max_parallel_buses_per_layer']) for segment in segments), default=0), 'actual_target_concurrency': int(min(len(control_sites), max_bus_capacity))}

def qft_qudit_verification__exact_state_limit(total_dimension: int, *, backend: str) -> bool:
    if backend == 'cirq':
        return total_dimension <= 262144
    return total_dimension <= 4096

def qft_qudit_verification__parse_positive_int_values(spec: str) -> list[int]:
    values = [int(value) for value in verification_common__parse_int_list(spec)]
    if not values:
        raise ValueError('Provide at least one positive integer value.')
    if any((value <= 0 for value in values)):
        raise ValueError('All sweep values must be positive integers.')
    ordered: list[int] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered

def qft_qudit_verification__depth_metric(row: dict[str, object]) -> float:
    if str(row.get('mode') or '') == 'swap_baseline':
        return float(row.get('swap_baseline_depth') or 0.0)
    return float(row.get('moment_count') or 0.0)

def qft_qudit_verification__concurrency_actual(row: dict[str, object]) -> float:
    mode = str(row.get('mode') or '')
    if mode in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    return float(row.get('max_parallel_edges_per_layer') or row.get('max_target_concurrency') or 1.0)

def qft_qudit_verification__dimension_bound(row: dict[str, object]) -> float:
    mode = str(row.get('mode') or '')
    if mode in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    configured_dimension = row.get('configured_dimension')
    if configured_dimension in (None, '', 'None'):
        return 1.0
    configured_value = max(2.0, float(configured_dimension))
    required_buses = max(1.0, float(row.get('required_buses_max') or row.get('max_target_concurrency') or 1.0))
    return float(min(required_buses, max(1.0, math.floor(math.log2(configured_value)))))

def qft_qudit_verification__bus_bound(row: dict[str, object]) -> float:
    mode = str(row.get('mode') or '')
    if mode in {'swap_baseline', 'routed_serialized'}:
        return 1.0
    required_buses = max(1.0, float(row.get('required_buses_max') or row.get('max_target_concurrency') or 1.0))
    bus_capacity_limit = row.get('bus_capacity_limit')
    if bus_capacity_limit in (None, '', 'None'):
        return required_buses
    return float(min(required_buses, max(1.0, float(bus_capacity_limit))))

def qft_qudit_verification__predicted_concurrency(row: dict[str, object]) -> tuple[float, float, float, str]:
    bus_bound = qft_qudit_verification__bus_bound(row)
    dimension_bound = qft_qudit_verification__dimension_bound(row)
    predicted = float(min(bus_bound, dimension_bound))
    if str(row.get('mode') or '') in {'swap_baseline', 'routed_serialized'}:
        limiting_factor = 'bus'
    elif predicted == dimension_bound and dimension_bound <= bus_bound:
        limiting_factor = 'dimension'
    else:
        limiting_factor = 'bus'
    return (bus_bound, dimension_bound, predicted, limiting_factor)

def qft_qudit_verification__annotate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        depth_metric = qft_qudit_verification__depth_metric(enriched)
        actual_concurrency = qft_qudit_verification__concurrency_actual(enriched)
        bus_bound, dimension_bound, predicted_concurrency, limiting_factor = qft_qudit_verification__predicted_concurrency(enriched)
        fidelity_mean = enriched.get('random_state_fidelity_mean')
        required_dimension_min = float(enriched.get('required_dimension_min') or 0.0)
        configured_dimension = float(enriched.get('configured_dimension') or 0.0)
        required_buses_max = float(enriched.get('required_buses_max') or 0.0)
        bus_capacity_limit = float(enriched.get('bus_capacity_limit') or required_buses_max or 0.0)
        dimension_ratio = None if required_dimension_min <= 0.0 else float(configured_dimension / required_dimension_min)
        bus_ratio = None if required_buses_max <= 0.0 else float(bus_capacity_limit / required_buses_max)
        satisfaction_candidates = [value for value in [dimension_ratio, bus_ratio] if value is not None and value > 0.0]
        resource_satisfaction_ratio = None if not satisfaction_candidates else float(min(satisfaction_candidates))
        prediction_efficiency = float(actual_concurrency / max(predicted_concurrency, 1.0))
        prediction_error_abs = float(abs(actual_concurrency - predicted_concurrency))
        enriched.update({
            'algorithm_family': 'qft',
            'depth_metric': depth_metric,
            'depth_prediction_proxy': float((float(enriched.get('serialized_interaction_rounds') or 0.0) + float(enriched.get('logical_qubits') or 0.0)) / max(predicted_concurrency, 1.0)),
            'c_bus': bus_bound,
            'c_dimension': dimension_bound,
            'c_pred': predicted_concurrency,
            'c_actual': actual_concurrency,
            'prediction_efficiency': prediction_efficiency,
            'prediction_error_abs': prediction_error_abs,
            'prediction_tight_10pct': bool(prediction_efficiency >= 0.9),
            'limiting_factor': limiting_factor,
            'dimension_ratio': dimension_ratio,
            'bus_ratio': bus_ratio,
            'resource_satisfaction_ratio': resource_satisfaction_ratio,
            'reconstruction_error_mean': None if fidelity_mean in (None, '', 'None') else float(max(0.0, 1.0 - float(fidelity_mean))),
            'exact_validation_available': bool(str(enriched.get('evaluation_mode') or '') == 'exact'),
            'failure_flag': bool('limited' in str(enriched.get('evaluation_mode') or '')),
        })
        annotated.append(enriched)
    return annotated

def qft_qudit_verification__theory_mapping_payload() -> dict[str, object]:
    return {
        'suite': 'qft',
        'equations': {
            'concurrency_bound': 'C_pred = min(C_bus, C_dimension)',
            'bus_bound_parallel': 'C_bus = min(required_buses_max, bus_capacity_limit) with unlimited bus_capacity_limit treated as required_buses_max',
            'dimension_bound_parallel': 'C_dimension = min(required_buses_max, floor(log2(configured_dimension)))',
            'depth_proxy': 'Depth_proxy ~ (serialized_interaction_rounds + logical_qubits) / C_pred',
            'exact_parallel_condition': 'configured_dimension >= required_dimension_min',
        },
        'mapping': {
            'spectral_dimension': 'configured_dimension and dimension_condition_met',
            'multi_bus_capacity': 'required_buses_max, bus_capacity_limit, max_parallel_buses_per_layer',
            'observed_parallelism': 'max_parallel_edges_per_layer',
            'congestion': 'edge_congestion_max',
            'depth': 'depth_metric',
            'fidelity': 'random_state_fidelity_mean',
        },
    }

def qft_qudit_verification__representative_rows(rows: Sequence[dict[str, object]], *, mode: str, backend: str | None=None) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows if str(row.get('mode')) == mode and (backend is None or str(row.get('backend')) == backend)]
    grouped: dict[tuple[int, str], dict[str, object]] = {}
    for row in candidates:
        key = (int(row.get('n_qubits') or 0), str(row.get('backend') or ''))
        score = (
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            int(row.get('bus_capacity_limit') or row.get('required_buses_max') or 0),
            int(row.get('configured_dimension') or 0),
            -qft_qudit_verification__depth_metric(row),
        )
        previous = grouped.get(key)
        if previous is None or score > previous.get('_score', (-1, -1, -1, 0.0)):
            row['_score'] = score
            grouped[key] = row
    result: list[dict[str, object]] = []
    for row in grouped.values():
        row.pop('_score', None)
        result.append(row)
    return sorted(result, key=lambda entry: (int(entry.get('n_qubits') or 0), str(entry.get('backend') or '')))

def qft_qudit_verification__build_seed_values(base_seed: int, seed_count: int) -> list[int]:
    if seed_count <= 0:
        raise ValueError('seed_count must be positive.')
    return [int(base_seed + 9973 * index) for index in range(seed_count)]

def qft_qudit_verification__fit_power_law(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, object]:
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

def qft_qudit_verification__seed_representative_rows(rows: Sequence[dict[str, object]], *, mode: str) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows if str(row.get('mode')) == mode]
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for row in candidates:
        key = (int(row.get('run_seed') or 0), int(row.get('n_qubits') or 0))
        score = (
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            int(row.get('bus_capacity_limit') or row.get('required_buses_max') or 0),
            int(row.get('configured_dimension') or 0),
            -qft_qudit_verification__depth_metric(row),
        )
        previous = grouped.get(key)
        if previous is None or score > previous.get('_score', (-1, -1, -1, 0.0)):
            row['_score'] = score
            grouped[key] = row
    result: list[dict[str, object]] = []
    for row in grouped.values():
        row.pop('_score', None)
        result.append(row)
    return sorted(result, key=lambda entry: (int(entry.get('n_qubits') or 0), int(entry.get('run_seed') or 0)))

def qft_qudit_verification__aggregate_mode_series(rows: Sequence[dict[str, object]], *, mode: str) -> list[dict[str, object]]:
    representative = qft_qudit_verification__seed_representative_rows(rows, mode=mode)
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in representative:
        grouped.setdefault(int(row.get('n_qubits') or 0), []).append(row)
    series: list[dict[str, object]] = []
    for n_qubits in sorted(grouped):
        bucket = grouped[n_qubits]
        depth_values = np.asarray([qft_qudit_verification__depth_metric(row) for row in bucket], dtype=float)
        concurrency_values = np.asarray([float(row.get('c_actual') or 0.0) for row in bucket], dtype=float)
        fidelity_values = np.asarray([float(row.get('random_state_fidelity_mean') or 0.0) for row in bucket if row.get('random_state_fidelity_mean') not in (None, '', 'None')], dtype=float)
        series.append({
            'n_qubits': int(n_qubits),
            'mode': mode,
            'sample_count': int(len(bucket)),
            'depth_mean': float(np.mean(depth_values)),
            'depth_std': float(np.std(depth_values)),
            'c_actual_mean': float(np.mean(concurrency_values)),
            'c_actual_std': float(np.std(concurrency_values)),
            'fidelity_mean': None if fidelity_values.size == 0 else float(np.mean(fidelity_values)),
            'fidelity_std': None if fidelity_values.size == 0 else float(np.std(fidelity_values)),
        })
    return series

def qft_qudit_verification__scaling_law_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
        series = qft_qudit_verification__aggregate_mode_series(rows, mode=mode)
        fit = qft_qudit_verification__fit_power_law([float(row['n_qubits']) for row in series], [float(row['depth_mean']) for row in series])
        summary_rows.append({'mode': mode, 'mode_label': MODE_LABELS[mode], **fit})
    return summary_rows

def qft_qudit_verification__make_seed_variability_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    seed_values = sorted({int(row.get('run_seed') or 0) for row in rows})
    if plt is None or len(seed_values) <= 1:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
        series = qft_qudit_verification__aggregate_mode_series(rows, mode=mode)
        if not series:
            continue
        axes[0].errorbar([int(row['n_qubits']) for row in series], [float(row['depth_mean']) for row in series], yerr=[float(row['depth_std']) for row in series], marker='o', capsize=3, label=MODE_LABELS[mode])
        fidelity_series = [row for row in series if row.get('fidelity_mean') not in (None, '', 'None')]
        if fidelity_series:
            axes[1].errorbar([int(row['n_qubits']) for row in fidelity_series], [float(row['fidelity_mean']) for row in fidelity_series], yerr=[float(row['fidelity_std'] or 0.0) for row in fidelity_series], marker='o', capsize=3, label=MODE_LABELS[mode])
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('Logical qubits')
    axes[1].set_ylabel('Fidelity')
    axes[1].set_title('Seed variability')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_efficiency_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    plot_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline' and row.get('prediction_efficiency') not in (None, '', 'None')]
    if plt is None or not plot_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for mode in ['routed_serialized', 'routed_parallel']:
        mode_rows = [row for row in plot_rows if str(row.get('mode')) == mode]
        if not mode_rows:
            continue
        axes[0].scatter([int(row.get('n_qubits') or 0) for row in mode_rows], [float(row.get('prediction_efficiency') or 0.0) for row in mode_rows], label=MODE_LABELS[mode], alpha=0.8)
        dim_rows = [row for row in mode_rows if row.get('configured_dimension') not in (None, '', 'None')]
        if dim_rows:
            axes[1].scatter([int(row.get('configured_dimension') or 0) for row in dim_rows], [float(row.get('prediction_efficiency') or 0.0) for row in dim_rows], label=MODE_LABELS[mode], alpha=0.8)
        ratio_rows = [row for row in mode_rows if row.get('resource_satisfaction_ratio') not in (None, '', 'None')]
        if ratio_rows:
            axes[2].scatter([float(row.get('resource_satisfaction_ratio') or 0.0) for row in ratio_rows], [float(row.get('prediction_efficiency') or 0.0) for row in ratio_rows], label=MODE_LABELS[mode], alpha=0.8)
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
    axes[2].set_xlabel('Resource satisfaction ratio')
    axes[2].set_ylabel('Efficiency')
    axes[2].set_title('Efficiency vs violation severity')
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_phase_diagram(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    phase_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel' and row.get('configured_dimension') not in (None, '', 'None')]
    if plt is None or not phase_rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    factor_colors = {'bus': 'tab:blue', 'dimension': 'tab:red'}
    backend_markers = {'cirq': 'o', 'qutip': 's'}
    for backend in sorted({str(row.get('backend') or '') for row in phase_rows}):
        backend_rows = [row for row in phase_rows if str(row.get('backend')) == backend]
        for factor in ['bus', 'dimension']:
            factor_rows = [row for row in backend_rows if str(row.get('limiting_factor')) == factor]
            if not factor_rows:
                continue
            ax.scatter([int(row.get('configured_dimension') or 0) for row in factor_rows], [int(row.get('n_qubits') or 0) for row in factor_rows], color=factor_colors[factor], marker=backend_markers.get(backend, 'o'), alpha=0.85, label=f'{backend} / {factor}')
    ax.set_xlabel('Configured dimension d')
    ax.set_ylabel('Logical qubits n')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small', ncol=2)
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__failure_summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    ideal_lookup: dict[tuple[int, str], dict[str, object]] = {}
    for row in [row for row in rows if str(row.get('mode')) == 'routed_parallel']:
        key = (int(row.get('n_qubits') or 0), str(row.get('backend') or ''))
        score = (
            1 if float(row.get('resource_satisfaction_ratio') or 0.0) >= 1.0 else 0,
            2 if str(row.get('evaluation_mode') or '') == 'exact' else 1 if 'reference' in str(row.get('evaluation_mode') or '') else 0,
            float(row.get('c_pred') or 0.0),
            float(row.get('prediction_efficiency') or 0.0),
            -qft_qudit_verification__depth_metric(row),
        )
        previous = ideal_lookup.get(key)
        if previous is None or score > previous.get('_score', (-1, -1, -1.0, -1.0, 0.0)):
            ideal_lookup[key] = dict(row, _score=score)
    failure_rows: list[dict[str, object]] = []
    for row in [row for row in rows if str(row.get('mode')) == 'routed_parallel' and row.get('resource_satisfaction_ratio') not in (None, '', 'None')]:
        ideal = ideal_lookup.get((int(row.get('n_qubits') or 0), str(row.get('backend') or '')))
        if ideal is None:
            continue
        ideal_depth = max(qft_qudit_verification__depth_metric(ideal), 1.0)
        failure_rows.append({
            'backend': row.get('backend'),
            'n_qubits': int(row.get('n_qubits') or 0),
            'configured_dimension': row.get('configured_dimension'),
            'resource_satisfaction_ratio': float(row.get('resource_satisfaction_ratio') or 0.0),
            'depth_blowup_ratio': float(qft_qudit_verification__depth_metric(row) / ideal_depth),
            'concurrency_loss_ratio': float(row.get('prediction_efficiency') or 0.0),
            'limiting_factor': row.get('limiting_factor'),
        })
    return failure_rows

def qft_qudit_verification__make_quantitative_failure_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    failure_rows = qft_qudit_verification__failure_summary_rows(rows)
    if plt is None or not failure_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    color_lookup = {'bus': 'tab:blue', 'dimension': 'tab:red'}
    axes[0].scatter([float(row['resource_satisfaction_ratio']) for row in failure_rows], [float(row['depth_blowup_ratio']) for row in failure_rows], c=[color_lookup.get(str(row.get('limiting_factor') or ''), 'tab:gray') for row in failure_rows], alpha=0.85)
    axes[1].scatter([float(row['resource_satisfaction_ratio']) for row in failure_rows], [float(row['concurrency_loss_ratio']) for row in failure_rows], c=[color_lookup.get(str(row.get('limiting_factor') or ''), 'tab:gray') for row in failure_rows], alpha=0.85)
    axes[0].set_xlabel('Violation severity = min(d/d_req, buses/buses_req)')
    axes[0].set_ylabel('Depth actual / depth ideal')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Violation severity = min(d/d_req, buses/buses_req)')
    axes[1].set_ylabel('Concurrency efficiency = C_actual / C_pred')
    axes[1].set_title('Quantitative degradation')
    axes[1].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_constraint_distribution_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    distribution_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel']
    if plt is None or not distribution_rows:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    factors = ['bus', 'dimension']
    backends = sorted({str(row.get('backend') or '') for row in distribution_rows})
    x = np.arange(len(backends))
    width = 0.35
    for offset_index, factor in enumerate(factors):
        heights = []
        for backend in backends:
            backend_rows = [row for row in distribution_rows if str(row.get('backend')) == backend]
            count = sum((1 for row in backend_rows if str(row.get('limiting_factor')) == factor))
            heights.append(0.0 if not backend_rows else float(count / len(backend_rows)))
        ax.bar(x + (offset_index - 0.5) * width, heights, width=width, label=factor)
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.set_ylabel('Fraction of runs')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__write_claim_summary(rows: Sequence[dict[str, object]], result_dir: Path, *, suite_stem: str, title_prefix: str) -> None:
    scaling_laws = qft_qudit_verification__scaling_law_summary(rows)
    modeled_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline']
    tight_fraction = 0.0 if not modeled_rows else float(sum((1 for row in modeled_rows if bool(row.get('prediction_tight_10pct')))) / len(modeled_rows))
    mean_efficiency = 0.0 if not modeled_rows else float(np.mean([float(row.get('prediction_efficiency') or 0.0) for row in modeled_rows]))
    routed_parallel_rows = [row for row in rows if str(row.get('mode')) == 'routed_parallel']
    dominance_counts = {factor: sum((1 for row in routed_parallel_rows if str(row.get('limiting_factor')) == factor)) for factor in ['bus', 'dimension']}
    noise_rows = [row for row in rows if str(row.get('backend')) == 'qutip' and row.get('random_state_fidelity_mean') not in (None, '', 'None')]
    parallel_noise = [row for row in noise_rows if str(row.get('mode')) == 'routed_parallel']
    serialized_noise = [row for row in noise_rows if str(row.get('mode')) == 'routed_serialized']
    scaling_lookup = {str(row.get('mode') or ''): row for row in scaling_laws}
    serialized_exponent = scaling_lookup.get('routed_serialized', {}).get('slope')
    parallel_exponent = scaling_lookup.get('routed_parallel', {}).get('slope')
    class_change = 'undetermined'
    if serialized_exponent is not None and parallel_exponent is not None:
        class_change = 'class-improving' if float(parallel_exponent) + 0.2 < float(serialized_exponent) else 'primarily constant-factor'
    lines = [
        f'{title_prefix} claim summary',
        f'Modeled runs: {len(modeled_rows)}',
        f'Tight prediction fraction (efficiency >= 0.9): {tight_fraction:.3f}',
        f'Mean concurrency efficiency C_actual/C_pred: {mean_efficiency:.3f}',
        f'Dominant constraint counts: bus={dominance_counts["bus"]}, dimension={dominance_counts["dimension"]}',
        f'Scaling interpretation: serialized={scaling_lookup.get("routed_serialized", {}).get("big_o")}, parallel={scaling_lookup.get("routed_parallel", {}).get("big_o")}, swap={scaling_lookup.get("swap_baseline", {}).get("big_o")}, verdict={class_change}',
    ]
    if parallel_noise and serialized_noise:
        lines.append(f'Noise-depth punchline: mean parallel depth={np.mean([qft_qudit_verification__depth_metric(row) for row in parallel_noise]):.3f}, mean serialized depth={np.mean([qft_qudit_verification__depth_metric(row) for row in serialized_noise]):.3f}, mean parallel fidelity={np.mean([float(row.get("random_state_fidelity_mean") or 0.0) for row in parallel_noise]):.3f}, mean serialized fidelity={np.mean([float(row.get("random_state_fidelity_mean") or 0.0) for row in serialized_noise]):.3f}')
    edge_rows = sorted(qft_qudit_verification__failure_summary_rows(rows), key=lambda row: float(row.get('resource_satisfaction_ratio') or 1.0))
    if edge_rows:
        worst = edge_rows[0]
        lines.append(f'Worst stress case: n={worst["n_qubits"]}, d={worst["configured_dimension"]}, violation={float(worst["resource_satisfaction_ratio"]):.3f}, depth_blowup={float(worst["depth_blowup_ratio"]):.3f}, efficiency={float(worst["concurrency_loss_ratio"]):.3f}, limiting={worst["limiting_factor"]}')
    (result_dir / f'{suite_stem}_claim_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def qft_qudit_verification__make_exact_validation_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    exact_rows = [row for row in rows if str(row.get('evaluation_mode') or '') == 'exact']
    if plt is None or not exact_rows:
        return
    modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    n_values = sorted({int(row['n_qubits']) for row in exact_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(modes))
    for n_qubits in n_values:
        mode_rows = []
        for mode in modes:
            candidates = [row for row in exact_rows if int(row['n_qubits']) == n_qubits and str(row['mode']) == mode]
            mode_rows.append(max(candidates, key=lambda row: (float(row.get('c_pred') or 0.0), -qft_qudit_verification__depth_metric(row))) if candidates else None)
        axes[0].plot(x, [float(row.get('random_state_fidelity_mean') or 0.0) if row is not None else 0.0 for row in mode_rows], marker='o', label=f'n={n_qubits}')
        axes[1].plot(x, [qft_qudit_verification__depth_metric(row) if row is not None else 0.0 for row in mode_rows], marker='o', label=f'n={n_qubits}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODE_LABELS[mode] for mode in modes], rotation=15)
    axes[0].set_ylabel('State fidelity')
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

def qft_qudit_verification__make_dimension_sweep_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    sweep_rows = [row for row in rows if row.get('configured_dimension') not in (None, '', 'None')]
    if plt is None or len({int(row['configured_dimension']) for row in sweep_rows}) <= 1:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for backend in sorted({str(row.get('backend') or '') for row in sweep_rows}):
        parallel_rows = sorted([row for row in sweep_rows if str(row.get('backend')) == backend and str(row.get('mode')) == 'routed_parallel'], key=lambda row: (int(row['n_qubits']), int(row['configured_dimension'])))
        if not parallel_rows:
            continue
        for n_qubits in sorted({int(row['n_qubits']) for row in parallel_rows}):
            n_rows = [row for row in parallel_rows if int(row['n_qubits']) == n_qubits]
            label = f'{backend}, n={n_qubits}'
            axes[0].plot([int(row['configured_dimension']) for row in n_rows], [qft_qudit_verification__depth_metric(row) for row in n_rows], marker='o', label=label)
            axes[1].plot([int(row['configured_dimension']) for row in n_rows], [float(row.get('c_actual') or 0.0) for row in n_rows], marker='o', label=label)
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

def qft_qudit_verification__make_scaling_gap_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    representative = {mode: qft_qudit_verification__aggregate_mode_series(rows, mode=mode) for mode in modes}
    if not representative['routed_parallel']:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode in modes:
        mode_rows = representative[mode]
        axes[0].errorbar([int(row['n_qubits']) for row in mode_rows], [float(row['depth_mean']) for row in mode_rows], yerr=[float(row['depth_std']) for row in mode_rows], marker='o', capsize=3, label=MODE_LABELS[mode])
    serialized_lookup = {int(row['n_qubits']): row for row in representative['routed_serialized']}
    parallel_lookup = {int(row['n_qubits']): row for row in representative['routed_parallel']}
    gap_points: list[tuple[int, float]] = []
    for n_qubits, serialized_row in serialized_lookup.items():
        parallel_row = parallel_lookup.get(n_qubits)
        if parallel_row is None:
            continue
        serialized_depth = max(float(serialized_row['depth_mean']), 1.0)
        parallel_depth = float(parallel_row['depth_mean'])
        gap_points.append((n_qubits, float((serialized_depth - parallel_depth) / serialized_depth)))
    if gap_points:
        gap_points.sort()
        axes[1].plot([point[0] for point in gap_points], [point[1] for point in gap_points], marker='o')
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('Logical qubits')
    axes[1].set_ylabel('(serialized - parallel) / serialized')
    axes[1].set_title('Parallel advantage gap')
    axes[1].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_prediction_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    plot_rows = [row for row in rows if str(row.get('mode') or '') != 'swap_baseline']
    if plt is None or not plot_rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for backend in sorted({str(row.get('backend') or '') for row in plot_rows}):
        backend_rows = [row for row in plot_rows if str(row.get('backend')) == backend]
        ax.scatter([float(row.get('c_pred') or 0.0) for row in backend_rows], [float(row.get('c_actual') or 0.0) for row in backend_rows], label=backend, alpha=0.8)
    max_axis = max([1.0] + [float(max(float(row.get('c_pred') or 0.0), float(row.get('c_actual') or 0.0))) for row in plot_rows])
    ax.plot([0.0, max_axis], [0.0, max_axis], linestyle='--', color='black', linewidth=1.0)
    ax.set_xlabel('Predicted concurrency C_pred')
    ax.set_ylabel('Measured concurrency C_actual')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_failure_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    failure_rows = [row for row in rows if bool(row.get('failure_flag')) or str(row.get('mode')) == 'routed_parallel']
    if plt is None or not failure_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for backend in sorted({str(row.get('backend') or '') for row in failure_rows}):
        backend_rows = sorted([row for row in failure_rows if str(row.get('backend')) == backend and str(row.get('mode')) == 'routed_parallel'], key=lambda row: (int(row.get('n_qubits') or 0), int(row.get('configured_dimension') or 0)))
        if not backend_rows:
            continue
        axes[0].scatter([int(row.get('configured_dimension') or 0) for row in backend_rows], [qft_qudit_verification__depth_metric(row) for row in backend_rows], label=backend, alpha=0.8)
        exact_rows = [row for row in backend_rows if row.get('random_state_fidelity_mean') not in (None, '', 'None')]
        if exact_rows:
            axes[1].scatter([int(row.get('configured_dimension') or 0) for row in exact_rows], [float(row.get('random_state_fidelity_mean') or 0.0) for row in exact_rows], label=backend, alpha=0.8)
    axes[0].set_xlabel('Configured dimension d')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('Configured dimension d')
    axes[1].set_ylabel('Fidelity')
    axes[1].set_title('Failure / breakdown regime')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_noise_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    noisy_rows = [row for row in rows if str(row.get('backend')) == 'qutip' and str(row.get('evaluation_mode') or '') == 'exact' and ((float(row.get('routing_gate_time') or 0.0) > 0.0) or (float(row.get('target_gate_time') or 0.0) > 0.0) or (float(row.get('local_gate_time') or 0.0) > 0.0) or (float(row.get('leakage_epsilon') or 0.0) > 0.0))]
    if plt is None or not noisy_rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for mode in ['routed_serialized', 'routed_parallel']:
        mode_rows = [row for row in noisy_rows if str(row.get('mode')) == mode]
        if not mode_rows:
            continue
        ax.scatter([qft_qudit_verification__depth_metric(row) for row in mode_rows], [float(row.get('random_state_fidelity_mean') or 0.0) for row in mode_rows], label=MODE_LABELS[mode], alpha=0.8)
    ax.set_xlabel('Depth / moments')
    ax.set_ylabel('Fidelity')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__make_unified_figure(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    representative = qft_qudit_verification__representative_rows(rows, mode='routed_parallel')
    if not representative:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    scaling_modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    for mode in scaling_modes:
        mode_rows = qft_qudit_verification__representative_rows(rows, mode=mode)
        axes[0].plot([int(row['n_qubits']) for row in mode_rows], [qft_qudit_verification__depth_metric(row) for row in mode_rows], marker='o', label=MODE_LABELS[mode])
    for backend in sorted({str(row.get('backend') or '') for row in representative}):
        backend_rows = [row for row in representative if str(row.get('backend')) == backend]
        axes[1].scatter([float(row.get('c_pred') or 0.0) for row in backend_rows], [float(row.get('c_actual') or 0.0) for row in backend_rows], label=backend, alpha=0.8)
    dominance_rows = sorted(representative, key=lambda row: int(row.get('n_qubits') or 0))
    factor_to_y = {'bus': 0.0, 'dimension': 1.0}
    axes[2].scatter([int(row.get('n_qubits') or 0) for row in dominance_rows], [factor_to_y.get(str(row.get('limiting_factor') or ''), 0.0) for row in dominance_rows], c=['tab:blue' if str(row.get('limiting_factor')) == 'bus' else 'tab:red' for row in dominance_rows], alpha=0.8)
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    max_axis = max([1.0] + [float(max(float(row.get('c_pred') or 0.0), float(row.get('c_actual') or 0.0))) for row in representative])
    axes[1].plot([0.0, max_axis], [0.0, max_axis], linestyle='--', color='black', linewidth=1.0)
    axes[1].set_xlabel('Predicted concurrency')
    axes[1].set_ylabel('Measured concurrency')
    axes[1].set_title('Prediction vs measurement')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[2].set_xlabel('Logical qubits')
    axes[2].set_yticks([0.0, 1.0])
    axes[2].set_yticklabels(['bus', 'dimension'])
    axes[2].set_ylabel('Limiting factor')
    axes[2].set_title('Constraint dominance')
    axes[2].grid(alpha=0.3)
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__write_analysis_bundle(rows: Sequence[dict[str, object]], result_dir: Path, *, suite_stem: str, title_prefix: str) -> None:
    mapping = qft_qudit_verification__theory_mapping_payload()
    scaling_laws = qft_qudit_verification__scaling_law_summary(rows)
    seed_statistics: list[dict[str, object]] = []
    for mode in ['swap_baseline', 'routed_serialized', 'routed_parallel']:
        seed_statistics.extend(qft_qudit_verification__aggregate_mode_series(rows, mode=mode))
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
    verification_common__write_csv(result_dir / f'{suite_stem}_scaling_laws.csv', scaling_laws)
    verification_common__write_json(result_dir / f'{suite_stem}_scaling_laws.json', {'suite': suite_stem, 'rows': scaling_laws})
    qft_qudit_verification__make_exact_validation_plot(rows, result_dir / f'{suite_stem}_exact_validation.png', f'{title_prefix} exact validation')
    qft_qudit_verification__make_dimension_sweep_plot(rows, result_dir / f'{suite_stem}_dimension_sweep.png', f'{title_prefix} dimension sweep')
    qft_qudit_verification__make_scaling_gap_plot(rows, result_dir / f'{suite_stem}_scaling_gap.png', f'{title_prefix} scaling')
    qft_qudit_verification__make_seed_variability_plot(rows, result_dir / f'{suite_stem}_seed_variability.png', f'{title_prefix} seed variability')
    qft_qudit_verification__make_efficiency_plot(rows, result_dir / f'{suite_stem}_efficiency.png', f'{title_prefix} efficiency')
    qft_qudit_verification__make_phase_diagram(rows, result_dir / f'{suite_stem}_phase_diagram.png', f'{title_prefix} dominant-constraint phase diagram')
    qft_qudit_verification__make_quantitative_failure_plot(rows, result_dir / f'{suite_stem}_quantitative_failure.png', f'{title_prefix} failure severity')
    qft_qudit_verification__make_constraint_distribution_plot(rows, result_dir / f'{suite_stem}_constraint_distribution.png', f'{title_prefix} dominant constraint distribution')
    qft_qudit_verification__make_prediction_plot(rows, result_dir / f'{suite_stem}_prediction_vs_measurement.png', f'{title_prefix} prediction')
    qft_qudit_verification__make_failure_plot(rows, result_dir / f'{suite_stem}_failure_regime.png', f'{title_prefix} failure regime')
    qft_qudit_verification__make_noise_plot(rows, result_dir / f'{suite_stem}_noise_validation.png', f'{title_prefix} noisy validation')
    qft_qudit_verification__make_unified_figure(rows, result_dir / f'{suite_stem}_unified_figure.png', f'{title_prefix} unified concurrency')
    qft_qudit_verification__write_claim_summary(rows, result_dir, suite_stem=suite_stem, title_prefix=title_prefix)

def qft_qudit_verification__make_plot(rows: Sequence[dict[str, object]], figure_path: Path, title: str) -> None:
    if plt is None or not rows:
        return
    modes = ['swap_baseline', 'routed_serialized', 'routed_parallel']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode in modes:
        mode_rows = qft_qudit_verification__representative_rows(rows, mode=mode)
        if not mode_rows:
            continue
        axes[0].plot([int(row['n_qubits']) for row in mode_rows], [qft_qudit_verification__depth_metric(row) for row in mode_rows], marker='o', label=MODE_LABELS[mode])
    axes[0].set_xlabel('Logical qubits')
    axes[0].set_ylabel('Depth / moments')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    parallel_rows = qft_qudit_verification__representative_rows(rows, mode='routed_parallel')
    if parallel_rows:
        axes[1].plot([int(row['n_qubits']) for row in parallel_rows], [float(row['c_actual'] if 'c_actual' in row else qft_qudit_verification__concurrency_actual(row)) for row in parallel_rows], marker='o', label='Observed concurrency')
        axes[1].plot([int(row['n_qubits']) for row in parallel_rows], [float(row['minimum_parallel_dimension_log2']) for row in parallel_rows], marker='s', linestyle='--', label='log2(d_min)')
    axes[1].set_xlabel('Logical qubits')
    axes[1].set_ylabel('Concurrency / dimension bits')
    axes[1].set_title('Routing concurrency requirement')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    verification_common__maybe_save_figure(fig, figure_path)

def qft_qudit_verification__add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--min-qubits', type=int, default=2)
    parser.add_argument('--max-qubits', type=int, default=6)
    parser.add_argument('--random-samples', type=int, default=3)
    parser.add_argument('--dimension', type=int, default=4, help='Configured local qudit dimension. Parallel routed QFT needs d >= 2^n.')
    parser.add_argument('--dimension-sweep', default='', help='Optional comma-separated dimension sweep, e.g. 2,4,8,16,32,64.')
    parser.add_argument('--bus-capacities', default='', help='Optional comma-separated routed-parallel bus capacities for ablation studies.')
    parser.add_argument('--optimization-level', type=int, default=2)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--seed-count', type=int, default=1, help='Repeat the routed experiment with this many deterministic seed offsets for mean/std error bars.')
    parser.add_argument('--workers', type=int, default=1, help='Worker processes for independent qudit configurations.')
    parser.add_argument('--native-threads-per-worker', type=int, default=None, help='Caps BLAS/OpenMP threads inside each worker. Defaults to 1 when workers > 1.')

def qft_qudit_verification__build_serialized_cirq_circuit(n_qubits: int, dimension: int):
    _init_qudit_cirq_verification()
    import cirq
    qudits = cirq.LineQid.range(n_qubits, dimension=dimension)
    circuit = cirq.Circuit()
    for target in range(n_qubits):
        for control in range(target + 1, n_qubits):
            theta = math.pi / 2 ** (control - target)
            circuit += qudit_cirq_verification__build_single_bus_circuit(qudits, qudit_cirq_verification__build_line_path(control, target), bus_index=1, unitary=np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128), label='CP')
        circuit.append(qudit_cirq_verification__LogicalQuditGate(dimension, qudit_cirq_verification__h_matrix(), 'H').on(qudits[target]))
    return (circuit, qudits, qudit_cirq_verification__circuit_metrics(circuit))

def qft_qudit_verification__build_parallel_cirq_circuit(n_qubits: int, dimension: int, *, max_bus_capacity: int | None=None):
    _init_qudit_cirq_verification()
    import cirq
    qudits = cirq.LineQid.range(n_qubits, dimension=dimension)
    circuit = cirq.Circuit()
    target_schedules = [qft_qudit_verification__parallel_target_schedule(target, n_qubits, max_bus_capacity=max_bus_capacity) for target in range(n_qubits)]
    for schedule in target_schedules:
        for segment in schedule['segments']:
            for layer in segment['layers']:
                circuit.append(cirq.Moment((qudit_cirq_verification__CombinedShiftGate(dimension, op['bus_specs']).on(qudits[int(op['source_site'])], qudits[int(op['target_site'])]) for op in layer)))
            if segment['bus_indices']:
                circuit.append(qudit_cirq_verification__MultiBusPhaseGate(dimension, segment['bus_indices'], segment['phase_angles'], label='CPMUX').on(qudits[int(schedule['target_site'])]))
            for layer in reversed(segment['layers']):
                circuit.append(cirq.Moment((qudit_cirq_verification__CombinedShiftGate(dimension, op['bus_specs'], inverse=True).on(qudits[int(op['source_site'])], qudits[int(op['target_site'])]) for op in layer)))
        circuit.append(qudit_cirq_verification__LogicalQuditGate(dimension, qudit_cirq_verification__h_matrix(), 'H').on(qudits[int(schedule['target_site'])]))
    aggregated = {'max_parallel_edges_per_layer': max((int(schedule['max_parallel_edges_per_layer']) for schedule in target_schedules), default=0), 'max_parallel_buses_per_layer': max((int(schedule['max_parallel_buses_per_layer']) for schedule in target_schedules), default=0), 'max_target_concurrency': max((int(schedule['actual_target_concurrency']) for schedule in target_schedules), default=0), 'target_schedules': target_schedules}
    return (circuit, qudits, qudit_cirq_verification__circuit_metrics(circuit), aggregated)

def qft_qudit_verification__evaluate_cirq_mode(*, mode: str, n_qubits: int, configured_dimension: int, random_samples: int, seed: int, optimization_level: int, bus_capacity_limit: int | None=None) -> dict[str, object]:
    _init_qudit_cirq_verification()
    structural = qft_qudit_verification__qft_structural_metrics(n_qubits)
    baseline = qft_qudit_verification__baseline_swap_metrics(n_qubits, seed=seed, optimization_level=optimization_level)
    row = {'n_qubits': n_qubits, 'backend': 'cirq', 'mode': mode, 'mode_label': MODE_LABELS[mode], 'configured_dimension': configured_dimension, 'required_dimension_min': None, 'dimension_condition_met': None, 'evaluation_mode': 'metrics_only', 'random_state_fidelity_mean': None, 'random_state_fidelity_min': None, 'logical_subspace_probability_mean': None, 'max_routing_population_mean': None, 'moment_count': None, 'operation_count': None, 'one_qudit_gate_count': None, 'two_qudit_gate_count': None, 'max_gate_width': None, 'max_parallel_edges_per_layer': 0, 'max_parallel_buses_per_layer': 0, 'swap_baseline_depth': None, 'swap_baseline_size': None, 'swap_baseline_swap_count': None, 'swap_baseline_cx_count': None, 'swap_baseline_available': baseline is not None, 'bus_capacity_limit': bus_capacity_limit, **structural}
    if baseline is not None:
        row.update({'swap_baseline_depth': baseline['transpiled_depth'], 'swap_baseline_size': baseline['transpiled_size'], 'swap_baseline_swap_count': baseline['swap_count'], 'swap_baseline_cx_count': baseline['cx_count']})
    if mode == 'swap_baseline':
        row['evaluation_mode'] = 'reference_metrics_only'
        return row
    if mode == 'routed_serialized':
        required_dimension = 4
    else:
        effective_buses = int(bus_capacity_limit) if bus_capacity_limit is not None else int(structural['required_buses_max'])
        bus_count = max(1, min(effective_buses, int(structural['required_buses_max'])))
        required_dimension = int(max(2, 2 ** (bus_count + 1)))
    row['required_dimension_min'] = required_dimension
    row['dimension_condition_met'] = configured_dimension >= required_dimension
    circuit_dimension = max(configured_dimension, required_dimension)
    if mode == 'routed_serialized':
        circuit, qudits, stats = qft_qudit_verification__build_serialized_cirq_circuit(n_qubits, circuit_dimension)
    else:
        circuit, qudits, stats, parallel_metrics = qft_qudit_verification__build_parallel_cirq_circuit(n_qubits, circuit_dimension, max_bus_capacity=bus_capacity_limit)
        row.update({'max_parallel_edges_per_layer': parallel_metrics['max_parallel_edges_per_layer'], 'max_parallel_buses_per_layer': parallel_metrics['max_parallel_buses_per_layer']})
    row.update({'moment_count': stats.moment_count, 'operation_count': stats.operation_count, 'one_qudit_gate_count': stats.one_qudit_gate_count, 'two_qudit_gate_count': stats.two_qudit_gate_count, 'max_gate_width': stats.max_gate_width})
    total_dimension = circuit_dimension ** n_qubits
    exact_allowed = configured_dimension >= required_dimension and qft_qudit_verification__exact_state_limit(total_dimension, backend='cirq')
    if not exact_allowed:
        row['evaluation_mode'] = 'metrics_only_dimension_limited' if configured_dimension < required_dimension else 'metrics_only_state_limited'
        return row
    rng = np.random.default_rng(seed)
    fidelities: list[float] = []
    fidelities_no_swaps: list[float] = []
    fidelities_with_swaps: list[float] = []
    logical_probabilities: list[float] = []
    routing_residuals: list[float] = []
    logical_sites = list(range(n_qubits))
    for _ in range(random_samples):
        input_state = qubit_reference__random_statevector(n_qubits, int(rng.integers(0, 1000000)))
        initial = qudit_cirq_verification__embed_qubit_state(input_state, dimension=circuit_dimension, total_sites=n_qubits, logical_sites=logical_sites)
        final_state = qudit_cirq_verification__simulate_circuit(circuit, qudits, initial)
        extracted = qudit_cirq_verification__extract_clean_logical_state(final_state, dimension=circuit_dimension, total_sites=n_qubits, logical_sites=logical_sites)
        probability = float(np.linalg.norm(extracted) ** 2)
        normalized = extracted / np.linalg.norm(extracted) if probability > 0.0 else extracted
        if probability > 0.0:
            reference_no_swaps = qubit_reference__qft_statevector(input_state, n_qubits)
            reference_with_swaps = qubit_reference__qft_statevector(input_state, n_qubits, with_swaps=True)
            fidelity_no_swaps = verification_common__statevector_fidelity(normalized, reference_no_swaps)
            fidelity_with_swaps = verification_common__statevector_fidelity(normalized, reference_with_swaps)
            fidelities.append(max(fidelity_no_swaps, fidelity_with_swaps))
            fidelities_no_swaps.append(fidelity_no_swaps)
            fidelities_with_swaps.append(fidelity_with_swaps)
        else:
            fidelities.append(0.0)
            fidelities_no_swaps.append(0.0)
            fidelities_with_swaps.append(0.0)
        logical_probabilities.append(probability)
        routing_residuals.append(qudit_cirq_verification__max_routing_population(final_state, total_sites=n_qubits, dimension=circuit_dimension))
    row.update({
        'evaluation_mode': 'exact',
        'random_state_fidelity_mean': float(np.mean(fidelities)),
        'random_state_fidelity_min': float(np.min(fidelities)),
        'random_state_fidelity_no_swaps_mean': float(np.mean(fidelities_no_swaps)),
        'random_state_fidelity_no_swaps_min': float(np.min(fidelities_no_swaps)),
        'random_state_fidelity_with_swaps_mean': float(np.mean(fidelities_with_swaps)),
        'random_state_fidelity_with_swaps_min': float(np.min(fidelities_with_swaps)),
        'logical_subspace_probability_mean': float(np.mean(logical_probabilities)),
        'max_routing_population_mean': float(np.mean(routing_residuals)),
    })
    return row

def qft_qudit_verification__evaluate_qutip_mode(*, mode: str, n_qubits: int, configured_dimension: int, random_samples: int, seed: int, optimization_level: int, routing_gate_time: float, target_gate_time: float, local_gate_time: float, t1_levels: Sequence[float], tphi_levels: Sequence[float], leakage_epsilon: float, monte_carlo_trajectories: int, bus_capacity_limit: int | None=None) -> dict[str, object]:
    _init_qudit_qutip_verification()
    structural = qft_qudit_verification__qft_structural_metrics(n_qubits)
    baseline = qft_qudit_verification__baseline_swap_metrics(n_qubits, seed=seed, optimization_level=optimization_level)
    row = {'n_qubits': n_qubits, 'backend': 'qutip', 'mode': mode, 'mode_label': MODE_LABELS[mode], 'configured_dimension': configured_dimension, 'required_dimension_min': None, 'dimension_condition_met': None, 'evaluation_mode': 'metrics_only', 'random_state_fidelity_mean': None, 'random_state_fidelity_min': None, 'logical_subspace_probability_mean': None, 'max_routing_population_mean': None, 'moment_count': None, 'operation_count': None, 'one_qudit_gate_count': None, 'two_qudit_gate_count': None, 'max_gate_width': None, 'max_parallel_edges_per_layer': 0, 'max_parallel_buses_per_layer': 0, 'swap_baseline_depth': None, 'swap_baseline_size': None, 'swap_baseline_swap_count': None, 'swap_baseline_cx_count': None, 'swap_baseline_available': baseline is not None, 'routing_gate_time': routing_gate_time, 'target_gate_time': target_gate_time, 'local_gate_time': local_gate_time, 'leakage_epsilon': leakage_epsilon, 'bus_capacity_limit': bus_capacity_limit, **structural}
    if baseline is not None:
        row.update({'swap_baseline_depth': baseline['transpiled_depth'], 'swap_baseline_size': baseline['transpiled_size'], 'swap_baseline_swap_count': baseline['swap_count'], 'swap_baseline_cx_count': baseline['cx_count']})
    if mode == 'swap_baseline':
        row['evaluation_mode'] = 'reference_metrics_only'
        return row
    if mode == 'routed_serialized':
        required_dimension = 4
    else:
        effective_buses = int(bus_capacity_limit) if bus_capacity_limit is not None else int(structural['required_buses_max'])
        bus_count = max(1, min(effective_buses, int(structural['required_buses_max'])))
        required_dimension = int(max(2, 2 ** (bus_count + 1)))
    row['required_dimension_min'] = required_dimension
    row['dimension_condition_met'] = configured_dimension >= required_dimension
    circuit_dimension = max(configured_dimension, required_dimension)
    if mode == 'routed_serialized':
        serialized_stats = {'moment_count': sum((2 * (control - target) + 2 for target in range(n_qubits) for control in range(target + 1, n_qubits))) + 1, 'operation_count': None, 'one_qudit_gate_count': None, 'two_qudit_gate_count': None, 'max_gate_width': 2}
        row.update(serialized_stats)
    else:
        schedules = [qft_qudit_verification__parallel_target_schedule(target, n_qubits, max_bus_capacity=bus_capacity_limit) for target in range(n_qubits)]
        row.update({'moment_count': int(sum((sum((2 * len(segment['layers']) + (1 if segment['control_sites'] else 0) for segment in schedule['segments'])) + 1 for schedule in schedules))), 'operation_count': None, 'one_qudit_gate_count': None, 'two_qudit_gate_count': None, 'max_gate_width': 2, 'max_parallel_edges_per_layer': max((int(schedule['max_parallel_edges_per_layer']) for schedule in schedules), default=0), 'max_parallel_buses_per_layer': max((int(schedule['max_parallel_buses_per_layer']) for schedule in schedules), default=0)})
    total_dimension = circuit_dimension ** n_qubits
    exact_allowed = configured_dimension >= required_dimension and qft_qudit_verification__exact_state_limit(total_dimension, backend='qutip')
    if not exact_allowed:
        row['evaluation_mode'] = 'metrics_only_dimension_limited' if configured_dimension < required_dimension else 'metrics_only_state_limited'
        return row
    dims = [circuit_dimension] * n_qubits
    h0 = qudit_qutip_verification__zero_hamiltonian(dims)
    collapse_ops = qudit_qutip_verification__build_relaxation_and_dephasing_ops(dims, t1_levels, tphi_levels)
    leakage_kraus = qudit_qutip_verification__build_leakage_kraus(circuit_dimension, leakage_epsilon)
    qudit_qutip_verification__configure_monte_carlo(seed, monte_carlo_trajectories)
    rng = np.random.default_rng(seed)
    fidelities: list[float] = []
    logical_probabilities: list[float] = []
    routing_residuals: list[float] = []
    logical_sites = list(range(n_qubits))
    for _ in range(random_samples):
        input_state = qubit_reference__random_statevector(n_qubits, int(rng.integers(0, 1000000)))
        state = qudit_qutip_verification__embed_qubit_state_as_ket(input_state, dimension=circuit_dimension, total_sites=n_qubits, logical_sites=logical_sites)
        if mode == 'routed_serialized':
            for target in range(n_qubits):
                for control in range(target + 1, n_qubits):
                    theta = math.pi / 2 ** (control - target)
                    state = qudit_qutip_verification__apply_routed_single_bus_unitary(state, path=qudit_qutip_verification__build_line_path(control, target), dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__phase_matrix(theta), routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, h0=h0, collapse_ops=collapse_ops, leakage_kraus=leakage_kraus)
                state = qudit_qutip_verification__apply_logical_unitary(state, site=target, dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        else:
            for schedule in [qft_qudit_verification__parallel_target_schedule(target, n_qubits, max_bus_capacity=bus_capacity_limit) for target in range(n_qubits)]:
                for segment in schedule['segments']:
                    for layer in segment['layers']:
                        layer_ops = [qudit_qutip_verification__embed_two_site(qudit_qutip_verification__combined_shift_matrix(circuit_dimension, op['bus_specs']), int(op['source_site']), int(op['target_site']), dims) for op in layer]
                        state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(layer_ops), routing_gate_time, h0, collapse_ops, leakage_sites=[int(op['target_site']) for op in layer], leakage_kraus=leakage_kraus) if layer_ops else state
                    if segment['bus_indices']:
                        target_gate = qudit_qutip_verification__embed_single_site(qt.Qobj(qudit_qutip_verification__multi_bus_phase_matrix(circuit_dimension, segment['bus_indices'], segment['phase_angles']), dims=[[circuit_dimension], [circuit_dimension]]).to('csr'), int(schedule['target_site']), dims)
                        state = qudit_qutip_verification__apply_global_gate_with_noise(state, target_gate, target_gate_time, h0, collapse_ops)
                    for layer in reversed(segment['layers']):
                        layer_ops = [qudit_qutip_verification__embed_two_site(qudit_qutip_verification__combined_shift_matrix(circuit_dimension, op['bus_specs'], inverse=True), int(op['source_site']), int(op['target_site']), dims) for op in layer]
                        state = qudit_qutip_verification__apply_global_gate_with_noise(state, qudit_qutip_verification__compose_layer_operator(layer_ops), routing_gate_time, h0, collapse_ops, leakage_sites=[int(op['target_site']) for op in layer], leakage_kraus=leakage_kraus) if layer_ops else state
                state = qudit_qutip_verification__apply_logical_unitary(state, site=int(schedule['target_site']), dimension=circuit_dimension, dims=dims, unitary=qudit_qutip_verification__h_matrix(), gate_time=local_gate_time, h0=h0, collapse_ops=collapse_ops)
        ideal = qudit_qutip_verification__embed_qubit_state_as_ket(qubit_reference__qft_statevector(input_state, n_qubits), dimension=circuit_dimension, total_sites=n_qubits, logical_sites=logical_sites)
        fidelities.append(qudit_qutip_verification__pure_state_overlap(state, ideal))
        logical_probabilities.append(qudit_qutip_verification__logical_subspace_population(state, dimension=circuit_dimension, total_sites=n_qubits, logical_sites=logical_sites))
        routing_residuals.append(max((qudit_qutip_verification__routing_population(state, site, circuit_dimension, dims) for site in range(n_qubits))))
    row.update({'evaluation_mode': 'exact', 'random_state_fidelity_mean': float(np.mean(fidelities)), 'random_state_fidelity_min': float(np.min(fidelities)), 'logical_subspace_probability_mean': float(np.mean(logical_probabilities)), 'max_routing_population_mean': float(np.mean(routing_residuals))})
    return row

def qft_qudit_verification__evaluate_cirq_configuration(task: dict[str, object]) -> list[dict[str, object]]:
    run_seed = int(task['run_seed'])
    n_qubits = int(task['n_qubits'])
    configured_dimension = int(task['configured_dimension'])
    random_samples = int(task['random_samples'])
    optimization_level = int(task['optimization_level'])
    bus_capacity_values = [int(value) for value in task.get('bus_capacity_values', ())]
    rows: list[dict[str, object]] = []
    baseline_row = qft_qudit_verification__evaluate_cirq_mode(mode='swap_baseline', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level)
    baseline_row['run_seed'] = run_seed
    rows.append(baseline_row)
    serialized_row = qft_qudit_verification__evaluate_cirq_mode(mode='routed_serialized', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level)
    serialized_row['run_seed'] = run_seed
    rows.append(serialized_row)
    if bus_capacity_values:
        for bus_capacity_limit in bus_capacity_values:
            parallel_row = qft_qudit_verification__evaluate_cirq_mode(mode='routed_parallel', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits + 1000 * bus_capacity_limit, optimization_level=optimization_level, bus_capacity_limit=bus_capacity_limit)
            parallel_row['run_seed'] = run_seed
            rows.append(parallel_row)
    else:
        parallel_row = qft_qudit_verification__evaluate_cirq_mode(mode='routed_parallel', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level)
        parallel_row['run_seed'] = run_seed
        rows.append(parallel_row)
    return rows

def qft_qudit_verification__evaluate_qutip_configuration(task: dict[str, object]) -> list[dict[str, object]]:
    run_seed = int(task['run_seed'])
    n_qubits = int(task['n_qubits'])
    configured_dimension = int(task['configured_dimension'])
    random_samples = int(task['random_samples'])
    optimization_level = int(task['optimization_level'])
    routing_gate_time = float(task['routing_gate_time'])
    target_gate_time = float(task['target_gate_time'])
    local_gate_time = float(task['local_gate_time'])
    t1_levels = [float(value) for value in task['t1_levels']]
    tphi_levels = [float(value) for value in task['tphi_levels']]
    leakage_epsilon = float(task['leakage_epsilon'])
    monte_carlo_trajectories = int(task['monte_carlo_trajectories'])
    bus_capacity_values = [int(value) for value in task.get('bus_capacity_values', ())]
    rows: list[dict[str, object]] = []
    baseline_row = qft_qudit_verification__evaluate_qutip_mode(mode='swap_baseline', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories)
    baseline_row['run_seed'] = run_seed
    rows.append(baseline_row)
    serialized_row = qft_qudit_verification__evaluate_qutip_mode(mode='routed_serialized', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories)
    serialized_row['run_seed'] = run_seed
    rows.append(serialized_row)
    if bus_capacity_values:
        for bus_capacity_limit in bus_capacity_values:
            parallel_row = qft_qudit_verification__evaluate_qutip_mode(mode='routed_parallel', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits + 1000 * bus_capacity_limit, optimization_level=optimization_level, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories, bus_capacity_limit=bus_capacity_limit)
            parallel_row['run_seed'] = run_seed
            rows.append(parallel_row)
    else:
        parallel_row = qft_qudit_verification__evaluate_qutip_mode(mode='routed_parallel', n_qubits=n_qubits, configured_dimension=configured_dimension, random_samples=random_samples, seed=run_seed + n_qubits, optimization_level=optimization_level, routing_gate_time=routing_gate_time, target_gate_time=target_gate_time, local_gate_time=local_gate_time, t1_levels=t1_levels, tphi_levels=tphi_levels, leakage_epsilon=leakage_epsilon, monte_carlo_trajectories=monte_carlo_trajectories)
        parallel_row['run_seed'] = run_seed
        rows.append(parallel_row)
    return rows

def qft_qudit_verification__main_cirq(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='Cirq qudit QFT verification as a routing stress test.')
    qft_qudit_verification__add_shared_arguments(parser)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.max_qubits = verification_common__prompt_int('Up to how many logical qubits do you want to test?', args.max_qubits)
    ctx = verification_common__setup_run_context(script_file)
    dimension_values = qft_qudit_verification__parse_positive_int_values(args.dimension_sweep) if args.dimension_sweep.strip() else [int(args.dimension)]
    bus_capacity_values = qft_qudit_verification__parse_positive_int_values(args.bus_capacities) if args.bus_capacities.strip() else []
    seed_values = qft_qudit_verification__build_seed_values(args.seed, args.seed_count)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'run_seed': int(run_seed), 'n_qubits': int(n_qubits), 'configured_dimension': int(configured_dimension), 'random_samples': int(args.random_samples), 'optimization_level': int(args.optimization_level), 'bus_capacity_values': list(bus_capacity_values)} for run_seed in seed_values for n_qubits in range(args.min_qubits, args.max_qubits + 1) for configured_dimension in dimension_values]
    print(f'[qft_cirq] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(qft_qudit_verification__evaluate_cirq_configuration, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[qft_cirq] parallel section complete rows={len(rows)}')
    rows = qft_qudit_verification__annotate_rows(rows)
    verification_common__write_csv(ctx.result_dir / 'qft_cirq.csv', rows)
    qft_qudit_verification__make_plot(rows, ctx.result_dir / 'qft_cirq.png', 'QFT routing depth comparison (Cirq)')
    qft_qudit_verification__write_analysis_bundle(rows, ctx.result_dir, suite_stem='qft_cirq', title_prefix='QFT (Cirq)')
    verification_common__write_json(ctx.result_dir / 'qft_cirq.json', {'suite': 'qft_cirq', 'focus': 'swap_vs_serialized_vs_parallel_routing', 'configured_dimension': args.dimension, 'rows': rows})
    print(f'Saved results to: {ctx.result_dir}')

def qft_qudit_verification__main_qutip(script_file: str) -> None:
    parser = argparse.ArgumentParser(description='QuTiP qudit QFT verification as a routing stress test.')
    qft_qudit_verification__add_shared_arguments(parser)
    parser.add_argument('--routing-gate-time', type=float, default=0.0)
    parser.add_argument('--target-gate-time', type=float, default=0.0)
    parser.add_argument('--local-gate-time', type=float, default=0.0)
    parser.add_argument('--t1-levels', default='inf')
    parser.add_argument('--tphi-levels', default='inf')
    parser.add_argument('--leakage-epsilon', type=float, default=0.0)
    parser.add_argument('--monte-carlo-trajectories', type=int, default=64)
    args = parser.parse_args()
    if verification_common__launched_without_cli():
        args.max_qubits = verification_common__prompt_int('Up to how many logical qubits do you want to test?', args.max_qubits)
    t1_levels = [math.inf] if args.t1_levels.strip().lower() == 'inf' else verification_common__parse_float_list(args.t1_levels)
    tphi_levels = [math.inf] if args.tphi_levels.strip().lower() == 'inf' else verification_common__parse_float_list(args.tphi_levels)
    ctx = verification_common__setup_run_context(script_file)
    dimension_values = qft_qudit_verification__parse_positive_int_values(args.dimension_sweep) if args.dimension_sweep.strip() else [int(args.dimension)]
    bus_capacity_values = qft_qudit_verification__parse_positive_int_values(args.bus_capacities) if args.bus_capacities.strip() else []
    seed_values = qft_qudit_verification__build_seed_values(args.seed, args.seed_count)
    workers, native_threads_per_worker = verification_common__resolve_parallelism(args.workers, args.native_threads_per_worker)
    tasks = [{'run_seed': int(run_seed), 'n_qubits': int(n_qubits), 'configured_dimension': int(configured_dimension), 'random_samples': int(args.random_samples), 'optimization_level': int(args.optimization_level), 'routing_gate_time': float(args.routing_gate_time), 'target_gate_time': float(args.target_gate_time), 'local_gate_time': float(args.local_gate_time), 't1_levels': [float(value) for value in t1_levels], 'tphi_levels': [float(value) for value in tphi_levels], 'leakage_epsilon': float(args.leakage_epsilon), 'monte_carlo_trajectories': int(args.monte_carlo_trajectories), 'bus_capacity_values': list(bus_capacity_values)} for run_seed in seed_values for n_qubits in range(args.min_qubits, args.max_qubits + 1) for configured_dimension in dimension_values]
    print(f'[qft_qutip] parallel workers={workers} native_threads_per_worker={native_threads_per_worker} tasks={len(tasks)}')
    rows = [row for task_rows in verification_common__parallel_map(qft_qudit_verification__evaluate_qutip_configuration, tasks, workers=workers, native_threads_per_worker=native_threads_per_worker) for row in task_rows]
    print(f'[qft_qutip] parallel section complete rows={len(rows)}')
    rows = qft_qudit_verification__annotate_rows(rows)
    verification_common__write_csv(ctx.result_dir / 'qft_qutip.csv', rows)
    qft_qudit_verification__make_plot(rows, ctx.result_dir / 'qft_qutip.png', 'QFT routing depth comparison (QuTiP)')
    qft_qudit_verification__write_analysis_bundle(rows, ctx.result_dir, suite_stem='qft_qutip', title_prefix='QFT (QuTiP)')
    verification_common__write_json(ctx.result_dir / 'qft_qutip.json', {'suite': 'qft_qutip', 'focus': 'swap_vs_serialized_vs_parallel_routing', 'configured_dimension': args.dimension, 'rows': rows})
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

main_cirq = qft_qudit_verification__main_cirq
main_qutip = qft_qudit_verification__main_qutip
main_aer = qft_verification__main_aer
main_theory = qft_verification__main_theory
SuiteStep = merged_suite_runner__SuiteStep
parse_include_list = merged_suite_runner__parse_include_list
run_suite_steps = merged_suite_runner__run_suite_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full QFT validation stack in one command.")
    parser.add_argument("--include", default="all", help="Comma-separated backends: theory,aer_cpu,aer_gpu,cirq,qutip or all.")
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=6)
    parser.add_argument("--random-samples", type=int, default=3)
    parser.add_argument("--dimension", type=int, default=4, help="Configured local qudit dimension.")
    parser.add_argument("--dimension-sweep", default="", help="Optional comma-separated dimension sweep for qudit backends.")
    parser.add_argument("--bus-capacities", default="", help="Optional comma-separated routed-parallel bus capacities for qudit ablations.")
    parser.add_argument("--topologies", default="alltoall,line,ring,grid")
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--seed-count", type=int, default=1, help="Repeat qudit runs with deterministic seed offsets for error bars.")
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
    theory_argv = ["--min-qubits", str(args.min_qubits), "--max-qubits", str(args.max_qubits), "--random-samples", str(args.random_samples), "--seed", str(args.seed), "--workers", str(args.workers)]
    if args.native_threads_per_worker is not None:
        theory_argv.extend(["--native-threads-per-worker", str(args.native_threads_per_worker)])
    aer_argv = list(theory_argv)
    aer_argv.extend(["--topologies", args.topologies, "--optimization-level", str(args.optimization_level)])
    cirq_argv = ["--min-qubits", str(args.min_qubits), "--max-qubits", str(args.max_qubits), "--random-samples", str(args.random_samples), "--dimension", str(args.dimension), "--optimization-level", str(args.optimization_level), "--seed", str(args.seed), "--seed-count", str(args.seed_count), "--workers", str(args.workers)]
    if args.native_threads_per_worker is not None:
        cirq_argv.extend(["--native-threads-per-worker", str(args.native_threads_per_worker)])
    if args.dimension_sweep.strip():
        cirq_argv.extend(["--dimension-sweep", args.dimension_sweep])
    if args.bus_capacities.strip():
        cirq_argv.extend(["--bus-capacities", args.bus_capacities])
    qutip_argv = list(cirq_argv)
    qutip_argv.extend(["--routing-gate-time", str(args.routing_gate_time), "--target-gate-time", str(args.target_gate_time), "--local-gate-time", str(args.local_gate_time), "--t1-levels", args.t1_levels, "--tphi-levels", args.tphi_levels, "--leakage-epsilon", str(args.leakage_epsilon), "--monte-carlo-trajectories", str(args.monte_carlo_trajectories)])
    step_map = {
        "theory": SuiteStep(name="theory", label="Qubit theory baseline", script_name="2_qft_theory.py", func=main_theory, argv=tuple(theory_argv)),
        "aer_cpu": SuiteStep(name="aer_cpu", label="Qubit Aer CPU baseline", script_name="2_qft_aer_cpu.py", func=lambda script_file: main_aer(script_file, device="CPU"), argv=tuple(aer_argv)),
        "aer_gpu": SuiteStep(name="aer_gpu", label="Qubit Aer GPU baseline", script_name="2_qft_aer_gpu.py", func=lambda script_file: main_aer(script_file, device="GPU"), argv=tuple(aer_argv)),
        "cirq": SuiteStep(name="cirq", label="Ideal qudit routed validation", script_name="2_qft_cirq.py", func=main_cirq, argv=tuple(cirq_argv)),
        "qutip": SuiteStep(name="qutip", label="Noisy qudit routed validation", script_name="2_qft_qutip.py", func=main_qutip, argv=tuple(qutip_argv)),
    }
    ordered_steps = [step_map[name] for name in include]
    result_dir = run_suite_steps(merged_script_file=__file__, suite_name="qft_all", steps=ordered_steps, stop_on_error=args.stop_on_error)
    print("Merged QFT run includes:")
    for step in ordered_steps:
        print(f"  - {step.name}: {step.label}")
    print(f"Merged QFT summary: {result_dir}")


if __name__ == "__main__":
    main()
