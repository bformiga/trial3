"""Run generated GEMM inputs through the Atalla-Sim tiled hardware platform."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atalla.sysarr_tpu_system import VLSFrontendBridge, _decode_lanes_u16, build_tpu_platform
from atalla.tensors import bf16_round, from_bf16_bits, to_bytes
from base.clocked_object import Clocked
from systolic_array.systolic_array_meissa import SystolicArrayMEISSA
from systolic_array.systolic_array_meissa_sparse import SystolicArrayMEISSASparse24, prune_2to4

SIZE = 16
VECTOR_LEN = 32
OUTPUT_REG = 128
ACCUM_REG = OUTPUT_REG + VECTOR_LEN
WORKLOADS = ("resnet50_conv1", "llama3_q_proj", "llama3_gate_proj")


class BF16LoadBridge(VLSFrontendBridge):
    """Convert the scratchpad's BF16 storage bits to Vector Core values."""

    def _load_values(self, lanes):
        return from_bf16_bits(_decode_lanes_u16(lanes, self.vc.vector_len)).tolist()


class MEISSABridge(Clocked):
    """Stream columns and activation rows between the GSAU and MEISSA."""

    def __init__(self, vc, array, sparse):
        super().__init__()
        self.vc = vc
        self.array = array
        self.sparse = sparse
        self.activation_width = SIZE * (2 if sparse else 1)
        self.weight_columns = 0
        self.output_index = 0

    def tick(self, time=None):
        req = self.vc.gsau.to_systolic.peek()
        weight = req is not None and req["is_weight"]
        self.array.set_control(weight_en=weight, mac_shift=not weight, start=not weight)
        if weight:
            values = req["vdata"]
            if self.sparse:
                # Four 2-bit indices in each BF16 lane, as an exact integer.
                indices = [(int(values[SIZE + i // 4]) >> (2 * (i % 4))) & 3
                           for i in range(SIZE)]
                accepted = self.array.enqueue_packed_weights(values[:SIZE], indices)
            else:
                accepted = self.array.enqueue_weights(values[:SIZE])
            if accepted:
                self.vc.pop_systolic_request()
                self.weight_columns += 1
        elif req is not None and self.array.enqueue(req["vdata"][:self.activation_width]):
            self.vc.pop_systolic_request()

        self.array.tick(time)
        rows = self.array.get_buffer()
        if self.output_index < len(rows):
            row = rows[self.output_index]
            if self.vc.push_systolic_response(
                {"vdata": row + [0.0] * (self.vc.vector_len - SIZE), "dtype": "bf16"}
            ):
                self.output_index += 1


def compute_path(vc, size, dtype, mirror, systolic_array):
    sparse = systolic_array == "meissa_sparse"
    cls = SystolicArrayMEISSASparse24 if sparse else SystolicArrayMEISSA
    array = cls(size=SIZE, dtype=dtype)
    return array, MEISSABridge(vc, array, sparse)


def weight_stream(weights, sparse):
    if not sparse:
        return weights.T[::-1]
    values, indices = SystolicArrayMEISSASparse24.pack_2to4(weights)
    groups = indices.reshape(SIZE // 4, 4, SIZE)
    selectors = np.sum(groups << (2 * np.arange(4)[None, :, None]), axis=1)
    return np.concatenate((values, selectors), axis=0).T[::-1]


@dataclass
class Pass:
    slot: int
    activations: np.ndarray
    weights: np.ndarray
    dma_done: set = field(default_factory=set)
    ready: dict = field(default_factory=lambda: {"act": set(), "weight": set()})
    loads_sent: dict = field(default_factory=lambda: {"act": 0, "weight": 0})

    def pad(self, kind):
        return 2 * self.slot + int(kind == "weight")

    def reg(self, kind, row):
        return self.pad(kind) * VECTOR_LEN + row


def prefetch(platform, job):
    for kind, values in (("act", job.activations), ("weight", job.weights)):
        pad = job.pad(kind)
        addr = 0x1000 * (pad + 1)
        platform.dram.write(addr, to_bytes(values, "bf16"))
        tx = platform.backends[pad].driver_to_backend_start_load(
            base_sp_addr=0, base_dram_addr=addr, rows=len(values), cols=values.shape[1],
            callback=lambda _tx, j=job, k=kind: j.dma_done.add(k),
        )
        if tx < 0:
            raise RuntimeError("could not enqueue tile DMA")


def issue_loads(vc, job):
    for kind, values in (("act", job.activations), ("weight", job.weights)):
        row = job.loads_sent[kind]
        if kind in job.dma_done and row < len(values) and vc.enqueue_memory({
            "kind": "load", "vls": job.pad(kind), "dst": job.reg(kind, row),
            "addr": row, "vl": values.shape[1], "dtype": "bf16",
            "meta": {"slot": job.slot, "kind": kind, "row": row},
        }):
            job.loads_sent[kind] += 1


def is_busy(vlsu):
    return vlsu.outstanding_loads() or any(
        not queue.is_empty()
        for queue in (vlsu.issue_q, vlsu.req_q, vlsu.rsp_q, vlsu.wb_q)
    )


def simulate(activations, weights, sparse, dram_latency, max_cycles):
    """One output tile, from initial DRAM staging to final partial-row writeback."""
    platform = build_tpu_platform(
        size=VECTOR_LEN, dtype="bf16", lane_count=16,
        backend_dram_latency=dram_latency,
        vls_bridge_cls=BF16LoadBridge, compute_path_builder=compute_path,
        systolic_array="meissa_sparse" if sparse else "meissa",
    )
    vc, bridge = platform.vc, platform.sysarr_bridge
    stride = SIZE * (2 if sparse else 1)
    m, k = activations.shape
    jobs = [
        Pass(i % 2, activations[:, start:start + stride],
             weight_stream(weights[start:start + stride], sparse))
        for i, start in enumerate(range(0, k, stride))
    ]
    for job in jobs[:2]:
        prefetch(platform, job)

    cycles = 0
    waits = Counter()
    histogram = Counter()
    busy_cycles = Counter()
    result = np.zeros((m, SIZE), dtype=np.float32)
    for row in range(m):
        vc.write_vreg(ACCUM_REG + row, [0.0] * VECTOR_LEN)
    for index, job in enumerate(jobs):
        weight_sent = activation_sent = output_rows = adds_sent = adds_done = 0
        weight_goal = (index + 1) * SIZE
        while adds_done < m:
            if cycles >= max_cycles:
                raise RuntimeError(
                    f"cycle limit at pass {index}: {output_rows} output rows, "
                    f"{adds_sent} adds issued, {adds_done} written back"
                )

            for prefetch_job in jobs[index:index + 2]:
                issue_loads(vc, prefetch_job)

            if activation_sent < m:
                if bridge.weight_columns < weight_goal:
                    if activation_sent in job.ready["act"]:
                        stage = ("weight_dma" if "weight" not in job.dma_done
                                 else "weight_vlsu" if len(job.ready["weight"]) < SIZE
                                 else "weight_array_load")
                        waits[("initial_" if index == 0 else "") + stage] += 1
                    else:
                        waits["both_operands_pending_cycles"] += 1
                    if weight_sent in job.ready["weight"] and vc.enqueue_scheduler_instruction({
                        "unit": "gsau", "src": job.reg("weight", weight_sent), "is_weight": True,
                    }):
                        weight_sent += 1
                elif activation_sent in job.ready["act"]:
                    if vc.enqueue_scheduler_instruction({
                        "unit": "gsau", "src": job.reg("act", activation_sent),
                        "dst": OUTPUT_REG + activation_sent,
                    }):
                        activation_sent += 1
                else:
                    waits["activation_wait_cycles"] += 1

            # The Vector Core has one datapath writeback hold entry. Keep one
            # add outstanding while prefetched VLSU results compete for writeback.
            can_accumulate = adds_sent == adds_done and adds_sent < output_rows
            if can_accumulate and vc.enqueue_scheduler_instruction({
                "unit": "datapath", "op": "add", "dst": ACCUM_REG + adds_sent,
                "src0": ACCUM_REG + adds_sent, "src1": OUTPUT_REG + adds_sent,
            }):
                adds_sent += 1

            platform.tick(cycles)
            cycles += 1
            busy = [i for i, unit in enumerate(vc.vls_units) if is_busy(unit)]
            histogram[len(busy)] += 1
            busy_cycles.update(busy)

            wb = vc.last_wb if vc.wb_valid else None
            if wb is None:
                continue
            if wb["source"] == "vlsu":
                meta = wb["meta"]
                target = job if meta["slot"] == job.slot else jobs[index + 1]
                target.ready[meta["kind"]].add(meta["row"])
            elif wb["source"] == "gsau":
                output_rows += 1
            elif wb["source"] == "datapath":
                result[wb["dst"] - ACCUM_REG] = wb["data"][:SIZE]
                adds_done += 1

        if index + 2 < len(jobs):
            prefetch(platform, jobs[index + 2])

    stages = ("weight_dma", "weight_vlsu", "weight_array_load")
    report = {
        "cycles": cycles,
        "weight_stall_cycles": sum(waits[s] for s in stages),
        "weight_stall_by_stage": {s: waits[s] for s in stages},
        "initial_weight_wait_cycles": sum(waits["initial_" + s] for s in stages),
        "both_operands_pending_cycles": waits["both_operands_pending_cycles"],
        "activation_wait_cycles": waits["activation_wait_cycles"],
        "weight_tile_loads": len(jobs),
        "weight_payload_bytes": sum(2 * j.weights.size for j in jobs),
        "vlsu_active_count_cycles": {str(i): histogram[i] for i in range(5)},
        "vlsu_busy_cycles": {f"vlsu{i}": busy_cycles[i] for i in range(4)},
    }
    return result, report


def add_counters(total, report):
    for key, value in report.items():
        if isinstance(value, dict):
            add_counters(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, default=WORKLOADS[0])
    parser.add_argument("--inputs", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dram-latency", type=int, default=6)
    parser.add_argument("--max-cycles-per-tile", type=int, default=1_000_000)
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args()

    a = np.load(args.inputs / f"{args.workload}_activations.npy", mmap_mode="r")
    w = np.load(args.inputs / f"{args.workload}_weights_dense.npy", mmap_mode="r")
    m, k = a.shape
    n = w.shape[1]
    if not args.full:
        m, n = min(m, args.rows), min(n, args.cols)
    if w.shape[0] != k or m < 1 or n < 1 or n % SIZE or k % (2 * SIZE):
        parser.error("expected matching K dimensions divisible by 32, positive M, and N divisible by 16")
    if args.dram_latency < 1 or args.max_cycles_per_tile < 1:
        parser.error("latency and cycle limit must be positive")

    report = {
        "workload": args.workload, "shape_MNK": [m, n, k],
        "array_size": SIZE, "dram_latency": args.dram_latency,
        "timing_scope": "sum_of_independent_output_tiles_from_dram_to_accumulator_writeback",
        "accumulation": "bf16_vector_datapath",
    }
    for mode in ("dense", "sparse"):
        stats = {}
        max_error = 0.0
        for col in range(0, n, SIZE):
            weights = bf16_round(prune_2to4(w[:, col:col + SIZE]))
            for row in range(0, m, VECTOR_LEN):
                activations = bf16_round(a[row:min(row + VECTOR_LEN, m)])
                got, counts = simulate(activations, weights, mode == "sparse",
                                       args.dram_latency, args.max_cycles_per_tile)
                stride = SIZE * (2 if mode == "sparse" else 1)
                expected = np.zeros_like(got)
                for i in range(0, k, stride):
                    partial = bf16_round(activations[:, i:i + stride] @ weights[i:i + stride])
                    expected = bf16_round(expected + partial)
                error = float(np.max(np.abs(got - expected)))
                if not np.allclose(got, expected, atol=0.01, rtol=0.001):
                    raise RuntimeError(f"{mode} result mismatch at tile ({row}, {col}): max error {error}")
                max_error = max(max_error, error)
                add_counters(stats, counts)
        report[mode] = {**stats, "max_abs_error_vs_pass_reference": max_error}

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
