import argparse
import sys

import numpy as np

sys.path[:0] = ["src", "tools"]

from atalla.tensors import bf16_round
from tools.run_sparsity_compare import (
    SystolicArrayMEISSA,
    SystolicArrayMEISSASparse24,
    _gemm,
    prune_2to4,
)

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Run every row and column")
args = parser.parse_args()

for name in ("resnet50_conv1", "llama3_q_proj", "llama3_gate_proj"):
    a = np.load(f"benchmarks/{name}_activations.npy", mmap_mode="r")
    w = np.load(f"benchmarks/{name}_weights_dense.npy", mmap_mode="r")
    if not args.full:
        a, w = a[:8], w[:, :16]

    print(f"\n{name}: M={a.shape[0]}, N={w.shape[1]}, K={a.shape[1]}", flush=True)

    for label, array, packed in [
        ("Dense MEISSA", SystolicArrayMEISSA, False),
        ("Sparse MEISSA", SystolicArrayMEISSASparse24, True),
    ]:
        total_cycles = 0
        max_error = 0.0
        for col in range(0, w.shape[1], 16):
            weights = bf16_round(prune_2to4(w[:, col:col + 16]))
            for row in range(0, a.shape[0], 32):
                activations = bf16_round(a[row:row + 32])
                m, k = activations.shape
                result, cycles, _ = _gemm(array, activations, weights, m, k, 16, packed)
                total_cycles += cycles
                max_error = max(max_error, float(np.max(np.abs(result - activations @ weights))))

        print(f"  {label}: {total_cycles:,} array cycles, max abs error {max_error:.5f}")