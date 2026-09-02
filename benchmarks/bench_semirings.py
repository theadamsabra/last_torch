"""Benchmark semiring operations: JAX (last) vs PyTorch (last_torch).

Measures wall-clock time and peak memory for plus, sum, and times across
Real, Log, and MaxTropical semirings at varying tensor sizes.

Run:
    python benchmarks/bench_semirings.py
"""

import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../last'))

import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import numpy as np
import torch

import last.semirings as jsr
import last_torch.semirings as tsr
from benchmarks.utils import time_fn, time_fn_jax, peak_memory_mb, save_csv, to_markdown_table

SIZES = [100, 1_000, 10_000, 100_000]
SEMIRINGS = [
    ('Real',        jsr.Real,        tsr.Real),
    ('Log',         jsr.Log,         tsr.Log),
    ('MaxTropical', jsr.MaxTropical,  tsr.MaxTropical),
]
OPS = ['plus', 'sum', 'times']

WARMUP = 1
REPEAT = 3
NUMBER = 20

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'semirings.csv')
HEADERS = ['semiring', 'op', 'size', 'framework', 'wall_ms', 'peak_mb']


def bench_jax(sr_jax, op: str, size: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(size).astype(np.float32)
    if op == 'plus':
        y_np = rng.standard_normal(size).astype(np.float32)
        x_j, y_j = jnp.array(x_np), jnp.array(y_np)
        fn = jax.jit(lambda a, b: sr_jax.plus(a, b))
        wall = time_fn_jax(lambda: fn(x_j, y_j), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: jax.block_until_ready(fn(x_j, y_j)))
    elif op == 'sum':
        x_j = jnp.array(x_np)
        fn = jax.jit(lambda a: sr_jax.sum(a, axis=0))
        wall = time_fn_jax(lambda: fn(x_j), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: jax.block_until_ready(fn(x_j)))
    else:  # times
        y_np = rng.standard_normal(size).astype(np.float32)
        x_j, y_j = jnp.array(x_np), jnp.array(y_np)
        fn = jax.jit(lambda a, b: sr_jax.times(a, b))
        wall = time_fn_jax(lambda: fn(x_j, y_j), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: jax.block_until_ready(fn(x_j, y_j)))
    return wall, mem


def bench_torch(sr_torch, op: str, size: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(size).astype(np.float32)
    if op == 'plus':
        y_np = rng.standard_normal(size).astype(np.float32)
        x_t, y_t = torch.tensor(x_np), torch.tensor(y_np)
        wall = time_fn(lambda: sr_torch.plus(x_t, y_t), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: sr_torch.plus(x_t, y_t))
    elif op == 'sum':
        x_t = torch.tensor(x_np)
        wall = time_fn(lambda: sr_torch.sum(x_t, dim=0), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: sr_torch.sum(x_t, dim=0))
    else:  # times
        y_np = rng.standard_normal(size).astype(np.float32)
        x_t, y_t = torch.tensor(x_np), torch.tensor(y_np)
        wall = time_fn(lambda: sr_torch.times(x_t, y_t), warmup=WARMUP, repeat=REPEAT, number=NUMBER)
        mem = peak_memory_mb(lambda: sr_torch.times(x_t, y_t))
    return wall, mem


def main():
    rows = []
    for sr_name, sr_jax, sr_torch in SEMIRINGS:
        for op in OPS:
            for size in SIZES:
                print(f'  {sr_name}.{op} size={size}', end=' ', flush=True)
                j_wall, j_mem = bench_jax(sr_jax, op, size)
                t_wall, t_mem = bench_torch(sr_torch, op, size)
                print(f'jax={j_wall:.3f}ms torch={t_wall:.3f}ms')
                rows.append({'semiring': sr_name, 'op': op, 'size': size,
                             'framework': 'jax', 'wall_ms': f'{j_wall:.4f}', 'peak_mb': f'{j_mem:.4f}'})
                rows.append({'semiring': sr_name, 'op': op, 'size': size,
                             'framework': 'torch', 'wall_ms': f'{t_wall:.4f}', 'peak_mb': f'{t_mem:.4f}'})

    save_csv(CSV_PATH, rows, HEADERS)
    print('\n' + to_markdown_table(rows, HEADERS))


if __name__ == '__main__':
    main()
