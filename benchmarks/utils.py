"""Shared utilities for LAST benchmarks."""

import os
import statistics
import time
import tracemalloc
from typing import Callable

import last_torch
import numpy as np
import torch

# jax/last are imported lazily by the helpers that need them, so this module
# stays importable in a torch-only venv (see bench_scan_impl.py).


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_fn(fn: Callable, warmup: int = 1, repeat: int = 3, number: int = 10) -> float:
    """Returns median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number * 1000)
    return statistics.median(times)


def time_fn_jax(fn: Callable, warmup: int = 1, repeat: int = 3, number: int = 10) -> float:
    """Returns median wall-clock time in milliseconds, blocking until JAX computation finishes."""
    import jax

    def _run():
        result = fn()
        jax.block_until_ready(result)

    for _ in range(warmup):
        _run()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            _run()
        times.append((time.perf_counter() - t0) / number * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def peak_memory_mb(fn: Callable) -> float:
    """Returns peak memory in MB using tracemalloc."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


# ---------------------------------------------------------------------------
# Lattice factories
# ---------------------------------------------------------------------------

def build_torch_table_lattice(vocab_size: int, context_size: int,
                               batch_size: int, max_num_frames: int,
                               rng: np.random.Generator):
    """Builds a PyTorch RecognitionLattice with TableWeightFn + NullCacher."""
    input_vocab_size = max_num_frames  # frames contain integer labels 0..T-1
    num_context_states = sum(
        vocab_size ** i for i in range(context_size + 1))
    table = rng.standard_normal(
        [batch_size, input_vocab_size, num_context_states, 1 + vocab_size]
    ).astype(np.float32)
    table_t = torch.tensor(table)

    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last_torch.weight_fns.TableWeightFn(table_t),
        weight_fn_cacher_factory=lambda _: last_torch.weight_fns.NullCacher(),
    )
    return lattice, table


def build_jax_table_lattice(vocab_size: int, context_size: int,
                             batch_size: int, max_num_frames: int,
                             table: np.ndarray):
    """Builds a JAX RecognitionLattice with TableWeightFn + NullCacher using an existing table."""
    import jax.numpy as jnp
    import last
    table_j = jnp.array(table)

    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last.weight_fns.TableWeightFn(table=table_j),
        weight_fn_cacher_factory=lambda _: last.weight_fns.NullCacher(),
    )
    return lattice


def build_torch_rnn_lattice(vocab_size: int, context_size: int):
    """Builds a PyTorch RecognitionLattice with JointWeightFn + SharedRNNCacher."""
    context = last_torch.contexts.FullNGram(
        vocab_size=vocab_size, context_size=context_size)

    def cacher_factory(_ctx):
        return last_torch.weight_fns.SharedRNNCacher(
            vocab_size=vocab_size, context_size=context_size,
            rnn_size=24, rnn_embedding_size=24)

    def weight_fn_factory(_ctx):
        _, vs = _ctx.shape()
        return last_torch.weight_fns.JointWeightFn(vocab_size=vs, hidden_size=16)

    lattice = last_torch.RecognitionLattice(
        context=context,
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=cacher_factory,
        weight_fn_factory=weight_fn_factory,
    )
    return lattice


def build_jax_rnn_lattice(vocab_size: int, context_size: int):
    """Builds a JAX RecognitionLattice with JointWeightFn + SharedRNNCacher."""
    import last
    context = last.contexts.FullNGram(
        vocab_size=vocab_size, context_size=context_size)

    def cacher_factory(_ctx):
        return last.weight_fns.SharedRNNCacher(
            vocab_size=vocab_size, context_size=context_size,
            rnn_size=24, rnn_embedding_size=24)

    def weight_fn_factory(_ctx):
        _, vs = _ctx.shape()
        return last.weight_fns.JointWeightFn(vocab_size=vs, hidden_size=16)

    lattice = last.RecognitionLattice(
        context=context,
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=cacher_factory,
        weight_fn_factory=weight_fn_factory,
    )
    return lattice


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def to_markdown_table(rows: list[dict], headers: list[str]) -> str:
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))
    sep = '| ' + ' | '.join('-' * col_widths[h] for h in headers) + ' |'
    header = '| ' + ' | '.join(h.ljust(col_widths[h]) for h in headers) + ' |'
    lines = [header, sep]
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers) + ' |')
    return '\n'.join(lines)


def save_csv(path: str, rows: list[dict], headers: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join(str(row.get(h, '')) for h in headers) + '\n')
    print(f'Saved {path}')
