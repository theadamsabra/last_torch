"""Benchmark the sequential scan (forward algorithm) at increasing sequence lengths.

Measures wall-clock time per frame — this isolates Python for-loop overhead
(PyTorch) vs XLA-compiled scan (JAX). Expected result: PyTorch cost grows
linearly with num_frames; JAX amortizes JIT overhead after warmup.

Run:
    python benchmarks/bench_scan.py
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

import last
import last_torch
from benchmarks.utils import time_fn, time_fn_jax, save_csv, to_markdown_table

NUM_FRAMES_LIST = [5, 10, 25, 50, 100, 200]
BATCH_SIZE = 4
VOCAB_SIZE = 4
CONTEXT_SIZE = 1
FEATURE_SIZE = 8  # small to isolate scan cost

WARMUP = 1
REPEAT = 3
NUMBER = 5

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'scan.csv')
HEADERS = ['framework', 'num_frames', 'total_ms', 'ms_per_frame']


def make_inputs_torch(num_frames: int, rng: np.random.Generator):
    frames = torch.tensor(
        rng.standard_normal([BATCH_SIZE, num_frames, FEATURE_SIZE]).astype(np.float32))
    num_frames_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32)
    return frames, num_frames_t


def make_inputs_jax(num_frames: int, rng: np.random.Generator):
    frames = jnp.array(
        rng.standard_normal([BATCH_SIZE, num_frames, FEATURE_SIZE]).astype(np.float32))
    num_frames_j = jnp.full([BATCH_SIZE], num_frames, dtype=jnp.int32)
    return frames, num_frames_j


def main():
    rng = np.random.default_rng(42)
    rows = []

    # Build JAX lattice + init params once with max-length input.
    max_frames = max(NUM_FRAMES_LIST)
    jax_lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=lambda ctx: last.weight_fns.SharedRNNCacher(
            vocab_size=ctx.vocab_size, context_size=ctx.context_size,
            rnn_size=16, rnn_embedding_size=16),
        weight_fn_factory=lambda ctx: last.weight_fns.JointWeightFn(
            vocab_size=ctx.shape()[1], hidden_size=16),
    )
    frames_init, nf_init = make_inputs_jax(max_frames, rng)
    labels_init = jnp.ones([BATCH_SIZE, max_frames], dtype=jnp.int32)
    nlab_init = jnp.full([BATCH_SIZE], max_frames, dtype=jnp.int32)
    _, jax_params = jax_lattice.init_with_output(
        jax.random.PRNGKey(0), frames=frames_init, num_frames=nf_init,
        labels=labels_init, num_labels=nlab_init)

    # Build PyTorch lattice.
    torch_lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=lambda ctx: last_torch.weight_fns.SharedRNNCacher(
            vocab_size=ctx.vocab_size, context_size=ctx.context_size,
            rnn_size=16, rnn_embedding_size=16),
        weight_fn_factory=lambda ctx: last_torch.weight_fns.JointWeightFn(
            vocab_size=ctx.shape()[1], hidden_size=16),
    )
    # Warm up torch to initialize lazy parameters.
    frames_t0, nf_t0 = make_inputs_torch(max_frames, rng)
    labels_t0 = torch.ones([BATCH_SIZE, max_frames], dtype=torch.float32)
    nlab_t0 = torch.full([BATCH_SIZE], max_frames, dtype=torch.float32)
    with torch.no_grad():
        torch_lattice(frames=frames_t0, num_frames=nf_t0, labels=labels_t0, num_labels=nlab_t0)

    @jax.jit
    def jax_forward(frames, num_frames):
        cache = jax_lattice.apply(jax_params, method=jax_lattice.build_cache)
        labels = jnp.ones([BATCH_SIZE, frames.shape[1]], dtype=jnp.int32)
        num_labels = jnp.full([BATCH_SIZE], frames.shape[1], dtype=jnp.int32)
        return jax_lattice.apply(jax_params, frames=frames, num_frames=num_frames,
                                  labels=labels, num_labels=num_labels, cache=cache)

    for num_frames in NUM_FRAMES_LIST:
        rng2 = np.random.default_rng(num_frames)
        frames_j, nf_j = make_inputs_jax(num_frames, rng2)
        frames_t, nf_t = make_inputs_torch(num_frames, rng2)
        labels_t = torch.ones([BATCH_SIZE, num_frames], dtype=torch.float32)
        nlab_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32)

        # JAX: warm up, then time.
        _ = jax.block_until_ready(jax_forward(frames_j, nf_j))
        j_ms = time_fn_jax(lambda: jax_forward(frames_j, nf_j),
                            warmup=WARMUP, repeat=REPEAT, number=NUMBER)

        # PyTorch.
        with torch.no_grad():
            t_ms = time_fn(lambda: torch_lattice(
                frames=frames_t, num_frames=nf_t, labels=labels_t, num_labels=nlab_t),
                warmup=WARMUP, repeat=REPEAT, number=NUMBER)

        print(f'  num_frames={num_frames:4d}  jax={j_ms:.2f}ms ({j_ms/num_frames:.3f}/frame)'
              f'  torch={t_ms:.2f}ms ({t_ms/num_frames:.3f}/frame)')

        rows.append({'framework': 'jax', 'num_frames': num_frames,
                     'total_ms': f'{j_ms:.4f}', 'ms_per_frame': f'{j_ms/num_frames:.5f}'})
        rows.append({'framework': 'torch', 'num_frames': num_frames,
                     'total_ms': f'{t_ms:.4f}', 'ms_per_frame': f'{t_ms/num_frames:.5f}'})

    save_csv(CSV_PATH, rows, HEADERS)
    print('\n' + to_markdown_table(rows, HEADERS))


if __name__ == '__main__':
    main()
