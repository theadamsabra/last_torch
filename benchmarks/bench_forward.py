"""Benchmark full RecognitionLattice forward pass + gradient.

Measures wall-clock time across varying batch sizes and sequence lengths,
using JointWeightFn + SharedRNNCacher (the realistic training configuration).

Run:
    python benchmarks/bench_forward.py
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

BATCH_SIZES = [1, 4, 16]
NUM_FRAMES_LIST = [10, 50, 100, 200]
FEATURE_SIZE = 80
VOCAB_SIZE = 4
CONTEXT_SIZE = 1

WARMUP = 1
REPEAT = 3
NUMBER = 3

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'forward.csv')
HEADERS = ['batch_size', 'num_frames', 'framework', 'pass', 'wall_ms']


def main():
    rows = []

    for batch_size in BATCH_SIZES:
        # ---- Build JAX lattice + init once at max frames ----
        max_frames = max(NUM_FRAMES_LIST)
        jax_lattice = last.RecognitionLattice(
            context=last.contexts.FullNGram(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE),
            alignment=last.alignments.FrameDependent(),
            weight_fn_cacher_factory=lambda ctx: last.weight_fns.SharedRNNCacher(
                vocab_size=ctx.vocab_size, context_size=ctx.context_size,
                rnn_size=24, rnn_embedding_size=24),
            weight_fn_factory=lambda ctx: last.weight_fns.JointWeightFn(
                vocab_size=ctx.shape()[1], hidden_size=16),
        )
        rng = np.random.default_rng(batch_size)
        frames_init = jnp.array(rng.standard_normal([batch_size, max_frames, FEATURE_SIZE]).astype(np.float32))
        nf_init = jnp.full([batch_size], max_frames, dtype=jnp.int32)
        labels_init = jnp.ones([batch_size, max_frames], dtype=jnp.int32)
        nlab_init = jnp.full([batch_size], max_frames, dtype=jnp.int32)
        _, jax_params = jax_lattice.init_with_output(
            jax.random.PRNGKey(batch_size), frames=frames_init,
            num_frames=nf_init, labels=labels_init, num_labels=nlab_init)

        @jax.jit
        def jax_fwd(params, frames, num_frames, labels, num_labels):
            return jax_lattice.apply(params, frames=frames, num_frames=num_frames,
                                      labels=labels, num_labels=num_labels)

        @jax.jit
        def jax_fwd_grad(params, frames, num_frames, labels, num_labels):
            def loss_fn(f):
                return jax_lattice.apply(params, frames=f, num_frames=num_frames,
                                          labels=labels, num_labels=num_labels).sum()
            return jax.value_and_grad(loss_fn)(frames)

        # ---- Build PyTorch lattice ----
        torch_lattice = last_torch.RecognitionLattice(
            context=last_torch.contexts.FullNGram(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE),
            alignment=last_torch.alignments.FrameDependent(),
            weight_fn_cacher_factory=lambda ctx: last_torch.weight_fns.SharedRNNCacher(
                vocab_size=ctx.vocab_size, context_size=ctx.context_size,
                rnn_size=24, rnn_embedding_size=24),
            weight_fn_factory=lambda ctx: last_torch.weight_fns.JointWeightFn(
                vocab_size=ctx.shape()[1], hidden_size=16),
        )
        # Warm up to initialize lazy parameters.
        frames_t0 = torch.tensor(rng.standard_normal([batch_size, max_frames, FEATURE_SIZE]).astype(np.float32))
        nf_t0 = torch.full([batch_size], max_frames, dtype=torch.float32)
        lab_t0 = torch.ones([batch_size, max_frames], dtype=torch.float32)
        nlab_t0 = torch.full([batch_size], max_frames, dtype=torch.float32)
        with torch.no_grad():
            torch_lattice(frames=frames_t0, num_frames=nf_t0, labels=lab_t0, num_labels=nlab_t0)

        for num_frames in NUM_FRAMES_LIST:
            rng2 = np.random.default_rng(batch_size * 1000 + num_frames)
            frames_np = rng2.standard_normal([batch_size, num_frames, FEATURE_SIZE]).astype(np.float32)

            frames_j = jnp.array(frames_np)
            nf_j = jnp.full([batch_size], num_frames, dtype=jnp.int32)
            labels_j = jnp.ones([batch_size, num_frames], dtype=jnp.int32)
            nlab_j = jnp.full([batch_size], num_frames, dtype=jnp.int32)

            frames_t = torch.tensor(frames_np)
            nf_t = torch.full([batch_size], num_frames, dtype=torch.float32)
            labels_t = torch.ones([batch_size, num_frames], dtype=torch.float32)
            nlab_t = torch.full([batch_size], num_frames, dtype=torch.float32)

            # --- JAX forward only ---
            _ = jax.block_until_ready(jax_fwd(jax_params, frames_j, nf_j, labels_j, nlab_j))
            j_fwd_ms = time_fn_jax(
                lambda: jax_fwd(jax_params, frames_j, nf_j, labels_j, nlab_j),
                warmup=WARMUP, repeat=REPEAT, number=NUMBER)

            # --- JAX forward + grad ---
            _ = jax.block_until_ready(jax_fwd_grad(jax_params, frames_j, nf_j, labels_j, nlab_j))
            j_grad_ms = time_fn_jax(
                lambda: jax_fwd_grad(jax_params, frames_j, nf_j, labels_j, nlab_j),
                warmup=WARMUP, repeat=REPEAT, number=NUMBER)

            # --- Torch forward only ---
            with torch.no_grad():
                t_fwd_ms = time_fn(
                    lambda: torch_lattice(frames=frames_t, num_frames=nf_t,
                                         labels=labels_t, num_labels=nlab_t),
                    warmup=WARMUP, repeat=REPEAT, number=NUMBER)

            # --- Torch forward + grad ---
            def _torch_grad():
                ft = frames_t.detach().requires_grad_(True)
                loss = torch_lattice(frames=ft, num_frames=nf_t, labels=labels_t, num_labels=nlab_t)
                loss.sum().backward()

            t_grad_ms = time_fn(_torch_grad, warmup=WARMUP, repeat=REPEAT, number=NUMBER)

            print(f'  batch={batch_size} frames={num_frames:4d} | '
                  f'jax fwd={j_fwd_ms:.1f}ms grad={j_grad_ms:.1f}ms | '
                  f'torch fwd={t_fwd_ms:.1f}ms grad={t_grad_ms:.1f}ms')

            rows.extend([
                {'batch_size': batch_size, 'num_frames': num_frames,
                 'framework': 'jax', 'pass': 'forward', 'wall_ms': f'{j_fwd_ms:.3f}'},
                {'batch_size': batch_size, 'num_frames': num_frames,
                 'framework': 'jax', 'pass': 'fwd+grad', 'wall_ms': f'{j_grad_ms:.3f}'},
                {'batch_size': batch_size, 'num_frames': num_frames,
                 'framework': 'torch', 'pass': 'forward', 'wall_ms': f'{t_fwd_ms:.3f}'},
                {'batch_size': batch_size, 'num_frames': num_frames,
                 'framework': 'torch', 'pass': 'fwd+grad', 'wall_ms': f'{t_grad_ms:.3f}'},
            ])

    save_csv(CSV_PATH, rows, HEADERS)
    print('\n' + to_markdown_table(rows, HEADERS))


if __name__ == '__main__':
    main()
