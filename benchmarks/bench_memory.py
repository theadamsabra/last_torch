"""Benchmark peak memory during forward+backward at increasing sequence lengths.

Shows how PyTorch's Python for-loop stores all scan intermediates (O(T) memory)
vs JAX's nn.remat policy which recomputes some activations to reduce memory.

Run:
    python benchmarks/bench_memory.py
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
from benchmarks.utils import peak_memory_mb, save_csv, to_markdown_table

NUM_FRAMES_LIST = [10, 25, 50, 100, 200]
BATCH_SIZE = 4
FEATURE_SIZE = 80
VOCAB_SIZE = 4
CONTEXT_SIZE = 1

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'memory.csv')
HEADERS = ['framework', 'num_frames', 'peak_mb']


def main():
    rows = []
    rng = np.random.default_rng(7)

    # ---- Build JAX lattice ----
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
    frames_init = jnp.array(rng.standard_normal([BATCH_SIZE, max_frames, FEATURE_SIZE]).astype(np.float32))
    nf_init = jnp.full([BATCH_SIZE], max_frames, dtype=jnp.int32)
    labels_init = jnp.ones([BATCH_SIZE, max_frames], dtype=jnp.int32)
    nlab_init = jnp.full([BATCH_SIZE], max_frames, dtype=jnp.int32)
    _, jax_params = jax_lattice.init_with_output(
        jax.random.PRNGKey(0), frames=frames_init, num_frames=nf_init,
        labels=labels_init, num_labels=nlab_init)

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
    frames_t0 = torch.tensor(rng.standard_normal([BATCH_SIZE, max_frames, FEATURE_SIZE]).astype(np.float32))
    nf_t0 = torch.full([BATCH_SIZE], max_frames, dtype=torch.float32)
    lab_t0 = torch.ones([BATCH_SIZE, max_frames], dtype=torch.float32)
    nlab_t0 = torch.full([BATCH_SIZE], max_frames, dtype=torch.float32)
    with torch.no_grad():
        torch_lattice(frames=frames_t0, num_frames=nf_t0, labels=lab_t0, num_labels=nlab_t0)

    for num_frames in NUM_FRAMES_LIST:
        rng2 = np.random.default_rng(num_frames + 100)
        frames_np = rng2.standard_normal([BATCH_SIZE, num_frames, FEATURE_SIZE]).astype(np.float32)

        frames_j = jnp.array(frames_np)
        nf_j = jnp.full([BATCH_SIZE], num_frames, dtype=jnp.int32)
        labels_j = jnp.ones([BATCH_SIZE, num_frames], dtype=jnp.int32)
        nlab_j = jnp.full([BATCH_SIZE], num_frames, dtype=jnp.int32)

        frames_t = torch.tensor(frames_np)
        nf_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32)
        labels_t = torch.ones([BATCH_SIZE, num_frames], dtype=torch.float32)
        nlab_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32)

        def jax_fwd_grad():
            def loss_fn(f):
                return jax_lattice.apply(jax_params, frames=f, num_frames=nf_j,
                                          labels=labels_j, num_labels=nlab_j).sum()
            val, grad = jax.value_and_grad(loss_fn)(frames_j)
            jax.block_until_ready((val, grad))

        def torch_fwd_grad():
            ft = frames_t.detach().requires_grad_(True)
            loss = torch_lattice(frames=ft, num_frames=nf_t, labels=labels_t, num_labels=nlab_t)
            loss.sum().backward()

        j_mb = peak_memory_mb(jax_fwd_grad)
        t_mb = peak_memory_mb(torch_fwd_grad)

        print(f'  num_frames={num_frames:4d}  jax={j_mb:.2f}MB  torch={t_mb:.2f}MB')

        rows.extend([
            {'framework': 'jax', 'num_frames': num_frames, 'peak_mb': f'{j_mb:.3f}'},
            {'framework': 'torch', 'num_frames': num_frames, 'peak_mb': f'{t_mb:.3f}'},
        ])

    save_csv(CSV_PATH, rows, HEADERS)
    print('\n' + to_markdown_table(rows, HEADERS))


if __name__ == '__main__':
    main()
