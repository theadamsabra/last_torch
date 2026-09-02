"""GPU vs CPU benchmark: JAX (last) vs PyTorch (last_torch).

Runs the forward pass and gradient computation on both CPU and GPU,
across increasing sequence lengths with batch=16 and feature_size=80.
This is the most representative training-time workload.

Run:
    .venv/bin/python3 -m benchmarks.bench_gpu
"""

import os
import sys
import statistics
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../last'))

import jax
import jax.numpy as jnp
import numpy as np
import torch

import last
import last_torch
from benchmarks.utils import save_csv, to_markdown_table

NUM_FRAMES_LIST = [10, 50, 100, 200, 500]
BATCH_SIZE = 16
FEATURE_SIZE = 80
VOCAB_SIZE = 4
CONTEXT_SIZE = 1

WARMUP = 3   # more warmup for GPU to stabilize
REPEAT = 5
NUMBER = 5

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'gpu.csv')
HEADERS = ['framework', 'device', 'num_frames', 'pass', 'wall_ms']


def time_torch(fn, warmup=WARMUP, repeat=REPEAT, number=NUMBER):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - t0) / number * 1000)
    return statistics.median(times)


def time_jax(fn, warmup=WARMUP, repeat=REPEAT, number=NUMBER):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) / number * 1000)
    return statistics.median(times)


def build_torch_lattice(device_str: str):
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE, device=device_str),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=lambda ctx: last_torch.weight_fns.SharedRNNCacher(
            vocab_size=ctx.vocab_size, context_size=ctx.context_size,
            rnn_size=24, rnn_embedding_size=24, device=device_str),
        weight_fn_factory=lambda ctx: last_torch.weight_fns.JointWeightFn(
            vocab_size=ctx.shape()[1], hidden_size=16, device=device_str),
        device=device_str,
    )
    return lattice


def build_jax_lattice():
    return last.RecognitionLattice(
        context=last.contexts.FullNGram(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=lambda ctx: last.weight_fns.SharedRNNCacher(
            vocab_size=ctx.vocab_size, context_size=ctx.context_size,
            rnn_size=24, rnn_embedding_size=24),
        weight_fn_factory=lambda ctx: last.weight_fns.JointWeightFn(
            vocab_size=ctx.shape()[1], hidden_size=16),
    )


def init_jax_params(lattice, max_frames, device_ctx=None):
    rng = np.random.default_rng(0)
    frames = jnp.array(rng.standard_normal([BATCH_SIZE, max_frames, FEATURE_SIZE]).astype(np.float32))
    nf = jnp.full([BATCH_SIZE], max_frames, dtype=jnp.int32)
    labels = jnp.ones([BATCH_SIZE, max_frames], dtype=jnp.int32)
    nlab = jnp.full([BATCH_SIZE], max_frames, dtype=jnp.int32)
    if device_ctx is not None:
        with device_ctx:
            _, params = lattice.init_with_output(
                jax.random.PRNGKey(0), frames=frames, num_frames=nf,
                labels=labels, num_labels=nlab)
    else:
        _, params = lattice.init_with_output(
            jax.random.PRNGKey(0), frames=frames, num_frames=nf,
            labels=labels, num_labels=nlab)
    return params


def warmup_torch_lattice(lattice, device_str, max_frames):
    rng = np.random.default_rng(0)
    frames = torch.tensor(
        rng.standard_normal([BATCH_SIZE, max_frames, FEATURE_SIZE]).astype(np.float32),
        device=device_str)
    nf = torch.full([BATCH_SIZE], max_frames, dtype=torch.float32, device=device_str)
    labels = torch.ones([BATCH_SIZE, max_frames], dtype=torch.float32, device=device_str)
    nlab = torch.full([BATCH_SIZE], max_frames, dtype=torch.float32, device=device_str)
    with torch.no_grad():
        lattice(frames=frames, num_frames=nf, labels=labels, num_labels=nlab)


def bench_config(framework: str, device_str: str, lattice, params_or_none,
                 num_frames: int, rows: list):
    rng = np.random.default_rng(num_frames * 13)
    frames_np = rng.standard_normal([BATCH_SIZE, num_frames, FEATURE_SIZE]).astype(np.float32)

    if framework == 'jax':
        jax_dev = jax.devices('gpu')[0] if device_str == 'cuda' else jax.devices('cpu')[0]
        with jax.default_device(jax_dev):
            frames_j = jnp.array(frames_np)
            nf_j = jnp.full([BATCH_SIZE], num_frames, dtype=jnp.int32)
            labels_j = jnp.ones([BATCH_SIZE, num_frames], dtype=jnp.int32)
            nlab_j = jnp.full([BATCH_SIZE], num_frames, dtype=jnp.int32)

        @jax.jit
        def fwd(p, f, nf, lab, nlab):
            return lattice.apply(p, frames=f, num_frames=nf, labels=lab, num_labels=nlab)

        @jax.jit
        def fwd_grad(p, f, nf, lab, nlab):
            def loss_fn(ff):
                return lattice.apply(p, frames=ff, num_frames=nf, labels=lab, num_labels=nlab).sum()
            return jax.value_and_grad(loss_fn)(f)

        # Warmup inside device context.
        with jax.default_device(jax_dev):
            jax.block_until_ready(fwd(params_or_none, frames_j, nf_j, labels_j, nlab_j))
            jax.block_until_ready(fwd_grad(params_or_none, frames_j, nf_j, labels_j, nlab_j))

        fwd_ms = time_jax(lambda: fwd(params_or_none, frames_j, nf_j, labels_j, nlab_j))
        grad_ms = time_jax(lambda: fwd_grad(params_or_none, frames_j, nf_j, labels_j, nlab_j))

    else:  # torch
        dev = torch.device(device_str)
        frames_t = torch.tensor(frames_np, device=dev)
        nf_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32, device=dev)
        labels_t = torch.ones([BATCH_SIZE, num_frames], dtype=torch.float32, device=dev)
        nlab_t = torch.full([BATCH_SIZE], num_frames, dtype=torch.float32, device=dev)

        with torch.no_grad():
            fwd_ms = time_torch(
                lambda: lattice(frames=frames_t, num_frames=nf_t, labels=labels_t, num_labels=nlab_t))

        def _grad():
            ft = frames_t.detach().requires_grad_(True)
            loss = lattice(frames=ft, num_frames=nf_t, labels=labels_t, num_labels=nlab_t)
            loss.sum().backward()

        grad_ms = time_torch(_grad)

    device_label = 'gpu' if device_str in ('cuda', 'cuda:0', 'cuda') else 'cpu'
    print(f'  {framework:5s} {device_label:3s} T={num_frames:4d} | fwd={fwd_ms:.2f}ms grad={grad_ms:.2f}ms')

    rows.append({'framework': framework, 'device': device_label, 'num_frames': num_frames,
                 'pass': 'forward', 'wall_ms': f'{fwd_ms:.3f}'})
    rows.append({'framework': framework, 'device': device_label, 'num_frames': num_frames,
                 'pass': 'fwd+grad', 'wall_ms': f'{grad_ms:.3f}'})


def main():
    rows = []
    max_frames = max(NUM_FRAMES_LIST)

    has_gpu = torch.cuda.is_available() and len(jax.devices('gpu')) > 0

    # ---- JAX CPU ----
    print('\n[JAX CPU]')
    with jax.default_device(jax.devices('cpu')[0]):
        jax_lat_cpu = build_jax_lattice()
        jax_params_cpu = init_jax_params(jax_lat_cpu, max_frames)
    for T in NUM_FRAMES_LIST:
        bench_config('jax', 'cpu', jax_lat_cpu, jax_params_cpu, T, rows)

    # ---- JAX GPU ----
    if has_gpu:
        print('\n[JAX GPU]')
        gpu_dev = jax.devices('gpu')[0]
        with jax.default_device(gpu_dev):
            jax_lat_gpu = build_jax_lattice()
            jax_params_gpu = init_jax_params(jax_lat_gpu, max_frames, jax.default_device(gpu_dev))
        for T in NUM_FRAMES_LIST:
            bench_config('jax', 'cuda', jax_lat_gpu, jax_params_gpu, T, rows)

    # ---- Torch CPU ----
    print('\n[Torch CPU]')
    torch_lat_cpu = build_torch_lattice('cpu')
    warmup_torch_lattice(torch_lat_cpu, 'cpu', max_frames)
    for T in NUM_FRAMES_LIST:
        bench_config('torch', 'cpu', torch_lat_cpu, None, T, rows)

    # ---- Torch GPU ----
    if has_gpu:
        print('\n[Torch GPU]')
        torch_lat_gpu = build_torch_lattice('cuda')
        warmup_torch_lattice(torch_lat_gpu, 'cuda', max_frames)
        for T in NUM_FRAMES_LIST:
            bench_config('torch', 'cuda', torch_lat_gpu, None, T, rows)

    save_csv(CSV_PATH, rows, HEADERS)
    print('\n' + to_markdown_table(rows, HEADERS))

    # Print speedup summary.
    print('\n=== Speedup Summary (GPU vs CPU) ===')
    for fw in ('jax', 'torch'):
        for p in ('forward', 'fwd+grad'):
            print(f'\n{fw} {p}:')
            for T in NUM_FRAMES_LIST:
                cpu_row = next((r for r in rows if r['framework'] == fw and
                                r['device'] == 'cpu' and int(r['num_frames']) == T and
                                r['pass'] == p), None)
                gpu_row = next((r for r in rows if r['framework'] == fw and
                                r['device'] == 'gpu' and int(r['num_frames']) == T and
                                r['pass'] == p), None)
                if cpu_row and gpu_row:
                    ratio = float(cpu_row['wall_ms']) / float(gpu_row['wall_ms'])
                    print(f'  T={T:4d}: CPU={cpu_row["wall_ms"]}ms GPU={gpu_row["wall_ms"]}ms  ({ratio:.1f}x speedup)')


if __name__ == '__main__':
    main()
