"""GPU vs CPU benchmark: JAX (last) vs PyTorch (last_torch).

Runs the forward pass and gradient computation on both CPU and GPU,
across increasing sequence lengths with batch=16 and feature_size=80.
Both frameworks return fresh gradients for frames and all model parameters.
Random labels and variable padding allow multiple alignment paths by default.
CUDA Graph capture is opt-in; graph setup is excluded from replay timings.

Run:
    .venv/bin/python3 -m benchmarks.bench_gpu
"""

import argparse
import json
import hashlib
from pathlib import Path
import platform
import os
import sys
import statistics
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../last'))

# Avoid reserving most GPU memory when comparing frameworks in one process.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

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

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'gpu_current.csv')
HEADERS = ['framework', 'device', 'num_frames', 'max_labels', 'label_ratio',
           'pass', 'wall_ms', 'peak_allocated_mb', 'capture_ms']


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


def make_inputs(num_frames, label_ratio=0.5):
    """Identical float32 frames and integer, variably padded labels in both APIs."""
    rng = np.random.default_rng(num_frames * 13)
    max_labels = max(1, int(num_frames * label_ratio))
    frames = rng.standard_normal((BATCH_SIZE, num_frames, FEATURE_SIZE)).astype(np.float32)
    nf = np.linspace(max(1, num_frames * 3 // 4), num_frames, BATCH_SIZE).astype(np.int32)
    nl = np.minimum((nf * label_ratio).astype(np.int32), max_labels)
    labels = rng.integers(1, VOCAB_SIZE + 1, (BATCH_SIZE, max_labels), dtype=np.int32)
    labels[np.arange(max_labels)[None, :] >= nl[:, None]] = 0
    return frames, nf, labels, nl


def bench_config(framework, device_str, lattice, params_or_none,
                 num_frames, rows, label_ratio=0.5, cuda_graph=False):
    arrays = make_inputs(num_frames, label_ratio)
    peak_mb = ''
    capture_ms = ''
    if framework == 'jax':
        dev = jax.devices('gpu' if device_str == 'cuda' else 'cpu')[0]
        inputs = tuple(jax.device_put(x, dev) for x in arrays)

        @jax.jit
        def fwd(p, f, nf, lab, nl):
            return lattice.apply(p, frames=f, num_frames=nf, labels=lab, num_labels=nl)

        def loss_fn(p, f, nf, lab, nl):
            return lattice.apply(p, frames=f, num_frames=nf, labels=lab, num_labels=nl).sum()

        # Both frameworks return fresh gradients for parameters AND frames.
        fwd_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        fwd_ms = time_jax(lambda: fwd(params_or_none, *inputs))
        grad_ms = time_jax(lambda: fwd_grad(params_or_none, *inputs))
    else:
        inputs = tuple(torch.as_tensor(x, device=device_str) for x in arrays)
        frames, nf, labels, nl = inputs
        # Initialize lazy parameters before collecting differentiation targets.
        with torch.no_grad():
            lattice(*inputs)
        if cuda_graph:
            from last_torch.cuda_graphs import make_graphed_lattice
            frames.requires_grad_(True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            lattice = make_graphed_lattice(lattice, inputs)
            torch.cuda.synchronize()
            capture_ms = round((time.perf_counter() - start) * 1000, 3)
        with torch.no_grad():
            fwd_ms = time_torch(lambda: lattice(*inputs))
        frames.requires_grad_(True)
        targets = (frames, *tuple(lattice.parameters()))

        def grad():
            loss = lattice(frames, nf, labels, nl).sum()
            return loss, torch.autograd.grad(loss, targets)

        grad_ms = time_torch(grad)
        if device_str == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            result = grad()
            torch.cuda.synchronize()
            peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 3)
            del result

    device_label = 'gpu' if device_str == 'cuda' else 'cpu'
    print(f'{framework:5s} {device_label:3s} T={num_frames:4d} '
          f'L={arrays[2].shape[-1]:4d} | fwd={fwd_ms:.3f}ms '
          f'fwd+grad={grad_ms:.3f}ms', flush=True)
    for name, ms in [('forward', fwd_ms), ('fwd+grad', grad_ms)]:
        rows.append(dict(framework=framework, device=device_label,
                         num_frames=num_frames, max_labels=arrays[2].shape[-1],
                         label_ratio=label_ratio, **{'pass': name},
                         wall_ms=f'{ms:.3f}',
                         peak_allocated_mb=peak_mb if name == 'fwd+grad' else '',
                         capture_ms=capture_ms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', choices=['jax', 'torch', 'both'], default='both')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--frames', nargs='+', type=int, default=NUM_FRAMES_LIST)
    parser.add_argument('--label-ratio', type=float, default=0.5)
    parser.add_argument('--output', default=CSV_PATH)
    parser.add_argument('--cuda-graph', action='store_true',
                        help='capture Torch forward/backward for each shape bucket')
    args = parser.parse_args()
    if not 0 < args.label_ratio <= 1 or any(t < 1 for t in args.frames):
        parser.error('frames must be positive and label-ratio must be in (0, 1]')
    if args.cuda_graph and (args.device != 'cuda' or args.framework == 'jax'):
        parser.error('--cuda-graph requires a Torch CUDA benchmark')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jax.config.update('jax_default_matmul_precision', 'highest')
    rows = []
    frameworks = ['jax', 'torch'] if args.framework == 'both' else [args.framework]
    for fw in frameworks:
        for length in args.frames:
            if fw == 'jax':
                dev = jax.devices('gpu' if args.device == 'cuda' else 'cpu')[0]
                with jax.default_device(dev):
                    lattice = build_jax_lattice()
                    params = init_jax_params(lattice, length)
            else:
                torch.manual_seed(0)
                torch._dynamo.reset()
                lattice, params = build_torch_lattice(args.device), None
            bench_config(fw, args.device, lattice, params, length, rows, args.label_ratio, args.cuda_graph)
            # Preserve completed results if a subsequent configuration fails.
            save_csv(args.output, rows, HEADERS)
            del lattice, params
    metadata = dict(torch=torch.__version__, jax=jax.__version__,
                    python=platform.python_version(), device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                    batch_size=BATCH_SIZE, feature_size=FEATURE_SIZE,
                    vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE,
                    torch_threads=1, dtype='float32', tf32=False,
                    gradient_targets='all parameters and frames; fresh returned gradients',
                    warmup=WARMUP, repeat=REPEAT, number=NUMBER,
                    seed=0, inputs_seed='13 * num_frames', arguments=vars(args),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *sorted(Path('last_torch').glob('*.py'))]})
    with open(args.output + '.json', 'w') as stream:
        json.dump(metadata, stream, indent=2)
    print(to_markdown_table(rows, HEADERS), flush=True)


if __name__ == '__main__':
    main()
