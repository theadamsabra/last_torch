"""Does torch.scan beat the per-step-compiled Python for-loop?

Three implementations of the same alpha recurrence -- the no_grad scan inside
_RecurrenceFn -- measured against each other:

  loop      current production. Python for-loop with torch.compile on
            alignment.forward only. Dispatches O(T) compiled kernels; compiles
            once, and the compilation does not depend on T.
  unrolled  the whole loop inside a single torch.compile. Can fuse across
            steps, but Dynamo specialises on the Python int T, so every
            distinct T is a fresh compile.
  scan      torch.scan inside torch.compile. combine_fn is traced once and
            reused for every T, at the cost of a mandatory .clone() on the
            stacked output (scan forbids the output aliasing an input).

scan only needs to work here because _RecurrenceFn.forward runs under no_grad --
its autograd support is undocumented and is not exercised.

Two measurements:
  fixed-T   per-T wall time, warm (post-compile).
  sweep     a run over many distinct T values, which is where unrolled pays for
            recompilation and scan does not.

Usage: .venv/bin/python3 -m benchmarks.bench_scan_impl
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

import last_torch
from last_torch import semirings
from last_torch.lattices import _init_context_state_weights, _padding_masks
from benchmarks.utils import save_csv

try:
  from torch._higher_order_ops import scan as torch_scan
except ImportError:  # pragma: no cover - depends on torch version
  torch_scan = getattr(torch, 'scan', None)

# Required, and undocumented. On torch 2.13.0 a scan under the default inductor
# backend dies with
#   InductorError: DataDependentOutputException: aten._local_scalar_dense.default
# even for the minimal example in the torch.scan docs. Dynamo traces it fine --
# backend='eager' and backend='aot_eager' both work -- so the failure is in
# inductor lowering, on CPU and CUDA alike. This flag is what unblocks it.
torch._dynamo.config.capture_scalar_outputs = True

# Matches bench_gpu.py. Duplicated rather than imported so this script runs in a
# bare torch venv -- bench_gpu imports jax, which the version-comparison venv
# deliberately does not have.
BATCH_SIZE = 16
FEATURE_SIZE = 80
VOCAB_SIZE = 4
CONTEXT_SIZE = 1

FIXED_T = [10, 50, 100, 200, 500]
# Unrolling emits one copy of the step body per frame, so Inductor compile time
# grows with T and becomes impractical well before T=500. Measured up to here.
UNROLLED_MAX_T = 200
# Deliberately many distinct lengths, as in a real batch-by-utterance loader.
SWEEP_T = [37, 61, 88, 113, 149, 176, 204, 233, 261, 290]
HEADERS = ['measurement', 'device', 'variant', 'num_frames', 'wall_ms',
           'unique_graphs']
CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'scan_impl.csv')


def build_torch_lattice(device_str: str):
  """Same configuration as bench_gpu.build_torch_lattice."""
  return last_torch.RecognitionLattice(
      context=last_torch.contexts.FullNGram(
          vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE, device=device_str),
      alignment=last_torch.alignments.FrameDependent(),
      weight_fn_cacher_factory=lambda ctx: last_torch.weight_fns.SharedRNNCacher(
          vocab_size=ctx.vocab_size, context_size=ctx.context_size,
          rnn_size=24, rnn_embedding_size=24, device=device_str),
      weight_fn_factory=lambda ctx: last_torch.weight_fns.JointWeightFn(
          vocab_size=ctx.shape()[1], hidden_size=16, device=device_str),
      device=device_str,
  ).to(device_str)


def _setup(lattice, T, device):
  """Precomputed arc weights + init carry, shared by all three variants."""
  frames = torch.rand([BATCH_SIZE, T, FEATURE_SIZE], device=device)
  num_frames = torch.full([BATCH_SIZE], T, dtype=torch.float32, device=device)
  with torch.no_grad():
    all_blank, all_lexical = lattice._precompute_weights(
        lattice.build_cache(), frames, 1)
  alpha = _init_context_state_weights(
      batch_dims=num_frames.shape,
      dtype=all_blank.dtype,
      num_states=lattice.context.shape()[0],
      start=lattice.context.start(),
      semiring=semirings.Log,
      device=device)
  return all_blank.detach(), all_lexical.detach(), num_frames, alpha


def _loop(lattice, all_blank, all_lexical, num_frames, alpha):
  """Current production shape: Python for-loop over compiled per-step kernels."""
  in_dim = 1
  T = all_blank.shape[in_dim]
  padding_list = _padding_masks(num_frames, T, all_blank.device)
  alpha_list = []
  for i in range(T):
    next_alpha = lattice._align_fwd(
        alpha=alpha,
        blank=[all_blank.select(in_dim, i)],
        lexical=[all_lexical.select(in_dim, i)],
        context=lattice.context,
        semiring=semirings.Log)
    alpha_list.append(alpha)
    alpha = torch.where(padding_list[i], alpha, next_alpha)
  return semirings.Log.sum(alpha, dim=-1), torch.stack(alpha_list, dim=in_dim)


def _unrolled_inner(lattice, all_blank, all_lexical, num_frames, alpha):
  """Same loop, but the whole thing is handed to one torch.compile region."""
  in_dim = 1
  T = all_blank.shape[in_dim]
  padding = (torch.arange(T, device=all_blank.device).view(T, 1) >=
             num_frames).unsqueeze(-1)
  alpha_list = []
  for i in range(T):
    next_alpha = lattice.alignment.forward(
        alpha=alpha,
        blank=[all_blank.select(in_dim, i)],
        lexical=[all_lexical.select(in_dim, i)],
        context=lattice.context,
        semiring=semirings.Log)
    alpha_list.append(alpha)
    alpha = torch.where(padding[i], alpha, next_alpha)
  return semirings.Log.sum(alpha, dim=-1), torch.stack(alpha_list, dim=in_dim)


def _scan_inner(lattice, all_blank, all_lexical, num_frames, alpha):
  """torch.scan: combine_fn traced once, reused for every T."""
  in_dim = 1
  T = all_blank.shape[in_dim]
  # scan wants the scanned axis leading, and real tensors rather than a closure
  # over the data (combine_fn must be pure).
  blank_T = all_blank.movedim(in_dim, 0)
  lexical_T = all_lexical.movedim(in_dim, 0)
  padding_T = (torch.arange(T, device=all_blank.device).view(T, 1) >=
               num_frames).unsqueeze(-1)

  def combine_fn(carry, x):
    blank_t, lexical_t, pad_t = x
    next_alpha = lattice.alignment.forward(
        alpha=carry,
        blank=[blank_t],
        lexical=[lexical_t],
        context=lattice.context,
        semiring=semirings.Log)
    # .clone() is mandatory: scan forbids an output aliasing an input.
    return torch.where(pad_t, carry, next_alpha), carry.clone()

  # scan compares carry metadata including stride and memory_format, not just
  # the shape/dtype the docs mention. _init_context_state_weights returns a
  # broadcast view (stride (0, 1)) while the recurrence produces a contiguous
  # tensor, so the init has to be materialised first.
  alpha_T, alphas = torch_scan(combine_fn, alpha.contiguous(),
                               (blank_T, lexical_T, padding_T))
  return semirings.Log.sum(alpha_T, dim=-1), alphas.movedim(0, in_dim)


def _unique_graphs():
  return torch._dynamo.utils.counters['stats'].get('unique_graphs', 0)


def _time(fn, warmup=2, repeat=3, number=5):
  for _ in range(warmup):
    fn()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  out = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    for _ in range(number):
      fn()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    out.append((time.perf_counter() - t0) / number * 1000)
  return statistics.median(out)


def _compile_time(fn, args):
  """Wall time of the first (cold) call, i.e. trace + compile + one execution."""
  t0 = time.perf_counter()
  with torch.no_grad():
    fn(*args)
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return (time.perf_counter() - t0) * 1000


def build_variants(lattice):
  variants = {'loop': _loop,
              'unrolled': torch.compile(_unrolled_inner)}
  if torch_scan is not None:
    variants['scan'] = torch.compile(_scan_inner)
  return variants


def check_agreement(lattice, device):
  """All variants must produce the same log_z and alphas as the plain loop."""
  args = _setup(lattice, 50, device)
  with torch.no_grad():
    ref = _loop(lattice, *args)
    for name, fn in build_variants(lattice).items():
      got = fn(lattice, *args)
      for a, b, what in zip(got, ref, ('log_z', 'alphas')):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6,
                                   msg=f'{name} {what} disagrees with loop')
  print('  all variants agree with the for-loop', flush=True)


def main():
  if torch_scan is None:
    print('torch.scan unavailable on this build; loop vs unrolled only.')
  print(f'torch {torch.__version__}', flush=True)
  rows = []
  devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

  def record(measurement, device, variant, T, ms, graphs=''):
    rows.append({'measurement': measurement, 'device': device,
                 'variant': variant, 'num_frames': T,
                 'wall_ms': round(ms, 3), 'unique_graphs': graphs})

  for device in devices:
    lattice = build_torch_lattice(device)
    print(f'\n=== agreement / {device} ===', flush=True)
    check_agreement(lattice, device)

    names = list(build_variants(lattice))
    print(f'\n=== warm per-call, ms / {device} '
          f'batch={BATCH_SIZE} feature={FEATURE_SIZE} ===', flush=True)
    print(f'{"T":>5} ' + ' '.join(f'{n:>12}' for n in names), flush=True)
    for T in FIXED_T:
      args = _setup(lattice, T, device)
      variants = build_variants(lattice)
      cells = []
      for name in names:
        if name == 'unrolled' and T > UNROLLED_MAX_T:
          cells.append(f'{"skipped":>12}')
          continue
        with torch.no_grad():
          ms = _time(lambda fn=variants[name]: fn(lattice, *args))
        record('warm', device, name, T, ms)
        cells.append(f'{ms:>10.2f}ms')
      print(f'{T:>5} ' + ' '.join(cells), flush=True)

    print(f'\n=== cold compile, ms (first call incl. trace+codegen) '
          f'/ {device} ===', flush=True)
    print(f'{"T":>5} ' + ' '.join(f'{n:>12}' for n in names), flush=True)
    for T in FIXED_T:
      cells = []
      for name in names:
        if name == 'unrolled' and T > UNROLLED_MAX_T:
          cells.append(f'{"skipped":>12}')
          continue
        torch._dynamo.reset()
        lat = build_torch_lattice(device)
        args = _setup(lat, T, device)
        ms = _compile_time(build_variants(lat)[name], (lat,) + args)
        record('cold', device, name, T, ms)
        cells.append(f'{ms:>10.1f}ms')
      print(f'{T:>5} ' + ' '.join(cells), flush=True)

    print(f'\n=== variable-T sweep, {len(SWEEP_T)} distinct lengths, cold '
          f'/ {device} ===', flush=True)
    for name in names:
      torch._dynamo.reset()
      torch._dynamo.utils.counters.clear()
      lat = build_torch_lattice(device)
      fn = build_variants(lat)[name]
      presets = [_setup(lat, T, device) for T in SWEEP_T]
      t0 = time.perf_counter()
      with torch.no_grad():
        for args in presets:
          fn(lat, *args)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      total = (time.perf_counter() - t0) * 1000
      graphs = _unique_graphs()
      print(f'  {name:>9}: {total:9.1f}ms total, {graphs:3d} unique graphs',
            flush=True)
      record('sweep', device, name, len(SWEEP_T), total, graphs)

  save_csv(CSV_PATH, rows, HEADERS)


if __name__ == '__main__':
  main()
