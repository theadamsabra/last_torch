"""Backs whitepaper 4.6 and 6.6: the _RecurrenceFn custom VJP vs plain autograd.

Two measurements, both before/after in the same process so the columns are
comparable:

  isolated  -- _forward_backward alone (4.6's table). Directly times the custom
               VJP against differentiating _forward with autograd.
  end-to-end-- the full loss, lattice(...) (6.6's tables). "before" is obtained
               by pointing _forward_backward back at plain-autograd _forward,
               which is exactly what it did prior to the custom VJP.

Usage: .venv/bin/python3 -m benchmarks.bench_vjp
"""
import functools
import os

import torch

import last_torch
from benchmarks.bench_gpu import (BATCH_SIZE, FEATURE_SIZE, build_torch_lattice,
                                  time_torch)
from benchmarks.utils import save_csv

NUM_FRAMES_LIST = [10, 50, 100, 200, 500]
HEADERS = ['scope', 'device', 'num_frames', 'variant', 'wall_ms']
CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'vjp.csv')


def _inputs(device, T):
  return (torch.rand([BATCH_SIZE, T, FEATURE_SIZE], device=device,
                     requires_grad=True),
          torch.full([BATCH_SIZE], T, dtype=torch.float32, device=device))


def bench_isolated(device, T):
  """_forward_backward on its own: autograd baseline vs custom VJP."""
  lat = build_torch_lattice(device)
  frames, num_frames = _inputs(device, T)
  lat._precompute_weights(lat.build_cache(), frames, 1)

  def run(fn):
    lat.zero_grad(set_to_none=True)
    frames.grad = None
    log_z, _ = fn(cache=lat.build_cache(), frames=frames,
                  num_frames=num_frames)
    torch.sum(log_z).backward()

  autograd = functools.partial(lat._forward, semiring=last_torch.semirings.Log)
  return (time_torch(lambda: run(autograd)),
          time_torch(lambda: run(lat._forward_backward)))


def bench_end_to_end(device, T):
  """Full loss. 'before' restores the pre-VJP passthrough to _forward."""
  lat = build_torch_lattice(device)
  frames, num_frames = _inputs(device, T)
  labels = torch.ones([BATCH_SIZE, T], dtype=torch.float32, device=device)
  num_labels = torch.full([BATCH_SIZE], T, dtype=torch.float32, device=device)

  def run():
    lat.zero_grad(set_to_none=True)
    frames.grad = None
    torch.sum(lat(frames, num_frames, labels, num_labels)).backward()

  after = time_torch(run)
  # The exact body _forward_backward had before _RecurrenceFn existed.
  original = lat._forward_backward
  lat._forward_backward = functools.partial(
      lat._forward, semiring=last_torch.semirings.Log)
  try:
    before = time_torch(run)
  finally:
    lat._forward_backward = original
  return before, after


def main():
  rows = []
  devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
  for scope, fn, cols in (
      ('isolated', bench_isolated, ('autograd', 'custom VJP')),
      ('end-to-end', bench_end_to_end, ('before VJP', 'after VJP')),
  ):
    for device in devices:
      print(f'\n=== {scope} / {device}  '
            f'batch={BATCH_SIZE} feature={FEATURE_SIZE} ===')
      print(f'{"T":>5} {cols[0]:>13} {cols[1]:>13} {"speedup":>9}')
      for T in NUM_FRAMES_LIST:
        before, after = fn(device, T)
        print(f'{T:>5} {before:>11.2f}ms {after:>11.2f}ms {before/after:>8.2f}x')
        for variant, ms in ((cols[0], before), (cols[1], after)):
          rows.append({'scope': scope, 'device': device, 'num_frames': T,
                       'variant': variant, 'wall_ms': round(ms, 3)})
  save_csv(CSV_PATH, rows, HEADERS)


if __name__ == '__main__':
  main()
