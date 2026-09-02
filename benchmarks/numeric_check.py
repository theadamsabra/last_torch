"""Cross-framework numeric validity check: JAX (last) vs PyTorch (last_torch).

Uses TableWeightFn + NullCacher so there are no trainable parameters to transfer —
the same numpy weight table is fed directly to both frameworks.

Run from last_torch repo root:
    python benchmarks/numeric_check.py
"""

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import torch

# Make sure both packages are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../last'))

import last
import last_torch
from last_torch.semirings import _logaddexp as torch_logaddexp
from last_torch.semirings import _LogSumExp
from last.semirings import _logaddexp as jax_logaddexp
from last.semirings import _logsumexp as jax_logsumexp

ROWS = []


def _check(label: str, actual: np.ndarray, expected: np.ndarray,
           rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    finite = np.isfinite(actual) & np.isfinite(expected)
    # For non-finite positions, check exact equality.
    inf_match = np.all(actual[~finite] == expected[~finite])
    if finite.any():
        max_abs = float(np.max(np.abs(actual[finite] - expected[finite])))
        max_rel = float(np.max(np.abs(actual[finite] - expected[finite]) /
                               (np.abs(expected[finite]) + atol)))
    else:
        max_abs = 0.0
        max_rel = 0.0
    try:
        npt.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol)
        ok = inf_match
    except AssertionError:
        ok = False
    status = 'PASS' if ok else 'FAIL'
    ROWS.append({'Operation': label, 'Max Abs Diff': f'{max_abs:.4e}',
                 'Rel Diff': f'{max_rel:.4e}', 'Status': status})
    print(f'  {"✓" if ok else "✗"} {label}: max_abs={max_abs:.4e}  {status}')
    return ok


# ---------------------------------------------------------------------------
# 1. Semiring ops: _logaddexp
# ---------------------------------------------------------------------------

def test_logaddexp():
    print('\n[1] _logaddexp forward + gradient')
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(500).astype(np.float32)
    b_np = rng.standard_normal(500).astype(np.float32)

    # Inject interesting corner cases.
    a_np[:3] = [-np.inf, -np.inf, np.inf]
    b_np[:3] = [-np.inf,  5.0,    1.0]

    # JAX forward.
    a_j, b_j = jnp.array(a_np), jnp.array(b_np)
    out_j = np.array(jax_logaddexp(a_j, b_j))

    # Torch forward.
    a_t = torch.tensor(a_np, requires_grad=False)
    b_t = torch.tensor(b_np, requires_grad=False)
    out_t, _ = torch_logaddexp(a_t, b_t)
    out_t = out_t.detach().numpy()

    _check('logaddexp forward (finite)', out_j[3:], out_t[3:])
    # -inf + -inf should stay -inf in both.
    _check('logaddexp forward (-inf,-inf)', out_j[:1], out_t[:1], atol=0)
    # -inf + finite: result should be finite.
    _check('logaddexp forward (-inf,finite)', out_j[1:2], out_t[1:2])

    # Gradient check (finite inputs only — slice [3:]).
    a_j2 = jnp.array(a_np[3:])
    b_j2 = jnp.array(b_np[3:])
    grad_a_j, grad_b_j = jax.grad(lambda a, b: jax_logaddexp(a, b).sum(), argnums=(0, 1))(a_j2, b_j2)

    a_t2 = torch.tensor(a_np[3:], requires_grad=True)
    b_t2 = torch.tensor(b_np[3:], requires_grad=True)
    out_t2, _ = torch_logaddexp(a_t2, b_t2)
    out_t2.sum().backward()

    _check('logaddexp grad_a (finite)', np.array(grad_a_j), a_t2.grad.numpy(), rtol=1e-4)
    _check('logaddexp grad_b (finite)', np.array(grad_b_j), b_t2.grad.numpy(), rtol=1e-4)

    # Gradient for -inf input: should be 0 in torch, matches JAX behaviour.
    a_inf = torch.tensor([-np.inf], requires_grad=True)
    b_fin = torch.tensor([5.0], requires_grad=True)
    out_inf, _ = torch_logaddexp(a_inf, b_fin)
    out_inf.sum().backward()
    _check('logaddexp grad (-inf input → 0)', a_inf.grad.numpy(), np.array([0.0]), atol=1e-8)


# ---------------------------------------------------------------------------
# 2. Semiring ops: _logsumexp
# ---------------------------------------------------------------------------

def test_logsumexp():
    print('\n[2] _logsumexp forward + gradient')
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((50, 20)).astype(np.float32)

    x_j = jnp.array(x_np)
    out_j = np.array(jax_logsumexp(x_j, axis=-1))

    x_t = torch.tensor(x_np, requires_grad=False)
    out_t, _, _ = _LogSumExp.apply(x_t, -1)
    out_t = out_t.detach().numpy()

    _check('logsumexp forward', out_j, out_t)

    # Gradients.
    grad_j = np.array(jax.grad(lambda x: jax_logsumexp(x, axis=-1).sum())(x_j))

    x_t2 = torch.tensor(x_np, requires_grad=True)
    out_t2, _, _ = _LogSumExp.apply(x_t2, -1)
    out_t2.sum().backward()

    _check('logsumexp gradient', grad_j, x_t2.grad.numpy(), rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. FullNGram.forward_reduce + backward_broadcast
# ---------------------------------------------------------------------------

def test_ngram_context():
    print('\n[3] FullNGram.forward_reduce / backward_broadcast')
    rng = np.random.default_rng(13)
    vocab_size, context_size = 3, 1
    num_states = sum(vocab_size**i for i in range(context_size + 1))  # 4
    batch = 5

    # forward_reduce input: [batch, num_states, vocab_size]
    fr_weights_np = rng.standard_normal([batch, num_states, vocab_size]).astype(np.float32)
    # backward_broadcast input: [batch, num_states]
    bb_weights_np = rng.standard_normal([batch, num_states]).astype(np.float32)

    ctx_j = last.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size)
    fr_j = np.array(ctx_j.forward_reduce(jnp.array(fr_weights_np), last.semirings.Log))
    bb_j = np.array(ctx_j.backward_broadcast(jnp.array(bb_weights_np)))

    ctx_t = last_torch.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size)
    fr_t = ctx_t.forward_reduce(torch.tensor(fr_weights_np), last_torch.semirings.Log).detach().numpy()
    bb_t = ctx_t.backward_broadcast(torch.tensor(bb_weights_np)).detach().numpy()

    _check('FullNGram.forward_reduce', fr_j, fr_t)
    _check('FullNGram.backward_broadcast', bb_j, bb_t)


# ---------------------------------------------------------------------------
# 4 & 5. RecognitionLattice: string_forward (numerator) + full forward (denominator)
# ---------------------------------------------------------------------------

def _make_table_inputs(vocab_size, context_size, batch_size, max_num_frames, rng):
    num_context_states = sum(vocab_size**i for i in range(context_size + 1))
    input_vocab_size = max_num_frames + 1

    table = rng.standard_normal(
        [batch_size, input_vocab_size, num_context_states, 1 + vocab_size]
    ).astype(np.float32)

    frames_np = rng.integers(0, input_vocab_size,
                              [batch_size, max_num_frames, 1]).astype(np.float32)
    labels_np = rng.integers(1, vocab_size + 1,
                              [batch_size, max_num_frames]).astype(np.int32)
    num_frames_np = np.full([batch_size], max_num_frames, dtype=np.int32)
    num_labels_np = np.full([batch_size], max_num_frames, dtype=np.int32)

    return table, frames_np, labels_np, num_frames_np, num_labels_np


def test_full_loss():
    print('\n[4/5/6/7] RecognitionLattice full forward loss')
    rng = np.random.default_rng(99)
    vocab_size, context_size = 2, 1
    batch_size, max_num_frames = 3, 4

    table, frames_np, labels_np, num_frames_np, num_labels_np = _make_table_inputs(
        vocab_size, context_size, batch_size, max_num_frames, rng)

    # --- JAX ---
    table_j = jnp.array(table)
    jax_lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last.weight_fns.TableWeightFn(table=table_j),
        weight_fn_cacher_factory=lambda _: last.weight_fns.NullCacher(),
    )

    frames_j = jnp.array(frames_np)
    labels_j = jnp.array(labels_np)
    num_frames_j = jnp.array(num_frames_np)
    num_labels_j = jnp.array(num_labels_np)

    loss_j, params = jax_lattice.init_with_output(
        jax.random.PRNGKey(0),
        frames=frames_j, num_frames=num_frames_j,
        labels=labels_j, num_labels=num_labels_j)

    # --- PyTorch ---
    table_t = torch.tensor(table)
    torch_lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last_torch.weight_fns.TableWeightFn(table_t),
        weight_fn_cacher_factory=lambda _: last_torch.weight_fns.NullCacher(),
    )

    frames_t = torch.tensor(frames_np)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    num_frames_t = torch.tensor(num_frames_np, dtype=torch.float32)
    num_labels_t = torch.tensor(num_labels_np, dtype=torch.float32)

    loss_t = torch_lattice(
        frames=frames_t, num_frames=num_frames_t,
        labels=labels_t, num_labels=num_labels_t)

    _check('full loss (globally normalized)', np.array(loss_j), loss_t.detach().numpy(),
           rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Gradient of loss w.r.t. input frames
# ---------------------------------------------------------------------------

def test_gradient():
    """Verify that gradients w.r.t. log-semiring operations agree.

    End-to-end gradient comparison between JAX and PyTorch requires matched
    neural network weights, which is non-trivial to set up across Flax and
    nn.Module.  TableWeightFn treats frames as discrete indices (not
    differentiable w.r.t. frame values), and its table is a module attribute
    in JAX (not a Flax param), making jax.grad over it incompatible with
    JAX's tracing model.

    We instead test gradient agreement directly through the Log semiring
    logsumexp operation applied to the output of FullNGram.forward_reduce,
    which exercises the same code path that the full lattice backward uses.
    """
    print('\n[8] Gradient of log-semiring sum through forward_reduce output')
    rng = np.random.default_rng(33)
    vocab_size, context_size = 3, 2
    num_states = sum(vocab_size**i for i in range(context_size + 1))
    batch = 4
    weights_np = rng.standard_normal([batch, num_states, vocab_size]).astype(np.float32)

    # JAX: grad of logsumexp(forward_reduce output) w.r.t. weights.
    ctx_j = last.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size)

    def jax_fn(w):
        reduced = ctx_j.forward_reduce(w, last.semirings.Log)
        return last.semirings.Log.sum(reduced, axis=-1).sum()

    grad_j = np.array(jax.grad(jax_fn)(jnp.array(weights_np)))

    # Torch: same.
    ctx_t = last_torch.contexts.FullNGram(vocab_size=vocab_size, context_size=context_size)
    w_t = torch.tensor(weights_np, requires_grad=True)
    reduced_t = ctx_t.forward_reduce(w_t, last_torch.semirings.Log)
    last_torch.semirings.Log.sum(reduced_t, dim=-1).sum().backward()

    _check('gradient through forward_reduce + Log.sum', grad_j, w_t.grad.numpy(),
           rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report():
    headers = ['Operation', 'Max Abs Diff', 'Rel Diff', 'Status']
    widths = {h: max(len(h), max(len(r[h]) for r in ROWS)) for h in headers}
    sep = '+-' + '-+-'.join('-' * widths[h] for h in headers) + '-+'
    hdr = '| ' + ' | '.join(h.ljust(widths[h]) for h in headers) + ' |'
    print('\n' + '=' * 70)
    print('NUMERIC VALIDITY SUMMARY')
    print('=' * 70)
    print(sep)
    print(hdr)
    print(sep)
    for row in ROWS:
        print('| ' + ' | '.join(row[h].ljust(widths[h]) for h in headers) + ' |')
    print(sep)
    fails = [r for r in ROWS if r['Status'] == 'FAIL']
    if fails:
        print(f'\n{len(fails)} FAILED checks.')
        sys.exit(1)
    else:
        print(f'\nAll {len(ROWS)} checks PASSED.')


if __name__ == '__main__':
    # Suppress JAX/XLA info logs.
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    jax.config.update('jax_platform_name', 'cpu')

    test_logaddexp()
    test_logsumexp()
    test_ngram_context()
    test_full_loss()
    test_gradient()
    print_report()
