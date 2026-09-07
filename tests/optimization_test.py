"""Value and gradient regression checks for GPU execution optimizations."""
import pytest
import torch

import last_torch
from last_torch import semirings

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def lattice_for(device, alignment=None):
    context = last_torch.contexts.FullNGram(3, 1, device=device)
    return last_torch.RecognitionLattice(
        context=context,
        alignment=alignment or last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=lambda c: last_torch.weight_fns.SharedRNNCacher(
            c.vocab_size, c.context_size, 8, 8, device=device),
        weight_fn_factory=lambda c: last_torch.weight_fns.JointWeightFn(
            c.vocab_size, 8, device=device), device=device)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('expansions', [0, 2])
def test_shared_weights_match_independent_loss_and_gradients(device, expansions):
    torch.manual_seed(5)
    alignment = (last_torch.alignments.FrameLabelDependent(expansions)
                 if expansions else last_torch.alignments.FrameDependent())
    lat = lattice_for(device, alignment)
    frames = torch.randn(4, 8, 6, device=device, requires_grad=True)
    nf = torch.tensor([8, 6, 3, 0], device=device)
    labels = torch.tensor([[1, 2, 1, 3], [2, 2, 1, 0],
                           [3, 1, 0, 0], [0, 0, 0, 0]], device=device)
    nl = torch.tensor([4, 3, 2, 0], device=device)
    lat._precompute_weights(lat.build_cache(), frames, 1)
    targets = (frames, *lat.parameters())
    cache = lat.build_cache()
    independent = (lat._forward_backward(cache, frames, nf)[0] -
                   lat._string_forward(cache, frames, nf, labels, nl, semirings.Log))
    # Nonuniform incoming cotangent exercises scaling and signs in both VJPs.
    scale = torch.tensor([0.5, -2., 1., 3.], device=device)
    ref_grad = torch.autograd.grad((independent * scale).sum(), targets)
    shared = lat(frames, nf, labels, nl)
    actual_grad = torch.autograd.grad((shared * scale).sum(), targets)
    torch.testing.assert_close(shared, independent, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(actual_grad, ref_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_complete_steps_match_autograd(device):
    torch.manual_seed(11)
    lat = lattice_for(device)
    frames = torch.randn(4, 9, 6, device=device, requires_grad=True)
    nf = torch.tensor([9, 6, 3, 0], device=device)
    labels = torch.tensor([[1, 2, 1, 3], [2, 1, 2, 0],
                           [3, 0, 0, 0], [0, 0, 0, 0]], device=device)
    nl = torch.tensor([4, 3, 1, 0], device=device)
    lat._precompute_weights(lat.build_cache(), frames, 1)
    targets = (frames, *lat.parameters())
    # Reference uses ordinary autograd through both original alignment scans.
    cache = lat.build_cache()
    str_bwd, lat._align_str_bwd = lat._align_str_bwd, None
    try:
        ref = (lat._forward(cache, frames, nf, semirings.Log)[0] -
               lat._string_forward(cache, frames, nf, labels, nl, semirings.Log))
        ref_grads = torch.autograd.grad(ref.sum(), targets)
    finally:
        lat._align_str_bwd = str_bwd
    actual = lat(frames, nf, labels, nl)
    grads = torch.autograd.grad(actual.sum(), targets)
    torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-6)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('order', [0, 1, 2, 4])
@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)])
def test_vectorized_context_walk_matches_transitions(device, order, batch_shape):
    torch.manual_seed(19)
    ctx = last_torch.contexts.FullNGram(3, order, device=device)
    for length in [0, 1, 17]:
        labels = torch.randint(0, 4, (*batch_shape, length), device=device)
        reference = last_torch.contexts.ContextDependency.walk_states(ctx, labels)
        torch.testing.assert_close(ctx.walk_states(labels), reference)
        zeros = torch.zeros_like(labels)
        torch.testing.assert_close(ctx.walk_states(zeros),
                                   torch.zeros((*batch_shape, length + 1),
                                               device=device, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Graph requires CUDA')
@pytest.mark.parametrize('length', [12, 200, 500])
def test_cuda_graph_replays_inputs_and_parameter_updates(length):
    from last_torch.cuda_graphs import make_graphed_lattice

    torch.manual_seed(23)
    lat = lattice_for('cuda')
    frames = torch.randn(4, length, 6, device='cuda', requires_grad=True)
    nf = torch.tensor([length, length * 3 // 4, length // 2, 0], device='cuda')
    labels = torch.randint(1, 4, (4, length // 2), device='cuda')
    nl = torch.tensor([length // 2, length // 3, length // 4, 0], device='cuda')
    captured = make_graphed_lattice(lat, (frames, nf, labels, nl))
    for step in range(3):
        f = (frames.detach() + step / 10).requires_grad_()
        lab = labels.roll(step, dims=-1)
        targets = (f, *lat.parameters())
        scale = torch.tensor([0.5, -1., 2., 0.], device='cuda')
        actual = captured(f, nf.roll(step), lab, nl.roll(step))
        grads = [x.clone() for x in torch.autograd.grad((actual * scale).sum(), targets)]
        actual = actual.clone()
        reference = lat(f, nf.roll(step), lab, nl.roll(step))
        expected = torch.autograd.grad((reference * scale).sum(), targets)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        for a, b in zip(grads, expected):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)
        with torch.no_grad():
            for p in lat.parameters():
                p.add_(0.001)
    with pytest.raises(ValueError, match='input bucket'):
        captured(frames[:, :-1], nf, labels, nl)


@pytest.mark.parametrize('device', DEVICES)
def test_one_lattice_accepts_many_sequence_lengths(device):
    """Do not exhaust one lattice's compiler cache during a length sweep."""
    lat = lattice_for(device)
    for length in (8, 13, 19, 27, 35, 43, 51, 63, 75, 89):
        frames = torch.randn(2, length, 6, device=device, requires_grad=True)
        nf = torch.tensor([length, length - 2], device=device)
        labels = torch.randint(1, 4, (2, length // 2), device=device)
        nl = torch.tensor([length // 2, length // 2 - 1], device=device)
        loss = lat(frames, nf, labels, nl)
        grads = torch.autograd.grad(loss.sum(), (frames, *lat.parameters()))
        assert torch.isfinite(loss).all()
        assert all(torch.isfinite(g).all() for g in grads)
