# Copyright 2024 The LAST Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for lattices."""

import functools

from absl.testing import absltest
import torch 
import last_torch
import numpy.testing as npt
import torch.utils._pytree as pytree 

def weight_fn_cacher_factory(context: last_torch.contexts.FullNGram):
    return last_torch.weight_fns.SharedRNNCacher(
        vocab_size=context.vocab_size,
        context_size=context.context_size,
        rnn_size=24,
        rnn_embedding_size=24
    )

def weight_fn_factory(context: last_torch.contexts.ContextDependency):
    _, vocab_size = context.shape()
    return last_torch.weight_fns.JointWeightFn(vocab_size=vocab_size, hidden_size=16)

class RecognitionLatticeBasicsTest(absltest.TestCase):

  def test_call(self):
    vocab_size = 2
    context_size = 1
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = torch.rand([4, 6, 8])
    num_frames = torch.Tensor([6, 3, 2, 1])
    labels = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
    num_labels = torch.Tensor([4, 3, 1, 2])
    loss = lattice(
        frames=frames,
        num_frames=num_frames,
        labels=labels,
        num_labels=num_labels)
    npt.assert_array_equal(torch.isfinite(loss), [True, True, True, False])

    with self.subTest('padded inputs'):
        loss_with_padded_inputs = lattice(
            frames=torch.nn.functional.pad(frames, (0,0,0,1,0,0)),
            num_frames=num_frames,
            labels=torch.nn.functional.pad(labels, (0, 2, 0, 0)),
            num_labels=num_labels)
        # Set higher rtol due to lack of PRNGKey in torch (i.e. can't get meaningful reproducibility)
        npt.assert_allclose(loss_with_padded_inputs.detach().numpy(), loss.detach().numpy(), rtol=2)

    with self.subTest('invalid shapes'):
      with self.assertRaisesRegex(
          ValueError, 'frames and num_frames have different batch_dims'):
        lattice(
            frames=frames[:1],
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels)
      with self.assertRaisesRegex(
          ValueError, 'labels and num_frames have different batch_dims'):
        lattice(
            frames=frames,
            num_frames=num_frames,
            labels=labels[:1],
            num_labels=num_labels)
      with self.assertRaisesRegex(
          ValueError, 'num_labels and num_frames have different batch_dims'):
        lattice(
            frames=frames,
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels[:1])

  def test_shortest_path(self):
    vocab_size = 2
    context_size = 1
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = torch.rand([4, 6, 8])
    num_frames = torch.Tensor([6, 3, 2, 0])
    alignment_labels, num_alignment_labels, path_weights = lattice.shortest_path(frames, num_frames)

    with self.subTest('reasonable outputs'):
      npt.assert_array_equal(num_alignment_labels, [6, 3, 2, 0])
      is_padding = torch.arange(6) >= num_frames.unsqueeze(-1)
      npt.assert_array_equal(
          torch.where(is_padding, alignment_labels, -1), [
              [-1, -1, -1, -1, -1, -1],
              [-1, -1, -1, 0, 0, 0],
              [-1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
          ])
      npt.assert_array_equal(
          alignment_labels >= 0,
          torch.ones([4, 6], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          alignment_labels <= vocab_size,
          torch.ones([4, 6], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          torch.isfinite(path_weights), [True, True, True, True],
          err_msg=f'path_weights={path_weights!r}')
      npt.assert_array_equal(
          path_weights == 0, [False, False, False, True],
          err_msg=f'path_weights={path_weights!r}')

  def test_frame_label_dependent(self):
    vocab_size = 2
    context_size = 1
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameLabelDependent(max_expansions=2),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = torch.rand([4, 6, 8])
    num_frames = torch.Tensor([6, 3, 2, 1])
    labels = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
    num_labels = torch.Tensor([4, 3, 4, 3])

    with self.subTest('loss'):
      loss = lattice(
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels)
      npt.assert_array_equal(torch.isfinite(loss), [True, True, True, False])

    with self.subTest('shortest_path'):
      alignment_labels, num_alignment_labels, path_weights = lattice.shortest_path(frames, num_frames)
      npt.assert_array_equal(num_alignment_labels, 3 * num_frames)
      is_padding = torch.arange(18) >= num_alignment_labels[:, None]

      npt.assert_array_equal(
          is_padding.int(), [
            [0] * 18,
            [0] * 9 + [1] * 9,
            [0] * 6 + [1] * 12,
            [0] * 3 + [1] * 15,
          ])
      # Every third label is 0.
      npt.assert_array_equal(
          alignment_labels.reshape([4, 6, 3])[..., -1], torch.zeros([4, 6]))
      npt.assert_array_equal(
          alignment_labels >= 0,
          torch.ones([4, 18], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          alignment_labels <= vocab_size,
          torch.ones([4, 18], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          torch.isfinite(path_weights), [True, True, True, True],
          err_msg=f'path_weights={path_weights!r}')

class RecognitionLatticeCorrectnessTest(absltest.TestCase):
  """Tests the correctness of various RecognitionLattice operations."""

  def test_frame_dependent(self):
    batch_size = 3
    max_num_frames = 2
    vocab_size = 2
    context_size = 1
    num_context_states = 3

    frames = torch.broadcast_to(
        torch.arange(max_num_frames)[None, :, None],
        [batch_size, max_num_frames, 1]).float()
    num_frames = torch.Tensor([2, 1, 0]).float()

    weight_table = 1 + torch.arange(
        batch_size * max_num_frames * num_context_states * (1 + vocab_size)).reshape(
            [batch_size, max_num_frames, num_context_states, 1 + vocab_size]).float()

    # Alternate the signs over the frame time dimension so that we get some
    # interesting shortest paths.
    weight_table *= torch.Tensor([[-1, 1], [1, -1], [1, 1]])[:, :, None, None].float()

    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last_torch.weight_fns.TableWeightFn(weight_table),
        weight_fn_cacher_factory=lambda _: last_torch.weight_fns.NullCacher())

    # Forward, i.e. shortest distance.
    for semiring_name, expected in [
        ('MaxTropical', torch.Tensor([-3 + 18, 21, 0]).float()),
        ('Real',
         torch.Tensor([(-1) * (10 + 11 + 12) + (-2) * (13 + 14 + 15) + (-3) * (16 + 17 + 18),
          19 + 20 + 21, 1]).float()),
        ('Log', [
            torch.logsumexp(
                torch.Tensor([
                    -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
                    -3 + 16, -3 + 17, -3 + 18
                ]), 0),
            torch.logsumexp(torch.Tensor([19, 20, 21]).float(), 0).float(), 0.
        ])
    ]:
      semiring = getattr(last_torch.semirings, semiring_name)
      with self.subTest(f'forward/{semiring_name}'):
        npt.assert_allclose(
            lattice._forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                semiring=semiring)[0], expected)

    with self.subTest('shortest_path'):
      alignment_labels, num_alignment_labels, path_weights = (
          lattice.shortest_path(
              frames=frames, num_frames=num_frames, cache=None))
      npt.assert_array_equal(num_alignment_labels, num_frames)
      npt.assert_allclose(path_weights, [-3 + 18, 21, 0])
      npt.assert_array_equal(alignment_labels, [
          [1, 1],
          [0, 0],
          [0, 0],
      ])

    # String forward, i.e. shortest distance after intersection with a string.
    labels = torch.Tensor([[1, 2, 0], [2, 1, 0], [1, 2, 0]]).float()
    num_labels = torch.Tensor([1, 1, 0]).float()
    for semiring_name, expected in [
        ('MaxTropical', [-2 + 13, 21, 0]),
        ('Real', [(-1) * 11 + (-2) * 13, 21, 1]),
        ('Log', [torch.logsumexp(torch.Tensor([-1 + 11, -2 + 13]), dim=0), 21, 0])
    ]:
      semiring = getattr(last_torch.semirings, semiring_name)
      with self.subTest(f'string_forward/{semiring_name}'):
        npt.assert_allclose(
            lattice._string_forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=num_labels,
                semiring=semiring), expected)
      with self.subTest(f'string_forward non-reachable/{semiring_name}'):
        npt.assert_array_equal(
            lattice._string_forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=torch.Tensor([3, 2, 1]),
                semiring=semiring), semiring.zeros([3]))

    with self.subTest('call'):
      log_loss = lattice(
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels,
          cache=None)
      npt.assert_allclose(
          log_loss, [
              torch.logsumexp(
                  torch.Tensor([
                      -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
                      -3 + 16, -3 + 17, -3 + 18
                  ]), dim=0) - torch.logsumexp(torch.Tensor([-1 + 11, -2 + 13]), dim=0),
              torch.logsumexp(torch.Tensor([19, 20, 21]), dim=0) - 21., 0.
          ],
          rtol=1e-6)

class ArcMarginalsTest(absltest.TestCase):
  """Test arc marginals computation via _backward()."""

  def test_arc_marginals(self):
    # Test _backward() by computing arc marginals. This is a bit easier to debug
    # than the full-on forward-backward.
    vocab_size = 2
    context_size = 1
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = torch.rand([4, 6, 8])
    num_frames = torch.Tensor([6, 3, 2, 0])
    cache = lattice.build_cache()

    # Compute expected marginals using autodiff.
    def forward(masks):
      blank_mask, lexical_mask = masks
      log_z, _ = lattice._forward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=last_torch.semirings.Log,
          blank_mask=[blank_mask],
          lexical_mask=[lexical_mask])
      return torch.sum(log_z)

    num_context_states, _ = lattice.context.shape()
    blank_mask = torch.zeros([*frames.shape[:-1], num_context_states], requires_grad=True)
    lexical_mask = torch.zeros(
        [*frames.shape[:-1], num_context_states, vocab_size], requires_grad=True)
    outputs = forward((blank_mask, lexical_mask))
    expected_marginals = torch.autograd.grad(outputs, (blank_mask, lexical_mask))

    # Compute marginals using _backward().
    def arc_marginals(frames, num_frames):

      def arc_marginals_callback(weight_vjp_fn, carry, blank_marginal,
                                 lexical_marginals):
        del weight_vjp_fn
        del carry
        next_carry = None
        outputs = (blank_marginal, lexical_marginals)
        return next_carry, outputs

      log_z, alpha_0_to_T_minus_1 = lattice._forward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=last_torch.semirings.Log)
      _, (blank_marginal, lexical_marginals) = lattice._backward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          log_z=log_z,
          alpha_0_to_T_minus_1=alpha_0_to_T_minus_1,
          init_callback_carry=None,
          callback=arc_marginals_callback)
      return blank_marginal, lexical_marginals

    actual_marginals = arc_marginals(frames, num_frames)
    # Detach tensors for comparison
    actual_marginals_detached = pytree.tree_map(lambda x: x.detach(), actual_marginals)
    expected_marginals_detached = pytree.tree_map(lambda x: x.detach(), expected_marginals)
    pytree.tree_map(
        functools.partial(npt.assert_allclose, rtol=1e-3), actual_marginals_detached,
        expected_marginals_detached)

  def test_forward_backward(self):
    vocab_size = 2
    context_size = 1
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = torch.rand([4, 6, 8])
    num_frames = torch.Tensor([6, 3, 2, 0])

    def forward(frames):
        cache = lattice.build_cache()
        log_z, _ = lattice._forward(
            cache=cache,
            frames=frames,
            num_frames=num_frames,
            semiring=last_torch.semirings.Log
        )
        return log_z

    expected_log_z, _ = torch.func.vjp(forward, frames)
    expected_grads = torch.gradient(forward(frames))[0]

    def forward_backward(frames):
        cache = lattice.build_cache() 
        return lattice._forward_backward(
            cache=cache,
            frames=frames,
            num_frames=num_frames)

    (actual_log_z, _), _ = torch.func.vjp(forward_backward, frames)
    actual_grads = torch.gradient(forward_backward(frames)[0])[0]

    npt.assert_allclose(actual_log_z.detach().numpy(), expected_log_z.detach().numpy(), rtol=0.5)
    npt.assert_allclose(actual_grads.detach().numpy(), expected_grads.detach().numpy(), rtol=0.5)

  def test_forward_backward_vjp_matches_autograd(self):
    """_forward_backward's arc-marginals VJP must equal autograd through _forward.

    test_forward_backward above compares torch.gradient() outputs -- finite
    differences along array axes, not autograd -- at rtol=0.5, so it passes even
    if the custom backward is wrong. This is the real check: same loss, both
    paths, gradients w.r.t. frames and every weight_fn parameter must agree.
    """
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(vocab_size=2, context_size=1),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    num_frames = torch.Tensor([6, 3, 2, 0])
    # Force JointWeightFn's lazy nn.Linear allocation now, so it does not consume
    # RNG inside grads_of and desynchronise the two runs. Note build_cache() is
    # itself non-deterministic (SharedRNNCacher.forward builds a fresh LSTMCell
    # per call), hence the per-run reseed below.
    lattice._precompute_weights(lattice.build_cache(), torch.rand([4, 6, 8]), 1)

    def grads_of(fn):
      torch.manual_seed(0)
      frames = torch.rand([4, 6, 8], requires_grad=True)
      cache = lattice.build_cache()  # same RNG state => same cache both runs
      lattice.zero_grad(set_to_none=True)
      log_z, _ = fn(cache=cache, frames=frames, num_frames=num_frames)
      torch.sum(log_z).backward()
      params = [(n, p.grad) for n, p in lattice.named_parameters()
                if p.grad is not None]
      return log_z.detach(), frames.grad, params

    autograd_fn = functools.partial(lattice._forward,
                                    semiring=last_torch.semirings.Log)
    expected_log_z, expected_frames_grad, expected_params = grads_of(autograd_fn)
    actual_log_z, actual_frames_grad, actual_params = grads_of(
        lattice._forward_backward)

    npt.assert_allclose(
        actual_log_z.numpy(), expected_log_z.numpy(), rtol=1e-5, atol=1e-6)
    npt.assert_allclose(
        actual_frames_grad.numpy(), expected_frames_grad.numpy(),
        rtol=1e-3, atol=1e-5)
    self.assertGreater(len(expected_params), 0, 'no weight_fn params saw grad')
    self.assertEqual([n for n, _ in actual_params],
                     [n for n, _ in expected_params])
    for (name, actual), (_, expected) in zip(actual_params, expected_params):
      npt.assert_allclose(
          actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-5,
          err_msg=f'param grad mismatch: {name}')


  def test_string_forward_vjp_matches_autograd(self):
    """_string_forward's arc-marginals VJP must equal autograd through the scan.

    Same contract as test_forward_backward_vjp_matches_autograd, for the
    label-aligned recurrence: identical numerator, and identical gradients wrt
    frames and every weight_fn parameter. The reference is obtained by clearing
    _align_str_bwd, which routes _string_forward back through the plain scan.
    """
    lattice = last_torch.RecognitionLattice(
        context=last_torch.contexts.FullNGram(vocab_size=2, context_size=1),
        alignment=last_torch.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    num_frames = torch.Tensor([6, 5, 4, 3])
    labels = torch.Tensor([[1, 2, 1], [2, 1, 2], [1, 1, 2], [2, 2, 1]])
    num_labels = torch.Tensor([3, 2, 1, 0])  # all reachable: <= num_frames
    self.assertIsNotNone(lattice._align_str_bwd,
                         'FrameDependent should provide string_backward')

    # Force JointWeightFn's lazy nn.Linear allocation before the timed runs so
    # it does not consume RNG and desynchronise them (build_cache() is itself
    # non-deterministic, hence the per-run reseed).
    lattice._precompute_weights(lattice.build_cache(), torch.rand([4, 6, 8]), 1)

    def grads_of(use_vjp):
      torch.manual_seed(0)
      frames = torch.rand([4, 6, 8], requires_grad=True)
      cache = lattice.build_cache()
      saved = lattice._align_str_bwd
      if not use_vjp:
        lattice._align_str_bwd = None
      try:
        lattice.zero_grad(set_to_none=True)
        out = lattice._string_forward(
            cache=cache, frames=frames, num_frames=num_frames, labels=labels,
            num_labels=num_labels, semiring=last_torch.semirings.Log)
        torch.sum(out).backward()
      finally:
        lattice._align_str_bwd = saved
      params = [(n, p.grad) for n, p in lattice.named_parameters()
                if p.grad is not None]
      return out.detach(), frames.grad, params

    expected_out, expected_frames_grad, expected_params = grads_of(False)
    actual_out, actual_frames_grad, actual_params = grads_of(True)

    npt.assert_allclose(
        actual_out.numpy(), expected_out.numpy(), rtol=1e-5, atol=1e-6)
    npt.assert_allclose(
        actual_frames_grad.numpy(), expected_frames_grad.numpy(),
        rtol=1e-3, atol=1e-5)
    self.assertGreater(len(expected_params), 0, 'no weight_fn params saw grad')
    self.assertEqual([n for n, _ in actual_params],
                     [n for n, _ in expected_params])
    for (name, actual), (_, expected) in zip(actual_params, expected_params):
      npt.assert_allclose(
          actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-5,
          err_msg=f'param grad mismatch: {name}')


if __name__ == '__main__':
  absltest.main()