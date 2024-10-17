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

    # with self.subTest('padded inputs'):
    #   loss_with_padded_inputs = lattice.apply(
    #       frames=jnp.pad(frames, [(0, 0), (0, 1), (0, 0)]),
    #       num_frames=num_frames,
    #       labels=jnp.pad(labels, [(0, 0), (0, 2)]),
    #       num_labels=num_labels)
    #   npt.assert_allclose(loss_with_padded_inputs, loss)

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

#   def test_frame_dependent(self):
#     batch_size = 3
#     max_num_frames = 2
#     vocab_size = 2
#     context_size = 1
#     num_context_states = 3

#     frames = torch.broadcast_to(
#         torch.arange(max_num_frames)[None, :, None],
#         [batch_size, max_num_frames, 1]).float()
#     num_frames = torch.Tensor([2, 1, 0]).float()

#     weight_table = 1 + torch.arange(
#         batch_size * max_num_frames * num_context_states * (1 + vocab_size)).reshape(
#             [batch_size, max_num_frames, num_context_states, 1 + vocab_size]).float()

#     # Alternate the signs over the frame time dimension so that we get some
#     # interesting shortest paths.
#     weight_table *= torch.Tensor([[-1, 1], [1, -1], [1, 1]])[:, :, None, None].float()

#     lattice = last_torch.RecognitionLattice(
#         context=last_torch.contexts.FullNGram(
#             vocab_size=vocab_size, context_size=context_size),
#         alignment=last_torch.alignments.FrameDependent(),
#         weight_fn_factory=lambda _: last_torch.weight_fns.TableWeightFn(weight_table),
#         weight_fn_cacher_factory=lambda _: last_torch.weight_fns.NullCacher())

#     # Forward, i.e. shortest distance.
#     for semiring_name, expected in [
#         ('MaxTropical', torch.Tensor([-3 + 18, 21, 0]).float()),
#         ('Real',
#          torch.Tensor([(-1) * (10 + 11 + 12) + (-2) * (13 + 14 + 15) + (-3) * (16 + 17 + 18),
#           19 + 20 + 21, 1]).float()),
#         ('Log', [
#             torch.logsumexp(
#                 torch.Tensor([
#                     -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
#                     -3 + 16, -3 + 17, -3 + 18
#                 ]).float(), 0),
#             torch.logsumexp(torch.Tensor([19, 20, 21]).float(), 0), 0.
#         ])
#     ]:
#       semiring = getattr(last_torch.semirings, semiring_name)
#       with self.subTest(f'forward/{semiring_name}'):
#         npt.assert_allclose(
#             lattice._forward(
#                 cache=None,
#                 frames=frames,
#                 num_frames=num_frames,
#                 semiring=semiring)[0], expected)

#     with self.subTest('shortest_path'):
#       alignment_labels, num_alignment_labels, path_weights = (
#           lattice.shortest_path(
#               frames=frames, num_frames=num_frames, cache=None))
#       npt.assert_array_equal(num_alignment_labels, num_frames)
#       npt.assert_allclose(path_weights, [-3 + 18, 21, 0])
#       npt.assert_array_equal(alignment_labels, [
#           [2, 2],
#           [2, 0],
#           [0, 0],
#       ])

#     # String forward, i.e. shortest distance after intersection with a string.
#     labels = torch.Tensor([[1, 2, 0], [2, 1, 0], [1, 2, 0]]).float()
#     num_labels = torch.Tensor([1, 1, 0]).float()
#     for semiring_name, expected in [
#         ('MaxTropical', [-2 + 13, 21, 0]),
#         ('Real', [(-1) * 11 + (-2) * 13, 21, 1]),
#         ('Log', [torch.logsumexp(torch.Tensor([-1 + 11, -2 + 13])), 21., 0.])
#     ]:
#       semiring = getattr(last_torch.semirings, semiring_name)
#       with self.subTest(f'string_forward/{semiring_name}'):
#         npt.assert_allclose(
#             lattice._string_forward(
#                 cache=None,
#                 frames=frames,
#                 num_frames=num_frames,
#                 labels=labels,
#                 num_labels=num_labels,
#                 semiring=semiring), expected)
#       with self.subTest(f'string_forward non-reachable/{semiring_name}'):
#         npt.assert_array_equal(
#             lattice._string_forward(
#                 cache=None,
#                 frames=frames,
#                 num_frames=num_frames,
#                 labels=labels,
#                 num_labels=torch.Tensor([3, 2, 1]),
#                 semiring=semiring), semiring.zeros([3]))

#     with self.subTest('call'):
#       log_loss = lattice(
#           frames=frames,
#           num_frames=num_frames,
#           labels=labels,
#           num_labels=num_labels,
#           cache=None)
#       npt.assert_allclose(
#           log_loss, [
#               torch.logsumexp(
#                   torch.Tensor([
#                       -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
#                       -3 + 16, -3 + 17, -3 + 18
#                   ])) - torch.logsumexp(torch.Tensor([-1 + 11, -2 + 13])),
#               torch.logsumexp(torch.Tensor([19, 20, 21])) - 21., 0.
#           ],
#           rtol=1e-6)

#   def test_arc_marginals(self):
#     # Test _backward() by computing arc marginals. This is a bit easier to debug
#     # than the full-on forward-backward.
#     vocab_size = 2
#     context_size = 1
#     lattice = last_torch.RecognitionLattice(
#         context=last_torch.contexts.FullNGram(
#             vocab_size=vocab_size, context_size=context_size),
#         alignment=last_torch.alignments.FrameDependent(),
#         weight_fn_cacher_factory=weight_fn_cacher_factory,
#         weight_fn_factory=weight_fn_factory)
#     frames = torch.rand([4, 6, 8])
#     num_frames = torch.Tensor([6, 3, 2, 0])
#     cache = lattice.build_cache()

#     # Compute expected marginals using autodiff.
#     def forward(masks):
#       blank_mask, lexical_mask = masks
#       log_z, _ = lattice._forward(
#           cache=cache,
#           frames=frames,
#           num_frames=num_frames,
#           semiring=last_torch.semirings.Log,
#           blank_mask=[blank_mask],
#           lexical_mask=[lexical_mask])
#       return torch.sum(log_z)

#     num_context_states, _ = lattice.context.shape()
#     blank_mask = torch.zeros([*frames.shape[:-1], num_context_states])
#     lexical_mask = torch.zeros(
#         [*frames.shape[:-1], num_context_states, vocab_size])
#     expected_marginals = torch.autograd.grad(forward)((blank_mask, lexical_mask))
#     # Compute marginals using _backward().
#     def arc_marginals(frames, num_frames):

#       def arc_marginals_callback(weight_vjp_fn, carry, blank_marginal,
#                                  lexical_marginals):
#         del weight_vjp_fn
#         del carry
#         next_carry = None
#         outputs = (blank_marginal, lexical_marginals)
#         return next_carry, outputs

#       log_z, alpha_0_to_T_minus_1 = lattice._forward(  # pylint: disable=invalid-name
#           cache=cache,
#           frames=frames,
#           num_frames=num_frames,
#           semiring=last_torch.semirings.Log)
#       _, (blank_marginal, lexical_marginals) = lattice._backward(
#           cache=cache,
#           frames=frames,
#           num_frames=num_frames,
#           log_z=log_z,
#           alpha_0_to_T_minus_1=alpha_0_to_T_minus_1,
#           init_callback_carry=None,
#           callback=arc_marginals_callback)
#       return blank_marginal, lexical_marginals

#     actual_marginals = arc_marginals(frames, num_frames)
#     pytree.tree_map(
#         functools.partial(npt.assert_allclose, rtol=1e-3), actual_marginals,
#         expected_marginals)

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

    expected_log_z, expected_vjp_fn = torch.func.vjp(forward, frames)

    def forward_backward(frames):
        cache = lattice.build_cache() 
        return lattice._forward_backward(
            cache=cache,
            frames=frames,
            num_frames=num_frames)

    (actual_log_z, _), actual_vjp_fn = torch.func.vjp(forward_backward, frames)

    npt.assert_allclose(actual_log_z.detach().numpy(), expected_log_z.detach().numpy(), rtol=0.2)

    # TODO: Debug backward w/ setup_context
    # for g in [
    #     torch.ones_like(expected_log_z),
    #     torch.rand(expected_log_z.shape)
    # ]:
    #     expected_grads = expected_vjp_fn(g)
    #     actual_grads = actual_vjp_fn(g)
    #     pytree.tree_map(
    #         functools.partial(npt.assert_allclose, rtol=1e-3, atol=1e-6),
    #         actual_grads, expected_grads)

if __name__ == '__main__':
  absltest.main()