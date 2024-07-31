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
