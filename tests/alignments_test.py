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

"""Tests for alignments."""

from absl.testing import absltest
import torch 
from last_torch import alignments
from last_torch import contexts
from last_torch import semirings
import numpy.testing as npt


class AlignmentsTest(absltest.TestCase):

  def test_shift_down(self):
    npt.assert_array_equal(
        alignments.shift_down(torch.Tensor([1, 2, 3]), semirings.Real), [0, 1, 2])
    npt.assert_array_equal(
        alignments.shift_down(
            torch.Tensor([[1, 2, 3], [4, 5, 6]]), semirings.Real),
        [[0, 1, 2], [0, 4, 5]])
    npt.assert_array_equal(
        alignments.shift_down(
            torch.Tensor([[1, 2, 3], [4, 5, 6]]).to(torch.float32),
            semirings.Log), [[-torch.inf, 1, 2], [-torch.inf, 4, 5]])

class FrameDependentTest(absltest.TestCase):

  def test_topology(self):
    alignment = alignments.FrameDependent()
    self.assertEqual(alignment.num_states(), 1)
    self.assertEqual(alignment.start(), 0)
    self.assertEqual(alignment.blank_next(0), 0)
    self.assertEqual(alignment.lexical_next(0), 0)
    self.assertListEqual(alignment.topological_visit(), [0])

  def test_forward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameDependent()
    alpha = torch.rand([3])
    blank = torch.rand([3])
    lexical = torch.rand([3, 2])

    # Single.
    next_alpha = alignment.forward(
        alpha=alpha,
        blank=[blank],
        lexical=[lexical],
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0],
        alpha[1] * blank[1] + torch.sum(alpha * lexical[:, 0]),
        alpha[2] * blank[2] + torch.sum(alpha * lexical[:, 1]),
    ])
    # Batched.
    batched_next_alpha = alignment.forward(
        alpha=alpha.unsqueeze(0),
        blank=[blank.unsqueeze(0)],
        lexical=[lexical.unsqueeze(0)],
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha.unsqueeze(0))

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.forward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          context=context,
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.forward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          context=context,
          semiring=semirings.Real)
