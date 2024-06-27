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

  def test_backward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameDependent()
    alpha = torch.rand([3])
    blank = torch.rand([3])
    lexical = torch.rand([3, 2])
    beta = torch.rand([3])
    z = torch.rand([])

    # bakcward() always uses the log semiring

    # Single.
    log_next_beta, [blank_marginal], [lexical_marginal] = (
        alignment.backward(
            alpha=torch.log(alpha),
            blank=[torch.log(blank)],
            lexical=[torch.log(lexical)],
            beta=torch.log(beta),
            log_z=torch.log(z),
            context=context))
    next_beta = torch.exp(log_next_beta)
    npt.assert_allclose(
        next_beta, [
            blank[0] * beta[0] + lexical[0, 0] * beta[1] +
            lexical[0, 1] * beta[2],
            blank[1] * beta[1] + lexical[1, 0] * beta[1] +
            lexical[1, 1] * beta[2],
            blank[2] * beta[2] + lexical[2, 0] * beta[1] +
            lexical[2, 1] * beta[2],
        ],
        rtol=1e-4)
    npt.assert_allclose(blank_marginal, alpha * blank * beta / z, rtol=1e-4)
    npt.assert_allclose(
        lexical_marginal, [
            [
                alpha[0] * lexical[0, 0] * beta[1] / z,
                alpha[0] * lexical[0, 1] * beta[2] / z
            ],
            [
                alpha[1] * lexical[1, 0] * beta[1] / z,
                alpha[1] * lexical[1, 1] * beta[2] / z
            ],
            [
                alpha[2] * lexical[2, 0] * beta[1] / z,
                alpha[2] * lexical[2, 1] * beta[2] / z
            ],
        ],
        rtol=1e-4)

    # Batched.
    batched_log_next_beta, _, _ = (
        alignment.backward(
            alpha=torch.log(alpha).unsqueeze(0),
            blank=[torch.log(blank).unsqueeze(0)],
            lexical=[torch.log(lexical).unsqueeze(0)],
            beta=torch.log(beta).unsqueeze(0),
            log_z=torch.log(z).unsqueeze(0),
            context=context))
    npt.assert_allclose(batched_log_next_beta, log_next_beta.unsqueeze(0))

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.backward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          beta=beta,
          log_z=z,
          context=context)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.backward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          beta=beta,
          log_z=z,
          context=context)

  def test_string_forward(self):
    alignment = alignments.FrameDependent()
    alpha = torch.rand([4])
    blank = torch.rand([4])
    lexical = torch.rand([4])

    # Single.
    next_alpha = alignment.string_forward(
        alpha=alpha, blank=[blank], lexical=[lexical], semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0],
        alpha[1] * blank[1] + alpha[0] * lexical[0],
        alpha[2] * blank[2] + alpha[1] * lexical[1],
        alpha[3] * blank[3] + alpha[2] * lexical[2],
    ])

    # Batched.
    batched_next_alpha = alignment.string_forward(
        alpha=alpha.unsqueeze(0),
        blank=[blank.unsqueeze(0)],
        lexical=[lexical.unsqueeze(0)],
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha.unsqueeze(0))

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          semiring=semirings.Real)