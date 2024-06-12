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

"""Tests for contexts."""

from absl.testing import absltest
import  torch
from last_torch import contexts
from last_torch import semirings
import numpy.testing as npt


class FullNGramTest(absltest.TestCase):

    def test_invalid_args(self):
        with self.assertRaisesRegex(ValueError, 'vocab_size should be > 0'):
            contexts.FullNGram(vocab_size=0, context_size=1)
        with self.assertRaisesRegex(ValueError, 'context_size should be >= 0'):
            contexts.FullNGram(vocab_size=1, context_size=-1)
    
    def test_invalid_inputs(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        with self.assertRaisesRegex(ValueError,
                                    r'weights\.shape\[-2:\] should be \(3, 2\)'):
            context.forward_reduce(torch.zeros([3, 4]), semirings.Real)
        with self.assertRaisesRegex(ValueError,
                                    r'weights\.shape\[-1\] should be 3'):
            context.backward_broadcast(torch.zeros([4]))
    
    def test_context_size_0_basics(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        self.assertEqual(context.num_states(), 1)
        self.assertEqual(context.shape(), (1, 3))
        self.assertEqual(context.start(), 0)
    
    def test_context_size_0_next_state(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        npt.assert_array_equal(context.next_state(torch.tensor(0), torch.tensor(1)), 0)
        npt.assert_array_equal(
            context.next_state(torch.Tensor([0, 0, 0]), torch.Tensor([0, 1, 2])),
            [0, 0, 0])
        npt.assert_array_equal(
            context.next_state(torch.Tensor([[0, 0, 0]]), torch.Tensor([[0, 1, 2]])),
            [[0, 0, 0]])
        # Epsilon transitions.
        npt.assert_array_equal(
            context.next_state(torch.Tensor([0, 1, 2]), torch.Tensor([0, 0, 0])),
            [0, 1, 2])
    
    def test_context_size_0_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        npt.assert_array_equal(
            context.forward_reduce(torch.Tensor([[1, 2, 3]]), semirings.Real), [6])
        npt.assert_array_equal(
            context.forward_reduce(
                torch.arange(6).reshape((2, 1, 3)), semirings.Real), [[3], [12]])
        npt.assert_array_equal(
            context.forward_reduce(
                torch.arange(6).reshape((1, 2, 1, 3)), semirings.Real), [[[3], [12]]])

    def test_context_size_0_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        npt.assert_array_equal(
            context.backward_broadcast(torch.Tensor([1])), [[1, 1, 1]])
        npt.assert_array_equal(
            context.backward_broadcast(torch.Tensor([[1], [2]])),
            [[[1, 1, 1]], [[2, 2, 2]]])
        npt.assert_array_equal(
            context.backward_broadcast(torch.Tensor([[[1], [2]]])),
            [[[[1, 1, 1]], [[2, 2, 2]]]]) 