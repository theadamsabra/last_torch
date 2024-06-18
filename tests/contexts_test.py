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
    
    def test_context_size_1_basics(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        self.assertEqual(context.num_states(), 3)
        self.assertEqual(context.shape(), (3, 2))
        self.assertEqual(context.start(), 0)

    def test_context_size_1_next_state(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        npt.assert_array_equal(context.next_state(torch.tensor(0), torch.tensor(1)), 1)
        npt.assert_array_equal(
            context.next_state(torch.Tensor([0, 1, 2]), torch.Tensor([1, 2, 1])),
            [1, 2, 1])
        npt.assert_array_equal(
            context.next_state(torch.Tensor([[0, 1, 2]]), torch.Tensor([[1, 2, 1]])),
            [[1, 2, 1]])
        # Epsilon transitions.
        npt.assert_array_equal(
            context.next_state(torch.Tensor([0, 1, 2]), torch.Tensor([0, 0, 0])),
            [0, 1, 2])

    def test_context_size_1_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        npt.assert_array_equal(
            context.forward_reduce(torch.arange(6).reshape((3, 2)), semirings.Real),
            [0, 0 + 2 + 4, 1 + 3 + 5])
        npt.assert_array_equal(
            context.forward_reduce(
                torch.arange(6).reshape((1, 3, 2)), semirings.Real),
            [[0, 0 + 2 + 4, 1 + 3 + 5]])
        npt.assert_array_equal(
            context.forward_reduce(
                torch.arange(6).reshape((1, 1, 3, 2)), semirings.Real),
            [[[0, 0 + 2 + 4, 1 + 3 + 5]]]) 
    
    def test_context_size_1_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        npt.assert_array_equal(
            context.backward_broadcast(torch.arange(3)), [[1, 2], [1, 2], [1, 2]])
        npt.assert_array_equal(
            context.backward_broadcast(torch.arange(3).reshape((1, 3))),
            [[[1, 2], [1, 2], [1, 2]]])
        npt.assert_array_equal(
            context.backward_broadcast(torch.arange(3).reshape((1, 1, 3))),
            [[[[1, 2], [1, 2], [1, 2]]]])

    def test_context_size_2_basics(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        self.assertEqual(context.num_states(), 13)
        self.assertEqual(context.shape(), (13, 3))
        self.assertEqual(context.start(), 0)

    def test_context_size_2_next_state(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        npt.assert_array_equal(
            context.next_state(
                torch.Tensor([0, 1, 3, 4, 12]), torch.Tensor([1, 2, 3, 1, 2])),
            [1, 5, 12, 4, 11])
        # Epsilon transitions.
        npt.assert_array_equal(
            context.next_state(
                torch.Tensor([0, 1, 3, 4, 12]), torch.Tensor([0, 0, 0, 0, 0])),
            [0, 1, 3, 4, 12])

    def test_context_size_2_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        npt.assert_array_equal(
            context.forward_reduce(
                torch.arange(39).reshape((1, 13, 3)), semirings.Real), [[
                    0, 0, 1, 2, 3 * 4 + 54, 4 * 4 + 54, 5 * 4 + 54, 6 * 4 + 54,
                    7 * 4 + 54, 8 * 4 + 54, 9 * 4 + 54, 10 * 4 + 54, 11 * 4 + 54
                ]])

    def test_context_size_2_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        npt.assert_array_equal(
            context.backward_broadcast(torch.arange(13).reshape((1, 13))),
            [[[1, 2, 3]] + [[4, 5, 6], [7, 8, 9], [10, 11, 12]] * 4])

    def test_walk_states(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        self.assertEqual(
            context.walk_states(torch.zeros([2, 3, 4], dtype=torch.int32)).shape,
            (2, 3, 5))
        npt.assert_array_equal(
            context.walk_states(torch.Tensor([2, 3, 1])), [0, 2, 9, 10])
        # Epsilon transitions.
        npt.assert_array_equal(
            context.walk_states(torch.Tensor([2, 0, 0, 3, 1])), [0, 2, 2, 2, 9, 10])


class NextStateTableTest(absltest.TestCase):

  def test_invalid_args(self):
    with self.assertRaisesRegex(ValueError,
                                'next_state_table should have a non-zero size'):
      contexts.NextStateTable(torch.zeros([1, 0], dtype=torch.int32))
    with self.assertRaisesRegex(ValueError,
                                'next_state_table should have a non-zero size'):
      contexts.NextStateTable(torch.zeros([0, 1], dtype=torch.int32))
    with self.assertRaisesRegex(ValueError,
                                'next_state_table should have shape'):
      contexts.NextStateTable(torch.zeros([1], dtype=torch.int32))
    with self.assertRaisesRegex(ValueError,
                                'next_state_table should be an int32 ndarray'):
      contexts.NextStateTable(torch.zeros([2, 3]))
    
  def test_from_full_n_gram(self):
    next_state_table = contexts.FullNGram(
        vocab_size=3, context_size=2).next_state_table()
    self.assertEqual(next_state_table.shape, (13, 3))
    context = contexts.NextStateTable(next_state_table.to(torch.int32))

    with self.subTest('basics'):
       self.assertEqual(context.shape(), (13,3))
       self.assertEqual(context.start(), 0)
    
    with self.subTest('next_state'):
      npt.assert_array_equal(
          context.next_state(
              torch.Tensor([0, 1, 3, 4, 12]), torch.Tensor([1, 2, 3, 1, 2])),
          [1, 5, 12, 4, 11])
      # Epsilon transitions.
      npt.assert_array_equal(
          context.next_state(
              torch.Tensor([0, 1, 3, 4, 12]), torch.Tensor([0, 0, 0, 0, 0])),
          [0, 1, 3, 4, 12])