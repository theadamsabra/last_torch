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

"""Tests for weight_fns."""

from last_torch import weight_fns
from absl.testing import absltest
import torch
import numpy.testing as npt


class WeightFnTets(absltest.TestCase):

    def test_hat_normalize(self):
        blank = torch.Tensor([2, 7])
        lexical = torch.Tensor([[0, 1], [3, 5]])
        expect_blank = torch.Tensor([-0.126928, -0.000912])
        expect_lexical = torch.Tensor([[-3.44019, -2.44019], [-9.12784, -7.12784]])
        actual_blank, actual_lexical = weight_fns.hat_normalize(blank, lexical)
        npt.assert_allclose(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
        npt.assert_allclose(actual_lexical, expect_lexical, rtol=1e-3, atol=1e-6)
    
    def test_log_softmax_normalize(self):
        blank = torch.Tensor([2, 7])
        lexical = torch.Tensor([[0, 1], [3, 5]])
        expect_blank = torch.Tensor([-0.407606, -0.142932])
        expect_lexical = torch.Tensor([[-2.407606, -1.407606], [-4.142932, -2.142932]])
        actual_blank, actual_lexical = weight_fns.log_softmax_normalize(blank, lexical)
        npt.assert_allclose(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
        npt.assert_allclose(actual_lexical, expect_lexical, rtol=1e-3, atol=1e-6)


class JointWeightFnTest(absltest.TestCase):

  def test_call(self):
    weight_fn = weight_fns.JointWeightFn(vocab_size=3, hidden_size=8)
    frame = torch.rand((2, 4))
    cache = torch.rand((6, 5))  # context embeddings.
    
    with self.subTest('all context states'):
      blank, lexical = weight_fn(cache, frame)
      npt.assert_equal(blank.shape, (2, 6))
      npt.assert_equal(lexical.shape, (2, 6, 3))

    # TODO: Add per state calculation with assert all close
    with self.subTest('per-state shapes'):
      state = torch.Tensor([2, 4])
      blank_per_state, lexical_per_state = weight_fn(cache, frame, state)
      npt.assert_equal(blank_per_state.shape, (2,))
      npt.assert_equal(lexical_per_state.shape, (2,3))


class SharedEmbCacher(absltest.TestCase):

  def test_call(self):
    NUM_CONTEXT_STATES = 4
    EMBEDDING_SIZE = 5
    cacher = weight_fns.SharedEmbCacher(num_context_states=NUM_CONTEXT_STATES, embedding_size=EMBEDDING_SIZE)
    embedding = cacher()
    idx = torch.Tensor([i for i in range(0, NUM_CONTEXT_STATES)]).long()

    npt.assert_equal(embedding(idx).shape, 
                    (NUM_CONTEXT_STATES, EMBEDDING_SIZE)
                    )