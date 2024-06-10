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

    def test_NullCacher(self):
       cacher = weight_fns.NullCacher()
       cache = cacher()
       self.assertIsNone(cache)

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


class SharedRNNCacher(absltest.TestCase):
   
   def test_call(self):
      pad = -2
      start = -1

      class FakeRNNCell(torch.nn.RNNCellBase):
        def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, device=None, dtype=None) -> None:
            super().__init__(input_size, hidden_size, bias, num_chunks, device, dtype)

        def forward(self, inputs, carry=None):
            if carry == None:
                carry = self.initialize_carry((1, self.hidden_size))

            carry = torch.concat(([carry[..., 1:], inputs[..., :1]]), dim=-1)
            return carry, carry

        def initialize_carry(self, input_shape):
            batch_dims = input_shape[:-1]
            return torch.full((*batch_dims, self.hidden_size), pad)

      with self.subTest('context_size=2'):
        embeddings = torch.broadcast_to(torch.Tensor([start, 1, 2, 3]).unsqueeze(-1), (4,6))
        cacher = weight_fns.SharedRNNCacher(
           vocab_size=3,
           context_size=2,
           rnn_size=4,
           rnn_embedding_size=6,
           rnn_cell=FakeRNNCell(input_size=3, hidden_size=4, bias=False,
                                num_chunks=1)
        )
        
        # update embeddings
        cacher.embedding = torch.nn.Embedding.from_pretrained(embeddings)

        npt.assert_array_equal(
            cacher(),
            [
                # Start.
                [pad, pad, pad, start],
                # Unigrams.
                [pad, pad, start, 1],
                [pad, pad, start, 2],
                [pad, pad, start, 3],
                # Bigrams.
                [pad, start, 1, 1],
                [pad, start, 1, 2],
                [pad, start, 1, 3],
                [pad, start, 2, 1],
                [pad, start, 2, 2],
                [pad, start, 2, 3],
                [pad, start, 3, 1],
                [pad, start, 3, 2],
                [pad, start, 3, 3],
            ],
        )
    
      with self.subTest('context_size=0'):
        cacher = weight_fns.SharedRNNCacher(
            vocab_size=3,
            context_size=0,
            rnn_size=4,
            rnn_embedding_size=6,
            rnn_cell=FakeRNNCell(input_size=3, hidden_size=4, bias=False,
                                num_chunks=1)
        )

        # update embeddings
        cacher.embedding = torch.nn.Embedding.from_pretrained(embeddings)

        npt.assert_array_equal(
            cacher(),
            [[pad, pad, pad, start]],
        )