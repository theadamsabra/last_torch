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

"""Weight functions."""

import abc
import torch
import einops

from typing import Generic, TypeVar, Optional, Callable
from torch import nn
from torch.utils._pytree import tree_map
from torch.nn import functional as F

# Weight functions are the only components in GNAT with trainable parameters. We
# implement weight functions in two parts: WeightFn and WeightFnCacher.
#
# A WeightFn is a neural network that computes the arc weights for a given
# frame. Sometimes it requires static data that doesn't depend on the frames but
# is expensive to compute (e.g. the context embeddings of the shared-rnn weight
# function). We avoid unnecessarily recomputing such static data by off-loading
# the computation of static data to a separate WeightFnCacher (e.g.
# SharedRNNCacher).
#
# This way, whenever we know the static data doesn't change (e.g. when the
# underlying model parameters don't change such as during inference), we can
# reuse the result from WeightFnCacher as cache.

T = TypeVar('T')

class WeightFn(nn.Module, Generic[T], abc.ABC):
  """Interface for weight functions.

  A weight function is a neural network that computes the arc weights from all
  or some context states for a given frame. A WeightFn is used in pair with a
  WeightFnCacher that produces the static data cache, e.g. JointWeightFn can be
  used with SharedEmbCacher or SharedRNNCacher.
  """  

  @abc.abstractmethod
  def forward(
    self,
    cache: T,
    frame: torch.Tensor,
    state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes arc weights for a given frame.

    Args:
      cache: Cached data from the corresponding WeightFnCacher.
      frame: [batch_dims..., feature_size] input frame.
      state: None or int32 array broadcastable to [batch_dims...]. If None,
        compute arc weights for all context states. Otherwise, compute arc
        weights for the specified context state.

    Returns:
      (blank, lexical) tuple.

      If state is None:
      - blank: [batch_dims..., num_context_states] weights for blank arcs.
        blank[..., p] is the weight of producing blank from context state p.
      - lexical: [batch_dims..., num_context_states, vocab_size] weights for
        lexical arcs. lexical[..., p, y] is the weight of producing label y from
        context state p.

      If state is not None:
      - blank: [batch_dims...] weights for blank arcs from the corresponding
        `state`.
      - lexical: [batch_dims..., vocab_size] weights for lexical arcs.
        lexical[..., y] is the weight of producing label y from the
        corresponding `state`.
    """
    raise NotImplementedError


class WeightFnCacher(nn.Module, Generic[T], abc.ABC):
  """Interface for weight function cachers.

  A weight function cacher prepares static data that may require expensive
  computational work. For example: the context state embeddings used by
  JointWeightFn can be from running an RNN on n-gram label sequences
  """

  @abc.abstractmethod
  def forward(self) -> T:
    """Builds the cached data."""


def hat_normalize(blank: torch.Tensor,
                  lexical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Local normalization used in the Hybrid Autoregressive Transducer (HAT) paper.

  The sigmoid of the blank weight is directly interpreted as the probability of
  blank. The lexical probability is then normalized with a log-softmax.

  Args:
    blank: [batch_dims...] blank weight.
    lexical: [batch_dims..., vocab_size] lexical weights.

  Returns:
    Normalized (blank, lexical) weights.
  """
  # Outside normalizer
  z = torch.log(1 + torch.exp(blank))
  normalized_blank = blank - z
  normalized_lexical = F.log_softmax(lexical, dim=-1) - torch.unsqueeze(z, -1) 
  return normalized_blank, normalized_lexical


def log_softmax_normalize(
    blank: torch.Tensor,
    lexical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Standard log-softmax local normalization.

  Weights are concatenated and then normalized together.

  Args:
    blank: [batch_dims...] blank weight.
    lexical: [batch_dims..., vocab_size] lexical weights.

  Returns:
    Normalized (blank, lexical) weights.
  """
  all_weights = torch.concatenate([torch.unsqueeze(blank, -1), lexical], dim=-1)
  all_weights = F.log_softmax(all_weights, dim=-1)
  return all_weights[..., 0], all_weights[..., 1:]


class LocallyNormalizedWeightFn(WeightFn[T]):
  """Wrapper for turning any weight function into a locally normalized one.

  This is the recommended way of obtaining a locally normalized weight function.
  Algorithms such as those that computes the sequence log-loss may rely on a
  weight function being of this type to eliminate unnecessary denominator
  computation.

  It is thus also important for the normalize function to be mathematically
  correct: let (blank, lexical) be the pair of weights produced by the normalize
  function, then `jnp.exp(blank) + jnp.sum(jnp.exp(lexical), axis=-1)` should be
  approximately equal to 1.

  Attributes:
    weight_fn: Underlying weight function.
    normalize: Callable that produces normalized log-probabilities from (blank,
      lexical) weights, e.g. hat_normalize() or log_softmax_normalize().
  """
  def __init__(self, weight_fn: WeightFn[T], 
               normalize: Callable[[torch.Tensor, torch.Tensor],
                      tuple[torch.Tensor, torch.Tensor]] = hat_normalize,
               *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.weight_fn = weight_fn
    self.normalize = normalize

  def forward(
      self,
      cache: T,
      frame: torch.Tensor,
      state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    blank, lexical = self.weight_fn(cache, frame, state)
    return self.normalize(blank, lexical)


class JointWeightFn(WeightFn[torch.Tensor]):
  r"""Common implementation of both the shared-emb and shared-rnn weight functions.

  To use shared-emb weight functions, pair this with a SharedEmbCacher. To use
  shared-rnn weight functions, pair this with a SharedRNNCacher. More generally,
  this weight function works with any WeightFnCacher that produces a
  [num_context_states, embedding_size] context embedding table.

  Attributes:
    vocab_size: Size of the lexical output vocabulary (not including the blank),
      i.e. $|\Sigma|$.
    hidden_size: Hidden layer size.
  """
  def __init__(self, vocab_size: int, hidden_size: int, 
               device: Optional[str] = 'cpu', *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.device = device

  def forward(
      self,
      cache: torch.Tensor,
      frame: torch.Tensor,
      state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    context_embeddings = cache
    context_embeddings.to(self.device)

    if state is None:
      frame = torch.unsqueeze(frame, 1) 
    else:
      context_embeddings = torch.index_select(context_embeddings, dim=0,
                                              index=state.long())

    context_projection = nn.Linear(context_embeddings.shape[-1], 
                                   self.hidden_size, bias=False, device=self.device)
    blank_projection = nn.Linear(frame.shape[-1], self.hidden_size,
                                 bias=False, device=self.device)

    # TODO: will follow the projection as shown in the JAX implementation.
    # Speak to Ke later if changes are made and track as needed.
    projected_context_embeddings = context_projection(context_embeddings)
    projected_frame = blank_projection(frame)

    joint = F.tanh(projected_context_embeddings + projected_frame)

    joint_projection_to_blank = nn.Linear(joint.shape[-1], 1, device=self.device)
    joint_projection_to_vocab = nn.Linear(joint.shape[-1], self.vocab_size, device=self.device)

    blank = torch.squeeze(
      joint_projection_to_blank(joint), dim=-1
    )
    lexical = joint_projection_to_vocab(joint)
    return blank, lexical


class SharedEmbCacher(WeightFnCacher[torch.Tensor]):
  """A randomly initialized, independent context embedding table.

  The result context embedding table can be used with JointWeightFn.
  """
  def __init__(self, num_context_states: int, embedding_size: int, device: Optional[str] = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_context_states = num_context_states
    self.embedding_size = embedding_size
    self.device = device if device else 'cpu'
  
  def forward(self) -> torch.Tensor:
    return torch.nn.Embedding(self.num_context_states, self.embedding_size).to(self.device)
  

class SharedRNNCacher(WeightFnCacher[torch.Tensor]):
  """Builds a context embedding table by running n-gram context labels through an RNN.

  This is usually used with last.contexts.FullNGram, where num_context_states =
  sum(vocab_size**i for i in range(context_size + 1)). The result context
  embedding table can be used with JointWeightFn.
  """
  def __init__(self, vocab_size: int, context_size: int, rnn_size: int, rnn_embedding_size:int,
               rnn_cell: Optional[nn.RNNCellBase] = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.vocab_size = vocab_size
    self.context_size = context_size
    self.rnn_size = rnn_size
    self.rnn_embedding_size = rnn_embedding_size
    self.rnn_cell = rnn_cell

  def _tile_rnn_state(self, state):
    return einops.repeat(state, 'n ... -> (n v) ...', v = self.vocab_size)

  def forward(self) -> torch.Tensor:
    if self.rnn_cell is None:
      rnn_cell = nn.LSTMCell(self.rnn_embedding_size, self.rnn_size)
    else:
      rnn_cell = self.rnn_cell

    feed_cell_state = rnn_cell._get_name() == 'LSTMCell'

    embed = nn.Embedding(self.vocab_size + 1, self.rnn_embedding_size)
    hidden_state, cell_state = rnn_cell(embed(torch.Tensor([0]).long()))
    parts = [cell_state]
    inputs = None
    for i in range(self.context_size):
      if i == 0:
        inputs = embed(torch.arange(1, self.vocab_size + 1))
      else:
        inputs = einops.repeat(inputs, 'n ... -> (v n) ...', v=self.vocab_size)

      tiled_hidden = tree_map(self._tile_rnn_state, hidden_state) 
      if feed_cell_state:
        tiled_cell_state = tree_map(self._tile_rnn_state, cell_state)
        hidden_state, cell_state = rnn_cell(
          inputs, (tiled_hidden, tiled_cell_state)
        )
      else:
        hidden_state, cell_state = rnn_cell(
          inputs, tiled_hidden 
        )
      parts.append(cell_state)
    
    return torch.concatenate(parts, axis=0)