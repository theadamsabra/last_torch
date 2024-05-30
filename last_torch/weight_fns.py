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

from typing import Generic, TypeVar, Optional, Callable
from torch import nn
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
  weight_fn: WeightFn[T]
  normalize: Callable[[torch.Tensor, torch.Tensor],
                      tuple[torch.Tensor, torch.Tensor]] = hat_normalize

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
  vocab_size: int
  hidden_size: int

  def forward(
      self,
      cache: torch.Tensor,
      frame: torch.Tensor,
      state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    context_embeddings = cache
    if state is None:
      # TODO: Intentionally breaking this code to debug in testing:
      frame = frame[..., torch.newaxis, :]
    else:
      context_embeddings = context_embeddings[state]
    # TODO: will follow the projection as shown in the JAX implementation.
    # Speak to Ke later if changes are made.
    # Will also debug during testing for in_features parameter:
    projected_context_embeddings = nn.Linear(
      in_features=self.hidden_size,
      out_features=self.hidden_size,
      bias=False
    )(context_embeddings)
    projected_frame = nn.Linear(
      in_features=self.hidden_size,
      out_features=self.hidden_size,
      bias=False
    )(frame)
