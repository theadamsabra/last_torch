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

"""Recognition lattice."""

from collections.abc import Callable, Sequence
import functools
from typing import Any, Generic, Optional, Protocol, TypeVar

import torch
import torch.nn as nn

from last_torch import alignments
from last_torch import contexts
from last_torch import semirings
from last_torch import weight_fns

DType = Any
T = TypeVar('T')

class RecognitionLattice(nn.Module, Generic[T]):
  """Recognition lattice in GNAT-style formulation and operations over it.

  A RecognitionLattice provides operations used in training and inference, such
  as computing the negative-log-probability loss, or finding the highest scoring
  alignment path.

  Following the formulation in GNAT, we combines the three modelling components
  to define a RecognitionLattice:
  -   Context dependency: The finite automaton that models output history. See
      last.contexts.ContextDependency for details.
  -   Alignment lattice: The finite automaton that models the alignment between
      input frames and output labels. See
      last.alignments.TimeSyncAlignmentLattice for details.
  -   Weight function: The neural network that produces arc weights from any
      context state given an input frame. See last.weight_fns for details.

  Given a sequence of `T` input frames, its recognition lattice is the following
  finite automaton:
  -   States: The states are pairs of an alignment state and a context state.
      For any alignment state (t, a) (t is the index of the current frame) and
      context state c, there is a state (t, a, c) in the recognition lattice.
  -   Start state: (0, s_a, s_c), where s_a is the start state in the
      frame-local alignment lattice (see
      last.contexts.TimeSyncAlignmentLattice), and s_c is the start state in the
      context dependency.
  -   Final states: (T, s_a, c) for any context state c.
  -   Arcs:
      -   Blank arcs: For any blank arc `(t, a) -> (t', a')` in the alignment
          lattice, and any context state c, there is an arc
          `(t, a, c) --blank-> (t', a', c)` in the recognition lattice.
      -   Lexical arcs: For any lexical arc `(t, a) -> (t', a')` in the
          alignment lattice, and any arc `c --y-> c'` in the context dependency,
          there is an arc `(t, a, c) --y-> (t', a', c')` in the recognition
          lattice.
  -   Arc weights: For each state (t, a, c), the weight function receives the
      t-th frame and the context state c, and produces arc weights for blank
      and lexical arcs. Notably while in principle weight functions can also
      depend on the alignment state, in practice we haven't yet encountered a
      compelling reason for such weight functions. Thus for the sake of
      simplicity, weight functions currently only depend on the frame and the
      context state.

  The arc weights can be used to model the conditional distribution of the paths
  in the recognition lattice, especially those that lead to the reference label
  sequence. A RecognitionLattice can be either a locally normalized model, or
  a globally normalized model,
  -   A locally normalized model uses last.weight_fns.LocallyNormalizedWeightFn,
      where the arc weights from the same recognition lattice state add up to 1
      after taking an exponential. The probability of
      P(alignment labels | frames) is simply the product of arc weights on the
      alignment path after exponential.
  -   A globally normalized model uses a WeightFn that is not a subclass of
      last.weight_fns.LocallyNormalizedWeightFn. To obtain a probabilistic
      interpretation, we normalize the path weights with the sum of the
      exponentiated weights of all possible paths in the recognition lattice.

  Globally normalized models are more expensive to train, but they have various
  advantages. See the GNAT paper for more details.

  Attributes:
    context: Context dependency.
    alignment: Alignment lattice.
    weight_fn_cacher_factory: Callable that builds a WeightFnCacher given the
      context dependency.
    weight_fn_factory: Callable that builds a WeightFn given the context
      dependency.
  """
  def __init__(self, context:contexts.ContextDependency, 
               alignment:alignments.TimeSyncAlignmentLattice,
               weight_fn_cacher_factory: Callable[[contexts.ContextDependency],
                                                  weight_fns.WeightFnCacher[T]],
                weight_fn_factory: Callable[[contexts.ContextDependency],
                                            weight_fns.WeightFn[T]]):
    self.context = context
    self.alignment = alignment
    self.weight_fn_cacher_factory = weight_fn_cacher_factory
    self.weight_fn_factory = weight_fn_factory

    self.weight_fn_cacher = self.weight_fn_cacher_factory(self.context)
    self.weight_fn = self.weight_fn(self.context)

  def build_cache(self) -> T:
    """Builds the weight function cache.

    Weight functions are implemented as a pair of WeightFn and WeightFnCacher to
    avoid unnecessary recomputation (see last.weight_fns for more details).
    build_cache() builds the cached static data that can be used in other public
    methods.

    Returns:
      Cached data.
    """
    return self.weight_fn_cacher()
  
  def forward(self, 
              frames: torch.Tensor,
              num_frames: torch.Tensor,
              labels: torch.Tensor,
              num_labels: torch.Tensor,
              cache: Optional[T] = None) -> torch.Tensor:
    """Compute the negative sequence log-probability loss.

    There can be multiple alignment paths from the input frames to the output
    labels. The conditional probability P(labels | frames) is thus the sum
    of probabilities P(alignment labels | frames) for all possible alignments
    that produce the given label sequence. Interpreting the arc weights as
    (possibly unnormalized) log-probabilities, this function computes
    -log P(labels | frames) for both locally and globally normalized models.

    Args:
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      labels: [batch_dims..., max_num_labels] padded label sequences.
      num_labels: [batch_dims...] number of labels.
      cache: Optional weight function cache data.

    Returns:
      [batch_dims...] negative sequence log-prob loss.
    """
    batch_dims = num_frames.shape
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    if labels.shape[:-1] != batch_dims:
      raise ValueError('labels and num_frames have different batch_dims: '
                       f'{labels.shape[:-1]} vs {batch_dims}')
    if num_labels.shape != batch_dims:
      raise ValueError('num_labels and num_frames have different batch_dims: '
                       f'{num_labels.shape} vs {batch_dims}')

    semiring = semirings.Log 
    if cache is None:
      cache = self.weight_fn_cacher()
    numerator = self._string_forward(
      cache = cache,
      frames = frames,
      num_frames = num_frames,
      semiring = semiring)
    if isinstance(self.weight_fn, weight_fns.LocallyNormalizedWeightFn):
      return -numerator
    denominator = self._forward_backward(
      cache=cache, frames=frames, num_frames=num_frames
    )
    return denominator - numerator

  def shortest_path(
      self,
      frames: torch.Tensor,
      num_frames: torch.Tensor,
      cache: Optional[T] = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the shortest path in the recognition lattice.

    The shortest path is the path with the highest score, in other words, the
    "shoretst" path under the max-tropical semiring.

    Args:
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      cache: Optional weight function cache data.

    Returns:
      (alignment_labels, num_alignment_labels, path_weights) tuple,
      -   alignment_labels: [batch_dims..., max_num_alignment_labels] padded
          alignment labels, either blank (0) or lexical (1 to vocab_size).
      -   num_alignment_labels: [batch_dims...] number of alignment labels.
      -   path_weights: [batch_dims...] path weights.
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    max_num_frames = frames.shape[-2]
    num_alignment_states = self.alignment.num_states()

    if cache is None:
      cache = self.weight_fn_cacher()

    # Find shortest path by differentiating shortest distance under the tropical
    # semiring. Called forward_helper to not be confused with forward pass in torch
    def forward_helper(lattice: 'RecognitionLattice', lexical_mask: torch.Tensor):
      path_weights, _ = lattice._forward(
        cache=cache,
        frames=frames,
        num_frames=num_frames,
        semiring=semirings.MaxTropical,
        lexical_mask=[
          lexical_mask[..., i, None, :]
          for i in range(num_alignment_states)
        ])
      return path_weights
    
    _, vocab_size = self.context.shape()
    lexical_mask = torch.zeros(
      [*batch_dims, max_num_frames, num_alignment_states, vocab_size]
    )
    path_weights, vjp_fn = torch.autograd.functional.vjp(
      forward_helper, lexical_mask
    )
    _, viterbi_lexical_mask = vjp_fn(torch.ones_like(path_weights))
    is_blank = torch.all(viterbi_lexical_mask == 0, dim=-1)
    alignment_labels = torch.where(is_blank, 0,
                                   1 + torch.argmax(viterbi_lexical_mask, dim=-1))
    alignment_labels = alignment_labels.reshape([*batch_dims, -1])
    num_alignment_labels = num_alignment_states * num_frames
    return alignment_labels, num_alignment_labels, path_weights
  
  # Private methods:
  def _string_forward(self, cache: T, frames: torch.Tensor,
                    num_frames: torch.Tensor, labels: torch.Tensor,
                    num_labels: torch.Tensor,
                    semiring: semirings.Semiring[torch.Tensor]) -> torch.Tensor:
    """Shortest distance on the intersection of the recognition lattice and an output string (the label sequence) computed using the forward algorithm.

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      labels: [batch_dims..., max_num_labels] padded label sequence.
      num_labels: [batch_dims...] number of labels.
      semiring: Semiring to use for shortest distance computation.

    Returns:
      [batch_dims...] shortest distance.
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                        f'{frames.shape[:-2]} vs {batch_dims}')
    if labels.shape[:-1] != batch_dims:
      raise ValueError('labels and num_frames have different batch_dims: '
                        f'{labels.shape[:-1]} vs {batch_dims}')
    if num_labels.shape != batch_dims:
      raise ValueError('num_labels and num_frames have different batch_dims: '
                        f'{num_labels.shape} vs {batch_dims}')

    # Calculate arc weights for all visited context states.
    #
    # We can't fit into memory all
    # O(batch_size * max_num_frames * (max_num_labels + 1) * (vocab_size + 1))
    # arcs, thus we use a scan loop over the (max_num_labels + 1) axis to
    # produce just the O(batch_size * max_num_frames * (max_num_labels + 1))
    # arcs actually needed later. This is better than scanning over the
    # max_num_frames axis because weight_fn can be vectorized over multiple
    # frames for the same state (weight_fn with states often involves
    # gathering).

    # Before vmaps
    # - frame is [batch_dims..., hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims...], [batch_dims..., vocab_size]).
    compute_weights = (
        lambda weight_fn, frame, state: weight_fn(cache, frame, state))
    # Add time dimension on frame
    # - frame is [batch_dims..., max_num_frames, hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims..., max_num_frames],
    # [batch_dims..., max_num_frames, vocab_size]).
    compute_weights = torch.vmap(
        compute_weights,
        in_dims=(-2, None),
        out_dims=(-1,-2))