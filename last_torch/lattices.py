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
from torch import utils
from torch._higher_order_ops import scan
import torch.nn as nn
import torch.utils._pytree as pytree 

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
    super().__init__()
    self.context = context
    self.alignment = alignment
    self.weight_fn_cacher_factory = weight_fn_cacher_factory
    self.weight_fn_factory = weight_fn_factory

    self.weight_fn_cacher = self.weight_fn_cacher_factory(self.context)
    self.weight_fn = self.weight_fn_factory(self.context)

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
      labels = labels,
      num_labels = num_labels,
      semiring = semiring)
    if isinstance(self.weight_fn, weight_fns.LocallyNormalizedWeightFn):
      return -numerator
    denominator, _ = self._forward_backward(
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
    def forward_helper(lexical_mask: torch.Tensor):
      path_weights, _ = self._forward(
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

    path_weights, vjp_fn = torch.func.vjp(
      forward_helper, lexical_mask
    )
    viterbi_lexical_mask = vjp_fn(torch.ones_like(path_weights))[0]
    is_blank = torch.all(viterbi_lexical_mask == 0, dim=-1)
    alignment_labels = torch.where(is_blank, 0,
                                   torch.argmax(viterbi_lexical_mask, dim=-1))
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

    # Add time dimension on frame
    # - frame is [batch_dims..., max_num_frames, hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims..., max_num_frames],
    # [batch_dims..., max_num_frames, vocab_size]).
    compute_weights = (
      lambda frame: self.weight_fn(cache, frame, self._state))
    # Add time dimension on frame
    # - frame is [batch_dims..., max_num_frames, hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims..., max_num_frames],
    # [batch_dims..., max_num_frames, vocab_size]).

    compute_weights = torch.vmap(
        compute_weights,
        randomness='same',
        in_dims=1,
        out_dims=-1
    )
    def make_safe_classes(y):
       return torch.where(y-1 < 0, 1, y)

    def gather_weight(weights, y):
      # weights: [batch_dims..., max_num_frames, vocab_size]
      # y: [batch_dims..., max_num_frames]
      # weights are for labels [1, vocab_size], so y-1 are the corresponding
      # indicies. one_hot(-1) is safe (all zeros).
      y = make_safe_classes(y)
      mask = torch.nn.functional.one_hot(y.long() - 1, weights.shape[-1])
      return torch.einsum('...TV,...V->...T', weights, mask.float())

    def weight_step(carry, inputs):
      self._state, next_label = inputs
      blank_weight, lexical_weights = compute_weights(frames)
      lexical_weight = gather_weight(lexical_weights.permute(0,2,1), 
                                     next_label)
      #      carry = None doesn't hold for pytorch's scan. We replace with
      #      torch.zeros(1)
      return carry, (blank_weight, lexical_weight)

    # [batch_dims..., max_num_labels + 1]
    context_states = self.context.walk_states(labels)
    context_next_labels = torch.concatenate(
        [labels, torch.ones_like(labels[..., :1])], dim=-1)
    # [batch_dims..., max_num_frames, max_num_labels+1]
    blank_weight, lexical_weight = weight_step_scan(
      weight_step, batch_dims, context_states, context_next_labels
    )

    # Dynamic program for summing up all alignment paths. Actual work is done by
    # alignment.string_forward(). This function mostly takes care of padding
    # frames.
    def shortest_distance_step(carry, inputs):
      # alpha: [batch_dims..., max_num_labels + 1]
      t, alpha = carry
      # blank, lexical: [batch_dims..., max_num_labels + 1]
      blank, lexical = inputs
      # We current only support alignment-state invariant weights.
      blank = [blank for _ in range(self.alignment.num_states())]
      lexical = [lexical for _ in range(self.alignment.num_states())]
      next_alpha = self.alignment.string_forward(
          alpha=alpha, blank=blank, lexical=lexical, semiring=semiring)
      is_padding = (t >= num_frames).unsqueeze(-1)
      next_alpha = torch.where(is_padding, alpha, next_alpha)
      return (t + 1, next_alpha), None

    num_alpha_states = labels.shape[-1] + 1
    init_alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        dtype=lexical_weight.dtype,
        num_states=num_alpha_states,
        start=0,
        semiring=semiring)
    (_, alpha), _ = shortest_distance_step_scan(
        shortest_distance_step,
        (0, init_alpha),
        pytree.tree_map(
            functools.partial(_to_time_major, num_batch_dims=len(batch_dims)),
            (blank_weight, lexical_weight))
        )
    is_final = num_labels.unsqueeze(-1) == torch.arange(num_alpha_states)
    return semiring.sum(
        torch.where(is_final, alpha, semiring.zeros([], alpha.dtype)), dim=-1)

  def _forward(
      self,
      cache: T,
      frames: torch.Tensor,
      num_frames: torch.Tensor,
      semiring: semirings.Semiring[torch.Tensor],
      blank_mask: Optional[Sequence[torch.Tensor]] = None,
      lexical_mask: Optional[Sequence[torch.Tensor]] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest distance on the recognition lattice computed using the forward algorithm.

    It is often useful to differentiate through shortest distance with respect
    to arc weights. For example, under the log semiring, that gives us arc
    marginals; whereas under the tropical semiring, that gives us shortest path.
    To make that possible while the arc weights are being computed on the fly,
    we can pass in zero-valued masks. The masks are added onto arc weights, and
    because d f(x + y) / dy | y=0 = d f(x) / dx, we can get gradients with
    respect to arc weights (i.e. x) by differentiating over the masks (i.e. y).

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      semiring: Semiring to use for shortest distance computation.
      blank_mask: Optional length num_alignment_states sequence whose elements
        are broadcastable to [batch_dims..., max_num_frames,
        num_context_states].
      lexical_mask: Optional length num_alignment_states sequence whose elements
        are broadcastable to [batch_dims..., max_num_frames, num_context_states,
        vocab_size].

    Returns:
      (shortest_distance, alpha_0_to_T_minus_1) tuple,
      -   shortest_distance: [batch_dims...] shortest distance.
      -   alpha_0_to_T_minus_1: [batch_dims..., max_num_frames,
          num_context_states] forward weights after observing 0 to T-1 frames
          (T is the number of frames in the sequence).
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    if blank_mask is not None and len(
        blank_mask) != self.alignment.num_states():
      raise ValueError(
          'The length of blank_mask should be equal to '
          f'{self.alignment.num_states()} (the number of alignment states), '
          f'but is {len(blank_mask)}')
    if lexical_mask is not None and len(
        lexical_mask) != self.alignment.num_states():
      raise ValueError(
          'The length of lexical_mask should be equal to '
          f'{self.alignment.num_states()} (the number of alignment states), '
          f'but is {len(lexical_mask)}')

    # Dynamic program for summing up all alignment paths.
    def step(weight_fn, carry, inputs):
      # alpha: [batch_dims..., num_context_states]
      t, alpha = carry
      # frame: [batch_dims..., hidden_size]
      # blank_mask: None or [batch_dims...]
      # lexical_mask: None or broadcastable to
      #   [batch_dims..., num_alignment_states, vocab_size]
      frame, blank_mask, lexical_mask = inputs
      # blank: [batch_dims..., num_context_states]
      # lexical: [batch_dims..., num_context_states, vocab_size]
      blank, lexical = weight_fn(cache, frame)
      # We currently only support alignment-state-invariant weights.
      blank = [blank for _ in range(self.alignment.num_states())]
      lexical = [lexical for _ in range(self.alignment.num_states())]
      if blank_mask is not None:
        blank = [b + m for b, m in zip(blank, blank_mask)]
      if lexical_mask is not None:
        lexical = [l + m for l, m in zip(lexical, lexical_mask)]
      next_alpha = self.alignment.forward(
          alpha=alpha,
          blank=blank,
          lexical=lexical,
          context=self.context,
          semiring=semiring)
      is_padding = (t >= num_frames).unsqueeze(-1)
      next_alpha = torch.where(is_padding, alpha, next_alpha)
      return (t + 1, next_alpha), alpha

    # Reduce memory footprint when using autodiff.
    #
    # For the log semiring, this is not as fast or memory efficient as forward-
    # backward, but still better than the defaults (i.e. no remat or no saving
    # intermediates at all).
    #
    # For the tropical semiring, this should be equivalent to no remat.
    def save_small(prim, *args, **params):
      y, _ = prim.abstract_eval(*args, **params)
      greater_than_1_dims = len([None for i in y.shape if i > 1])
      save = greater_than_1_dims <= (len(batch_dims) + 1)
      return save


    init_t = torch.Tensor([0])
    init_alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        # TODO(wuke): Find a way to do this with jax.eval_shape.
        dtype=self.weight_fn(cache, frames[..., 0, :])[0].dtype,
        num_states=self.context.shape()[0],
        start=self.context.start(),
        semiring=semiring)
    init_carry = (init_t, init_alpha)

    inputs = (frames, blank_mask, lexical_mask)
    in_dim = len(batch_dims)
    out_dim = len(batch_dims)

    (_, alpha_T), alpha_0_to_T_minus_1 = scan_step_forward( 
        step,
        self.weight_fn, 
        init_carry, inputs, in_dim, out_dim, self.alignment.num_states())
    return semiring.sum(alpha_T, dim=-1), alpha_0_to_T_minus_1

  def _forward_backward(self, cache: T, frames: torch.Tensor,
                        num_frames: torch.Tensor) -> torch.Tensor:
    """Shortest distance under the log semiring with gradients computed using the backward algorithm.

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      init_callback_carry: PyTree of initial carry value for the callback.

    Returns:
      [batch_dims...] shortest distance.
    """
    semiring = semirings.Log

    class ForwardBackward(torch.autograd.Function):
      @staticmethod
      def forward(cache, frames):
        log_z, alpha_0_to_T_minus_1 = self._forward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=semiring
        )
        return log_z, alpha_0_to_T_minus_1

      @staticmethod
      def setup_context(ctx, inputs, output):
        pass

      @staticmethod
      def backward(ctx, grad_output):
        """Computes arc marginals under the log semiring using the backward algorithm.

        Under the log semiring, arc weights can be viewed as unnormalized log
        probabilities, and a conditional distribution over paths can be defined by
        normalizing with respect to the exponentiated shortest distance (i.e. sum of
        unnormalized path probabilities). The marginal probability of each arc can
        then be computed with the backward algorithm.

        Mathematically, under the log semiring, arc marginals are equal to the
        gradients of shortest distance with respect to arc weights. The backward
        algorithm offers a slightly more efficient method for computing these
        gradients than reverse mode automatic differentiation with gradient
        rematerialization:
        -   Both methods compute the arc weights twice: once in the forward pass,
            once in the backward pass.
        -   Both methods carry out the "backward-broadcast" operation, i.e.
            broadcasting the backward weights from a destination state to all source
            states, once in the backward pass.
        -   Autodiff carries out the "forward-reduce" operation, i.e. summing up
            path weights to the same destination state, twice: once in the forward
            pass, once in the backward pass.
        -   Forward-backward only carries out the "forward-reduce" operation once,
            in the forward pass.

        In other words, forward-backward saves one "forward-reduce" operation. The
        savings can be significant when the "forward-reduce" call is often
        expensive, which is the main justification for all this added complexity.

        Args:
          cache: Weight function cache data.
          frames: [batch_dims..., max_num_frames, feature_size] padded frame
            sequences.
          num_frames: [batch_dims...] number of frames.
          log_z: [batch_dims...] shortest distance from _forward(). Under the log
            semiring, the shortest distance is the log-normalizer, thus the name.
          alpha_0_to_T_minus_1: [batch_dims..., max_num_frames, num_context_states]
            forward weights from _forward().
          callback: Callback used in the backward algorithm loop.

        Returns:
          (final_callback_carry, callback_outputs) tuple.
        """
        log_z, alpha_0_to_T_minus_1 = grad_output

        batch_dims = num_frames.shape
        if frames.shape[:-2] != batch_dims:
          raise ValueError('frames and num_frames have different batch_dims: '
                          f'{frames.shape[:-2]} vs {batch_dims}')
        if log_z.shape != batch_dims:
          raise ValueError('log_z and num_frames have different batch_dims: '
                          f'{log_z.shape} vs {batch_dims}')
        if alpha_0_to_T_minus_1.shape[:-2] != batch_dims:
          raise ValueError(
              'alpha_0_to_T_minus_1 and num_frames have different '
              f'batch_dims: {alpha_0_to_T_minus_1.shape[:-2]} vs {batch_dims}')

        def step(lattice, carry, inputs):
          # beta: [batch_dims..., num_context_states]
          t, beta, callback_carry = carry
          # alpha: [batch_dims..., num_context_states]
          # frame: [batch_dims..., hidden_size]
          alpha, frame = inputs
          # blank: [batch_dims..., num_context_states]
          # lexical: [batch_dims..., num_context_states, vocab_size]
          (blank, lexical), weight_vjp_fn = nn.vjp(
              lambda lattice, cache, frame: lattice.weight_fn(cache, frame),
              lattice, cache, frame)
          # We currently only support alignment-state-invariant weights.
          blank = [blank for _ in range(self.alignment.num_states())]
          lexical = [lexical for _ in range(self.alignment.num_states())]
          next_beta, blank_marginal, lexical_marginals = self.alignment.backward(
              alpha=alpha,
              blank=blank,
              lexical=lexical,
              beta=beta,
              log_z=log_z,
              context=self.context)
          # We currently only support alignment-state-invariant weights.
          blank_marginal = torch.sum(torch.stack(blank_marginal), axis=0)
          lexical_marginals = torch.sum(torch.stack(lexical_marginals), axis=0)
          # Mask out marginals on padding positions.
          is_padding = (t >= num_frames)[..., torch.newaxis]
          next_beta = torch.where(is_padding, beta, next_beta)
          blank_marginal = torch.where(is_padding, 0, blank_marginal)
          lexical_marginals = torch.where(is_padding[..., torch.newaxis], 0,
                                        lexical_marginals)
          next_callback_carry, callback_outputs = callback(
              weight_vjp_fn=weight_vjp_fn,
              carry=callback_carry,
              blank_marginal=blank_marginal,
              lexical_marginals=lexical_marginals)

          return (t - 1, next_beta, next_callback_carry), callback_outputs


        num_context_states, _ = self.context.shape()
        init_beta = semirings.Log.ones([*batch_dims, num_context_states],
                                      log_z.dtype)
        init_t = torch.Tensor(frames.shape[-2] - 1)
        init_carry = (init_t, init_beta, init_callback_carry)

        inputs = (alpha_0_to_T_minus_1, frames)
        (_, _, final_callback_carry), callback_outputs = scan_step_backward(
            step,
            init_carry, 
            inputs
        )

        return final_callback_carry, callback_outputs

    _fwd_bwd = ForwardBackward.apply    
    return _fwd_bwd(cache, frames)



def _init_context_state_weights(
    batch_dims: Sequence[int], dtype: DType, num_states: int, start: int,
    semiring: semirings.Semiring[torch.Tensor]) -> torch.Tensor:
  is_start = torch.arange(num_states) == start
  weights = torch.where(is_start, semiring.ones([], dtype),
                      semiring.zeros([], dtype))
  return torch.broadcast_to(weights, (*batch_dims, num_states))


def _to_time_major(x: torch.Tensor, num_batch_dims: int) -> torch.Tensor:
  # [batch_dims..., time, ...] -> [time, batch_dims..., ...]
  axes = [
      num_batch_dims,
      *range(num_batch_dims),
      *range(num_batch_dims + 1, x.ndim),
  ]
  return torch.permute(x, tuple(axes))


def _to_batch_major(x: torch.Tensor, num_batch_dims: int) -> torch.Tensor:
  # [time, batch_dims..., ...] -> [batch_dims..., time, ...]
  axes = [
      *range(1, num_batch_dims + 1),
      0,
      *range(num_batch_dims + 1, x.ndim),
  ]
  return torch.transpose(x, axes)


def weight_step_scan(weight_step, batch_dims, context_states, context_next_labels):
  assert context_states.shape == context_next_labels.shape
  carry = torch.zeros(len(batch_dims))
  blank_weight = torch.Tensor()
  lexical_weight = torch.Tensor()

  for last_dim_i in range(context_next_labels.shape[-1]):
    inputs = (
      context_states[:, last_dim_i],
      context_next_labels[:, last_dim_i]
    )
    _, (blank_weight_i, lexical_weight_i) = weight_step(carry, inputs)
    blank_weight = torch.concat([blank_weight, blank_weight_i.unsqueeze(-1)], dim=-1)
    lexical_weight = torch.concat([lexical_weight, lexical_weight_i.unsqueeze(-1)],dim=-1)

  return blank_weight, lexical_weight

def shortest_distance_step_scan(shortest_distance_step, init, xs):
  t, alpha = init
  blank_weight, lexical_weight = xs

  for i in range(blank_weight.shape[0]):
    (t, alpha), _ = shortest_distance_step((t, alpha), (blank_weight[i,:], lexical_weight[i,:]))

  return (t, alpha), None

def scan_step_forward(scan_fn, weight_fn, init_carry, inputs, in_dim, out_dim, num_alignment_states=None):
  t, alpha = init_carry

  alpha_0_to_t_minus_1 = torch.tensor(())

  frames, blank_mask, lexical_mask = inputs

     
  for i in range(frames.shape[in_dim]):
    frame = torch.index_select(frames, in_dim, torch.tensor(i)).squeeze(in_dim)

    if lexical_mask != None:
      lexical_mask_framed = torch.index_select(lexical_mask[0], in_dim, torch.tensor(i)).squeeze(in_dim)
      (t, alpha), alpha_t_minus_1 = scan_fn(weight_fn, 
                                      (t, alpha),
                                      (frame, blank_mask, lexical_mask_framed))
    
    else:
      (t, alpha), alpha_t_minus_1 = scan_fn(weight_fn, 
                                      (t, alpha),
                                      (frame, blank_mask, lexical_mask))

    alpha_0_to_t_minus_1 = torch.cat((alpha_0_to_t_minus_1, alpha_t_minus_1), dim=out_dim)  

  alpha_0_to_t_minus_1 = alpha_0_to_t_minus_1.reshape(
    frames.shape[:-1] + (alpha.shape[-1],)  
  )

  return (t, alpha), alpha_0_to_t_minus_1 

def scan_step_backward(scan_fn, weight_fn, init_carry, inputs):
  pass