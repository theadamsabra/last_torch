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
from typing import Any, Generic, Optional, Protocol, TypeVar

import torch
import torch.nn as nn
import optree

from last_torch import alignments
from last_torch import contexts
from last_torch import semirings
from last_torch import weight_fns

DType = Any
T = TypeVar('T')


class _RecurrenceFn(torch.autograd.Function):
  """Custom VJP for the alpha recurrence under the Log semiring.

  Wraps *only* the recurrence scan, taking already-precomputed arc weights as
  inputs. Weight precomputation stays outside under normal autograd, so the
  returned grads flow back through the vmap to the weight_fn parameters without
  any manual VJP through the network.

  Two savings over differentiating the recurrence with autograd:
  -   forward runs under no_grad, so no autograd graph is built for the T*K
      semiring ops; only alpha_0..T-1 is stored.
  -   backward is a direct arc-marginals reverse scan, which performs the
      expensive "forward-reduce" once instead of twice (see _backward's
      docstring for the full argument).
  """

  @staticmethod
  def forward(all_blank, all_lexical, num_frames, lattice):
    # all_blank: [batch_dims..., T, num_context_states]
    # all_lexical: [batch_dims..., T, num_context_states, vocab_size]
    batch_dims = num_frames.shape
    in_dim = len(batch_dims)
    T = all_blank.shape[in_dim]
    num_states = lattice.alignment.num_states()

    with torch.no_grad():
      alpha = _init_context_state_weights(
          batch_dims=batch_dims,
          dtype=all_blank.dtype,
          num_states=lattice.context.shape()[0],
          start=lattice.context.start(),
          semiring=semirings.Log,
          device=lattice.device)
      padding_list = _padding_masks(num_frames, T, lattice.device)

      alpha_list = []
      for i in range(T):
        blank_t = all_blank.select(in_dim, i)
        lexical_t = all_lexical.select(in_dim, i)
        next_alpha = lattice._align_fwd(
            alpha=alpha,
            blank=[blank_t for _ in range(num_states)],
            lexical=[lexical_t for _ in range(num_states)],
            context=lattice.context,
            semiring=semirings.Log)
        alpha_list.append(alpha)
        alpha = torch.where(padding_list[i], alpha, next_alpha)

      alpha_0_to_T_minus_1 = torch.stack(alpha_list, dim=in_dim)
      log_z = semirings.Log.sum(alpha, dim=-1)
    return log_z, alpha_0_to_T_minus_1

  @staticmethod
  def setup_context(ctx, inputs, output):
    all_blank, all_lexical, num_frames, lattice = inputs
    log_z, alpha_0_to_T_minus_1 = output
    ctx.save_for_backward(all_blank, all_lexical, num_frames, log_z,
                          alpha_0_to_T_minus_1)
    ctx.lattice = lattice

  @staticmethod
  def backward(ctx, grad_log_z, _grad_alphas):
    all_blank, all_lexical, num_frames, log_z, alphas = ctx.saved_tensors
    lattice = ctx.lattice
    batch_dims = num_frames.shape
    in_dim = len(batch_dims)
    T = all_blank.shape[in_dim]
    num_states = lattice.alignment.num_states()
    num_context_states, _ = lattice.context.shape()

    beta = semirings.Log.ones([*batch_dims, num_context_states],
                              log_z.dtype).to(lattice.device)
    padding_list = _padding_masks(num_frames, T, lattice.device)
    # d log_z / d arc_weight is the arc marginal, scaled by the incoming grad.
    scale = grad_log_z.unsqueeze(-1)

    grad_blank_steps = [None] * T
    grad_lexical_steps = [None] * T
    for t in range(T - 1, -1, -1):
      blank_t = all_blank.select(in_dim, t)
      lexical_t = all_lexical.select(in_dim, t)
      next_beta, blank_marginal, lexical_marginals = lattice._align_bwd(
          alpha=alphas.select(in_dim, t),
          blank=[blank_t for _ in range(num_states)],
          lexical=[lexical_t for _ in range(num_states)],
          beta=beta,
          log_z=log_z,
          context=lattice.context)
      # We currently only support alignment-state-invariant weights. For
      # FrameDependent (num_states == 1) index directly rather than allocating a
      # stack and a sum per step.
      if num_states == 1:
        blank_marginal = blank_marginal[0]
        lexical_marginals = lexical_marginals[0]
      else:
        blank_marginal = torch.sum(torch.stack(blank_marginal), dim=0)
        lexical_marginals = torch.sum(torch.stack(lexical_marginals), dim=0)

      is_padding = padding_list[t]
      beta = torch.where(is_padding, beta, next_beta)
      grad_blank_steps[t] = torch.where(is_padding, 0, blank_marginal) * scale
      grad_lexical_steps[t] = torch.where(
          is_padding.unsqueeze(-1), 0, lexical_marginals) * scale.unsqueeze(-1)

    return (torch.stack(grad_blank_steps, dim=in_dim),
            torch.stack(grad_lexical_steps, dim=in_dim), None, None)


def _padding_masks(num_frames: torch.Tensor, T: int,
                   device: str) -> list[torch.Tensor]:
  """Per-step [batch_dims..., 1] padding masks, as one comparison for all T."""
  step_idx = torch.arange(T, device=device).view((T,) +
                                                 (1,) * len(num_frames.shape))
  return list((step_idx >= num_frames).unsqueeze(-1).unbind(0))

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
                                            weight_fns.WeightFn[T]],
                device:str = 'cpu'):
    super().__init__()
    self.context = context
    self.alignment = alignment
    self.weight_fn_cacher_factory = weight_fn_cacher_factory
    self.weight_fn_factory = weight_fn_factory

    self.weight_fn_cacher = self.weight_fn_cacher_factory(self.context)
    self.weight_fn = self.weight_fn_factory(self.context)
    self.device = device
    self._align_fwd = torch.compile(self.alignment.forward)
    self._align_str_fwd = torch.compile(self.alignment.string_forward)
    # Must be compiled: uncompiled, per-step Python dispatch in _RecurrenceFn's
    # explicit backward loop cancels the arc-marginals saving entirely.
    self._align_bwd = torch.compile(self.alignment.backward)

  def _precompute_weights(self, cache: T, frames: torch.Tensor,
                          in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes all T (blank, lexical) arc weight pairs in one batched call."""
    # Warm up lazy weight init (JointWeightFn allocates nn.Linear on first call);
    # vmap cannot run random ops so init must happen before we enter vmap.
    self.weight_fn(cache, frames.select(in_dim, 0))
    # ponytail: vmap over time dim; cache is captured, not vmapped.
    frames_T = frames.movedim(in_dim, 0)  # [T, batch..., features]
    all_blank_T, all_lexical_T = torch.func.vmap(
        lambda f: self.weight_fn(cache, f), in_dims=0, out_dims=0)(frames_T)
    # [batch..., T, num_ctx] and [batch..., T, num_ctx, vocab]
    return all_blank_T.movedim(0, in_dim), all_lexical_T.movedim(0, in_dim)

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
      if cache is not None:
        cache = cache.to(self.device)
    numerator = self._string_forward(
      cache = cache,
      frames = frames.to(self.device),
      num_frames = num_frames.to(self.device),
      labels = labels.to(self.device),
      num_labels = num_labels.to(self.device),
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
      [*batch_dims, max_num_frames, num_alignment_states, vocab_size],
      device=self.device
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

    # Precompute arc weights for all (label-state, frame) pairs via double vmap.
    context_states = self.context.walk_states(labels)
    context_next_labels = torch.concatenate(
        [labels, torch.ones_like(labels[..., :1])], dim=-1)

    # Warm up lazy weight init before vmap (same reason as in _forward).
    _ = self.weight_fn(cache, frames.select(-2, 0), context_states.select(-1, 0))

    # ponytail: double vmap (outer=label states L+1, inner=frames T).
    def _weight_for_state(state):
      return torch.func.vmap(
          lambda frame: self.weight_fn(cache, frame, state),
          in_dims=1, out_dims=-1)(frames)

    all_blank, all_lex_VT = torch.func.vmap(
        _weight_for_state, in_dims=-1, out_dims=-1)(context_states)
    # all_blank:   [batch..., T, L+1]
    # all_lex_VT:  [batch..., vocab, T, L+1]

    # Gather lexical weight for each label position using context_next_labels.
    T = frames.shape[-2]
    all_lex = all_lex_VT.permute(
        *range(len(batch_dims)), len(batch_dims)+1, len(batch_dims)+2, len(batch_dims))
    # all_lex: [batch..., T, L+1, vocab]
    l_idx = context_next_labels.unsqueeze(-2).expand(
        *batch_dims, T, context_next_labels.shape[-1])  # [batch..., T, L+1]
    safe_idx = (l_idx.long() - 1).clamp(min=0)
    blank_weight = all_blank  # [batch..., T, L+1]
    lexical_weight = torch.gather(
        all_lex, -1, safe_idx.unsqueeze(-1)).squeeze(-1)  # [batch..., T, L+1]

    # Sequential forward scan over frames using precomputed weights.
    num_alpha_states = labels.shape[-1] + 1
    init_alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        dtype=blank_weight.dtype,
        num_states=num_alpha_states,
        start=0,
        semiring=semiring,
        device=self.device)
    num_states = self.alignment.num_states()

    # Precompute per-step padding masks — avoids T separate comparison kernels.
    in_dim_str = len(batch_dims)  # time is the next-to-last dim
    step_idx_str = torch.arange(T, device=self.device).view(
        (T,) + (1,) * len(batch_dims))
    padding_list_str = list((step_idx_str >= num_frames).unsqueeze(-1).unbind(0))

    alpha = init_alpha
    for i in range(T):
      b = blank_weight.select(in_dim_str, i)
      l = lexical_weight.select(in_dim_str, i)
      next_alpha = self._align_str_fwd(
          alpha=alpha,
          blank=[b for _ in range(num_states)],
          lexical=[l for _ in range(num_states)],
          semiring=semiring)
      alpha = torch.where(padding_list_str[i], alpha, next_alpha)
    is_final = num_labels.unsqueeze(-1) == torch.arange(num_alpha_states).to(self.device)
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

    in_dim = len(batch_dims)
    out_dim = len(batch_dims)
    T = frames.shape[in_dim]

    all_blank, all_lexical = self._precompute_weights(cache, frames, in_dim)

    alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        dtype=all_blank.dtype,
        num_states=self.context.shape()[0],
        start=self.context.start(),
        semiring=semiring,
        device=self.device)

    has_both = blank_mask is not None and lexical_mask is not None
    has_lex_only = blank_mask is None and lexical_mask is not None
    num_states = self.alignment.num_states()

    padding_list = _padding_masks(num_frames, T, self.device)

    alpha_list = []
    for i in range(T):
      blank_t = all_blank.select(in_dim, i)
      lexical_t = all_lexical.select(in_dim, i)
      blank = [blank_t for _ in range(num_states)]
      lexical = [lexical_t for _ in range(num_states)]
      if has_both:
        bm = blank_mask[0].select(in_dim, i)
        lm = lexical_mask[0].select(in_dim, i)
        blank = [b + m for b, m in zip(blank, [bm])]
        lexical = [l + m for l, m in zip(lexical, [lm])]
      elif has_lex_only:
        lm = lexical_mask[0].select(in_dim, i)
        # ponytail: lm as raw tensor — zip iterates batch dim, stops at len(lexical)=1.
        lexical = [l + m for l, m in zip(lexical, lm)]
      next_alpha = self._align_fwd(
          alpha=alpha,
          blank=blank,
          lexical=lexical,
          context=self.context,
          semiring=semiring)
      alpha_list.append(alpha)
      alpha = torch.where(padding_list[i], alpha, next_alpha)

    alpha_T = alpha
    alpha_0_to_T_minus_1 = torch.stack(alpha_list, dim=out_dim)
    return semiring.sum(alpha_T, dim=-1), alpha_0_to_T_minus_1


  class BackwardStepCallback(Protocol):
    """Callback signature used in the backward algorithm loop."""

    # Type names.
    # pylint: disable=invalid-name
    Blank = torch.Tensor 
    Lexical = torch.Tensor 
    ParamsGrad = Any
    CacheGrad = Any
    FrameGrad = torch.Tensor 
    Carry = TypeVar('Carry')
    Output = Any
    # pylint: enable=invalid-name

    def __call__(self, weight_vjp_fn: Callable[[Blank, Lexical],
                                              tuple[ParamsGrad, CacheGrad,
                                                    FrameGrad]], carry: Carry,
                blank_marginal: Blank,
                lexical_marginals: Lexical) -> tuple[Carry, Output]:
      """Callback used in the backward algorithm loop.

      Standard backward algorithm simply computes the arc marginals and backward
      weights. Through a custom callback, we can perform on the fly processing
      beyond these without having to store all the arc marginals. An example is
      accumulating the gradients with respect to weight function parameters (see
      _forward_backward).

      Args:
        weight_vjp_fn: VJP function of the weight function. Callable of the
          signature (blank_grad, lexical_grad) -> (params_grad, cache_grad,
          frame_grad).
        carry: PyTree of custom callback carry data.
        blank_marginal: [batch_dims..., num_context_states] marginal probability
          of blank arcs.
        lexical_marginals: [batch_dims..., num_context_states, vocab_size]
          marginal probability of lexical arcs.

      Returns:
        next_carry and step outputs.
      """
      raise NotImplementedError

  def _backward(
      self,
      cache: T,
      frames: torch.Tensor, 
      num_frames: torch.Tensor,
      log_z: torch.Tensor,
      alpha_0_to_T_minus_1: torch.Tensor,
      init_callback_carry: ...,
      callback: BackwardStepCallback) -> tuple[Any, Any]:
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
      init_callback_carry: PyTree of initial carry value for the callback.
      callback: Callback used in the backward algorithm loop.

    Returns:
      (final_callback_carry, callback_outputs) tuple.
    """
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

      def step(carry, inputs):
        # beta: [batch_dims..., num_context_states]
        t, beta, callback_carry = carry
        # alpha: [batch_dims..., num_context_states]
        # frame: [batch_dims..., hidden_size]
        alpha, frame = inputs
        # blank: [batch_dims..., num_context_states]
        # lexical: [batch_dims..., num_context_states, vocab_size]
        (blank, lexical), weight_vjp_fn = torch.func.vjp(
            lambda cache, frame: self.weight_fn(cache, frame),
            cache, frame)
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
        is_padding = (t >= num_frames).unsqueeze(-1)
        next_beta = torch.where(is_padding, beta, next_beta)
        blank_marginal = torch.where(is_padding, 0, blank_marginal)
        lexical_marginals = torch.where(is_padding.unsqueeze(-1), 0,
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
      init_t = torch.Tensor([frames.shape[-2] - 1])
      init_carry = (init_t.to(self.device), 
                    init_beta.to(self.device), 
                    init_callback_carry
                    )

      inputs = (alpha_0_to_T_minus_1, frames)
      (_, _, final_callback_carry), callback_outputs = scan_step_backward(
        step, init_carry, inputs, in_dim=len(batch_dims), out_dim=len(batch_dims), reverse=True
      )

      return final_callback_carry, callback_outputs

  def _forward_backward(self, cache: T, frames: torch.Tensor,
                        num_frames: torch.Tensor) -> torch.Tensor:
    """Shortest distance under the log semiring, with a custom VJP.

    Same value as _forward() under semirings.Log, but the recurrence runs under
    no_grad inside _RecurrenceFn and is differentiated by an explicit
    arc-marginals backward scan instead of autograd. Weight precomputation stays
    outside the Function, so grads reach the weight_fn parameters through the
    usual vmap.

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.

    Returns:
      (log_z, alpha_0_to_T_minus_1) tuple.
    """
    in_dim = len(num_frames.shape)
    all_blank, all_lexical = self._precompute_weights(cache, frames, in_dim)
    return _RecurrenceFn.apply(all_blank, all_lexical, num_frames, self)


def _init_context_state_weights(
    batch_dims: Sequence[int], dtype: DType, num_states: int, start: int,
    semiring: semirings.Semiring[torch.Tensor],
    device:str='cpu') -> torch.Tensor:
  is_start = torch.arange(num_states) == start
  is_start = is_start.to(device) 

  weights = torch.where(is_start, semiring.ones([], dtype, device),
                      semiring.zeros([], dtype, device))
  return torch.broadcast_to(weights, (*batch_dims, num_states))




def scan_step_backward(scan_fn, init_carry, inputs, in_dim, out_dim, reverse):
  alphas, frames = inputs
  T = frames.shape[in_dim]
  indices = range(T - 1, -1, -1) if reverse else range(T)
  outputs_list = []

  for i in indices:
    frame = frames.select(in_dim, i)   # view, no allocation
    alpha = alphas.select(in_dim, i)
    init_carry, callback_outputs = scan_fn(init_carry, (alpha, frame))
    if callback_outputs is not None:
      outputs_list.append(callback_outputs)

  if reverse and outputs_list:
    outputs_list = outputs_list[::-1]

  if outputs_list:
    stacked_outputs = optree.tree_map(
        lambda *xs: torch.stack(xs, dim=in_dim),
        *outputs_list
    )
  else:
    stacked_outputs = None

  return init_carry, stacked_outputs