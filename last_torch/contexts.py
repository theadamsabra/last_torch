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

"""Context dependencies."""

import abc
import dataclasses

import torch

from last_torch import semirings


class ContextDependency(abc.ABC):
  r"""Interface for context dependencies.

  A context dependency is a deterministic finite automaton (DFA) that accepts
  $\Sigma^*$ ($\Sigma$ is the lexical output vocabulary). The state ids in [0,
  num_states) of a context dependency encodes the output history. See Sections 3
  and 4 of the GNAT paper for more details.

  Note: we assume all context dependency states to be final.

  Subclasses should implement the following methods:
  - shape
  - start
  - next_state
  - forward_reduce
  - backward_broadcast
  """

  @abc.abstractmethod
  def shape(self) -> tuple[int, int]:
    r"""Shape of a context dependency.

    Returns:
      (num_states, vocab_size) tuple:
      - num_states: The number of states in the context dependency DFA.
      - vocab_size: The size of the output vocabulary, $|\Sigma|$.
    """

  @abc.abstractmethod
  def start(self) -> int:
    """The start state id."""

  @abc.abstractmethod
  def next_state(self, state: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Takes a transition in the DFA.

    Note: because 0 is the epsilon label, it is normally not fed to
    `next_state`. For consistency, we require that `next_state` should return
    `state[i]` when `label[i] == 0`.

    Args:
      state: [batch_dims...] int32 source state ids.
      label: [batch_dims...] int32 output labels in the range [0, vocab_size].

    Returns:
      [batch_dims...] next state ids.
    """

  @abc.abstractmethod
  def forward_reduce(self, weights: torch.Tensor,
                     semiring: semirings.Semiring[torch.Tensor]) -> torch.Tensor:
    """The reduction used in the forward algorithm.

    For each state q, we sum over all source states p and labels y that lead to
    state q, i.e.

    result[..., q] = sum_{p-y->q} weights[..., p, y]

    Args:
      weights: [batch_dims..., num_states, vocab_size] weights.
      semiring: The semiring for carrying out the summation.

    Returns:
      [batch_dims..., num_states] reduced weights.
    """

  @abc.abstractmethod
  def backward_broadcast(self, weights: torch.Tensor) -> torch.Tensor:
    """The broadcast used in the backward algorithm.

    For each state q, we broadcast its weight to all the (source state p, label
    y) pairs leading to state q, i.e.

    result[..., p, y] = weights[..., q]

    Args:
      weights: [batch_dims..., num_states] weights.

    Returns:
      [batch_dims..., num_states, vocab_size] broadcasted weights.
    """

  # Methods below are implemented using the basic operations above.

  def walk_states(self, labels:torch.Tensor) -> torch.Tensor:
    """Walks a context dependency following label sequences.

    Args:
      labels: [batch_dims..., num_labels] int32 label sequences. Each element is
        in the range [0, vocab_size].

    Returns:
      [batch_dims..., num_labels + 1] int32 context states. states[..., 0]
      equals to the start state of the context dependency; states[..., i] for
      i > 0 is the state after observing labels[..., i - 1].
    """
    batch_dims = labels.shape[:-1]
    start = torch.broadcast_to(self.start(), batch_dims)

    def step(state, label):
      next_state = self.next_state(state, label)
      return next_state, next_state
    
    time_major_labels = torch.transpose(
      labels, [len(batch_dims), *range(len(batch_dims))])
    # TODO: torch.scan is not a real function. will need to implement or find
    # equivalent.
    _, time_major_states = torch.scan(step, start, time_major_labels)
    states = torch.transpose(time_major_states, [*range(1, labels.ndim), 0])
    return torch.concatenate([torch.unsqueeze(start, dim=-1), states], dim=-1)