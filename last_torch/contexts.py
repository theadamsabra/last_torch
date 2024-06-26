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
    start = torch.broadcast_to(torch.tensor(self.start()), 
                               batch_dims)

    # Define custom scan specifially for next step:
    def scan(f, init, xs):
      carry = init
      ys = torch.Tensor()
      for x in xs:
        carry, y = f(carry, x)
        # Check if zero dimensional tensor to help with concatenation:
        if (torch.numel(y) == 1) and (y.shape == torch.Size([])):
          y = torch.unsqueeze(y, 0)

        ys = torch.concatenate([ys, y])
      return carry, ys.reshape(xs.shape)

    def step(state, label):
      next_state = self.next_state(state, label)
      return next_state, next_state
    
    time_major_labels = torch.permute(
      labels, [len(batch_dims), *range(len(batch_dims))])
    _, time_major_states = scan(step, start, time_major_labels)
    states = torch.permute(time_major_states, [*range(1, labels.ndim), 0])
    return torch.concatenate([torch.unsqueeze(start, dim=-1), states], dim=-1)


@dataclasses.dataclass(frozen=True)
class FullNGram(ContextDependency):
  """Full n-gram context dependency as described in Section 4.1 of the GNAT paper.

  For a given vocab_size > 0, context_size >= 0,
  -   The set of states represents the set of all possible n-grams from length 0
      to length context_size for an output vocabulary of vocab_size.
  -   Each n-gram is assigned their lexicographic order as the id. The empty
      n-gram is state 0, followed by the vocab_size unigrams as states 1 to
      vocab_size, and so on.
  -   The start state is 0 (the empty n-gram).
  -   All states are final.
  -   From each n-gram state, there is an arc for each label in the vocabulary
      leading to the n-gram with the label appended to the end, with the length
      of the n-gram capped at context_size.

  Attributes:
    vocab_size: Lexical output vocabulary size.
    context_size: Maximum n-gram context size.
  """

  vocab_size: int
  context_size: int

  def __post_init__(self):
    if self.vocab_size <= 0:
      raise ValueError('vocab_size should be > 0, but got '
                       f'vocab_size={self.vocab_size}')
    if self.context_size < 0:
      raise ValueError('context_size should be >= 0, but got '
                       f'context_size={self.context_size}')

  def num_states(self) -> int:
    return sum(int(self.vocab_size ** i) for i in range(self.context_size + 1))
  
  def shape(self) -> tuple[int, int]:
    return self.num_states(), self.vocab_size
  
  def start(self) -> int:
    return 0
  
  def next_state(self, state: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    num_ascending_states = sum(
      self.vocab_size**i for i in range(self.context_size))
    ascend_nextstate = state * self.vocab_size + label
    if self.context_size == 0:
      full_nextstate = torch.zeros_like(ascend_nextstate)
    else:
      full_nextstate = (
        (state - num_ascending_states) %
        (self.vocab_size**(self.context_size - 1)) * self.vocab_size +
        num_ascending_states + label - 1)
    nextstate = torch.where(state < num_ascending_states, ascend_nextstate,
                            full_nextstate)
    # Remain where we were for epsilons
    nextstate = torch.where(label == 0, state, nextstate)
    return nextstate
  
  def forward_reduce(self, weights: torch.Tensor, 
                     semiring: semirings.Semiring[torch.Tensor]) -> torch.Tensor:
    batch_dims = weights.shape[:-2]
    if weights.shape[-2:] != self.shape():
      raise ValueError(f'weights.shape[-2:] should be {self.shape()} but got'
                       f' {weights.shape[-2:]}')

    # weights can be partitioned into two blocks, those leading to
    # ascending states, and those leading to the full context_size order states.
    next_accum_parts = []
    if self.context_size > 0:
      next_accum_parts.append(semiring.zeros(batch_dims + (1,), weights.dtype))

    num_states_going_into_ascending_states = sum(
        self.vocab_size**i for i in range(0, self.context_size - 1))
    next_accum_parts.append(weights[
        ..., :num_states_going_into_ascending_states, :].reshape(batch_dims +
                                                                 (-1,)))
    next_accum_parts.append(
        semiring.sum(
            weights[..., num_states_going_into_ascending_states:, :].reshape(
                batch_dims + (-1, self.vocab_size**self.context_size)),
            dim=-2))
    return torch.concatenate(next_accum_parts, dim=-1)
  
  def backward_broadcast(self, weights: torch.Tensor) -> torch.Tensor:
    batch_dims = weights.shape[:-1]
    num_states = weights.shape[-1]
    if num_states != self.num_states():
      raise ValueError(f'weights.shape[-1] should be {self.num_states()} but '
                       f'got {num_states}')

    if self.context_size == 0:
      return torch.broadcast_to(weights.unsqueeze(-1),
                              weights.shape + (self.vocab_size,))

    # Non-start ascending states have a unique incoming arc, thus a unique
    # incoming state.
    num_ascending_states = sum(
        self.vocab_size**i for i in range(self.context_size))
    part_a = weights[..., 1:num_ascending_states].reshape(batch_dims +
                                                          (-1, self.vocab_size))

    # States with arcs into full context_size order states.
    part_b = torch.broadcast_to(
        weights[..., None, num_ascending_states:], batch_dims +
        (1 + self.vocab_size, self.vocab_size**
         self.context_size)).reshape(batch_dims + (-1, self.vocab_size))

    return torch.concatenate([part_a, part_b], dim=-2)
  
  def next_state_table(self) -> torch.Tensor:
    """Generates the next state table (see NextStateTable)."""
    num_states, vocab_size = self.shape()
    return self.next_state(
        torch.arange(num_states).unsqueeze(-1),
        torch.arange(vocab_size).unsqueeze(0) + 1)


@dataclasses.dataclass(frozen=True)
class NextStateTable(ContextDependency):
  """Context dependency described as a transition lookup table.

  Attributes:
    next_state_table: [num_states, vocab_size] int32 array. next_state_table[p,
      y - 1] is the state reached from p with label y.
  """
  next_state_table: torch.Tensor 

  def __post_init__(self):
    if self.next_state_table.ndim != 2:
      raise ValueError(
          'next_state_table should have shape [num_states, vocab_size], but'
          f'got shape {self.next_state_table.shape}')
    if 0 in self.next_state_table.size():
      raise ValueError('next_state_table should have a non-zero size, but '
                       f'got shape {self.next_state_table.shape}')
    if self.next_state_table.dtype != torch.int32:
      raise ValueError('next_state_table should be an int32 ndarray, but '
                       f'got dtype {self.next_state_table.dtype}')
  
  def shape(self) -> tuple[int, int]:
    return self.next_state_table.shape
  
  def start(self) -> int:
    return 0
  
  def next_state(self, state: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # Note: lexical labels are in the range [1, vocab_size].
    is_epsilon = label == 0
    zero_based_label = torch.where(is_epsilon, 0, label - 1)
    nextstate = self.next_state_table[state.long(), zero_based_label.long()]
    # Remain where we were for epsilons.
    nextstate = torch.where(is_epsilon, state, nextstate)
    return nextstate
  
  def forward_reduce(self, weights: torch.Tensor,
                     semiring: semirings.Semiring[torch.Tensor]) -> torch.Tensor:
    batch_dims = weights.shape[:-2]
    if weights.shape[-2:] != self.shape():
      raise ValueError(f'weights.shape[-2:] should be {self.shape()} but got'
                       f' {weights.shape[-2:]}')
    num_states, vocab_size = self.shape()
    # Build the scatter operation.
    operand = semiring.zeros(batch_dims + (num_states,) + (vocab_size,), weights.dtype)
    updates = weights
    scatter_indices = torch.unsqueeze(self.next_state_table, dim=0)
    scatter_dims_to_operand_dims = len(batch_dims)
    return operand.scatter_reduce(scatter_dims_to_operand_dims, 
                                scatter_indices.to(torch.int64),
                                updates, 'sum').max(dim=-1).values
  
  def backward_broadcast(self, weights: torch.Tensor) -> torch.Tensor:
    num_states = weights.shape[-1]
    if num_states != self.shape()[0]:
      raise ValueError(f'weights.shape[-1] should be {self.shape()[0]} but '
                       f'got {num_states}')
    return weights[..., self.next_state_table]