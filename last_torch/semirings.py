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

'''Semirings.'''
from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Callable, Generic, Optional, TypeVar

import torch
import torch.utils._pytree as pytree 
from functorch import jacrev, jacfwd

# Types for documentation purposes
DType = Any
PyTree = Any
# Type vars for semiring values.
T = TypeVar('T')
S = TypeVar('S')

def value_shape(x:PyTree) -> tuple[int, ...]:
  """Obtains the shape of a semiring value.

  A semiring value is a PyTree of one or more identically shaped ndarrays.
  The shape of a semiring value is thus the common of shape of its leaves.

  Args:
    x: Some semiring value.

  Returns:
    The common shape of the leaves of x.

  Raises:
    ValueError: If the leaves of x do not have a common shape.
  """
  # Handle both None and normal entries:
  shapes = []
  for i in pytree.tree_leaves(x):
    if i == None:
      raise ValueError(
          f'No common shape can be derived for an empty PyTree: {x!r}'
      )
    else:
      shapes.append(tuple(i.shape))
  # Verify that shapes are consistent:
  result = shapes[0]
  for i in shapes[1:]:
    if i != result:
      raise ValueError(
          'A semiring value must consist of ndarrays of a common shape. '
          f'Got inconsistent shapes {result} vs {i} for PyTree: {x!r}'
      )
  return result

def value_dtype(x:PyTree) -> DType:
  """Obtains the dtypes of a semiring value.

  Different leaves of a semiring value may have different dtypes. Methods
  such as Semiring.{zeros,ones} can take a PyTree of dtypes in the same
  structure as the corresponding semiring values. This function can be used
  to extract such a dtype PyTree from a semiring value.

  Args:
    x: Some semiring value.

  Returns:
    dtypes in the same structure as x.
  """
  return pytree.tree_map(lambda x_: x_.dtype, x)

class Semiring(Generic[T]):
  """Base Semiring interface.

  See https://en.wikipedia.org/wiki/Semiring for what a semiring is. A Semiring
  object holds methods that implement the semiring operations. To simplify
  non-semiring operations on the semiring values, the semiring values are not
  typed: for most basic semirings, each value is a single ndarray; for some more
  complex semirings (e.g. Expectation or Cartesian), the values can be a tuple
  of ndarrays.

  In general, a semiring value under some semiring is represented as a PyTree
  of identically shaped ndarrays, with possibly different dtypes. The shape
  and dtypes of a semiring value can be obtained with methods
  `last.semirings.value_shape()` and `last.semirings.value_dtype()`.

  Semiring is not an abstract base class because we allow operations to be
  unimplemented (e.g. `prod`, is not commonly used).
  """

  def zeros(self, shape: Sequence[int], dtype: Optional[DType] = None) -> T:
    """Semiring zeros in the given shape and dtype.

    Args:
      shape: Desired output shape.
      dtype: Optional PyTree of dtypes.

    Returns:
      If dtype is None, semiring zero values in the specified shape with
      reasonable default dtypes. Otherwise, semiring zero values in the
      specified shape with the specified dtypes.
    """
    raise NotImplementedError

  def ones(self, shape: Sequence[int], dtype: Optional[DType] = None) -> T:
    """Semiring ones in the given shape and dtype.

    Args:
      shape: Desired output shape.
      dtype: Optional PyTree of dtypes.

    Returns:
      If dtype is None, semiring one values in the specified shape with
      reasonable default dtypes. Otherwise, semiring one values in the
      specified shape with the specified dtypes.
    """
    raise NotImplementedError

  def times(self, a: T, b: T) -> T:
    """Semiring multiplication between two values."""
    raise NotImplementedError

  def plus(self, a: T, b: T) -> T:
    """Semiring addition between two values."""
    raise NotImplementedError

  def prod(self, a: T, dim: int) -> T:
    """Semiring multiplication along a single axis."""
    raise NotImplementedError

  def sum(self, a: T, dim: int) -> T:
    """Semiring addition along a single axis."""
    raise NotImplementedError

class _Real(Semiring[torch.Tensor]):
  """Real semiring."""

  @staticmethod
  def zeros(
      shape: Sequence[int], dtype: Optional[DType] = None
  ) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)
  
  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype)
  
  @staticmethod
  def times(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    return a * b

  @staticmethod
  def plus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

  @staticmethod
  def prod(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.prod(a, dim)

  @staticmethod
  def sum(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(a, dim)


Real = _Real() 


def _check_axis(a: torch.Tensor, axis: int) -> None:
  if not isinstance(axis, int):
    raise ValueError(f'Only int axis is supported, got axis={axis!r}')
  if not -a.ndim <= axis < a.ndim:
    raise ValueError(
        f'Invalid reduction axis={axis!r} for input shape {a.shape}')


class _Log(Semiring[torch.Tensor]):
  """Log semiring."""

  @staticmethod
  def zeros(
    shape: Sequence[int], dtype: Optional[DType] = None
  ) -> torch.Tensor:
    return torch.full(shape, -torch.inf, dtype=dtype)

  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)
  
  @staticmethod
  def times(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b
  
  @staticmethod
  def plus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = torch.broadcast_tensors(a, b)
    return _logaddexp(a,b)

  @staticmethod
  def prod(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(a, dim)
  
  @classmethod
  def sum(cls, a: torch.Tensor, dim: int) -> torch.Tensor:
    _check_axis(a, dim)
    # Special handling for safe gradients:
    if torch.numel(a) > 0:
      return _logsumexp(a, dim)
    # Summing empty input should result in zeros:
    if dim < 0:
      dim += a.ndim
    result_shape = a.shape[:dim] + a.shape[dim + 1:]
    return cls.zeros(result_shape, a.dtype)

# Specialized log{add,sum}exp with safe gradients.
#
# Scenarios:
# -   All operands are finite: As expected.
# -   All operands are -inf: Sum should be -inf. Gradient should be 0.
# -   All operands are +inf: Sum should be +inf. Gradient should be NaN.
# -   Mixed finite & -inf operands: Sum as expected. Gradient should be 0 for
#     -inf; non-0 for others.
# -   Mixed finite & +inf operands: Sum should +inf. Gradient should be NaN for
#     +inf; 0 for others.
# -   Mixed -inf & +inf operands: Sum should be +inf. Gradient should be NaN for
#     +inf; 0 for -inf.
# -   Mixed finite, -inf & +inf operands: Sum should be +inf. Gradient should be
#     NaN for +inf; 0 for others.
#
# The different treatment of -inf & +inf comes from their different sources.
# -   +inf is an indicator of a true error, e.g. an overflow somewhere. It's
#     thus desirabled to not silence such issues.
# -   -inf often arises from perfectly legitimate computations such as
#     `logaddexp(-inf, -inf + x)`, where `x` should not receive a NaN gradient.


class _LogAddExp(torch.autograd.Function):
  """Specialized log add exp with safe gradients."""

  @staticmethod
  def forward(ctx, a, b):
    c = torch.max(a, b)
    safe = torch.isfinite(c)
    c = torch.where(safe, c, 0)
    ea = torch.exp(a - c)
    eb = torch.exp(b - c)
    z = ea + eb
    ctx.save_for_backward((ea, eb, z))
    return c + torch.log(z)
  
  @staticmethod
  def backward(ctx, grad):
    ea, eb, z, = ctx.saved_tensors
    safe = z != 0
    z = torch.where(safe, z, 1)
    scale = grad / z
    return scale * ea, scale * eb


_logaddexp = _LogAddExp.apply


class _LogSumExp(torch.autograd.Function):
  """Specialized log add exp with safe gradients."""

  @staticmethod
  def forward(ctx, a, dim):
    c = torch.max(a)
    safe = torch.isfinite(c)
    c = torch.where(safe, c, 0)
    e = torch.exp(a - c)
    z = torch.sum(e, dim=dim, keepdim=True)
    r = torch.squeeze(c, dim=dim) + torch.log(torch.squeeze(z, dim=dim))
    ctx.save_for_backward((e, z, dim))
    return r
  
  @staticmethod
  def backward(ctx, g):
    e, z, dim, = ctx.saved_tensors
    safe = z != 0
    z = torch.where(safe, z, 1)
    g = torch.unsqueeze(g, dim=dim)
    return (g / z * e,)


_logsumexp = _LogSumExp.apply

Log = _Log()


class _MaxTropical(Semiring):
  """Max tropical semiring.

  The gradients of `plus` and `sum` is guaranteed to be non-zero on exactly 1
  input element, even in the event of a tie.
  """

  @staticmethod
  def zeros(
    shape: Sequence[int], dtype: Optional[DType] = None
  ) -> torch.Tensor:
    return torch.full(shape, -torch.inf, dtype=dtype)

  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)
  
  @staticmethod
  def times(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b
  
  @staticmethod
  def plus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = torch.broadcast_tensors(a, b)
    return _maximum(a, b)

  @staticmethod
  def prod(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(a, dim=dim)

  @classmethod
  def sum(cls, a: torch.Tensor, dim: int) -> torch.Tensor:
    _check_axis(a, dim)
    # Special handling is used for safe gradients.
    if torch.numel(a) > 0:
      return _max(a, dim=dim)
    # Summing empty input should result in zeros:
    if dim < 0:
      dim += a.ndim
    result_shape = a.shape[:dim] + a.shape[dim + 1:]
    return cls.zeros(result_shape, a.dtype)


MaxTropical = _MaxTropical()


class Maximum(torch.autograd.Function):

  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a >= b)
    return torch.max(a, b)
  
  @staticmethod
  def backward(ctx, g):
    choose_a, = ctx.saved_tensors
    return g * choose_a, g * (1 - choose_a)
  
_maximum = Maximum.apply

class Max(torch.autograd.Function):

  @staticmethod
  def forward(ctx, a, dim):
    argmax = torch.argmax(a, dim=dim)
    width = a.shape[dim]
    ctx.save_for_backward((argmax, width, dim))
    return torch.max(a, dim=dim)
  
  @staticmethod
  def backward(ctx, g):
    argmax, width, dim, = ctx.saved_tensors
    mask = torch.nn.functional.one_hot(argmax, width)
    g = torch.unsqueeze(g, dim=dim)
    return (g * mask,)


