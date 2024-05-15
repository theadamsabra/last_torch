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
  shapes = [tuple(i.shape) for i in pytree.tree_leaves(x)]
  if not shapes:
    raise ValueError(
        f'No common shape can be derived for an empty PyTree: {x!r}'
    )
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

  def prod(self, a: T, axis: int) -> T:
    """Semiring multiplication along a single axis."""
    raise NotImplementedError

  def sum(self, a: T, axis: int) -> T:
    """Semiring addition along a single axis."""
    raise NotImplementedError
