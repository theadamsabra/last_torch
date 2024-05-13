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