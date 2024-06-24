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

"""Tests for alignments."""

from absl.testing import absltest
import torch 
from last_torch import alignments
from last_torch import contexts
from last_torch import semirings
import numpy.testing as npt


class AlignmentsTest(absltest.TestCase):

  def test_shift_down(self):
    npt.assert_array_equal(
        alignments.shift_down(torch.Tensor([1, 2, 3]), semirings.Real), [0, 1, 2])
    npt.assert_array_equal(
        alignments.shift_down(
            torch.Tensor([[1, 2, 3], [4, 5, 6]]), semirings.Real),
        [[0, 1, 2], [0, 4, 5]])
    npt.assert_array_equal(
        alignments.shift_down(
            torch.Tensor([[1, 2, 3], [4, 5, 6]]).to(torch.float32),
            semirings.Log), [[-torch.inf, 1, 2], [-torch.inf, 4, 5]])