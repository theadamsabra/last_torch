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

"""Tests for weight_fns."""

from last_torch import weight_fns
from absl.testing import absltest
import torch
import numpy.testing as npt


class WeightFnTets(absltest.TestCase):

    def test_hat_normalize(self):
        blank = torch.Tensor([2, 7])
        lexical = torch.Tensor([[0, 1], [3, 5]])
        expect_blank = torch.Tensor([-0.126928, -0.000912])
        expect_lexical = torch.Tensor([[-3.44019, -2.44019], [-9.12784, -7.12784]])
        actual_blank, actual_lexical = weight_fns.hat_normalize(blank, lexical)
        npt.assert_allclose(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
        npt.assert_allclose(actual_lexical, expect_lexical, rtol=1e-3, atol=1e-6)