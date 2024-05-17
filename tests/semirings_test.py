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

"""Tests for semirings."""

from absl.testing import absltest

from last_torch import semirings
import torch
import numpy.testing as npt



def zero_and_one_test(semiring):
  one = semiring.ones([3])
  zero = semiring.zeros([3])
  xs = torch.Tensor([1., 2., 3.])

  for args in [(one, xs), (xs, one)]:
    npt.assert_array_equal(semiring.times(*args), xs)
    npt.assert_array_equal(semiring.prod(torch.stack(args), dim=0), xs)

    npt.assert_array_equal(
        semiring.times(semiring.ones((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.zeros((1, 2)), semiring.ones((3, 1))),
        semiring.zeros((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.ones((1, 2)), semiring.ones((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))

    npt.assert_array_equal(
        semiring.plus(semiring.ones((1, 2)), semiring.zeros((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.plus(semiring.zeros((1, 2)), semiring.ones((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.plus(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))

    # TODO: Uncomment when cls is figured out for Log Semiring
    # npt.assert_array_equal(
    #     semiring.sum(torch.zeros([3, 0]), dim=0), torch.zeros([0]))
    npt.assert_array_equal(
        semiring.prod(torch.zeros([3, 0]), dim=0), torch.zeros([0]))

    # TODO: Uncomment when cls is figured out for Log Semiring
    # npt.assert_array_equal(semiring.sum(torch.zeros([3, 0]), dim=1), zero)
    npt.assert_array_equal(semiring.prod(torch.zeros([3, 0]), dim=1), one)


class SemiringTest(absltest.TestCase):

  def test_value_shape(self):
    self.assertEqual(semirings.value_shape(torch.zeros([1, 2])), (1, 2))
    self.assertEqual(
        semirings.value_shape({'a': torch.zeros([1, 2]), 'b': torch.ones([1, 2])}),
        (1, 2),
    )
    with self.assertRaisesRegex(
        ValueError, 'No common shape can be derived for an empty PyTree'
    ):
      semirings.value_shape(None)
    with self.assertRaisesRegex(
        ValueError,
        'A semiring value must consist of ndarrays of a common shape',
    ):
      semirings.value_shape({'a': torch.zeros([1, 2]), 'b': torch.ones([2])})

class RealTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(semirings.Real.times(torch.Tensor([2]), torch.Tensor([3])), 6)
    npt.assert_array_equal(semirings.Real.prod(torch.Tensor([2, 3]), dim=0), 6)
    npt.assert_array_equal(semirings.Real.plus(torch.Tensor([2]), torch.Tensor([3])), 5)
    npt.assert_array_equal(semirings.Real.sum(torch.Tensor([2, 3]), dim=0), 5)
    zero_and_one_test(semirings.Real)


class LogTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(semirings.Log.times(torch.Tensor([2]), torch.Tensor([3])), 5)
    self.assertEqual(semirings.Log.prod(torch.Tensor([2, 3]), dim=0), 5)
    npt.assert_allclose(
        semirings.Log.plus(torch.Tensor([2]), torch.Tensor([3])), 3.31326169
    )
    # TODO: uncomment test after asking about cls
    # npt.assert_allclose(
    #     semirings.Log.sum(torch.Tensor([2, 3]), dim=0), 3.31326169)
    zero_and_one_test(semirings.Log)


if __name__ == '__main__':
  absltest.main()