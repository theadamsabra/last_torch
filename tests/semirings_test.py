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

    npt.assert_array_equal(
        semiring.sum(torch.zeros([3, 0]), dim=0), torch.zeros([0]))
    npt.assert_array_equal(
        semiring.prod(torch.zeros([3, 0]), dim=0), torch.zeros([0]))

    npt.assert_array_equal(semiring.sum(torch.zeros([3, 0]), dim=1), zero)
    npt.assert_array_equal(semiring.prod(torch.zeros([3, 0]), dim=1), one)


def expected(op, x, y):
  expected_z, expected_vjp_fn = torch.func.vjp(
      lambda x, y: op(*torch.broadcast_tensors(x, y)), x, y
  )
  expected_dx, expected_dy = expected_vjp_fn(torch.ones_like(expected_z))
  return expected_z, expected_dx, expected_dy


def binary_op_broadcasting_test_times(semiring):
  # TODO: implement for broadcasting test plus
  for op in [semiring.times]:
    for shapes in [
        ([], [2]),
        ([1], [2]),
        ([1, 2], [3, 2]),
        ([2, 1], [2, 3]),
        ([3], [2, 3]),
    ]:
      for shape_x, shape_y in [shapes, shapes[::-1]]:
        err_msg = f'op={op} shapes={(shape_x, shape_y)}'
        x = semiring.ones(shape_x)
        y = semiring.ones(shape_y)
        z, vjp_fn = torch.func.vjp(op, x, y)
        dx, dy = vjp_fn(torch.ones_like(z))
        expected_z, expected_dx, expected_dy = expected(op, x, y)
        npt.assert_allclose(z, expected_z, err_msg=err_msg)
        npt.assert_allclose(dx, expected_dx, err_msg=err_msg)
        npt.assert_allclose(dy, expected_dy, err_msg=err_msg)


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
    binary_op_broadcasting_test_times(semirings.Real)


def check_sum_axis(self, semiring):
  """Checks that semiring sum handles axes correctly."""
  xs = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape([2, 3, 4, 5]).requires_grad_()

  with self.subTest('forward'):
    self.assertEqual(semiring.sum(xs, dim=0).shape, (3, 4, 5))
    self.assertEqual(semiring.sum(xs, dim=1).shape, (2, 4, 5))
    self.assertEqual(semiring.sum(xs, dim=2).shape, (2, 3, 5))
    self.assertEqual(semiring.sum(xs, dim=3).shape, (2, 3, 4))
    self.assertEqual(semiring.sum(xs, dim=-1).shape, (2, 3, 4))
    self.assertEqual(semiring.sum(xs, dim=-2).shape, (2, 3, 5))
    self.assertEqual(semiring.sum(xs, dim=-3).shape, (2, 4, 5))
    self.assertEqual(semiring.sum(xs, dim=-4).shape, (3, 4, 5))
    with self.assertRaisesRegex(ValueError, 'Invalid reduction axis'):
      semiring.sum(xs, dim=4)
    with self.assertRaisesRegex(ValueError, 'Invalid reduction axis'):
      semiring.sum(xs, dim=-5)
    with self.assertRaisesRegex(ValueError, 'Only int axis'):
      semiring.sum(xs, dim=None)  # type: ignore

  with self.subTest('backward'):
    def f(xs, dim):
      zs = semiring.sum(xs, dim=dim)
      while zs.shape:
        zs = torch.sum(zs, dim=0)
      return zs

    for dim in range(-4, 4):
      y = f(xs, dim=dim)
      grad = torch.autograd.grad(y, xs)[0]
      self.assertEqual(grad.shape, xs.shape)


def check_sum_zero_sized(self, semiring):
  """Checks that semiring sum handles zero-sized dimensions correctly."""
  xs = torch.zeros([0, 2])

  npt.assert_array_equal(semiring.sum(xs, dim=0), semiring.zeros([2]))
  npt.assert_array_equal(semiring.sum(xs, dim=-2), semiring.zeros([2]))

  self.assertEqual(semiring.sum(xs, dim=1).shape, (0,))
  self.assertEqual(semiring.sum(xs, dim=-1).shape, (0,))


class LogTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(semirings.Log.times(torch.Tensor([2]), torch.Tensor([3])), 5)
    self.assertEqual(semirings.Log.prod(torch.Tensor([2, 3]), dim=0), 5)
    npt.assert_allclose(
        semirings.Log.plus(torch.Tensor([2]), torch.Tensor([3])), 3.31326169
    )
    npt.assert_allclose(
        semirings.Log.sum(torch.Tensor([2, 3]), dim=0), 3.31326169)
    zero_and_one_test(semirings.Log)
    binary_op_broadcasting_test_times(semirings.Log)


class MaxTropicalTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(
      semirings.MaxTropical.times(torch.Tensor([2]), torch.Tensor([3])), 5
    )
    npt.assert_array_equal(
      semirings.MaxTropical.prod(torch.Tensor([2, 3]), dim=0), 5
    )
    npt.assert_array_equal(
      semirings.MaxTropical.plus(torch.Tensor([2]), torch.Tensor([3])), 3
    )
    npt.assert_array_equal(
      semirings.MaxTropical.sum(torch.Tensor([2,3]), dim=0), 3
    )
    zero_and_one_test(semirings.MaxTropical)
    binary_op_broadcasting_test_times(semirings.MaxTropical)

  def test_plus_grad(self):
    fun = lambda a: torch.sum(semirings.MaxTropical.plus(a[0], a[1]))
    a = torch.Tensor([[1., 2., 3.,], [0., 2., 4.]]).requires_grad_()
    y = fun(a)
    gradient = torch.autograd.grad(y, a)[0]
    npt.assert_array_equal(gradient, torch.Tensor([[1., 1., 0.], [0., 0., 1.]]))

  def test_sum_grad(self):
    fun = lambda a: torch.sum(semirings.MaxTropical.sum(a, dim=0))
    a = torch.Tensor([[1., 2., 3.,], [0., 2., 4.]]).requires_grad_()
    y = fun(a)
    output = torch.Tensor([[1., 1., 0.], [0., 0., 1.]])
    gradient = torch.autograd.grad(y, a)[0]
    npt.assert_array_equal(gradient, output)


    fun = lambda a: torch.sum(semirings.MaxTropical.sum(a, dim=-1))
    a = torch.Tensor([[1., 2., 3.,], [0., 2., 4.]]).requires_grad_().T
    y = fun(a)
    output = torch.Tensor([[1., 1., 0.], [0., 0., 1.]]).T
    gradient = torch.autograd.grad(y, a)[0]
    npt.assert_array_equal(gradient, output)

  def test_sum_axis(self):
    check_sum_axis(self, semirings.MaxTropical)
  
  def test_sum_zero_sized(self):
    check_sum_zero_sized(self, semirings.MaxTropical)

if __name__ == '__main__':
  absltest.main()