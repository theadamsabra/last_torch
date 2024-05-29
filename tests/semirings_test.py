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
from torch.utils._pytree import tree_map


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


class ExpectationTest(absltest.TestCase):

  def test_basics(self):
    one = semirings.LogLogExpectation.ones([])
    zero = semirings.LogLogExpectation.zeros([])
    for wx in [
      semirings.LogLogExpectation.weighted(torch.Tensor([1]), torch.Tensor([2])),
      one, zero
    ]:
      with self.subTest(str(wx)):
        tree_map(npt.assert_array_equal,
                 semirings.LogLogExpectation.times(wx, one), wx)
        tree_map(npt.assert_array_equal,
                 semirings.LogLogExpectation.times(one, wx), wx)                 
        tree_map(npt.assert_array_equal,
                 semirings.LogLogExpectation.plus(wx, zero), wx)
        tree_map(npt.assert_array_equal,
                 semirings.LogLogExpectation.plus(zero, wx), wx)
  
  def test_shape_dtypes(self):
    one = semirings.LogLogExpectation.ones([1, 2], (torch.float32, torch.bfloat16))
    self.assertEqual(semirings.value_shape(one), (1, 2))
    self.assertEqual(semirings.value_dtype(one), (torch.float32, torch.bfloat16))
    zero = semirings.LogLogExpectation.zeros([], (torch.bfloat16, torch.float32))
    self.assertEqual(semirings.value_shape(zero), ())
    self.assertEqual(semirings.value_dtype(zero), (torch.bfloat16, torch.float32))

  def test_weighted(self):
    w, x = semirings.LogLogExpectation.weighted(
        torch.log(torch.Tensor([0, 1, 2])), torch.log(torch.Tensor([3, 4, 5])))
    npt.assert_allclose(torch.exp(w), [0, 1, 2])
    npt.assert_allclose(torch.exp(x), [0 * 3, 1 * 4, 2 * 5])

  def test_weighted_safety(self):
    w = torch.Tensor([float('-inf')])
    v = torch.Tensor([float('inf')])
    w, x = semirings.LogLogExpectation.weighted(w, v)
    npt.assert_array_equal(w, float('-inf'))
    npt.assert_array_equal(x, float('-inf'))

  def test_sum(self):
    w, x = semirings.LogLogExpectation.sum(
        semirings.LogLogExpectation.weighted(
            torch.log(torch.Tensor([[0, 1], [2, 3]])),
            torch.log(torch.Tensor([[4, 5], [6, 7]]))),
        axis=1)
    npt.assert_allclose(torch.exp(w), [0 + 1, 2 + 3])
    npt.assert_allclose(torch.exp(x), [0 * 4 + 1 * 5, 2 * 6 + 3 * 7], rtol=1e-6)

  def test_entropy(self):
    probs = torch.Tensor([0.25, 0.25, 0.5])
    log_probs = torch.log(probs)
    wx = semirings.LogLogExpectation.weighted(log_probs, torch.log(-log_probs))
    log_z, log_sum = semirings.LogLogExpectation.sum(wx, axis=0)
    npt.assert_allclose(log_z, 0)
    entropy = torch.exp(log_sum)
    npt.assert_allclose(entropy, -torch.sum(probs * log_probs))

    new_probs = torch.Tensor([0.25, 0.5, 0.25])
    new_log_probs = torch.log(new_probs)
    new_wx = semirings.LogLogExpectation.weighted(new_log_probs,
                                                  torch.log(-new_log_probs))
    log_z, log_sum = semirings.LogLogExpectation.sum(
        semirings.LogLogExpectation.times(wx, new_wx), axis=0)
    npt.assert_allclose(torch.exp(log_z), torch.sum(probs * new_probs))
    entropy = log_z + torch.exp(log_sum - log_z)
    npt.assert_allclose(
        entropy, -torch.sum(probs * new_probs * torch.exp(-log_z) *
                          (log_probs + new_log_probs - log_z)))

class CartesianTest(absltest.TestCase):

  def test_basics(self):
    semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
    one = semiring.ones([])
    zero = semiring.zeros([])
    for wx in [(torch.tensor(1.0), torch.tensor(2.0)), one, zero]:
      with self.subTest(str(wx)):
        tree_map(
            npt.assert_array_equal, semiring.times(wx, one), wx
        )
        tree_map(
            npt.assert_array_equal, semiring.times(one, wx), wx
        )
        tree_map(
            npt.assert_array_equal, semiring.plus(wx, zero), wx
        )
        tree_map(
            npt.assert_array_equal, semiring.plus(zero, wx), wx
        )

  def test_shape_dtypes(self):
    semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
    one = semiring.ones([1, 2], (torch.float32, torch.bfloat16))
    self.assertEqual(semirings.value_shape(one), (1, 2))
    self.assertEqual(semirings.value_dtype(one), (torch.float32, torch.bfloat16))
    zero = semiring.zeros([], (torch.bfloat16, torch.float32))
    self.assertEqual(semirings.value_shape(zero), ())
    self.assertEqual(semirings.value_dtype(zero), (torch.bfloat16, torch.float32))

  def test_arithmetics(self):
      semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
      a = (torch.tensor(2.0), torch.tensor(1.0))
      b = (torch.tensor(3.0), torch.tensor(4.0))
      c = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

      with self.subTest('times'):
        a_times_b = semiring.times(a, b)
        self.assertIsInstance(a_times_b, tuple)
        self.assertLen(a_times_b, 2)
        npt.assert_array_equal(a_times_b[0], 6.0)
        npt.assert_array_equal(a_times_b[1], 5.0)

      with self.subTest('plus'):
        a_plus_b = semiring.plus(a, b)
        self.assertIsInstance(a_plus_b, tuple)
        self.assertLen(a_plus_b, 2)
        npt.assert_array_equal(a_plus_b[0], 5.0)
        npt.assert_array_equal(a_plus_b[1], 4.0)

      with self.subTest('sum'):
        sum_c = semiring.sum(c, axis=0)
        self.assertIsInstance(sum_c, tuple)
        self.assertLen(sum_c, 2)
        npt.assert_array_equal(sum_c[0], 3.0)
        npt.assert_array_equal(sum_c[1], 4.0)

      with self.subTest('prod'):
        prod_c = semiring.prod(c, axis=0)
        self.assertIsInstance(prod_c, tuple)
        self.assertLen(prod_c, 2)
        npt.assert_array_equal(prod_c[0], 2.0)
        npt.assert_array_equal(prod_c[1], 7.0)

if __name__ == '__main__':
  absltest.main()