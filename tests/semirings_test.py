from absl.testing import absltest
from last_torch import semirings
import torch
import numpy.testing as npt

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
    npt.assert_array_equal(semirings.Real.prod(torch.Tensor([2, 3]), axis=0), 6)
    npt.assert_array_equal(semirings.Real.plus(torch.Tensor([2]), torch.Tensor([3])), 5)
    npt.assert_array_equal(semirings.Real.sum(torch.Tensor([2, 3]), axis=0), 5)

if __name__ == '__main__':
  absltest.main()