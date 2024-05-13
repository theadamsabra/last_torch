from absl.testing import absltest
from last_torch import semirings
import torch

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