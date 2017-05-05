from unittest import TestCase
from teafacto.blocks.pool import Pool, MaxPool, SumPool, AvgPool
from teafacto.core.base import Val
import numpy as np


class TestPool(TestCase):
    def test_total_pool(self):
        xval = np.random.random((10, 5))
        x = Val(xval)
        xpooled = MaxPool((None,), axis=(1,))(x)
        self.assertTrue(np.allclose(xpooled.eval(), xval.max(axis=1)))
        xpooled = SumPool((None,), axis=(1,))(x)
        self.assertTrue(np.allclose(xpooled.eval(), xval.sum(axis=1)))

    def test_fail_wrong_args(self):
        self.assertRaises(AssertionError, lambda : Pool((None, None), axis=(1,)))

    def test_normal_pool(self):
        xval = np.random.random((10, 6))
        x = Val(xval)
        xpooled = AvgPool((2, 2))(x)
        self.assertEqual(xpooled.eval().shape, (5, 3))

    def test_high_d_pool(self):
        xval = np.random.random((10, 6, 18, 14))
        x = Val(xval)
        xpooled = AvgPool((2, 2), axis=(2, 1))(x)
        self.assertEqual(xpooled.eval().shape, (10, 3, 9, 14))
        xpooled = AvgPool((2, 3), axis=(1, 2))(x)
        self.assertEqual(xpooled.eval().shape, (10, 3, 6, 14))
        xpooled = AvgPool((2, 3), axis=(2, 1))(x)
        self.assertEqual(xpooled.eval().shape, (10, 2, 9, 14))

# TODO: test strides and pads

