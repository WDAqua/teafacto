from unittest import TestCase
from teafacto.core.base import *
from teafacto.core.base import tensorops as T


class TestWrapAndAsblock(TestCase):
    def setUp(self):
        x = param((10, 20)).uniform()
        ab = asblock(lambda y: T.dot(y, x))
        inputdata = np.random.random((100, 10))
        self.outvals = ab.predict(inputdata)

    def test_wrap_output_shape(self):
        self.assertEqual(self.outvals.shape, (100, 20))
