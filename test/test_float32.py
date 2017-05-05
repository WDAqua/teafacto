from unittest import TestCase
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed
import theano, numpy as np

class TestFloat32(TestCase):
    def setUp(self):
        theano.config.floatX = "float32"

    def test_idxtoonehot(self):
        m = IdxToOneHot(100)
        self.assertEqual(m.W.dtype, "float32")

    def test_vectorembed_init(self):
        m = VectorEmbed(indim=100, dim=100)
        self.assertEqual(m.W.dtype, "float32")

    def test_vectorembed_cast(self):
        md = np.random.random((100, 100))
        m = VectorEmbed(indim=100, dim=100, value=md)
        self.assertEqual(m.W.dtype, "float32")

