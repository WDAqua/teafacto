from unittest import TestCase
from teafacto.blocks.basic import IdxToOneHot, MatDot, Linear, Softmax
import numpy as np

class TestBasic(TestCase):
    def test_idx_to_one_hot(self):
        ioh = IdxToOneHot(25)
        data = np.arange(0, 25).astype("int32")
        expout = np.eye(25, 25)
        outioh = ioh.predict(data)
        self.assertEqual(np.linalg.norm(expout - outioh), 0)


class TestMatDot(TestCase):
    def setUp(self):
        self.matdot = MatDot(indim=10, dim=15)
        self.data = np.random.random((100, 10))
        self.matdotout = self.matdot.predict(self.data)

    def test_matdot_shapes(self):
        self.assertEqual(self.matdotout.shape, (100, 15))

    def test_matdot_output(self):
        self.assertTrue(np.allclose(self.matdotout, np.dot(self.data, self.matdot.W.d.get_value())))


class TestLinear(TestCase):
    def setUp(self):
        self.linear = Linear(indim=10, dim=15)
        self.data = np.random.random((100, 10))
        self.out = self.linear.predict(self.data)

    def test_linear_shapes(self):
        self.assertEqual(self.out.shape, (100, 15))

    def test_linear_output(self):
        self.assertTrue(np.allclose(self.out, np.dot(self.data, self.linear.W.d.get_value()) + self.linear.b.d.get_value()))

