from unittest import TestCase
from teafacto.blocks.match import GenDotDistance, LinearDistance
import numpy as np

class TestGenDotDistance(TestCase):
    def test_shape(self):
        batsize = 10
        ldim = 5
        rdim = 4
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = GenDotDistance(ldim, rdim)
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))


class TestLinearGateDistance(TestCase):
    def test_shape(self):
        batsize = 10
        ldim = 5
        rdim = 4
        aggdim = 7
        l = np.random.random((batsize, ldim))
        r = np.random.random((batsize, rdim))
        b = LinearDistance(ldim + rdim, aggdim)
        pred = b.predict(l, r)
        self.assertEqual(pred.shape, (batsize,))
