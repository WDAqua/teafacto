from unittest import TestCase
from teafacto.core.base import *


class TestCoreMask(TestCase):
    def test_mask_autogen(self):
        xval = np.random.randint(0, 5, (4, 3))
        x = Val(xval)
        x.makemask()
        self.assertTrue(np.allclose(xval == 0, x.mask.v))
        self.assertEqual(xval.shape, x.mask.v.shape)
