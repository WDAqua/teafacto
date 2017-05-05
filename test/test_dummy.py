from unittest import TestCase

from teafacto.examples.dummy import run
import numpy as np


class TestDummy(TestCase):
    def test_dummy_run(self):
        preds = run(epochs=0)
        print preds
        self.assertTrue(np.allclose(0.0005*np.ones_like(preds), preds, rtol=1e-2))
