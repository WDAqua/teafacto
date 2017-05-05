from unittest import TestCase
from teafacto.blocks.basic import Dropout, Val
import numpy as np


class TestDropout(TestCase):
    def test_trainmode_access(self):
        d = Dropout(p=0.5)
        o = d(Val(np.random.random((25,))), _trainmode=True)
        print o.eval()