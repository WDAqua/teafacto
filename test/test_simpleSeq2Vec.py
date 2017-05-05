from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeq2Sca, SimpleSeq2MultiVec
from teafacto.blocks.match import MatchScore


class TestSimpleSeq2Vec(TestCase):
    def setUp(self):
        enc = SimpleSeq2Vec(indim=100, inpembdim=10, innerdim=20)
        x = np.random.randint(0, 100, (33, 5))
        o = enc.autobuild(x)
        self.o = o[1][0]
        m = MatchScore(enc, enc)
        mo = m.autobuild(x, x)
        self.mo = mo[1][0]

    def test_output(self):
        print self.mo, self.mo.ndim
        #theano.printing.pydotprint(self.mo.d, "debug.png")

    def test_mask(self):
        np.random.seed(1337)
        enc = SimpleSeq2Vec(indim=100, inpembdim=10, innerdim=4, maskid=-1, layers=2).all_outputs()
        x = np.random.randint(0, 100, (33, 5))
        maskr = np.random.randint(1, x.shape[1], (x.shape[0],))
        for i in range(x.shape[0]):
            x[i, maskr[i]:] = -1
        pred = enc.predict(x)
        print maskr
        print x
        print pred
        print pred.shape


class TestSimpleSeq2Sca(TestCase):
    def test_shape(self):
        enc = SimpleSeq2Sca(indim=100, inpembdim=10, innerdim=20)
        x = np.random.randint(0, 10, (33, 5))
        prd, mask = enc.predict(x)
        self.assertEqual(prd.shape, (33, 5))
        self.assertEqual(mask.shape, (33, 5))


class TestSimpleSeq2MultiVec(TestCase):
    def test_shape(self):
        enc = SimpleSeq2MultiVec(indim=100, inpembdim=10,
                                 innerdim=20, numouts=2, mode="seq")
        x = np.random.randint(0, 100, (33, 5))
        prd = enc.predict(x)
        self.assertEqual(prd.shape, (33, 2, 20))


