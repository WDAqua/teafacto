from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.oldseqproc import SimpleSeqTransDec
from teafacto.blocks.seq.rnn import RecStack
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import Softmax, VectorEmbed
from teafacto.core.base import Input, param, asblock, tensorops as T
from teafacto.core.stack import stack
from teafacto.modelusers import RecPredictor


class TestRecurrentStack(TestCase):
    def setUp(self):
        self.dims = [50, 20, 30, 40]
        grus = [GRU(dim=self.dims[i], innerdim=self.dims[i+1]) for i in range(len(self.dims)-1)]
        self.s = stack(*grus)
        self.paramnames = ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]

    def test_rnu_stack_parameter_propagation(self):
        o = self.s(Input(ndim=3, dtype="float32", name="stack_input"))
        allps = [x.name for x in o.allparams]
        for paramname in self.paramnames:
            self.assertEqual(allps.count(paramname), len(self.dims)-1)


class TestRecurrentStackRecappl(TestCase):
    def test_recappl(self):
        batsize = 100
        self.dims = [50, 20, 30, 40]
        recstack = RecStack(*[GRU(dim=self.dims[i], innerdim=self.dims[i+1]) for i in range(len(self.dims)-1)])
        mu = RecPredictor(recstack)
        for i in range(3):
            inpval = np.random.random((batsize, 50)).astype("float32")
            outpvals = mu.feed(inpval)
            self.assertEqual(outpvals.shape, (batsize, self.dims[-1]))


class TestSeqTransDecRecappl(TestCase):     # TODO: move this test
    def test_recappl_shapes_model_user(self):
        batsize = 100
        model = SimpleSeqTransDec(indim=200, outdim=50, inpembdim=20, outembdim=20, innerdim=[40, 30])
        mu = RecPredictor(model)
        inpval2 = np.random.randint(0, 50, (batsize,)).astype("int32")
        for i in range(5):
            inpval = np.random.randint(0, 200, (batsize,)).astype("int32")
            outpval = mu.feed(inpval, inpval2)
            inpval2 = np.argmax(outpval, axis=1).astype("int32")
            self.assertEqual(outpval.shape, (batsize, 50))



class TestBlockStack(TestCase):
    def setUp(self):
        dim=50
        self.vocabsize=2000
        data = np.arange(0, self.vocabsize).astype("int32")
        self.O = param((dim, self.vocabsize)).uniform()
        self.W = VectorEmbed(indim=self.vocabsize, dim=50)
        self.out = stack(self.W,
              asblock(lambda x: T.dot(self.O, x)),
              Softmax())(Input(ndim=1, dtype="int32"))

    def test_param_propagation(self):
        self.assertSetEqual(set(self.out.allparams), {self.O, self.W.W})

