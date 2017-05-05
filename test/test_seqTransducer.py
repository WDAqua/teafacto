from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.oldseqproc import SimpleSeqTransDec
from teafacto.blocks.seq.trans import SimpleSeqTrans


def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")


class TestSeqTransducer(TestCase):
    def test_output_shape(self):
        # settings
        batsize = 10
        seqlen = 5
        invocsize = 50
        inembdim = 50
        innerdim = 11
        outvocsize = 17

        # data
        traindata = np.random.randint(0, invocsize, (batsize, seqlen))
        traingold = np.random.randint(0, outvocsize, (batsize, seqlen))

        # model
        m = SimpleSeqTrans(indim=invocsize, embdim=inembdim, innerdim=innerdim, outdim=outvocsize)

        pred = m.predict(traindata)
        self.assertEqual(pred.shape, (batsize, seqlen, outvocsize))


class TestSeqTransDec(TestCase):
    def test_output_shape(self):
        # settings
        batsize = 10
        seqlen = 5
        invocsize = 50
        inembdim = 50
        outembdim = 40
        innerdim = 11
        outvocsize = 17

        # data
        traindata = np.random.randint(0, invocsize, (batsize, seqlen))
        traingold = np.random.randint(0, outvocsize, (batsize, seqlen))

        # model
        m = SimpleSeqTransDec(indim=invocsize, inpembdim=inembdim, outpembdim=outembdim, innerdim=innerdim, outdim=outvocsize)

        pred = m.predict(traindata, shiftdata(traingold))
        self.assertEqual(pred.shape, (batsize, seqlen, outvocsize))
