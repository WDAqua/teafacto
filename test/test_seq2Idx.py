from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.enc import SimpleSeq2Idx


class TestSeq2Idx(TestCase):
    def test_output_shape_and_probness(self):
        batsize = 10
        seqlen = 7
        invocsize = 12
        inpembdim = 9
        innerdim = 5
        innerdim2 = 6
        outvocsize = 17

        data = np.random.randint(0, invocsize, (batsize, seqlen)) - 1
        gold = np.random.randint(0, outvocsize, (batsize,))

        m = SimpleSeq2Idx(indim=invocsize-1, outdim=outvocsize, inpembdim=inpembdim, innerdim=[innerdim, innerdim2], maskid=-1)

        mpred = m.predict(data)
        self.assertEqual(mpred.shape, (batsize, outvocsize))
        self.assertTrue(np.allclose(mpred.sum(axis=1), np.ones((mpred.shape[0]))))

