from unittest import TestCase
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
import numpy as np


class TestSimpleSeqEncDecAtt(TestCase):
    def test_unidir_shapes(self):
        self.do_test_shapes(False)

    def test_bidir_shapes(self):
        self.do_test_shapes(True)

    def do_test_shapes(self, bidir=False):
        inpvocsize = 100
        outvocsize = 13
        inpembdim = 10
        outembdim = 5
        encdim = 9
        decdim = 7
        attdim = 8
        batsize = 11
        inpseqlen = 7
        outseqlen = 5

        m = SimpleSeqEncDecAtt(inpvocsize=inpvocsize,
                               inpembdim=inpembdim,
                               outvocsize=outvocsize,
                               outembdim=outembdim,
                               encdim=encdim,
                               decdim=decdim,
                               attdim=attdim,
                               bidir=bidir)

        inpseq = np.random.randint(0, inpvocsize, (batsize, inpseqlen)).astype("int32")
        outseq = np.random.randint(0, outvocsize, (batsize, outseqlen)).astype("int32")

        predenco, _, _ = m.enc.predict(inpseq)
        self.assertEqual(predenco.shape, (batsize, encdim if not bidir else encdim*2))

        pred = m.predict(inpseq, outseq)
        self.assertEqual(pred.shape, (batsize, outseqlen, outvocsize))