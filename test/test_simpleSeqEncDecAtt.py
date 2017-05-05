from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt


class TestSimpleSeqEncDecAtt(TestCase):
    def test_vector_out(self):
        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, outconcat=False, decdim=110)
        encdata = np.random.randint(0, 19, (1, 5))
        decdata = np.random.randint(0, 17, (1, 5))
        pred = encdec.predict(encdata, decdata)
        self.assertEqual(pred.shape, (1, 5, 17))

        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, vecout=True, outconcat=False, decdim=110)
        pred = encdec.predict(encdata, decdata)
        print pred.shape
        self.assertEqual(pred.shape, (1, 5, 110))
