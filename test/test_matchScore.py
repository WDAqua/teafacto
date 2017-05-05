from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.enc import SeqUnroll
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.match import SeqMatchScore, CosineDistance


class TestMatchScore(TestCase):

    def test_seq_scoring(self):
        vocsize = 100
        dim = 10
        numsam = 17
        seqlen = 5
        ve = VectorEmbed(vocsize, dim)
        m = SeqMatchScore(SeqUnroll(ve), SeqUnroll(ve), scorer=CosineDistance())

        data = np.random.randint(0, vocsize, (numsam, seqlen))
        #print data.shape
        pred = m.predict(data, data)
        #print pred
        self.assertTrue(np.allclose(np.ones_like(pred)*seqlen*1., pred))

