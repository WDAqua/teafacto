from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SeqDecoder
from teafacto.blocks.seq.rnu import GRU, LSTM
from teafacto.blocks.basic import IdxToOneHot


class TestSimpleRNNDecoder(TestCase):
    def setUp(self):
        self.batsize = 70
        self.vocsize = 27
        self.encdim = 44
        self.hdim = 33
        self.statedim = 50
        self.seqlen = 30
        self.encodings_data = np.random.random((self.batsize, self.encdim)).astype("float32")
        self.sequence_data = np.random.randint(0, self.vocsize, (self.batsize, self.seqlen))
        self.dec = SeqDecoder(
            [IdxToOneHot(self.vocsize), GRU(dim=self.vocsize+self.encdim, innerdim=self.hdim), LSTM(dim=self.hdim, innerdim=self.statedim)],
            inconcat=True,
            innerdim=self.statedim
        )

    def test_rnndecoder_output_shape(self):
        outvals = self.dec.predict(self.encodings_data, self.sequence_data)
        self.assertEqual(outvals.shape, (self.batsize, self.seqlen, self.vocsize))