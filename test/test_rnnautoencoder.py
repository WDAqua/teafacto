from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import RNNAutoEncoder, RewAttRNNEncDecoder, RewAttSumDecoder, FwdAttRNNEncDecoder, FwdAttSumDecoder, BiFwdAttSumDecoder, BiRewAttSumDecoder


def shiftdata(x):
    return np.concatenate([np.zeros_like(x[:, 0:1]), x[:, :-1]], axis=1)

# TODO: test params

class TestRNNAutoEncoder(TestCase):
    def setUp(self):
        vocsize = 24
        innerdim = 200
        encdim = 190
        batsize = 500
        seqlen = 5
        self.exppredshape = (batsize, seqlen, vocsize)
        self.rae = self.get_rae(vocsize=vocsize, outvocsize=vocsize, innerdim=innerdim, encdim=encdim)
        self.dummydata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.dummypred = self.rae.predict(self.dummydata, shiftdata(self.dummydata))

    def get_rae(self, **kwargs):
        return RNNAutoEncoder(**kwargs)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)


class AttentionRNNAutoEncoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return RewAttRNNEncDecoder(**kwargs)


class RNNAttWSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return RewAttSumDecoder(**kwargs)


class FwdAttWSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return FwdAttSumDecoder(**kwargs)


class FwdAttRNNDecoderTEst(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return FwdAttRNNEncDecoder(**kwargs)


class BiFwdAttSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return BiFwdAttSumDecoder(**kwargs)

class BiRewAttSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return BiRewAttSumDecoder(**kwargs)