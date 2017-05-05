from unittest import TestCase
from teafacto.blocks.mrnu import XLTM
import numpy as np


class TestXLTM(TestCase):
    def test_output_shape(self):
        # create XLTM
        indim = 20
        memsize = 50
        innerdim = 30
        xltm = XLTM(indim=indim, memsize=memsize, innerdim=innerdim, discrete=False)

        # generate some data
        batsize = 100
        seqlen = 15
        x = np.random.random((batsize, seqlen, indim)).astype("float32")

        # generate prediction
        #prediction = xltm.predict(x)

