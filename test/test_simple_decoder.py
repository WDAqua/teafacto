from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SimpleEncoderDecoder


class TestSimpleEncoderDecoder(TestCase):
    def setUp(self):
        seqlen = 5
        batsize = 33
        inpvocsize = 27
        innerdim = 7
        outvocsize = 17

        self.sed = SimpleEncoderDecoder(innerdim=innerdim, input_vocsize=inpvocsize, output_vocsize=outvocsize)
        inpseqv = np.random.randint(0, inpvocsize, (batsize, seqlen))
        outseqv = np.random.randint(0, outvocsize, (batsize, seqlen))
        self.sedoutv = self.sed.predict(inpseqv, outseqv)
        self.sedoutvshape = (batsize, seqlen, outvocsize)

    def test_output_shapes(self):
        self.assertEqual(self.sedoutv.shape, self.sedoutvshape)

    def test_propagated_param_names(self):
        expectedparams = {"w": 2,
                          "u": 2,
                          "b": 2,
                          "wm": 2,
                          "um": 2,
                          "bm": 2,
                          "whf": 2,
                          "uhf": 2,
                          "bhf": 2,
                          "matdot": 1}
        for param in [x.name for x in self.sed.output.allparams]:
            expectedparams[param] -= 1
        for k, v in expectedparams.items():
            self.assertEqual(v, 0)