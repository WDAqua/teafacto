from unittest import TestCase
import numpy as np
from IPython import embed
from teafacto.examples.dummy import Dummy


class TestExternalValidators(TestCase):
    def test_external_validator(self):
        vocabsize = 1000
        m = Dummy(indim=vocabsize, dim=10, outdim=2000)
        numbats = 20
        lr = 0.8
        data = np.arange(0, vocabsize).astype("int32")
        gdata = np.random.randint(0, 2000, (vocabsize,))

        mpredf = m.predict

        def extacc(*sampleinp):
            pred = mpredf(*sampleinp[:-1])
            ret = np.sum(np.argmax(pred, axis=1) == sampleinp[-1])
            return [ret * 1. / sampleinp[-1].shape[0]]

        _, err, verr, _, _ = \
            m.train([data], gdata).adadelta(lr=lr).cross_entropy() \
             .autovalidate().cross_entropy().accuracy().extvalid(extacc) \
            .train(numbats=numbats, epochs=10, returnerrors=True)

        verr = np.asarray(verr)
        verr = verr[:, 1] + verr[:, 2]
        self.assertTrue(np.allclose(verr, np.ones_like(verr)))
