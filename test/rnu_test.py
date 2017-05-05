import os
from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnu import GRU, LSTM, IFGRU
from teafacto.core.base import param, Input


class TestGRU(TestCase):
    def setUp(self):
        self.dim = 20
        self.innerdim = 50
        self.rnu = self.makernu()
        self.batsize = 50
        self.seqlen = 10
        self.datashape = (self.batsize, self.seqlen, self.dim)
        self.testdata = np.random.random(self.datashape).astype("float32")
        self.paramnames = self.getparamnames()
        self.wshape = (self.dim, self.innerdim)
        self.ushape = (self.innerdim, self.innerdim)
        self.bshape = (self.innerdim, )
        self.rnu.autobuild(self.testdata)
        self.toremovefiles = []

    def tearDown(self):
        for p in self.toremovefiles:
            os.remove(p)

    def makernu(self):
        return GRU(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]

    def getwparamnames(self):
        return ["wm", "whf", "w"]

    def getuparamnames(self):
        return ["um", "uhf", "u"]

    def getbparamnames(self):
        return ["bm", "bhf", "b"]

    def test_params_owned(self):
        for param in self.paramnames:
            self.assertTrue(hasattr(self.rnu, param))

    def test_param_dims(self):
        for param in self.getwparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.wshape)
        for param in self.getuparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.ushape)
        for param in self.getbparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.bshape)

    def test_params_propagated_to_outvar(self):
        outpvar = self.rnu.output
        gruparamset = set([getattr(self.rnu, paramname) for paramname in self.paramnames])
        varparamset = set(outpvar.allparams)
        self.assertSetEqual(gruparamset, varparamset)

    def test_params_propagated_through_rnu(self):
        O = param((self.dim, self.dim), name="bigo").uniform()
        i = Input(ndim=2, dtype="int32")
        x = O[i, :]
        out = self.rnu(x)
        self.assertIn(O, out.allparams)

    def test_output_shape_predict(self):
        outpv = self.rnu.predict(self.testdata)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen, self.innerdim))

    def test_input_other_batsize(self):
        othershape = (self.batsize*25, self.seqlen, self.dim) # 25 times more batches
        data = np.random.random(othershape).astype("float32")
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize*25, self.seqlen, self.innerdim))

    def test_input_other_seqlen(self):
        othershape = (self.batsize, self.seqlen*25, self.dim) # 25 times longer sequences
        data = np.random.random(othershape).astype("float32")
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen*25, self.innerdim))

    def test_input_fail_other_dims(self):
        othershape = (self.batsize, self.seqlen, self.dim*25)
        data = np.random.random(othershape)
        self.assertRaises(Exception, self.rnu.predict, data)

    def test_save_load_predict(self):
        outpv = self.rnu.predict(self.testdata)
        path = self.rnu.save()
        self.toremovefiles.append(path)
        loaded = self.rnu.__class__.load(path)
        self.assertTrue(np.allclose(outpv, loaded.predict(self.testdata)))


class TestLSTM(TestGRU):
    def makernu(self):
        return LSTM(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["wf", "rf", "bf", "wi", "ri", "bi", "wo", "ro", "bo", "w", "r", "b", "pf", "pi", "po"]

    def getwparamnames(self):
        return ["wf", "wi", "wo", "w"]

    def getuparamnames(self):
        return ["rf", "ri", "ro", "r"]

    def getbparamnames(self):
        return ["bf", "bi", "bo", "b"]


class TestGRUnobias(TestGRU):
    def makernu(self):
        return GRU(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "u", "w"]

    def getbparamnames(self):
        return []

    def test_has_zero_bias_params(self):
        for bias in super(self.__class__, self).getbparamnames():
            self.assertEqual(getattr(self.rnu, bias), 0)

    def test_output_var_has_no_bias_params(self):
        outpvarparamnames = [x.name for x in self.rnu.output.allparams]
        for bias in super(self.__class__, self).getbparamnames():
            self.assertNotIn(bias, outpvarparamnames)


class TestLSTMnobias(TestLSTM, TestGRUnobias):
    def makernu(self):
        return LSTM(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["wf", "rf", "wi", "ri", "wo", "ro", "w", "r", "pf", "pi", "po"]

    def getbparamnames(self):
        return []


class TestIFGRU(TestGRU):
    def makernu(self):
        return IFGRU(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w", "bm", "bhf", "bif", "b"]

    def getwparamnames(self):
        return ["wm", "whf", "w"]

    def getuparamnames(self):
        return ["uhf", "um", "u"]

    def getbparamnames(self):
        return ["bhf", "bm", "b"]

    def test_special_param_shapes(self):
        self.assertEqual(self.rnu.bif.shape, (self.dim,))
        self.assertEqual(self.rnu.wif.shape, (self.dim, self.dim))
        self.assertEqual(self.rnu.uif.shape, (self.innerdim, self.dim))


class TestIFGRUnobias(TestIFGRU, TestGRUnobias):
    def makernu(self):
        return IFGRU(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w"]

    def getbparamnames(self):
        return []

    def test_special_param_shapes(self):
        self.assertEqual(self.rnu.bif, 0)
        self.assertEqual(self.rnu.wif.shape, (self.dim, self.dim))
        self.assertEqual(self.rnu.uif.shape, (self.innerdim, self.dim))





