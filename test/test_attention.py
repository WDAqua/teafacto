from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, LinearSumAttentionGenerator, LinearGateAttentionGenerator, DotprodAttGen, GenDotProdAttGen, ForwardAttGen
from teafacto.blocks.seq.rnn import SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot


class TestGenDotProdAttGen(TestCase):
    def test_shapes(self):
        batsize, seqlen, datadim, critdim = 100, 7, 50, 40
        crit = np.random.random((batsize, critdim))
        data = np.random.random((batsize, seqlen, datadim))
        m = GenDotProdAttGen(indim=datadim, memdim=critdim)
        pred = m.predict(crit, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))


class TestForwardAttGen(TestCase):
    def test_shapes(self):
        batsize, seqlen, datadim, critdim, attdim = 100, 7, 50, 40, 60
        crit = np.random.random((batsize, critdim))
        data = np.random.random((batsize, seqlen, datadim))
        m = ForwardAttGen(indim=datadim, memdim=critdim, attdim=attdim)
        pred = m.predict(crit, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))


class AttentionGenTest(TestCase):
    def test_shapes(self):
        batsize, seqlen = 100, 7
        criterionshape = (batsize, 10)
        datashape = (batsize, seqlen, 10)
        attgen = DotprodAttGen()
        # generate data
        criterion = np.random.random(criterionshape)
        data = np.random.random(datashape)
        # predict and test
        pred = attgen.predict(criterion, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(pred.sum(axis=1), np.ones((pred.shape[0],))))

    def test_mask(self):
        batsize, seqlen = 100, 7
        criterionshape = (batsize, 10)
        datashape = (batsize, seqlen, 15)
        attgen = LinearGateAttentionGenerator(indim=25, attdim=12)
        # generate data
        criterion = np.random.random(criterionshape)
        data = np.random.random(datashape)
        mask = np.ones((batsize, seqlen))
        maskids = np.random.randint(2, seqlen+1, (batsize,))
        for i in range(maskids.shape[0]):
            mask[i, maskids[i]:] = 0
        # predict and test
        pred = attgen.predict(criterion, data, mask)
        maskthrough = np.not_equal(pred, 0)
        self.assertTrue(np.all(maskthrough == mask))



class DummyAttentionGeneratorConsumerTest(TestCase):
    def setUp(self):
        criteriondim = 30
        datadim = 20
        innerdim = 25
        batsize = 33
        seqlen = 11
        self.attgenshape = (batsize, seqlen)
        self.attconshape = (batsize, datadim)
        self.attgenc = self.getattgenc()
        self.attgen = self.attgenc(indim=criteriondim + datadim, innerdim=innerdim)
        self.attgenparams = self.getattgenparams()
        self.attcon = WeightedSumAttCon()
        self.att = Attention(self.attgen, self.attcon)
        self.criterion_val = np.random.random((batsize, criteriondim)).astype("float32")
        self.data_val = np.random.random((batsize, seqlen, datadim)).astype("float32")

    def getattgenc(self):
        return LinearSumAttentionGenerator

    def getattgenparams(self):
        return {self.attgen.W}

    def test_generator_shape(self):
        pred = self.attgen.predict(self.criterion_val, self.data_val)
        self.assertEqual(pred.shape, self.attgenshape)

    def test_generator_param_prop(self):
        self.attgen.predict(self.criterion_val, self.data_val)
        allparams = self.attgen.output.allparams
        self.assertSetEqual(allparams, self.attgenparams)

    def test_consumer_shape(self):
        pred = self.att.predict(self.criterion_val, self.data_val)
        self.assertEqual(pred.shape, self.attconshape)

    def test_consumer_param_prop(self):
        self.att.predict(self.criterion_val, self.data_val)
        allparams = self.att.output.allparams
        self.assertSetEqual(allparams, self.attgenparams)


class LinearAggAttentionGenTest(DummyAttentionGeneratorConsumerTest):
    def getattgenc(self):
        return LinearGateAttentionGenerator

    def getattgenparams(self):
        return {self.attgen.W, self.attgen.U}


class TestAttentionRNNDecoder(TestCase):
    def setUp(self):
        vocsize = 10
        innerdim = 50
        encdim = 30
        seqlen = 5
        batsize = 77
        self.att = Attention(LinearSumAttentionGenerator(indim=innerdim + encdim), WeightedSumAttCon())
        self.decwatt = SeqDecoder(
            [IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
            inconcat=True,
            attention=self.att,
            innerdim=innerdim
        )
        self.decwoatt = SeqDecoder(
            [IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
            inconcat=True,
            innerdim=innerdim
        )
        self.attdata = np.random.random((batsize, seqlen, encdim)).astype("float32")
        self.data = np.random.random((batsize, encdim)).astype("float32")
        self.seqdata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.predshape = (batsize, seqlen, vocsize)

    def test_shape(self):
        pred = self.decwatt.predict(self.attdata, self.seqdata)
        self.assertEqual(pred.shape, self.predshape)

    def test_shape_wo_att(self):
        pred = self.decwoatt.predict(self.data, self.seqdata)
        self.assertEqual(pred.shape, self.predshape)

    def test_attentiongenerator_param_in_allparams(self):
        self.decwatt.predict(self.attdata, self.seqdata)
        allparams = self.decwatt.output.allparams
        self.assertIn(self.att.attentiongenerator.W, allparams)

    def test_attentiongenerator_param_not_in_params_of_dec_wo_att(self):
        self.decwoatt.predict(self.data, self.seqdata)
        allparams = self.decwoatt.output.allparams
        self.assertNotIn(self.att.attentiongenerator.W, allparams)
