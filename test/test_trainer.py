from unittest import TestCase

import numpy as np

from teafacto.examples.dummy import *
from teafacto.core.trainer import ModelTrainer

from teafacto.core.base import Block
from teafacto.blocks.match import MatchScore, DotDistance, CosineDistance, EuclideanDistance
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.lang.wordvec import Glove

'''
    pred = ae.predict(pdata)
    print pred.shape
    print np.argmax(pred, axis=1)
    #print err, verr
'''

class TestTrainingPause(TestCase):
    def test_second_training_cycle(self):
        dim = 50
        epochs1 = 3
        epochs2 = 3
        self.vocabsize = 2000
        normalize = True
        ae = Dummy(indim=self.vocabsize, dim=dim, normalize=normalize)
        aefresh = Dummy.unfreeze(ae.freeze())
        frozen, self.err1 = self.trainfreeze(ae, epochs1)
        ae = Dummy.unfreeze(frozen)
        frozen, self.err2 = self.trainfreeze(ae, epochs2)
        frozen, self.err1p2 = self.trainfreeze(aefresh, epochs1+epochs2)
        self.assertLess(self.err2, self.err1)
        self.assertTrue(np.allclose(self.err1p2, self.err2))

    def trainfreeze(self, ae, epochs):
        numbats = 100
        lr = 0.2
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = ae.train([data], data).adadelta(lr=lr).cross_entropy().train(numbats=numbats, epochs=epochs, returnerrors=True)
        frozen = ae.freeze()
        err = self.err[-1]
        return frozen, err


class TestModelTrainer(TestCase):
    def setUp(self):
        dim=50
        self.epochs=4
        self.vocabsize=2000
        self.lrthresh = 2
        normalize=True
        self.ae = Dummy(indim=self.vocabsize, dim=dim, normalize=normalize)
        self.train()

    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).cross_entropy() \
                .autovalidate().cross_entropy().accuracy()\
            .train(numbats=numbats, epochs=self.epochs, returnerrors=True)

    def test_embeddings_normalized(self):
        pdata = range(self.vocabsize)
        pembs = self.ae.W.predict(pdata)
        norms = np.linalg.norm(pembs, axis=1)
        expectednorms = np.ones((self.vocabsize,))
        self.assertTrue(np.allclose(norms, expectednorms))

    def test_adaptive_learning_rate(self):
        differrs = self.err[:self.lrthresh]
        sameerrs = self.err[self.lrthresh:]
        for i in range(len(differrs)-1):
            for j in range(i+1, len(differrs)):
                self.assertFalse(np.allclose(differrs[i], differrs[j]))
        for i in range(len(sameerrs)):
            for j in range(len(sameerrs)):
                self.assertTrue(np.allclose(sameerrs[i], sameerrs[j]))


class TestModelTrainerNovalidate(TestModelTrainer):

    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).cross_entropy() \
            .train(numbats=numbats, epochs=self.epochs, returnerrors=True)

    def test_embeddings_normalized(self):
        pass


class TestModelTrainerValidsplit(TestModelTrainerNovalidate):
    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).cross_entropy() \
            .split_validate(5, random=True).cross_entropy() \
            .train(numbats=numbats, epochs=self.epochs, returnerrors=True)


class TestModelTrainerCrossValid(TestModelTrainerNovalidate):
    def train(self):
        numbats=100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).cross_entropy() \
            .cross_validate(5, random=True).cross_entropy() \
            .train(numbats=numbats, epochs=self.epochs, returnerrors=True)


class TestModelTrainerAutovalidate(TestModelTrainerNovalidate):
    def train(self):
        numbats = 100
        lr=0.02
        lr *= numbats
        data = np.arange(0, self.vocabsize).astype("int32")
        _, self.err, self.verr, _, _ = \
            self.ae.train([data], data).adadelta(lr=lr).dlr_thresh(thresh=self.lrthresh).cross_entropy() \
            .autovalidate(splits=5, random=True).cross_entropy() \
            .train(numbats=numbats, epochs=self.epochs, returnerrors=True)


class TestObjectives(TestCase):
    pass


class TestNSModelTrainer(TestCase):
    def test_ns_training(self):
        num = 2000
        self.expshape = (num, 50)
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        self.glove = Glove(self.expshape[1], self.expshape[0])
        self.cemb = VectorEmbed(indim=self.expshape[0]+1, dim=self.expshape[1])
        self.assertRaises(Exception, self.glove.block.predict, [num+1])
        self.assertRaises(Exception, self.cemb.predict, [num+1])

        m = MatchScore(self.glove.block, self.cemb, scorer=CosineDistance())
        mg = MatchScore(self.glove.block, self.glove.block)     # TODO factor out matchscore tests
        idxs = np.arange(num+1)

        # glove against glove
        self.assertTrue(np.allclose(mg.predict([num, 100], [num, 100]),
                                   [np.linalg.norm(self.glove % num)**2, np.linalg.norm(self.glove % 100)**2]))

        class NegIdxGen():
            def __init__(self, num): self.n = num
            def __call__(self, l, r): return l, np.random.randint(0, self.n, r.shape)

        m = m.nstrain([idxs, idxs]).negsamplegen(NegIdxGen(num+1)).negrate(5)\
            .adagrad(lr=0.1)\
            .train(numbats=50, epochs=50)

        print m.predict([num, num-1, num-2, num-1], [num, num-1, num-2, num-2])

        mrr = 0.0
        recat10 = 0.0
        recat1 = 0.0
        tot = num + 1
        for a in range(tot):
            abc = zip(range(num+1), list(m.predict([a]*(num+1), np.arange(0, num+1))))
            abc = sorted(abc, key=lambda (x, y): y, reverse=True)
            #print abc[:10]
            for i in range(len(abc)):
                if abc[i][0] == a:
                    #print i
                    mrr += 1./(1+i)
                    if i < 10:
                        recat10 += 1
                    if i < 1:
                        recat1 += 1
                    break

        mrr /= tot
        recat10 /= tot
        recat1 /= tot
        print "%.3f MRR,\t%.3f MR@10,\t%.3f MR@1" % (mrr, recat10, recat1)
        self.assertGreater(mrr, 0.85)
        self.assertGreater(recat10, 0.9)





