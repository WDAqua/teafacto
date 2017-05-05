from unittest import TestCase
import numpy as np
from teafacto.blocks.match import MatchScore, CosineDistance
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.lang.wordvec import Glove
from teafacto.util import ticktock


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

        vdata = np.arange(num)
        negrate = 5
        def obj(p, n):
            return n - p
        m, err, verr, _, _ = m.nstrain([idxs, idxs]).negsamplegen(NegIdxGen(num+1)).negrate(negrate)\
            .adagrad(lr=0.1).objective(obj) \
            .validate_on([vdata, vdata]).extvalid(geteval(m.predict, num, negrate)).validinter(30) \
            .train(numbats=50, epochs=29, returnerrors=True)
        #.writeresultstofile("testingresultswriter.tsv") \

        tdata = np.arange(num)
        tt = ticktock("eval")
        tt.tick()
        mrr, recat1, recat10 = geteval(m.predict, num, 1)(tdata)
        tt.tock("evaluated test data")
        print "%.4f MRR,\t%.4f MR@10,\t%.4f MR@1" % (mrr, recat10, recat1)
        self.assertGreater(mrr, 0.85)
        self.assertGreater(recat10, 0.9)
        print verr
        self.assertTrue(np.allclose(np.asarray([mrr, recat1, recat10]), np.asarray(verr[-1][1:])))


def geteval(predf, num, negrate):
    def inner(*inps):
        dx = inps[0]
        mrr = 0.0
        recat10 = 0.0
        recat1 = 0.0
        c = 0
        for a in list(dx[:dx.shape[0]/negrate]):
            abc = zip(range(num + 1), list(predf([a] * (num + 1), np.arange(0, num + 1))))
            abc = sorted(abc, key=lambda (x, y): y, reverse=True)
            # print abc[:10]
            for i in range(len(abc)):
                if abc[i][0] == a:
                    # print i
                    mrr += 1. / (1 + i)
                    if i < 10:
                        recat10 += 1
                    if i < 1:
                        recat1 += 1
                    break
            c += 1
        mrr /= c
        recat10 /= c
        recat1 /= c
        return mrr, recat1, recat10
    return inner
