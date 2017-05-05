from math import ceil

import numpy as np
import theano
from teafacto.core.trainutil import SGDBase, Saveable, Profileable, Predictor, Normalizable
from theano import tensor as T

from teafacto.blocks.seq.rnn import RNUBase

__author__ = 'denis'

# Knowledge Model - SoftMax objective # !!! DON'T USE ?!?8?!?
# ?? is this only modelling knowledge (no not for a E R E R E RWR in graph)?
# + KMSM only looks at the last result of the sequence

class KMSM(SGDBase, Saveable, Profileable, Predictor, Normalizable):
    #region SMBase
    def __init__(self, vocabsize=10, negrate=None, margin=None, **kw):
        super(KMSM, self).__init__(**kw)
        self.vocabsize = vocabsize

    def train(self, trainX, labels, evalinter=10):
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        model = self.defmodel() # the last element is list of inputs, all others go to geterr()
        tErr = self.geterr(*model[:-1])
        tReg = self.getreg()
        #embed()
        tCost = tErr + tReg
        #showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(model[-1], [tErr, tCost], tCost)
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def definnermodel(self, pathidxs):
        raise NotImplementedError("use subclass")

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.parameters))

    @property
    def ownparameters(self):
        return []

    @property
    def depparameters(self):
        return []

    def getnormf(self):
        return None
    #endregion SMBase


    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")
        zidx = T.ivector("zidx") # (batsize)
        scores = self.definnermodel(pathidxs) # ? scores: float(batsize, vocabsize)
        probs = T.nnet.softmax(scores) # row-wise softmax, ? probs: float(batsize, vocabsize)
        return probs, zidx, [pathidxs, zidx]

    def geterr(self, probs, gold): # cross-entropy
        return -T.mean(T.log(probs[T.arange(self.batsize), gold]))

    @property
    def ownparameters(self):
        return []

    @property
    def depparameters(self):
        return []

    def getsamplegen(self, trainX, labels):
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx].astype("int32")
            return [trainXsample, labelsample]     # start + path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        probs, gold, inps = self.defmodel()
        score = probs[T.arange(gold.shape[0]), gold]
        scoref = theano.function(inputs=[inps[0], inps[1]], outputs=score)
        def pref(path, o):
            args = [np.asarray(i).astype("int32") for i in [path, o]]
            return scoref(*args)
        return pref


class EKMSM(KMSM, Normalizable):
    def __init__(self, dim=10, **kw):
        super(EKMSM, self).__init__(**kw)
        offset=0.5
        scale=1.
        self.dim = dim
        self.W = theano.shared((np.random.random((self.vocabsize, self.dim)).astype("float32")-offset)*scale, name="W")

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    @property
    def printname(self):
        return super(EKMSM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparameters(self):
        return [self.W]

    @property
    def depparameters(self):
        return []

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, pathidxs):
        pathembs, = self.embed(pathidxs) # pathembs: (batsize, seqlen, edim); zemb: (batsize, edim)
        return self.innermodel(pathembs)

    def innermodel(self, pathembs):
        raise NotImplementedError("use subclass")


class RNNEKMSM(EKMSM):

    def innermodel(self, pathembs): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs)
        om = oseq[:, -1, :] # om is (batsize, innerdims)  ---> last output
        scores = T.dot(om, self.Wout) # --> (batsize, vocabsize)
        return scores

    @property
    def printname(self):
        return super(RNNEKMSM, self).printname + "+" + self.rnnu.__class__.__name__+ ":" + str(self.rnnu.innerdim) + "D"

    @property
    def depparameters(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(EKMSM, self).__add__(other)


    @property
    def ownparameters(self):
        return super(RNNEKMSM, self).ownparameters + [self.Wout]

    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.vocabsize)).astype("float32")-offset)*scale, name="Wout")


class AutoRNNEKMSM(RNNEKMSM):
    def innermodel(self, pathembs): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs)
        om = oseq[:, -1, :] # om is (batsize, innerdims)  ---> last output
        scores = T.dot(om, self.Wout) # --> (batsize, dim)
        scores = T.dot(scores, self.W.T) # --> (batsize, vocabsize)
        return scores

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.dim)).astype("float32")-offset)*scale, name="Wout")