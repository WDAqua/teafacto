import inspect
from math import ceil

import numpy as np
import theano
from teafacto.core.trainutil import SGDBase, Saveable, Profileable, Predictor, Normalizable
from theano import tensor as T

from teafacto.blocks.seq.rnn import RNUBase

__author__ = 'denis'

# Sequence Model - SoftMax objective
# used for knowledge modelling anyway
# all models here assume single index space
# i.c.t. KMSM, SMSM feedbacks on all elements from sequence


class SMSM(SGDBase, Saveable, Profileable, Predictor, Normalizable):
    #region SMBase
    def __init__(self, vocabsize=10, negrate=None, margin=None, **kw):
        super(SMSM, self).__init__(**kw)
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
                                   self.ownparams))

    @property
    def ownparams(self):
        return []

    @property
    def depparams(self):
        return []

    def getnormf(self):
        return None
    #endregion SMBase

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")  # integers of (batsize, seqlen)
        zidxs = T.imatrix("zidxs") # integers of (batsize, seqlen)
        occluder = T.imatrix("occluder")
        scores = self.definnermodel(pathidxs) #predictions, floats of (batsize, seqlen, vocabsize)
        #probs = T.nnet.softmax(scores) # row-wise softmax; probs: (batsize, seqlen, vocabsize) #softmax doesn't work on tensor3D
        probs, _ = theano.scan(fn=T.nnet.softmax,
                            sequences=scores,
                            outputs_info=[None])
        return probs, zidxs, occluder, [pathidxs, zidxs, occluder]

    def geterr(self, *args): # cross-entropy; probs: floats of (batsize, seqlen, vocabsize), gold: indexes of (batsize, seqlen)
        probs = args[0]
        golds = args[1]
        occluder = args[2]
        return -T.sum(
                    occluder *
                    T.log(
                        probs[T.arange(probs.shape[0])[:, None],
                              T.arange(probs.shape[1])[None, :],
                              golds])) / occluder.norm(1) # --> prob: floats of (batsize, seqlen) # is mean of logs of all matrix elements correct?

    def getsamplegen(self, trainX, labels): # trainX and labels must be of same dimensions
        batsize = self.batsize

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx, :].astype("int32")
            occluder = (labelsample > 0).astype("int32")
            return [trainXsample, labelsample, occluder]     # input seq, output seq
        return samplegen

    def getprobfunction(self): # occlusion is ignored
        probs, golds, occ, inps = self.defmodel()
        probs = probs[:, -1, :]
        scoref = theano.function(inputs=[inps[0]], outputs=probs)
        def probf(paths, occl=None):
            arg = np.asarray(paths).astype("int32")
            probsvals = scoref(arg)
            return probsvals
        return probf

    def getpredictfunction(self): # TOTEST
        probf = self.getprobfunction()
        def pref(path, o, occl=None):
            probvals = probf(path, occl)
            return probvals[np.arange(probvals.shape[0]), o]
        return pref

    def getsamplefunction(self):
        probf = self.getprobfunction()
        def samplef(path, occl=None):
            arg = np.asarray(path).astype("int32")
            probsvals = probf(arg, occl)
            ret = []
            for i in range(arg.shape[0]): #iterate over examples
                ret.append(np.random.choice(np.arange(probsvals.shape[1]), p=probsvals[i, :]))
            return ret
        return samplef

    def genseq(self, start, endsym):
        samplef = self.getsamplefunction()
        seq = [[]]
        current = start
        seq[0].append(current)
        while current != endsym:
            next = samplef(seq)[0]
            seq[0].append(next)
            current = next
        return seq


class ESMSM(SMSM, Normalizable): # identical to EKMSM since the same prediction part
    def __init__(self, dim=10, **kw):
        super(ESMSM, self).__init__(**kw)
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
        return super(ESMSM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparams(self):
        return [self.W]

    @property
    def depparams(self):
        return []

    def embed(self, *idxs):
        return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, pathidxs):
        pathembs, = self.embed(pathidxs) # pathembs: (batsize, seqlen, edim); zemb: (batsize, edim)
        return self.innermodel(pathembs)

    def innermodel(self, pathembs):
        raise NotImplementedError("use subclass")


class RNNESMSM(ESMSM):

    @classmethod
    def loadfrom(cls, src):
        self = cls()
        self.W = src.W
        self.Wout = src.Wout
        self.rnnu = src.rnnu
        self.vocabsize = src.vocabsize
        self.batsize = src.batsize
        return self

    def innermodel(self, pathembs): #pathemb: (batsize, seqlen, dim)
        oseq = self.rnnu(pathembs) # oseq is (batsize, seqlen, innerdims)  ---> last output
        scores = T.dot(oseq, self.Wout) # --> (batsize, seqlen, vocabsize)
        return scores

    @property
    def printname(self):
        return super(RNNESMSM, self).printname + "+" + self.rnnu.__class__.__name__+ ":" + str(self.rnnu.innerdim) + "D"

    @property
    def depparams(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(RNNESMSM, self).__add__(other)


    @property
    def ownparams(self):
        return super(RNNESMSM, self).ownparams + [self.Wout]

    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.vocabsize)).astype("float32")-offset)*scale, name="Wout")


# used for partially occluded prediction with models trained using RNNESMSM
# RNNESMSMShort supports random occlusion during training
class RNNESMSMShort(RNNESMSM):
    def __init__(self, occlusion=0.1, **kw): # randomly occlude portion of sequence elements
        self.occlusion = occlusion
        super(RNNESMSMShort, self).__init__(**kw)

    def defmodel(self):
        pathidxs = T.imatrix("pathidxs")  # integers of (batsize, seqlen)
        zidxs = T.imatrix("zidxs") # integers of (batsize, seqlen)
        occlusion = T.fmatrix("occlusion") # (batsize, seqlen)

        numstates = len(inspect.getargspec(self.rnnu.rec).args) - 2
        initstate = T.zeros((pathidxs.shape[0], self.rnnu.innerdim))
        initstate2 = T.zeros((pathidxs.shape[0], self.vocabsize))
        outputs, _ = theano.scan(fn=self.step, # --> iterate over seqlen
                                 sequences=[pathidxs.T, occlusion[:, :-1].T],
                                 outputs_info=[None]+[initstate2]+[initstate]*numstates)
        probs = outputs[0].dimshuffle(1, 0, 2)
        return probs, zidxs, occlusion, [pathidxs, zidxs, occlusion]

    def step(self, pathidx, occlusion, prevout, *rnustates): # pathidx, occlusion: (batsize,),   each state: (batsize, innerdim) #TODO: check dims
        pathemb, = self.embed(pathidx) # (batsize, dim)
        inp = (occlusion*pathemb.T).T + ((T.ones_like(occlusion)-occlusion)*T.dot(prevout, self.W).T).T
        rnnuouts = self.rnnu.rec(inp, *rnustates)
        rnuout = rnnuouts[0]
        rnnstates = rnnuouts[1:]
        newprevout = T.nnet.softmax(T.dot(rnuout, self.Wout)) # (batsize, vocabsize)
        probs = newprevout
        return [probs, newprevout] + rnnstates

    def geterr(self, probs, golds, occlusion): # cross-entropy; probs: floats of (batsize, seqlen, vocabsize), gold: indexes of (batsize, seqlen)
        r = occlusion[:, 1:] * T.log(probs[T.arange(probs.shape[0])[:, None],
                                           T.arange(probs.shape[1])[None, :],
                                           golds]) # --> result: floats of (batsize, seqlen)
        return -T.sum(r)/occlusion[:, 1:].norm(1)


    def getsamplegen(self, trainX, labels): # trainX and labels must be of same dimensions
        batsize = self.batsize
        occlprob = self.occlusion

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, trainX.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = trainX[nonzeroidx, :].astype("int32")
            labelsample = labels[nonzeroidx, :].astype("int32")
            occluder = (trainXsample > 0).astype("int32")
            occlusion = np.random.choice([0, 1], size=trainXsample.shape, p=[occlprob, 1-occlprob]).astype("float32")
            occlusion = occluder * occlusion
            lastoccluder = np.expand_dims((labelsample[:, -1] > 0).astype("float32"), 1)
            occlusion = np.append(occlusion, lastoccluder, axis=1).astype("float32")
            return [trainXsample, labelsample, occlusion]     # input seq, output seq
        return samplegen

    def getprobfunction(self):
        probs, golds, occ, inps = self.defmodel()
        probs = probs[:, -1, :] # last output --> (numsam, vocabsize)
        scoref = theano.function(inputs=[inps[0], inps[2]], outputs=probs)
        def probf(paths, occl=None):
            if occl is None:
                occl = np.ones_like(paths)
            paths = np.asarray(paths).astype("int32")
            occl = np.asarray(occl).astype("float32")
            assert paths.shape == occl.shape
            occl = np.append(occl, np.expand_dims(np.ones_like(occl[:, 0]), axis=1), axis=1).astype("float32")
            probsvals = scoref(paths, occl)
            return probsvals
        return probf


class AutoRNNESMSMShort(RNNESMSMShort): # also inherit from AutoRNNESMSM
    def step(self, pathidx, occlusion, prevout, *rnustates): # pathidx, occlusion: (batsize,),   each state: (batsize, innerdim) #TODO: check dims
        pathemb, = self.embed(pathidx) # (batsize, dim)
        inp = (occlusion*pathemb.T).T + ((1-occlusion)*T.dot(prevout, self.W).T).T
        rnnuouts = self.rnnu.rec(inp, *rnustates)
        rnuout = rnnuouts[0]
        rnnstates = rnnuouts[1:]
        newprevout = T.dot(rnuout, self.Wout) # (batsize, edim)
        probs = T.nnet.softmax(T.dot(newprevout, self.W.T)) # (batsize, vocabsize)
        return [probs, newprevout] + rnnstates