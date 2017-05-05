from math import ceil

import numpy as np
import theano
from teafacto.core.trainutil import SGDBase, Saveable, Profileable, Normalizable, Predictor, uniform
from theano import tensor as T

from teafacto.blocks.seq.rnn import RNUBase

__author__ = 'denis'

# Knowledge Model - Margin objective
# all models here assume single index space


class KMM(SGDBase, Predictor, Profileable, Saveable):
    def __init__(self, vocabsize=10, numrels=0, negrate=1, margin=1.0, **kw):
        super(KMM, self).__init__(**kw)
        self.vocabsize = vocabsize
        self.negrate = negrate
        self.margin = margin
        self.numrels = numrels

    @property
    def printname(self):
        return super(KMM, self).printname + "+n"+str(self.negrate)

    def defproblem(self):
        pdot, ndot, inps = self.defmodel()
        tErr = self.geterr(pdot, ndot)
        tReg = self.getreg()
        tCost = tErr + tReg
        return inps, tErr, tCost

    def ___train(self, trainX, labels, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = int(ceil(trainX.shape[0]*1./self.numbats))
        self.tbatsize = theano.shared(np.int32(self.batsize))
        pdot, ndot, inps = self.defmodel()
        tErr = self.geterr(pdot, ndot)
        tReg = self.getreg()
        #embed()
        tCost = tErr + tReg
        #showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        validf = self.getvalidf(inps, [tErr])
        err = self.trainloop(trainf=self.getbatchloop(trainf, self.getsamplegen(trainX, labels)),
                             evalinter=evalinter,
                             normf=self.getnormf(),
                             validf=validf)
        return err

    def defmodel(self):
        sidx = T.ivector("sidx")
        pathidxs = T.imatrix("pathidxs")
        zidx, nzidx = T.ivectors("zidx", "nzidx") # rhs corruption only
        dotp, ndotp = self.definnermodel(sidx, pathidxs, zidx, nzidx)
        return dotp, ndotp, [sidx, pathidxs, zidx, nzidx]

    def definnermodel(self, sidx, pathidxs, zidx, nzidx):
        raise NotImplementedError("use subclass")

    def getreg(self, regf=lambda x: T.sum(x**2), factor=1./2):
        return factor * reduce(lambda x, y: x + y,
                               map(lambda x: regf(x) * self.twreg,
                                   self.parameters))

    def geterr(self, pdot, ndot): # max margin
        comp = T.clip(self.margin - pdot + ndot, 0, np.infty)
        return T.sum(comp)

    @property
    def ownparameters(self):
            return []

    @property
    def depparameters(self):
        return []

    def getnormf(self):
        return None

    def getsamplegen(self, data, labels, onebatch=False):
        batsize = self.batsize if not onebatch else data.shape[0]
        negrate = self.negrate

        def samplegen():
            nonzeroidx = sorted(np.random.randint(0, data.shape[0], size=(batsize,)).astype("int32"))
            trainXsample = data[nonzeroidx, :].astype("int32")
            trainXsample = np.repeat(trainXsample, negrate, axis=0)
            labelsample = labels[nonzeroidx].astype("int32")
            labelsample = np.repeat(labelsample, negrate, axis=0)
            corruptedlabels = np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32")
            for i in range(negrate-1):
                corruptedlabels = np.append(corruptedlabels, np.random.randint(0, self.vocabsize, size=(batsize,)).astype("int32"), axis=0)
            return [trainXsample[:, 0], trainXsample[:, 1:], labelsample, corruptedlabels]     # start, path, target, bad_target
        return samplegen

    def getpredictfunction(self):
        pdot, _, inps = self.defmodel()
        scoref = theano.function(inputs=[inps[0], inps[1], inps[2]], outputs=pdot)
        def pref(s, path, o):
            args = [np.asarray(i).astype("int32") for i in [s, path, o]]
            return scoref(*args)
        return pref


class EKMM(KMM, Normalizable):
    def __init__(self, dim=10, **kw):
        super(EKMM, self).__init__(**kw)
        self.dim = dim
        self.initvars()

    def initvars(self):
        self.W = theano.shared(uniform((self.vocabsize, self.dim)), name="W")
        return [self.W]

    def getnormf(self):
        if self._normalize is True:
            upds = []
            for normparam in self.getnormparams():
                norms = normparam.norm(2, axis=1).reshape((normparam.shape[0], 1))
                upds.append((normparam, normparam/norms))
            return theano.function(inputs=[], outputs=[], updates=upds)
        else:
            return None

    def getnormparams(self):
        return [self.W]

    @property
    def printname(self):
        return super(EKMM, self).printname + "+E" + str(self.dim)+"D"

    @property
    def ownparameters(self):
        return [self.W]

    def embed(self, *idxs):
        if len(idxs) == 1:
            return self.W[idxs[0], :]
        else:
            return tuple(map(lambda x: self.W[x, :], idxs))

    def definnermodel(self, sidx, pathidxs, zidx, nzidx):#pathemb: (batsize, seqlen), *emb: (batsize)
        om, _ = theano.scan(fn=self.traverse,
                         sequences=pathidxs.T, # --> (seqlen, batsize)
                         outputs_info=[None] + self.start(sidx)
                         )
        om = om[0] # --> (seqlen, batsize, dim)
        om = om[-1, :, :] # --> (batsize, dim)
        dotp = self.membership(zidx, om, pathidxs)
        ndotp = self.membership(nzidx, om, pathidxs)
        return dotp, ndotp

    def start(self, sidx):
        return [self.embed(sidx)]

    def traverse(self, x_t, *states):
        raise NotImplementedError("use subclass")

    def membership(self, t, h_tm1, rels):
        raise NotImplementedError("use subclass")


class DistMemberEKMM(EKMM):
    def membership(self, t, h, rels):
        o = self.embed(t)
        return -T.sum(T.sqr(h - o), axis=1)


class DotMemberEKMM(EKMM):
    def membership(self, t, h, rels):
        o = self.embed(t)
        return T.batched_dot(o, h)


class CosMemberEKMM(EKMM):
    def membership(self, t, h, rels):
        o = self.embed(t)
        return T.batched_dot(o, h) / (o.norm(2, axis=1) * h.norm(2, axis=1))


class AddEKMM(DistMemberEKMM):                # TransE
    def traverse(self, x_t, *h_tm1): # x_t: (batsize, dim), h_tm1: (batsize, dim)
        h = h_tm1[0] + self.embed(x_t)
        return [h, h]

class AddAddEKMM(DistMemberEKMM): # TransD with m=n
    def __init__(self, innerdim=None, **kw):
        super(AddAddEKMM, self).__init__(**kw)
        self.innerdim = self.dim if innerdim is None else innerdim

    def initvars(self):
        super(AddAddEKMM, self).initvars()
        self.X = theano.shared(uniform((self.vocabsize, self.dim)), name="X")
        self.Wr = theano.shared(uniform((self.numrels, self.dim)), name="Wr")
        self.Xr = theano.shared(uniform((self.numrels, self.dim)), name="X")

    @property
    def ownparameters(self):
        return super(AddAddEKMM, self).ownparameters + [self.X, self.Wr, self.Xr]

    def getnormparams(self):
        return super(AddAddEKMM, self).getnormparams() + [self.X, self.Wr, self.Xr]

    def start(self, sidx):
        return [self.W[sidx, :], self.X[sidx, :]]

    def traverse(self, x_t, *states):
        x_t = x_t - self.vocabsize + self.numrels
        h_tm1, h_ptm1 = states
        r = self.Wr[x_t, :]
        rp = self.Xr[x_t, :]
        emod = T.batched_dot(h_tm1, h_ptm1)
        h_tm1 = T.concatenate([h_tm1, T.zeros((h_tm1.shape[0], r.shape[1] - h_tm1.shape[1]), dtype="float32")], axis=1) # pad with zeros
        hproj = (h_tm1.T + rp.T * emod).T
        h = hproj + r
        return [h, h, h]

    def membership(self, t, h, rels):
        temb = self.embed(t)
        lastrelidx = rels[:, -1] - self.vocabsize + self.numrels
        rp = self.Xr[lastrelidx, :]
        emod = T.batched_dot(temb, self.X[t, :])
        temb = T.concatenate([temb, T.zeros((temb.shape[0], rp.shape[1] - temb.shape[1]), dtype="float32")], axis=1) # pad with zeros
        tproj = (temb.T + rp.T * emod).T
        return -T.sum(T.sqr(h - tproj), axis=1)



class GateAddEKMM(DistMemberEKMM):
    def __init__(self, **kw):
        super(GateAddEKMM, self).__init__(**kw)
        self.R = theano.shared(uniform((self.numrels, self.dim)), name="R")

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        gate = self.R[x_t - self.vocabsize + self.numrels, :]
        gate = T.nnet.sigmoid(gate)
        h = h_tm1 * gate + self.embed(x_t)
        return [h, h]

    @property
    def ownparameters(self):
        return super(GateAddEKMM, self).ownparameters + [self.R]

class FracAddEKMM(GateAddEKMM):
    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        gate = self.R[x_t - self.vocabsize + self.numrels, :]
        gate = T.nnet.sigmoid(gate)
        h = h_tm1 * gate + self.embed(x_t) * (1-gate)
        return [h, h]


class EModAddEKMM(DistMemberEKMM): # better than TransE
    def __init__(self, **kw):
        super(EModAddEKMM, self).__init__(**kw)
        self.T = theano.shared(uniform((self.dim, self.dim)), name="T")

    @property
    def ownparameters(self):
        return super(EModAddEKMM, self).ownparameters + [self.T]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        gate = T.dot(h_tm1, self.T)
        gate = T.nnet.sigmoid(gate)
        add = self.embed(x_t) * gate
        h = h_tm1 + add
        return [h, h]

class EModRFracAddEKMM(DistMemberEKMM):
    def initvars(self):
        super(EModRFracAddEKMM, self).initvars()
        self.X = theano.shared(uniform((self.numrels, self.dim)), name="X")
        self.F = theano.shared(uniform((self.dim, self.dim)), name="F")

    @property
    def ownparameters(self):
        return super(EModRFracAddEKMM, self).ownparameters + [self.X, self.F]

    def getnormparams(self):
        return super(EModRFracAddEKMM, self).getnormparams() + [self.X]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xemb1 = self.W[x_t, :]
        xemb2 = self.X[x_t - self.vocabsize + self.numrels, :]
        gate = T.dot(h_tm1, self.F)
        gate = T.nnet.sigmoid(gate)
        add = xemb2 * (1-gate) + xemb1 * gate
        h = h_tm1 + add
        return [h, h]

class EModRFrac3AddEKMM(DistMemberEKMM):
    def initvars(self):
        super(EModRFrac3AddEKMM, self).initvars()
        self.X = theano.shared(uniform((self.numrels, self.dim)), name="X")
        self.Y = theano.shared(uniform((self.numrels, self.dim)), name="Y")
        self.F = theano.shared(uniform((self.dim, self.dim)), name="F")
        self.G = theano.shared(uniform((self.dim, self.dim)), name="G")

    @property
    def ownparameters(self):
        return super(EModRFrac3AddEKMM, self).ownparameters + [self.X, self.Y, self.F, self.G]

    def getnormparams(self):
        return super(EModRFrac3AddEKMM, self).getnormparams() + [self.X, self.Y]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xemb1 = self.W[x_t, :]
        xemb2 = self.X[x_t - self.vocabsize + self.numrels, :]
        xemb3 = self.Y[x_t - self.vocabsize + self.numrels, :]
        gate1 = T.dot(h_tm1, self.F)
        gate1 = T.nnet.sigmoid(gate1)
        gate2 = T.dot(h_tm1, self.G)
        gate2 = T.nnet.sigmoid(gate2)
        add = (xemb2 * (1-gate1) + xemb1 * gate1) * gate2 + (1-gate2) * xemb3
        h = h_tm1 + add
        return [h, h]


class ERModAddEKMM(DistMemberEKMM):
    def __init__(self, **kw):
        super(ERModAddEKMM, self).__init__(**kw)
        self.R = theano.shared(uniform((self.dim, self.dim)), name="R")
        self.T = theano.shared(uniform((self.dim, self.dim)), name="T")

    @property
    def ownparameters(self):
        return super(ERModAddEKMM, self).ownparameters + [self.R, self.T]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xembs = self.embed(x_t)
        gate = T.dot(xembs, self.R)
        gate += T.dot(h_tm1, self.T)
        gate = T.nnet.sigmoid(gate)
        add = xembs * gate
        h = h_tm1 + add
        return [h, h]

class RModEModAddEKMM(DistMemberEKMM): ############## -- ***** --
    def initvars(self):
        super(RModEModAddEKMM, self).initvars()
        self.R = theano.shared(uniform((self.numrels, self.dim, self.dim)), name="R")

    @property
    def ownparameters(self):
        return super(RModEModAddEKMM, self).ownparameters + [self.R]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xembs = self.embed(x_t)
        rmod = self.R[x_t-self.vocabsize+self.numrels, :, :]
        emod = T.batched_dot(h_tm1, rmod)
        gate = T.nnet.sigmoid(emod)
        ad = xembs * gate
        h = h_tm1 + ad
        return [h, h]

class ERModRFracAddEKMM(DistMemberEKMM):
    def initvars(self):
        super(ERModRFracAddEKMM, self).initvars()
        self.R = theano.shared(uniform((self.numrels, self.dim)))
        self.X = theano.shared(uniform((self.numrels, self.dim)))

    @property
    def ownparameters(self):
        return super(ERModRFracAddEKMM, self).ownparameters + [self.X, self.R]

    def getnormparams(self):
        return super(ERModRFracAddEKMM, self).getnormparams() + [self.X]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xembs1 = self.W[x_t, :]
        xembs2 = self.X[x_t-self.vocabsize+self.numrels, :]
        rmod = self.R[x_t-self.vocabsize+self.numrels, :]
        emod = h_tm1 + rmod
        gate = T.nnet.sigmoid(emod)
        add = xembs2 * (1-gate) + xembs1 * gate
        h = h_tm1 + add
        return [h, h]


class RModEModRFracAddEKMM(DistMemberEKMM):
    def initvars(self):
        super(RModEModRFracAddEKMM, self).initvars()
        self.R = theano.shared(uniform((self.numrels, self.dim, self.dim)))
        self.X = theano.shared(uniform((self.numrels, self.dim)))

    @property
    def ownparameters(self):
        return super(RModEModRFracAddEKMM, self).ownparameters + [self.X, self.R]

    def getnormparams(self):
        return super(RModEModRFracAddEKMM, self).getnormparams() + [self.X]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        xembs1 = self.W[x_t, :]
        xembs2 = self.X[x_t-self.vocabsize+self.numrels, :]
        rmod = self.R[x_t-self.vocabsize+self.numrels, :, :]
        emod = T.batched_dot(h_tm1, rmod)
        gate = T.nnet.sigmoid(emod)
        add = xembs2 * (1-gate) + xembs1 * gate
        h = h_tm1 + add
        return [h, h]





class VecMulEKMM(DotMemberEKMM):    # Bilinear Diag
    def traverse(self, x_t, *h_tm1): # x_t: (batsize, dim), h_tm1: (batsize, dim)
        h_tm1 = h_tm1[0]
        h = self.embed(x_t) * h_tm1
        return [h, h]


class VecMulEKMMDist(DistMemberEKMM, VecMulEKMM):
    pass


class MatMulEKMM(DotMemberEKMM):    # RESCAL
    def __init__(self, **kw):
        super(MatMulEKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
        self.R = theano.shared((np.random.random((self.numrels, self.dim, self.dim)).astype("float32")-offset)*scale, name="R")

    @property
    def ownparameters(self):
        return super(MatMulEKMM, self).ownparameters + [self.R]

    def traverse(self, x_t, *h_tm1): # x_t : (batsize, dim, dim), h_tm1 : (batsize, dim)
        h_tm1 = h_tm1[0]
        h = T.batched_dot(self.embedR(x_t-self.vocabsize+self.numrels), h_tm1)
        return [h, h]

    def embedR(self, idxs): # pathidxs: (batsize)
        return self.R[idxs, :] # return: (batsize, dim, dim)


class MatMulEKMMCos(CosMemberEKMM, MatMulEKMM):
    pass

class TransAddEKMM(DotMemberEKMM):
    def __init__(self, innerdim=10, **kw):
        super(TransAddEKMM, self).__init__(**kw)
        offset = 0.5
        scale = 1.
        self.innerdim = innerdim
        self.Rtrans = theano.shared((np.random.random((self.numrels, self.dim, self.innerdim)).astype("float32")-offset)*scale, name="Rtrans")
        self.Radd = theano.shared((np.random.random((self.numrels, self.innerdim)).astype("float32")-offset)*scale, name="Radd")
        self.Rtransinv = theano.shared((np.random.random((self.numrels, self.innerdim, self.dim)).astype("float32")-offset)*scale, name="Rtransinv")

    @property
    def ownparameters(self):
        return super(TransAddEKMM, self).ownparameters + [self.Rtrans, self.Radd, self.Rtransinv]

    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        x_t = x_t - self.vocabsize + self.numrels
        h = T.batched_dot(T.batched_dot(h_tm1, self.Rtrans[x_t, :]) + self.Radd[x_t, :], self.Rtransinv[x_t, :])
        return [h, h]

class RNNEKMM(DotMemberEKMM):

    def traverse(self, x_t, *h_tm1):         # x_t: (batsize, dim), h_tm1: (batsize, dim)
        h_tm1 = h_tm1[0]
        return self.rnnu.rec(self.embed(x_t), h_tm1)

    @property
    def printname(self):
        return super(RNNEKMM, self).printname + "+" + self.rnnu.__class__.__name__

    @property
    def depparameters(self):
        return self.rnnu.parameters

    def __add__(self, other):
        if isinstance(other, RNUBase):
            self.rnnu = other
            self.onrnnudefined()
            return self
        else:
            return super(EKMM, self).__add__(other)

    def onrnnudefined(self):
        pass

class ERNNEKMM(RNNEKMM):
    def traverse(self, x_t, *h_tm1):
        h_tm1 = h_tm1[0]
        return self.rnnu.rec(x_t - self.vocabsize + self.numrels, h_tm1)


class RNNEOKMM(RNNEKMM):    # is this still useful? TODO
    def onrnnudefined(self):
        self.initwout()

    def initwout(self):
        offset = 0.5
        scale = 0.1
        self.Wout = theano.shared((np.random.random((self.rnnu.innerdim, self.dim)).astype("float32")-offset)*scale, name="Wout")

    def membership(self, o, t):
        om = T.dot(o, self.Wout)
        return T.batched_dot(om, t)

    def membership_add(self, o, t):
        om = T.dot(o, self.Wout)
        return -T.sum(T.sqr(om - t), axis=1)

    @property
    def ownparameters(self):
        return super(RNNEOKMM, self).ownparameters + [self.Wout]

    @property
    def printname(self):
        return super(RNNEKMM, self).printname + "+" + self.rnnu.__class__.__name__ + ":" + str(self.rnnu.innerdim) + "D"