from teafacto.core.trainutil import Saveable, Trainable, Predictor, GradientDescent, Batched, AdaDelta, Embedder

__author__ = 'denis'
from theano import tensor as T
import numpy as np, math
from teafacto.core.utils import *
from teafacto.blocks.seq.rnn import GRU

# !!!!! NOT USED?

class RNNTFSGDC(Embedder, Trainable, Predictor, GradientDescent, Batched, Saveable):
    def __init__(self, rnnuc=GRU, dims=10, vocabsize=10, wreg=0.0, negrate=1, corruption="rhs", **kw):
        pass
        super(RNNTFSGDC, self).__init__(**kw)
        self.dims = dims
        self.vocabsize = vocabsize
        self.rnnuc = rnnuc
        self.wreg = wreg
        self.negrate = negrate
        self.corruption = corruption

        offset = 0.5
        self.W = theano.shared(np.random.random((self.vocabsize, self.dims)).astype("float32") - offset)
        self.rnnu = self.rnnuc(dim=self.dims, indim=self.dims, wreg=self.wreg)

    def train(self, X, evalinter=10): # X: z, x, y, v OR r, s, o
        batsize = self.getbatsize(X)
        outps, inps = self.defmodel()
        tErr = self.geterr(*outps)
        tReg = self.getreg(*inps)
        tCost = tErr + tReg
        trainf = self.gettrainf(inps, [tErr, tCost], tCost)
        err = self.trainloop(X=X,
                             trainf=self.getbatchloop(trainf, self.getsamplegen(X, batsize)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def geterr(self, x, y):
        return T.sum(y - x)

    def getreg(self, *inp): # l2 regularization
        return (1./2.) * (T.sum(self.W**2) * self.wreg) + self.rnnu.getreg()

    def defmodel(self):
        winp, rinp, hinp = T.ivectors("winp", "rinp", "hinp")
        nwinp, nrinp, nhinp = T.ivectors("nwinp", "nrinp", "nhinp")
        dotp = self.builddot(winp, rinp, hinp, self.rnnu)
        ndotp = self.builddot(nwinp, nrinp, nhinp, self.rnnu)
        dotp = dotp.reshape((dotp.shape[0], 1))
        ndotp = ndotp.reshape((ndotp.shape[0], 1))
        return [dotp, ndotp], [rinp, winp, hinp, nrinp, nwinp, nhinp]


    def builddot(self, winp, rinp, hinp, rnnu):
        hemb = self.W[hinp, :]
        om = self.prebuilddot(winp, rinp, rnnu)
        omdot = T.sum(om * hemb, axis=1)
        return T.nnet.sigmoid(omdot)

    def prebuilddot(self, winp, rinp, rnnu):
        wemb = self.W[winp, :] # (batsize, dims)
        remb = self.W[rinp, :] # (batsize, dims)
        iseq = T.stack(wemb, remb) # (2, batsize, dims)?
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = rnnu(iseq)
        om = oseq[:, np.int32(-1), :]
        return om

    @property
    def depparameters(self):
        return self.rnnu.parameters

    @property
    def ownparameters(self):
        return [self.W]

    def gettrainf(self, inps, outps, tCost):
        # get gradients
        params = self.parameters
        grads = map(lambda x: T.grad(tCost, x).astype("float32"), params)
        updates = map(lambda (p, g): (p, (p - self.lr * self.numbats * g).astype("float32")), zip(params, grads))
        trainf = theano.function(
            inputs=inps,
            outputs=outps,
            updates=updates,
            profile=self._profiletheano
        )
        return trainf

    def getpredictfunction(self):
        prf = self.getpreddotf()

        def pf(x, y, z):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [x, y, z]]
            return prf(*args)[0][0]

        return pf

    def getpredf(self):             # function to compute the predicted vector given entity and relation
        winp, rinp = T.ivectors("winpp", "rinpp")
        om = self.prebuilddot(winp, rinp, self.rnnu)
        return theano.function(inputs=[rinp, winp], outputs=[om])

    def getpreddotf(self):          # function to compute the score for a triple (array) given the indexes
        winp, rinp, hinp = T.ivectors("winppp", "rinppp", "hinppp")
        om = self.builddot(winp, rinp, hinp, self.rnnu)
        return theano.function(inputs=[rinp, winp,   hinp], outputs=[om])

    def getnormf(self):
        if self._normalize is True:
            norms = self.W.norm(2, axis=1).reshape((self.W.shape[0], 1))
            upd = (self.W, self.W/norms)
            return theano.function(inputs=[], outputs=[], updates=[upd])
        else:
            return None

    def getsamplegen(self, X, batsize):
        '''
        get sample generator
        :param X: indexes of nonzeroes of original input tensor. X is a ([int*]*)
        :param batsize: size of batch (number of samples generated)
        :return:
        '''
        negrate = self.negrate
        dims = X.shape
        print(dims)
        corruptrange = []
        corruptionmap = {
            "full": [0,1,2],
            "nlhs": [0,2],
            "rhs":  [2],
            "nmhs": [1,2],
            "nrhs": [0,1],
            "mhs":  [0],
            "lhs":  [1]
        }
        corruptrange = corruptionmap[self.corruption]
        xkeys = X.keys
        zvals = list(set(xkeys[:, 0]))
        print("corruptrange: ", corruptrange)

        def samplegen(): # vanilla samplegen
            corruptaxis = np.random.choice(corruptrange) # random axis to corrupt from corruptrange
            nonzeroidx = sorted(np.random.randint(0, len(X), (batsize,)).astype("int32"))
            possamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            negsamples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            corrupted = np.random.randint(0, dims[corruptaxis], (batsize,)).astype("int32")
            negsamples[corruptaxis] = corrupted
            return possamples + negsamples
        return samplegen


class RNNTFSM(Embedder, Saveable, Trainable, Predictor, Batched):
    # DONE: implement Adadelta: http://arxiv.org/pdf/1212.5701.pdf ?
    #       probably after-epoch normalization breaks Adadelta
    #       ==> remove norm, use wreg, exact norm doesn't matter much for prediction anyway
    #       TODO: check implementation
    # TODO: hierarchical softmax
    # TODO: think about XE
    # TODO: validation
    def __init__(self, rnnuc=GRU, **kw):
        super(RNNTFSM, self).__init__(**kw)
        self.rnnuc = rnnuc
        self.rnnu = self.rnnuc(dim=self.dims, indim=self.dims, wreg=self.wreg)
        scale = 0.1
        offset = 0.5
        self.smlayer = theano.shared((np.random.random((self.dims, self.vocabsize)).astype("float32")-offset)*scale, name="sm")

    def train(self, X, evalinter=10): # X: z, x, y, v OR r, s, o, v
        self.batsize = self.getbatsize(X)
        batsize = self.batsize
        probs, inps, gold = self.defmodel()
        tErr = self.geterr(probs, gold)
        tReg = self.getreg()
        tCost = tErr + tReg
        showgraph(tCost)
        #embed() # tErr.eval({inps[0]: [0], inps[1]:[10], gold: [1]})

        trainf = self.gettrainf(inps+[gold], [tErr, tCost], tCost)
        err = self.trainloop(X=X,
                             trainf=self.getbatchloop(trainf, self.getsamplegen(X, batsize)),
                             evalinter=evalinter,
                             normf=self.getnormf())
        return err

    def defmodel(self):
        sidx, ridx, oidx = T.ivectors("sidx", "ridx", "oidx")
        outp = self.builddot(sidx, ridx, self.rnnu) # (batsize, dims)
        Nclasses = int(math.ceil(math.sqrt(self.vocabsize)))
        Noutsperclass = int(math.ceil(math.sqrt(self.vocabsize)))
        ''' H-sm
        self.sm1w = theano.shared(np.random.random((self.dims, Nclasses)).astype("float32")*scale-offset)
        self.sm1b = theano.shared(np.random.random((Nclasses,)).astype("float32")*scale-offset)
        self.sm2w = theano.shared(np.random.random((Nclasses, self.dims, Noutsperclass)).astype("float32")*scale-offset)
        self.sm2b = theano.shared(np.random.random((Nclasses, Noutsperclass)).astype("float32")*scale-offset)'''
        ''' H-sm
        probs = h_softmax(outp, self.batsize, self.vocabsize, Nclasses, Noutsperclass, self.sm1w, self.sm1b, self.sm2w, self.sm2b, oidx)'''
        outdot = T.dot(outp, self.smlayer)
        probs = T.nnet.softmax(outdot)
        #showgraph(probs)
        return probs, [sidx, ridx], oidx # probs: (batsize, vocabsize)

    def builddot(self, sidx, ridx, rnnu):
        semb = self.embed(sidx) # (batsize, dims)
        remb = self.embed(ridx) # (batsize, dims)
        iseq = T.stack(semb, remb) # (2, batsize, dims)
        iseq = iseq.dimshuffle(1, 0, 2) # (batsize, 2, dims)
        oseq = rnnu(iseq)
        om = oseq[:, np.int32(-1), :] # om is (batsize, dims)
        #om = semb + remb
        #showgraph(om)
        return om

    def getsamplegen(self, X, batsize):
        xkeys = X.keys
        indices = range(len(X))
        def samplegen():
            nonzeroidx = sorted(np.random.choice(indices, size=(batsize,), replace=False).astype("int32"))
            samples = [xkeys[nonzeroidx][ax].astype("int32") for ax in range(X.numdims)]
            return [samples[1], samples[0], samples[2]]     # [[s*], [r*], [o*]]
        return samplegen

    def geterr(self, probs, gold): # probs: (nb_samples, vocabsize)
        return -T.sum(T.log(probs[:, gold])) / self.batsize

    def getreg(self):
        def regf(x):
            return T.sum(x**2)
        reg = (1./2.) * reduce(lambda x, y: x+y,
                               map(lambda x: regf(x)*self.wreg,
                                   self.ownparameters)
                               )
        return reg

    @property
    def ownparameters(self):
        return [self.W, self.smlayer]
        #return [self.W, self.sm1w, self.sm1b, self.sm2b, self.sm2w]

    @property
    def depparameters(self):
        return self.rnnu.parameters

    def getpredictfunction(self):
        probs, inps, gold = self.defmodel()
        score = probs[:, gold]
        scoref = theano.function(inputs=inps+[gold], outputs=score)
        def pref(s, r, o):
            args = [np.asarray(i).reshape((1,)).astype("int32") for i in [s, r, o]]
            return scoref(*args)
        return pref

class RNNTFADASM(AdaDelta, RNNTFSM):
    pass

class RNNTFSGDSM(GradientDescent, RNNTFSM):
    def __init__(self, **kw):
        super(RNNTFSGDSM, self).__init__(**kw)

def showgraph(var):
    theano.printing.pydotprint(var, outfile="/home/denis/logreg_pydotprint_prediction.png", var_with_name_simple=True)

if __name__ == "__main__":
    m = RNNTFSGDC.load("../models/RNNTFSGDSM.2015-12-10=22:39.auto")
    #embed()
    print m.predict(11329, 9325, 7156)
