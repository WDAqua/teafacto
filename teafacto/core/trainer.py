import sys, gc
from pympler import asizeof
from datetime import datetime as dt
from IPython import embed

import numpy as np
import theano
from lasagne.objectives import *
from lasagne.regularization import l1, l2
from lasagne.updates import *
from theano import tensor as tensor
from theano.compile.nanguardmode import NanGuardMode

#from core import Input
from teafacto.core.datafeed import DataFeeder, SplitIdxIterator
from teafacto.util import ticktock as TT, issequence


class DynamicLearningParam(object):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, lr, epoch, maxiter, terrs, verrs): # get new learning rate based on old one, epoch, maxiter, training error, validation errors
        raise NotImplementedError("use subclass")


class thresh_lr(DynamicLearningParam):
    def __init__(self, lr, thresh=5):
        super(thresh_lr, self).__init__(lr)
        self.thresh = thresh

    def __call__(self, lr, epoch, maxiter, terrs, verrs):
        return lr if epoch < self.thresh else 0.


class ModelTrainer(object):
    def __init__(self, model, gold):
        self.model = model
        self.goldvar = gold
        self.validsetmode= False
        self.average_err = True # TODO: do we still need this?
        self._autosave = False
        self._autosavepath = None
        self._autosaveblock = None
        # training settings
        self.learning_rate = None
        self.dynamic_lr = None
        self.objective = None
        self.regularizer = None
        self.optimizer = None
        self.traindata = None
        self.traingold = None
        self.gradconstraints = []
        # validation settings
        self._validinter = 1
        self.trainstrategy = self._train_full
        self.validsplits = 0
        self.validrandom = False
        self.validata = None
        self.validgold = None
        self.validation = None
        self.validators = []
        self.external_validators = []
        self.tt = TT("FluentTrainer")
        # taking best
        self.besttaker = None
        self.bestmodel = None
        self.savebest = None
        self.smallerbetter = True
        # writing
        self._writeresultspath = None


    #region ====================== settings =============================

    #region ################### LOSSES ##########################

    def _set_objective(self, obj):
        if self.validsetmode is False:
            self.objective = obj
        else:
            self.validators.append(obj)

    def linear_objective(self): # multiplies prediction with gold, assumes prediction is already the loss
                                # (this is for negative sampling models where the training model already computes the loss)
        self._set_objective(lambda x, y: x * y)
        return self

    def cross_entropy(self):
        """ own implementation of categorical cross-entropy """
        self._set_objective(self._inner_cross_entropy)
        return self

    @classmethod
    def _inner_cross_entropy(cls, probs, gold):
        if gold.ndim == 1:
            return tensor.nnet.categorical_crossentropy(probs, gold) #-tensor.log(probs[tensor.arange(gold.shape[0]), gold])
        elif gold.ndim == 2:    # sequences
            return cls._inner_seq_neg_log_prob(probs, gold)

    def seq_cross_entropy(self): # probs (batsize, seqlen, vocsize) + gold: (batsize, seqlen) ==> sum of neg log-probs of correct seq
        """ Own implementation of categorical cross-entropy, applied to a sequence of probabilities that should be multiplied """
        self._set_objective(self._inner_seq_neg_log_prob)
        return self

    @classmethod
    def _inner_seq_neg_log_prob(cls, probs, gold):   # probs: (batsize, seqlen, vocsize) probs, gold: (batsize, seqlen) idxs
        #print "using inner seq neg log prob"
        def _f(probsmat, goldvec):      # probsmat: (seqlen, vocsize), goldvec: (seqlen,)
            ce = tensor.nnet.categorical_crossentropy(probsmat, goldvec) #-tensor.log(probsmat[tensor.arange(probsmat.shape[0]), goldvec])
            return tensor.sum(ce)
        o, _ = theano.scan(fn=_f, sequences=[probs, gold], outputs_info=None)      # out: (batsize,)
        return o

    def squared_error(self):
        self._set_objective(squared_error)
        return self

    def squared_loss(self):
        self._set_objective(lambda x, y: (1 - x * y) ** 2)        # [-1, +1](batsize, )
        return self

    def binary_cross_entropy(self): # theano binary cross entropy (through lasagne), probs: (batsize,) float, gold: (batsize,) float
        self._set_objective(binary_crossentropy)
        return self

    def bin_accuracy(self, sep=0):
        self._set_objective(lambda x, y: theano.tensor.eq(x > sep, y > sep))
        return self

    def accuracy(self, top_k=1):
        def categorical_accuracy(predictions, targets, top_k=1): # !!! copied from Lasagne # TODO: import properly
            if targets.ndim == predictions.ndim:
                targets = theano.tensor.argmax(targets, axis=-1)
            elif targets.ndim != predictions.ndim - 1:
                raise TypeError('rank mismatch between targets and predictions')

            if top_k == 1:
                # standard categorical accuracy
                top = theano.tensor.argmax(predictions, axis=-1)
                return theano.tensor.eq(top, targets)
            else:
                # top-k accuracy
                top = theano.tensor.argsort(predictions, axis=-1)
                # (Theano cannot index with [..., -top_k:], we need to simulate that)
                top = top[[slice(None) for _ in range(top.ndim - 1)] +
                          [slice(-top_k, None)]]
                targets = theano.tensor.shape_padaxis(targets, axis=-1)
                return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)
        self._set_objective(lambda x, y: 1-categorical_accuracy(x, y, top_k=top_k))
        return self

    def seq_accuracy(self): # sequences must be exactly the same
        def inner(probs, gold):
            if gold.ndim == probs.ndim:
                gold = tensor.argmax(gold, axis=-1)
            elif gold.ndim != probs.ndim - 1:
                raise TypeError('rank mismatch between targets and predictions')
            top = tensor.argmax(probs, axis=-1)
            assert(gold.ndim == 2 and top.ndim == 2)
            diff = tensor.sum(abs(top - gold), axis=1)
            return tensor.eq(diff, tensor.zeros_like(diff))
        self._set_objective(lambda x, y: 1-inner(x, y))
        return self


    def hinge_loss(self, margin=1., labelbin=True): # gold must be -1 or 1 if labelbin if False, otherwise 0 or 1
        def inner(preds, gold):     # preds: (batsize,), gold: (batsize,)
            if labelbin is True:
                gold = 2 * gold - 1
            return tensor.nnet.relu(margin - gold * preds)
        self._set_objective(inner)
        return self

    def multiclass_hinge_loss(self, margin=1.):
        def inner(preds, gold):     # preds: (batsize, numclasses) scores, gold: int:(batsize)
            pass
        self._set_objective(inner)
        return self

    def log_loss(self):
        """ NOT cross-entropy, BUT log(1+e^(-t*y))"""
        def inner(preds, gold):     # preds: (batsize,) float, gold: (batsize,) float
            return tensor.nnet.softplus(-gold*preds)
        self._set_objective(inner)
        return self
    #endregion

    #region ################### GRADIENT CONSTRAINTS ############ --> applied in the order that they were added
    def grad_total_norm(self, max_norm, epsilon=1e-7):
        self.gradconstraints.append(lambda allgrads: total_norm_constraint(allgrads, max_norm, epsilon=epsilon))
        return self

    def grad_add_constraintf(self, f):
        self.gradconstraints.append(f)
        return self

    def _gradconstrain(self, allgrads):
        ret = allgrads
        for gcf in self.gradconstraints:
            ret = gcf(ret)
        return ret

    # !!! can add more
    #endregion

    #region #################### REGULARIZERS ####################
    def _regul(self, regf, amount, params):
        return amount * reduce(lambda x, y: x+y, [regf(x.d)*x.regmul for x in params], 0)

    def l2(self, amount):
        self.regularizer = lambda x: self._regul(l2, amount, x)
        return self

    def l1(self, amount):
        self.regularizer = lambda x: self._regul(l1, amount, x)
        return self
    #endregion

    #region ###################  LEARNING RATE ###################
    def _setlr(self, lr):
        if isinstance(lr, DynamicLearningParam):
            self.dynamic_lr = lr
            lr = lr.lr
        self.learning_rate = theano.shared(np.cast[theano.config.floatX](lr))

    def _update_lr(self, epoch, maxepoch, terrs, verrs):
        if self.dynamic_lr is not None:
            self.learning_rate.set_value(np.cast[theano.config.floatX](self.dynamic_lr(self.learning_rate.get_value(), epoch, maxepoch, terrs, verrs)))

    def dlr_thresh(self, thresh=5):
        self.dynamic_lr = thresh_lr(self.learning_rate, thresh=thresh)
        return self
    #endregion

    #region #################### OPTIMIZERS ######################
    def sgd(self, lr):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: sgd(x, y, learning_rate=l)
        return self

    def momentum(self, lr, mome=0.9):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: momentum(x, y, learning_rate=l, momentum=mome)
        return self

    def nesterov_momentum(self, lr, momentum=0.9):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: nesterov_momentum(x, y, learning_rate=l, momentum=momentum)
        return self

    def adagrad(self, lr=1.0, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adagrad(x, y, learning_rate=l, epsilon=epsilon)
        return self

    def rmsprop(self, lr=1., rho=0.9, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: rmsprop(x, y, learning_rate=l, rho=rho, epsilon=epsilon)
        return self

    def adadelta(self, lr=1., rho=0.95, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adadelta(x, y, learning_rate=l, rho=rho, epsilon=epsilon)
        return self

    def adam(self, lr=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adam(x, y, learning_rate=l, beta1=b1, beta2=b2, epsilon=epsilon)
        return self
    #endregion

    #region ################### VALIDATION ####################### --> use one of following

    def validinter(self, validinter=1):
        self._validinter = validinter
        return self

    def autovalidate(self, splits=5, random=True): # validates on the same data as training data
        self.validate_on(self.traindata, self.traingold, splits=splits, random=random)
        self.validsetmode = True
        return self

    def split_validate(self, splits=5, random=True):
        self.trainstrategy = self._train_split
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def validate_on(self, data, gold=None, splits=1, random=True):
        self.trainstrategy = self._train_validdata
        self.validdata = data
        self.validgold = gold
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def cross_validate(self, splits=5, random=False):
        self.trainstrategy = self._train_cross_valid
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def extvalid(self, evaluator):  # adds external, non-symbolic validator
        self.external_validators.append(evaluator)
        return self

    #endregion

    #region ######################### SELECTING THE BEST ######################
    def takebest(self, f=None, save=False, smallerbetter=True):
        if f is None:
            f = lambda x: x[1]   # pick the model with the best first validation score
        self.besttaker = f
        self.bestmodel = (None, float("inf"))
        self.savebest = save
        self.smallerbetter = smallerbetter
        return self
    #endregion
    #endregion

    #region ====================== execution ============================

    #region ######################### ACTUAL TRAINING #########################
    def traincheck(self):
        assert(self.optimizer is not None)
        assert(self.objective is not None)
        assert(self.traindata is not None)
        assert(self.traingold is not None)

    def train(self, numbats, epochs, returnerrors=False, _skiptrain=False):
        self.traincheck()
        self.numbats = numbats
        self.maxiter = epochs
        errors = self.trainstrategy(_skiptrain=_skiptrain)       # trains according to chosen training strategy, returns errors
        if self.besttaker is not None and self.savebest is None:      # unfreezes best model if best choosing was chosen
            self.model = self.model.__class__.unfreeze(self.bestmodel[0])
            self.tt.tock("unfroze best model (%.3f) - " % self.bestmodel[1]).tick()
        ret = self.model
        if returnerrors:
            ret = (ret,) + errors
        return ret

    def autobuild_model(self, model, *data, **kw):
        return model.autobuild(*data, **kw)

    def buildtrainfun(self, model):
        self.tt.tick("compiling training function")
        with model.trainmode(True):
            inps, out = self.autobuild_model(model, *self.traindata, _trainmode=True)
            if issequence(out):
                out = out[0]
            params = out.allparams #model.output.allparams
            inputs = inps #model.inputs
            scanupdates = out.allupdates #model.output.allupdates
            loss, newinp = self.buildlosses(out, [self.objective])
            loss = loss[0]
            if newinp is not None:
                inputs = newinp
            if self.regularizer is not None:
                reg = self.regularizer(params)
                cost = loss+reg
            else:
                cost = loss
            # theano.printing.debugprint(cost)
            # theano.printing.pydotprint(cost, outfile="pics/debug.png")
            updates = []
            print "params:\n " + "".join(map(lambda x: "\t%s\n" % str(x), params)) + "\n\t\t (in Block, base.py)\n"
            self.tt.msg("computing gradients")
            #grads = []
            #for x in params:
            #    self.tt.msg("computing gradient for %s" % str(x))
            #    grads.append(tensor.grad(cost, x.d))
            grads = tensor.grad(cost, [x.d for x in params])  # compute gradient
            self.tt.msg("computed gradients")
            grads = self._gradconstrain(grads)
            for param, grad in zip(params, grads):
                upds = self.optimizer([grad], [param.d], self.learning_rate*param.lrmul)
                for upd in upds:
                    broken = False
                    for para in params:
                        if para.d == upd:
                            updates.append((upd, para.constraintf()(upds[upd])))
                            broken = True
                            break
                    if not broken:
                        updates.append((upd, upds[upd]))
            #print updates
            #embed()
            allupdates = updates + scanupdates.items()
            #embed()
            trainf = theano.function(
                inputs=[x.d for x in inputs]+[self.goldvar],
                outputs=[cost],
                updates=allupdates,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
            )
            # TODO: add givens for transferring dataset to GPU --> must reimplement parts of trainer (batch generation, givens, ...)
            self.tt.tock("training function compiled")
        return trainf

    def buildlosses(self, out, objs):
        return [aggregate(obj(out.d, self.goldvar), mode='mean' if self.average_err is True else 'sum') for obj in objs], None

    def getvalidfun(self, model):
        symbolic_validfun = self.buildvalidfun(model)
        if len(self.external_validators) == 0:
            return symbolic_validfun
        else:
            extravalid = self.external_validators
            def validfun(*sampleinps):
                ret = []
                if symbolic_validfun is not None:
                    for x in symbolic_validfun(*sampleinps):
                        ret.append(x)
                for ev in extravalid:
                    a = ev(*sampleinps)
                    if not issequence(a):
                        a = [a]
                    else:
                        if isinstance(a, tuple):
                            a = list(a)
                    ret += a
                return ret
            return validfun

    def buildvalidfun(self, model):
        self.tt.tick("compiling validation function")
        inps, out = self.autobuild_model(model, *self.traindata, _trainmode=False)
        if issequence(out):
            out = out[0]
        metrics, newinp = self.buildlosses(out, self.validators)
        inputs = newinp if newinp is not None else inps
        ret = None
        if len(metrics) > 0:
            ret = theano.function(inputs=[x.d for x in inputs] + [self.goldvar],
                                  outputs=metrics,
                                  mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
                                  )
        else:
            self.tt.msg("NO VALIDATION METRICS DEFINED, RETURNS NONE")
        self.tt.tock("validation function compiled")
        return ret
    #endregion

    #region ################## TRAINING STRATEGIES ############
    def _train_full(self, _skiptrain=False): # train on all data, no validation
        trainf = self.buildtrainfun(self.model)
        err, _ = self.trainloop(
                trainf=self.getbatchloop(trainf,
                DataFeeder(*(self.traindata + [self.traingold])).numbats(self.numbats)),
                _skiptrain=_skiptrain)
        return err, None, None, None

    def _train_validdata(self, _skiptrain=False):
        validf = self.getvalidfun(self.model)
        trainf = self.buildtrainfun(self.model)
        df = DataFeeder(*(self.traindata + [self.traingold])).numbats(self.numbats)
        vdf = DataFeeder(*(self.validdata + [self.validgold]), random=False)
        vdf.batsize = df.batsize
        #embed()
        #dfvalid = df.osplit(split=self.validsplits, random=self.validrandom)
        err, verr = self.trainloop(
                trainf=self.getbatchloop(trainf, df),
                validf=self.getbatchloop(validf, vdf),
                _skiptrain=_skiptrain)
        return err, verr, None, None

    def _train_split(self, _skiptrain=False):
        trainf = self.buildtrainfun(self.model)
        validf = self.getvalidfun(self.model)
        df = DataFeeder(*(self.traindata + [self.traingold]))
        dftrain, dfvalid = df.split(self.validsplits, self.validrandom, df_randoms=(True, False))
        dftrain.numbats(self.numbats)
        dfvalid.batsize = dftrain.batsize
        err, verr = self.trainloop(
                trainf=self.getbatchloop(trainf, dftrain),
                validf=self.getbatchloop(validf, dfvalid),
                _skiptrain=_skiptrain)
        return err, verr, None, None

    def _train_cross_valid(self, _skiptrain=False):
        df = DataFeeder(*(self.traindata + [self.traingold]))
        splitter = SplitIdxIterator(df.size, split=self.validsplits, random=self.validrandom, folds=self.validsplits)
        err = []
        verr = []
        c = 0
        for splitidxs in splitter:
            trainf = self.buildtrainfun(self.model)
            validf = self.getvalidfun(self.model)
            tf, vf = df.isplit(splitidxs, df_randoms=(True, False))
            tf.numbats(self.numbats)
            vf.batsize = tf.batsize
            serr, sverr = self.trainloop(
                trainf=self.getbatchloop(trainf, tf),
                validf=self.getbatchloop(validf, vf),
                _skiptrain=_skiptrain)
            err.append(serr)
            verr.append(sverr)
            self.resetmodel(self.model)
        err = np.asarray(err)
        avgerr = np.mean(err, axis=0)
        verr = np.asarray(verr)
        avgverr = np.mean(verr, axis=0)
        self.tt.tock("done")
        return avgerr, avgverr, err, verr
    #endregion

    @staticmethod
    def resetmodel(model):
        params = model.output.allparams
        for param in params:
            param.reset()

    #region ############# TRAINING LOOPS ##################
    def trainloop(self, trainf, validf=None, _skiptrain=False):
        self.tt.tick("training")
        err = []
        verr = []
        stop = self.maxiter == 0
        self.currentiter = 1
        evalinter = self._validinter
        evalcount = evalinter
        tt = TT("iter")
        prevverre = [float("inf")] * len(self.validators)

        writeresf = None
        if self._writeresultspath is not None:
            writeresf = open(self._writeresultspath, "w", 1)

        while not stop:     # loop over epochs
            tt.tick("%d/%d" % (self.currentiter, int(self.maxiter)))
            if _skiptrain:
                tt.msg("skipping training")
                erre = [0.]
            else:
                erre = trainf()
            if self.currentiter == self.maxiter:
                stop = True
            self.currentiter += 1
            err.append(erre)
            #print "done training"
            verre = prevverre
            restowrite = ""
            if self._autosave:
                self.save()
            if validf is not None and self.currentiter % evalinter == 0: # validate and print
                verre = validf()
                prevverre = verre
                verr.append(verre)
                ttmsg = "training error: %s \t validation error: %s" \
                       % ("%.4f" % erre[0],
                          " - ".join(map(lambda x: "%.4f" % x, verre)))
                restowrite = "\t".join(map(str, erre[0:1] + verre))
            else:
                ttmsg = "training error: %s" % " - ".join(map(lambda x: "%.4f" % x, erre))
                restowrite = str(erre[0])
            if writeresf is not None:
                writeresf.write("{}\t{}\n".format(self.currentiter - 1, restowrite))
            # retaining the best
            if self.besttaker is not None:
                modelscore = self.besttaker(([erre]+verre+[self.currentiter]))
                smallerbettermult = 1 if self.smallerbetter else -1
                if smallerbettermult * modelscore < smallerbettermult * self.bestmodel[1]:
                    if self.savebest:
                        self.save(suffix=".best")
                        self.bestmodel = (None, modelscore)
                    else:
                        #tt.tock("freezing best with score %.3f (prev: %.3f)" % (modelscore, self.bestmodel[1]), prefix="-").tick()
                        self.bestmodel = (self.save(freeze=True, filepath=False), modelscore)
            tt.tock(ttmsg + "\t", prefix="-")
            self._update_lr(self.currentiter, self.maxiter, err, verr)
            evalcount += 1
            #embed()
        if writeresf is not None:
            writeresf.close()
        self.tt.tock("trained").tick()
        return err, verr

    def getbatchloop(self, trainf, datafeeder, verbose=True):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        sampletransf = self._transformsamples

        def batchloop():
            c = 0
            numex = 0
            prevperc = -1.
            terr = [0.0]
            numdigs = 2
            tt = TT("iter progress", verbose=verbose)
            tt.tick()
            while datafeeder.hasnextbatch():
                perc = round(c*100.*(10**numdigs)/datafeeder._numbats)/(10**numdigs)
                if perc > prevperc:
                    terr0 = terr[0] * 1.0 / numex if numex > 0 else 0.0
                    s = ("%."+str(numdigs)+"f%% \t error: %.3f") % (perc, terr0)
                    tt.live(s)
                    prevperc = perc
                sampleinps, batsize = datafeeder.nextbatch(withbatchsize=True)
                numex += batsize
                sampleinps = sampletransf(*sampleinps)
                eterr = trainf(*sampleinps)
                if len(terr) != len(eterr) and terr.count(0.0) == len(terr):
                    terr = [0.0]*len(eterr)     # ensure compatible size of terr (number of output scores)
                if self.average_err is True:
                    terr = [xterr + xeterr * batsize for xterr, xeterr in zip(terr, eterr)]
                else:
                    terr = [xterr + xeterr for xterr, xeterr in zip(terr, eterr)]
                c += 1
            tt.stoplive()
            if self.average_err is True:
                terr = [xterr * 1.0 / numex for xterr in terr]
            return terr
        return batchloop

    def _transformsamples(self, *s):
        return s
    #endregion
    #endregion

    @property
    def autosave(self):
        self._autosave = True
        return self

    def autosavethis(self, block, p):
        self._autosave = True
        self._autosaveblock = block
        self._autosavepath = p
        return self

    def writeresultstofile(self, p):
        self._writeresultspath = p
        return self

    def save(self, model=None, filepath=None, suffix="", freeze=False):
        model = model if model is not None else \
            self.model if self._autosaveblock is None else \
                self._autosaveblock
        if filepath is not False:
            filepath = filepath if filepath is not None else self._autosavepath
            model.save(filepath=filepath + suffix)
        if freeze:
            return model.freeze()


class NSModelTrainer(ModelTrainer):
    """ Model trainer using negative sampling """
    def __init__(self, model, gold, nrate, nsamgen, nrate_valid=None):
        super(NSModelTrainer, self).__init__(model, gold)
        self.ns_nrate = nrate
        self.ns_nrate_valid = nrate if nrate_valid is None else nrate_valid
        self.ns_nsamgen = nsamgen

    def _transformsamples(self, *s):
        """ apply negative sampling function and neg sam rate """
        psams = s[:-1]
        acc = []
        for i in range(self.ns_nrate):
            nsams = self.ns_nsamgen(*psams)
            news = psams + nsams + (s[-1],)
            ret = []
            if len(acc) == 0:       # first one
                ret = news
            else:
                for x, y in zip(acc, news):
                    ret.append(np.concatenate([x, y], axis=0))
            acc = ret
        return acc

    def autobuild_model(self, model, *data, **kw):
        return model.autobuild(*(data + data), **kw)
