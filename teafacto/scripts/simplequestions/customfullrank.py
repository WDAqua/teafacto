from teafacto.scripts.simplequestions.fullrank import readdata, SeqEncDecRankSearch, FullRankEval, shiftdata, EntEncRep, EntEnc, EntEmbEnc
from teafacto.util import argprun, ticktock
import numpy as np, os, sys, math
from IPython import embed
from teafacto.core.base import Val, tensorops as T, Block

from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeq2Sca, SeqUnroll
from teafacto.blocks.match import SeqMatchScore, GenDotDistance, DotDistance, CosineDistance
from teafacto.blocks.basic import VectorEmbed, MatDot


class CustomSeq2Pair(Block):
    def __init__(self, inpemb, encdim=100, scadim=100, maskid=0, bidir=False, scalayers=1, enclayers=1, outdim=100, **kw):
        super(CustomSeq2Pair, self).__init__(**kw)
        self.tosca = SimpleSeq2Sca(inpemb=inpemb, inpembdim=inpemb.outdim, innerdim=scadim, maskid=maskid, bidir=bidir, layers=scalayers)
        self.subjenc = SimpleSeq2Vec(inpemb=inpemb, inpembdim=inpemb.outdim, innerdim=encdim, maskid=maskid, bidir=bidir, layers=enclayers)
        self.predenc = SimpleSeq2Vec(inpemb=inpemb, inpembdim=inpemb.outdim, innerdim=encdim, maskid=maskid, bidir=bidir, layers=enclayers)
        self.subjmd = MatDot(self.subjenc.outdim, outdim)
        self.predmd = MatDot(self.predenc.outdim, outdim)

    def apply(self, x):
        weights, mask = self.tosca(x)
        subjenco = self.subjenc(x, weights=weights)[:, np.newaxis, :]
        predenco = self.predenc(x, weights=(1-weights))[:, np.newaxis, :]
        ret = T.concatenate([self.subjmd(subjenco), self.predmd(predenco)], axis=1)
        return ret

    @property
    def outdim(self):
        assert(self.subjenc.outdim == self.predenc.outdim)
        return self.subjenc.outdim


class CustomEntEnc(Block):
    def __init__(self, subjenc, predenc, offset, **kw):
        super(CustomEntEnc, self).__init__(**kw)
        self.subjenc = subjenc
        self.predenc = predenc
        self.offset = offset

    def apply(self, x): # (batsize, 2, syms)
        subjenco = self.subjenc(x[:, 0, :]).dimshuffle(0, "x", 1)
        predenco = self.predenc(x[:, 1, 0] - self.offset).dimshuffle(0, "x", 1)
        ret = T.concatenate([subjenco, predenco], axis=1)
        return ret

    @property
    def outdim(self):
        return self.subjenc.outdim


class CustomRankSearch(object):
    def __init__(self, model, canenc, scorer, agg, relstarts=0, *buildargs, **kw):
        super(CustomRankSearch, self).__init__(**kw)
        self.model = model
        self.scorer = scorer
        self.canenc = canenc
        self.agg = agg
        self.tt = ticktock("RankSearch")
        self.ott = ticktock("RankSearch")
        self.relstarts = relstarts

    def search(self, inpseqs, maxlen=100, candata=None,
               canids=None, transform=None, debug=False, split=1):
        assert(candata is not None and canids is not None)
        totret = []
        totsco = []
        splitsize = int(math.ceil(inpseqs.shape[0]*1./split))

        for isplit in range(split):
            self.ott.tick()
            inpseq = inpseqs[isplit*splitsize: min(inpseqs.shape[0], (isplit+1)*splitsize)]
            pred = self.model.predict(inpseq)
            scores = []
            outs = []
            for i in range(pred.shape[0]):
                canidsi = canids[i+isplit*splitsize]
                canidsii = [None, None]
                subjcans = filter(lambda x: x < self.relstarts, canidsi)
                predcans = filter(lambda x: x >= self.relstarts, canidsi)

                canidsii[0] = subjcans + [0] * max(0, len(predcans) - len(subjcans))
                canidsii[1] = predcans + [self.relstarts] * max(0, - len(predcans) + len(subjcans))
                canidsii = np.asarray(canidsii, dtype="int32").T
                canvecs = self.canenc.predict.transform(transform)(canidsii)

                entvecs = canvecs[:len(subjcans), 0, :]
                predvecs = canvecs[:len(predcans), 0, :]

                subjinpveci = np.repeat(pred[[i], 0], len(subjcans), axis=0)
                predinpveci = np.repeat(pred[[i], 1], len(predcans), axis=0)

                subjscores = self.scorer.predict(subjinpveci, entvecs)
                predscores = self.scorer.predict(predinpveci, predvecs)

                subjswscores = sorted(zip(subjcans, list(subjscores)), key=lambda (x, y): y, reverse=True)
                predswscores = sorted(zip(predcans, list(predscores)), key=lambda (x, y): y, reverse=True)

                if debug:
                    print i
                    print subjswscores
                    print predswscores

                best = [subjswscores[0][0] if len(subjswscores) > 0 else -1,
                        predswscores[0][0] if len(predswscores) > 0 else -1]
                scor = [subjswscores[0][1] if len(subjswscores) > 0 else 0,
                        predswscores[0][1] if len(predswscores) > 0 else 0]

                scores.append(scor)
                outs.append(best)

                self.tt.progress(i, pred.shape[0], live=True)
            scores = np.sum(np.asarray(scores, dtype="float32"), axis=1)
            outs = np.asarray(outs, dtype="int32")
            assert (outs.shape[0] == inpseq.shape[0] and outs.shape[1] <= maxlen)
            totret.append(outs)
            totsco.append(scores)
            self.ott.tock("done {}/{} splits".format(isplit+1, split))
        return np.concatenate(totret, axis=0), np.concatenate(totsco, axis=0)


def run(
        epochs=50,
        mode="char",    # "char" or "word" or "charword"
        numbats=1000,
        lr=0.1,
        wreg=0.000001,
        bidir=False,
        layers=1,
        encdim=200,
        decdim=200,
        embdim=100,
        negrate=1,
        margin=1.,
        hingeloss=False,
        debug=False,
        preeval=False,
        sumhingeloss=False,
        checkdata=False,        # starts interactive shell for data inspection
        printpreds=False,
        subjpred=False,
        predpred=False,
        specemb=-1,
        usetypes=False,
        evalsplits=50,
        cosine=False,
        loadmodel=False,
    ):
    if debug:       # debug settings
        sumhingeloss = True
        numbats = 10
        lr = 0.02
        epochs = 10
        printpreds = True
        whatpred = "all"
        if whatpred == "pred":
            predpred = True
        elif whatpred == "subj":
            subjpred = True
        preeval = True
        #specemb = 100
        margin = 1.
        evalsplits = 1
        #usetypes=True
        #mode = "charword"
        #checkdata = True
    # load the right file
    maskid = -1
    tt = ticktock("script")
    specids = specemb > 0
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat, relstarts, canids, wordmat, chardic\
        = readdata(mode, testcans="testcans.pkl", debug=debug, specids=True,
                   usetypes=usetypes, maskid=maskid)
    entmat = entmat.astype("int32")

    if checkdata:
        rwd = {v: k for k, v in worddic.items()}
        red = {v: k for k, v in entdic.items()}
        def p(xids):
            return (" " if mode == "word" else "").join([rwd[xid] if xid > -1 else "" for xid in xids])
        embed()

    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")

    numwords = max(worddic.values()) + 1
    numents = max(entdic.values()) + 1
    print "%d words, %d entities" % (numwords, numents)


    if bidir:
        encinnerdim = [encdim / 2] * layers
    else:
        encinnerdim = [encdim] * layers

    memembdim = embdim
    memlayers = layers
    membidir = bidir
    if membidir:
        decinnerdim = [decdim / 2] * memlayers
    else:
        decinnerdim = [decdim] * memlayers


    emb = VectorEmbed(numwords, embdim)

    subjenc = EntEnc(SimpleSeq2Vec(invocsize=numwords,
                                   inpembdim=embdim,
                                  innerdim=decinnerdim,
                                  maskid=maskid,
                                  bidir=membidir))

    numentembs = len(np.unique(entmat[:, 0]))
    repsplit = entmat[relstarts, 0]
    if specids:  # include vectorembedder
        subjenc = EntEmbEnc(subjenc, numentembs, specemb)
    predenc = VectorEmbed(indim=numents-relstarts+1, dim=subjenc.outdim, init="zero")
    entenc = CustomEntEnc(subjenc, predenc, repsplit)

    inpenc = CustomSeq2Pair(inpemb=emb, encdim=encinnerdim, scadim=encinnerdim,
                            enclayers=layers, scalayers=layers, bidir=bidir,
                            maskid=maskid, outdim=subjenc.outdim)

        # adjust params for enc/dec construction
        # encinnerdim[-1] += specemb
        # innerdim[-1] += specemb

    dist = DotDistance() if not cosine else CosineDistance()
    scorerkwargs = {"argproc": lambda x, y: ((x,), (y,)),
                   "scorer": dist}
    if sumhingeloss:
        scorerkwargs["aggregator"] = lambda x: x  # no aggregation of scores
    scorer = SeqMatchScore(inpenc, entenc, **scorerkwargs)

    class PreProc(object):
        def __init__(self, entmat, wordmat=None):
            self.f = PreProcE(entmat)
            self.w = PreProcL(wordmat) if wordmat is not None else wordmat

        def __call__(self, encdata, decgold):  # gold: idx^(batsize, seqlen)
            if self.w is not None:
                encdata = self.w(encdata)[0][0]
            if self.f is not None:
                decgold = self.f(decgold)[0][0]
            return (encdata, decgold), {}

    class PreProcE(object):
        def __init__(self, entmat):
            self.em = Val(entmat)

        def __call__(self, x):
            ret = self.em[x]
            return (ret,), {}

    class PreProcL(object):
        def __init__(self, wordmat):
            self.em = Val(wordmat)

        def __call__(self, x):
            ret = self.em[x]
            return (ret,), {}

    transf = PreProc(entmat)

    class NegIdxGen(object):
        def __init__(self, rng, midsplit):
            self.min = 0
            self.max = rng
            self.midsplit = midsplit

        def __call__(self, datas, gold):
            entrand = np.random.randint(self.min, self.midsplit, (gold.shape[0], 1))
            relrand = np.random.randint(self.midsplit, self.max, (gold.shape[0], 1))
            ret = np.concatenate([entrand, relrand], axis=1)
            return datas, ret.astype("int32")

    #embed()

    obj = lambda p, n: n - p
    if hingeloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)
    if sumhingeloss:  #
        obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)

    # embed()
    # eval
    if preeval:
        tt.tick("pre-evaluating")
        s = CustomRankSearch(inpenc, entenc, scorer.s, scorer.agg, relstarts=relstarts)
        eval = FullRankEval()
        pred, scores = s.search(testdata, testgold.shape[1],
                                candata=entmat, canids=canids, split=evalsplits,
                                transform=transf.f, debug=printpreds)
        evalres = eval.eval(pred, testgold, debug=debug)
        for k, evalre in evalres.items():
            print("{}:\t{}".format(k, evalre))
        tt.tock("pre-evaluated")

    if not loadmodel:
        tt.tick("training")
        nscorer = scorer.nstrain([traindata, traingold]).transform(transf) \
            .negsamplegen(NegIdxGen(numents, relstarts)).negrate(negrate).objective(obj) \
            .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0) \
            .validate_on([validdata, validgold]) \
            .train(numbats=numbats, epochs=epochs)
        tt.tock("trained")
        scorer.save("customfullrank.scorer.save")
    else:
        scorer = SeqMatchScore.load("customfullrank.scorer.save")

    # eval
    tt.tick("evaluating")

    s = CustomRankSearch(inpenc, entenc, scorer.s, scorer.agg, relstarts=relstarts)
    eval = FullRankEval()
    pred, scores = s.search(testdata, testgold.shape[1],
                            candata=entmat, canids=canids, split=evalsplits,
                            transform=transf.f, debug=printpreds)
    if printpreds:
        print pred
    debugarg = "subj" if subjpred else "pred" if predpred else False
    evalres = eval.eval(pred, testgold, debug=debugarg)
    for k, evalre in evalres.items():
        print("{}:\t{}".format(k, evalre))
    tt.tock("evaluated")

    # save
    basename = os.path.splitext(os.path.basename(__file__))[0]
    dirname = basename + ".results"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savenamegen = lambda i: "{}/{}.res".format(dirname, i)
    savename = None
    for i in xrange(1000):
        savename = savenamegen(i)
        if not os.path.exists(savename):
            break
        savename = None
    if savename is None:
        raise Exception("exceeded number of saved results")
    with open(savename, "w") as f:
        f.write("{}\n".format(" ".join(sys.argv)))
        for k, evalre in evalres.items():
            f.write("{}:\t{}\n".format(k, evalre))


if __name__ == "__main__":
    argprun(run)