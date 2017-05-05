import numpy as np, os, pickle, sys, math
from IPython import embed

from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.seq.enc import SimpleSeq2Vec, SeqUnroll

from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.match import SeqMatchScore, GenDotDistance
from teafacto.core.base import Val, tensorops as T, Block
from teafacto.eval.metrics import ClassAccuracy
from teafacto.modelusers import RecPredictor, SeqEncDecPredictor
from teafacto.util import argprun, ticktock


def readdata(mode, testcans=None, debug=False, specids=False, usetypes=False, maskid=0):  # if none, included in file
    if debug:
        testcans = None
    if mode == "char":
        if debug:
            p = "../../../data/simplequestions/datamat.char.mini.pkl"
        else:
            p = "../../../data/simplequestions/datamat.char.mem.fb2m.pkl"
    elif mode == "word" or mode == "charword":
        # BEWARE: all entities without labels have only the token "RARE" as label in entmat
        p = "../../../data/simplequestions/datamat.word.mem.fb2m.pkl"
    else:
        raise Exception("unknown mode")
    if usetypes:
        enttyp = {}
        typdic = {"None": 0}
        with open("../../../data/simplequestions/datamat.fb2m.types.map") as f:
            for line in f:
                e, t = line[:-1].split("\t")
                enttyp[e] = t
    x = pickle.load(open(p))

    worddic = x["worddic"] if mode == "word" or mode == "charword" else x["chardic"]
    entdic = x["entdic"]
    numents = x["numents"]

    entmat = x["entmat"]

    if specids: # prepend special id's (non-unique for entities, unique for predicates, unique for others)
        prependmat = np.zeros((entmat.shape[0], 1), dtype="int32")
        mid = 1
        if usetypes:
            reventdic = {v: k for k, v in entdic.items()}
            for i in range(1, numents):
                t = enttyp[reventdic[i]]
                if t not in typdic:
                    typdic[t] = len(typdic)
                    mid = len(typdic)
                prependmat[i] = typdic[t]
        # same ids for all entitites
        # unique predicate ids
        for i in range(numents, entmat.shape[0]):
            prependmat[i] = mid
            mid += 1
        entmat = np.concatenate([prependmat, entmat], axis=1)
        if debug:
            pass
            #embed()

    train = x["train"]
    valid = x["valid"]
    test  = x["test"]

    wordmat, chardic = buildwordmat(worddic, maskid=maskid) if mode == "charword" else None, None

    if testcans is None:
        canids = x["testcans"]
    else:
        canids = pickle.load(open(testcans))
    for canidl in canids:  # include all relations
        canidl.extend(range(numents, entmat.shape[0]))

    return train, valid, test, worddic, entdic, entmat, numents, canids, \
           wordmat, chardic


def buildwordmat(wd, maskid=-1, topmaxlen=40):
    rwd = {v: k for k, v in wd.items()}
    wordmat = maskid * np.ones((len(rwd), topmaxlen)).astype("int32")
    for i in range(len(rwd)):
        w = rwd[i]
        w = w[:min(topmaxlen, len(w))]
        wordmat[i, :len(w)] = [ord(c) for c in w]
    uniqueords = set(np.unique(wordmat)).difference({maskid})
    chardic = dict(zip([chr(x) for x in sorted(uniqueords)], range(len(uniqueords))))
    wordmat = np.vectorize(lambda x: chardic[chr(x)])(wordmat)
    return wordmat, chardic


class SeqEncDecRankSearch(object):
    def __init__(self, model, canenc, scorer, agg, beamsize=1, *buildargs, **kw):
        super(SeqEncDecRankSearch, self).__init__(**kw)
        self.model = model
        self.beamsize = beamsize
        self.mu = SeqEncDecPredictor(model, *buildargs)
        self.scorer = scorer
        self.canenc = canenc
        self.agg = agg
        self.tt = ticktock("RankSearch")
        self.ott = ticktock("RankSearch")

    def decode(self, inpseqs, maxlen=100, candata=None,
               canids=None, transform=None, debug=False, split=1):
        assert(candata is not None and canids is not None)
        totret = []
        totsco = []
        splitsize = int(math.ceil(inpseqs.shape[0]*1./split))

        for isplit in range(split):
            self.ott.tick()
            inpseq = inpseqs[isplit*splitsize: min(inpseqs.shape[0], (isplit+1)*splitsize)]
            self.mu.reset()
            self.mu.setbuildargs(inpseq)
            self.mu.settransform(transform)
            stop = False
            j = 0
            curout = np.repeat([self.mu.startsym], inpseq.shape[0]).astype("int32")
            accscores = []
            outs = []
            while not stop:
                curvectors = self.mu.feed(curout)
                curout = np.ones_like(curout, dtype=curout.dtype)
                accscoresj = np.zeros((inpseq.shape[0],))
                self.tt.tick()
                for i in range(curvectors.shape[0]):    # for each example, find the highest scoring suited cans and their scores
                    #print len(canids[i])
                    if len(canids[i]) == 0:
                        curout[i] = -1
                    else:
                        canidsi = canids[i]
                        candatai = candata[canidsi]
                        canrepsi = self.canenc.predict(candatai)
                        curvectori = np.repeat(curvectors[np.newaxis, i, ...], canrepsi.shape[0], axis=0)
                        scoresi = self.scorer.predict(curvectori, canrepsi)
                        curout[i] = canidsi[np.argmax(scoresi)]
                        accscoresj[i] += np.max(scoresi)
                        if debug:
                            print i+isplit*splitsize, sorted(zip(canidsi, scoresi), key=lambda (x, y): y, reverse=True)
                            print sorted(filter(lambda (x, y): x < 4711, zip(canidsi, scoresi)), key=lambda (x, y): y, reverse=True)
                            print sorted(filter(lambda (x, y): x >= 4711, zip(canidsi, scoresi)), key=lambda (x, y): y,
                                         reverse=True)
                        #embed()
                    self.tt.progress(i, curvectors.shape[0], live=True)
                accscores.append(accscoresj[:, np.newaxis])
                outs.append(curout)
                j += 1
                stop = j == maxlen
                self.tt.tock("done one timestep")
            accscores = np.sum(np.concatenate(accscores, axis=1), axis=1)
            ret = np.stack(outs).T
            assert (ret.shape[0] == inpseq.shape[0] and ret.shape[1] <= maxlen)
            totret.append(ret)
            totsco.append(accscores)
            self.ott.tock("done {}/{} splits".format(isplit+1, split))
        return np.concatenate(totret, axis=0), np.concatenate(totsco, axis=0)


class FullRankEval(object):
    def __init__(self):
        self.metrics = {"all": ClassAccuracy(),
                        "subj": ClassAccuracy(),
                        "pred": ClassAccuracy()}

    def eval(self, pred, gold, debug=False):
        for i in range(pred.shape[0]):
            self.metrics["all"].accumulate(gold[i], pred[i])
            if debug == "subj":
                self.metrics["subj"].accumulate(gold[i][0], pred[i][0])
                self.metrics["pred"].accumulate(gold[i][0], gold[i][0])     #dummy
            elif debug == "pred":
                self.metrics["pred"].accumulate(gold[i][0], pred[i][0])
                self.metrics["subj"].accumulate(gold[i][0], gold[i][0])     #dummy
            else:
                self.metrics["subj"].accumulate(gold[i][0], pred[i][0])
                self.metrics["pred"].accumulate(gold[i][1], pred[i][1])
        return self.metrics


class EntEnc(Block):
    def __init__(self, enc, **kw):
        self.enc = enc
        super(EntEnc, self).__init__(**kw)

    def apply(self, x, mask=None):
        return self.enc(x[:, 1:], mask=mask)

    @property
    def outdim(self):
        return self.enc.outdim


class EntEmbEnc(Block):
    def __init__(self, enc, vocsize, embdim, **kw):
        super(EntEmbEnc, self).__init__(**kw)
        self.enc = enc
        self.emb = VectorEmbed(vocsize, embdim)

    def apply(self, x, mask=None):
        enco = self.enc(x, mask=mask)
        embo = self.emb(x[:, 0])
        ret = T.concatenate([enco, embo], axis=1)                   # (?, encdim+embdim)
        return ret

    @property
    def outdim(self):
        return self.enc.outdim + self.emb.outdim


class EntEncRep(Block):
    ''' Replaces encoding with embedding based on repsplit index '''
    def __init__(self, enc, vocsize, repsplit, **kw):
        super(EntEncRep, self).__init__(**kw)
        self.enc = enc
        self.emb = VectorEmbed(vocsize, enc.outdim)
        self.split = repsplit

    def apply(self, x, mask=None):
        enco = self.enc(x, mask=mask)
        embo = self.emb(x[:, 0])
        mask = x[:, 0] >= self.split      # (batsize, )
        mask = mask[:, np.newaxis]
        #mask = T.repeat(mask, enco.shape[1], axis=1)
        #ret = T.switch(x[:, 0] >= self.split, embo, enco)
        ret = (1-mask) * enco + mask * embo
        return ret

    @property
    def outdim(self):
        return self.emb.outdim


def shiftdata(d):  # idx (batsize, seqlen)
    ds = np.zeros_like(d)
    ds[:, 1:] = d[:, :-1]
    return ds


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
        balancednegidx=False,
        usetypes=False,
        evalsplits=50,
        relembrep=False,
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
        #preeval = True
        specemb = 100
        margin = 1.
        balancednegidx = True
        evalsplits = 1
        relembrep = True
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

    #embed()

    if subjpred is True and predpred is False:
        traingold = traingold[:, [0]]
        validgold = validgold[:, [0]]
        testgold = testgold[:, [0]]
    if predpred is True and subjpred is False:
        traingold = traingold[:, [1]]
        validgold = validgold[:, [1]]
        testgold = testgold[:, [1]]


    if checkdata:
        rwd = {v: k for k, v in worddic.items()}
        red = {v: k for k, v in entdic.items()}
        def p(xids):
            return (" " if mode == "word" else "").join([rwd[xid] if xid > -1 else "" for xid in xids])
        embed()

    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")

    # *data: matrix of word ids (-1 filler), example per row
    # *gold: vector of true entity ids
    # entmat: matrix of word ids (-1 filler), entity label per row, indexes according to *gold
    # *dic: from word/ent-fbid to integer id, as used in data

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
        decinnerdim = [decdim/2]*memlayers
    else:
        decinnerdim = [decdim]*memlayers

    entenc = EntEnc(SimpleSeq2Vec(indim=numwords,
                         inpembdim=memembdim,
                         innerdim=decinnerdim,
                         maskid=maskid,
                         bidir=membidir))

    numentembs = len(np.unique(entmat[:, 0]))
    if specids:     # include vectorembedder
        entenc = EntEmbEnc(entenc, numentembs, specemb)
    if relembrep:
        repsplit = entmat[relstarts, 0]
        entenc = EntEncRep(entenc, numentembs, repsplit)

        # adjust params for enc/dec construction
        #encinnerdim[-1] += specemb
        #innerdim[-1] += specemb

    encdec = SimpleSeqEncDecAtt(inpvocsize=numwords, inpembdim=embdim,
                    encdim=encinnerdim, bidir=bidir, outembdim=entenc,
                    decdim=decinnerdim, vecout=True, statetrans="matdot")

    scorerargs = ([encdec, SeqUnroll(entenc)],
                  {"argproc": lambda x, y, z: ((x, y), (z,)),
                   "scorer": GenDotDistance(decinnerdim[-1], entenc.outdim)})
    if sumhingeloss:
        scorerargs[1]["aggregator"] = lambda x: x  # no aggregation of scores
    scorer = SeqMatchScore(*scorerargs[0], **scorerargs[1])

    #scorer.save("scorer.test.save")

    # TODO: below this line, check and test
    class PreProc(object):
        def __init__(self, entmat):
            self.f = PreProcE(entmat)

        def __call__(self, encdata, decsg, decgold):        # gold: idx^(batsize, seqlen)
            return (encdata, self.f(decsg), self.f(decgold)), {}

    class PreProcE(object):
        def __init__(self, entmat):
            self.em = Val(entmat)

        def __call__(self, x):
            return self.em[x]

    transf = PreProc(entmat)

    class NegIdxGen(object):
        def __init__(self, rng, midsplit=None):
            self.min = 0
            self.max = rng
            self.midsplit = midsplit

        def __call__(self, datas, sgold, gold):    # the whole target sequence is corrupted, corruption targets the whole set of entities and relations together
            if self.midsplit is None or not balancednegidx:
                return datas, sgold, np.random.randint(self.min, self.max, gold.shape).astype("int32")
            else:
                entrand = np.random.randint(self.min, self.midsplit, gold.shape)
                relrand = np.random.randint(self.midsplit, self.max, gold.shape)
                mask = np.random.randint(0, 2, gold.shape)
                ret = entrand * mask + relrand * (1 - mask)
                return datas, sgold, ret.astype("int32")

    obj = lambda p, n: n - p
    if hingeloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)
    if sumhingeloss:    #
        obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)

    traingoldshifted = shiftdata(traingold)
    validgoldshifted = shiftdata(validgold)

    #embed()
    # eval
    if preeval:
        tt.tick("pre-evaluating")
        s = SeqEncDecRankSearch(encdec, entenc, scorer.s, scorer.agg)
        eval = FullRankEval()
        pred, scores = s.decode(testdata, testgold.shape[1],
                                candata=entmat, canids=canids, split=evalsplits,
                                transform=transf.f, debug=printpreds)
        evalres = eval.eval(pred, testgold, debug=debug)
        for k, evalre in evalres.items():
            print("{}:\t{}".format(k, evalre))
        tt.tock("pre-evaluated")

    negidxgenargs = ([numents], {"midsplit": relstarts})
    if debug:
        pass
        #negidxgenargs = ([numents], {})

    tt.tick("training")
    nscorer = scorer.nstrain([traindata, traingoldshifted, traingold]).transform(transf) \
        .negsamplegen(NegIdxGen(*negidxgenargs[0], **negidxgenargs[1])).negrate(negrate).objective(obj) \
        .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0) \
        .validate_on([validdata, validgoldshifted, validgold]) \
        .train(numbats=numbats, epochs=epochs)
    tt.tock("trained")

    #scorer.save("scorer.test.save")

    # eval
    tt.tick("evaluating")
    s = SeqEncDecRankSearch(encdec, entenc, scorer.s, scorer.agg)
    eval = FullRankEval()
    pred, scores = s.decode(testdata, testgold.shape[1],
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

    #scorer.save(filepath=savename)


if __name__ == "__main__":
    argprun(run, debug=True)
