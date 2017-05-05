from teafacto.util import ticktock, argprun, inp, tokenize
import os, pickle, random
from teafacto.procutil import *
from IPython import embed
from scipy import sparse

from teafacto.blocks.lang.wordvec import Glove, WordEmb
from teafacto.blocks.lang.sentenc import TwoLevelEncoder
from teafacto.blocks.seq.rnn import RNNSeqEncoder, MaskMode
from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeq2MultiVec
from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.memory import MemVec
from teafacto.blocks.match import SeqMatchScore, CosineDistance, DotDistance, GenDotDistance, MatchScore

from teafacto.core.base import Block, tensorops as T, Val, TransWrapBlock


def readdata(p="../../../../data/simplequestions/clean/datamat.word.fb2m.pkl",
             entinfp="../../../../data/simplequestions/clean/subjs-counts-labels-types.fb2m.tsv",
             cachep=None, #"subjpredcharns.readdata.cache.pkl",
             maskid=-1,
             debug=False,
             numtestcans=None,
             wordlevel=False,
             ):
    tt = ticktock("dataloader")
    if cachep is not None and os.path.isfile(cachep):      # load
        tt.tick("loading from cache")
        ret = pickle.load(open(cachep))
        tt.tock("loaded from cache")
    else:
        tt.tick("loading datamat")
        x = pickle.load(open(p))
        tt.tock("datamat loaded")
        worddic = x["worddic"]
        entdic = x["entdic"]
        entmat = x["entmat"]
        numents = x["numents"]
        traindata, traingold = x["train"]
        validdata, validgold = x["valid"]
        testdata, testgold = x["test"]
        traingold[:, 1] -= numents
        validgold[:, 1] -= numents
        testgold[:, 1] -= numents

        # TODO: add surrogates here?

        rwd = {v: k for k, v in worddic.items()}

        subjdic = {k: v for k, v in entdic.items() if v < numents}
        reldic = {k: v - numents for k, v in entdic.items() if v >= numents}

        subjmat = entmat[:numents]
        ssubjmat = np.sum(subjmat != maskid, axis=1)
        if np.any(ssubjmat == 0):
            for i in list(np.argwhere(ssubjmat == 0)[:, 0]):
                subjmat[i, 0] = worddic["<RARE>"]

        relmat = entmat[numents:]
        if debug:
            embed()

        if not wordlevel:
            traindata = wordmat2wordchartensor(traindata, rwd=rwd, maskid=maskid)
            validdata = wordmat2wordchartensor(validdata, rwd=rwd, maskid=maskid)
            testdata = wordmat2wordchartensor(testdata, rwd=rwd, maskid=maskid)

            subjmat = wordmat2charmat(subjmat, rwd=rwd, maskid=maskid, raretoken="<RARE>", maxlen=75)

        ret = ((traindata, traingold), (validdata, validgold),
               (testdata, testgold), (subjmat, relmat), (subjdic, reldic),
               worddic)
        if cachep is not None:
            tt.tick("dumping to cache")
            pickle.dump(ret, open(cachep, "w"))
            tt.tock("dumped to cache")

    subjinfo = loadsubjinfo(entinfp, subjdic)
    testsubjcans = loadsubjtestcans(numcans=numtestcans)
    testrelcans, relspersubj = loadreltestcans(testgold, subjdic, reldic)
    if debug:
        embed()
    return ret + (subjinfo, (testsubjcans, relspersubj))


def loadreltestcans(testgold, subjdic, reldic, relsperentp="../../../../data/simplequestions/allrelsperent.dmp"):
    tt = ticktock("test rel can loader")
    testsubjs = testgold[:, 0]
    relsperent = {} #{k: ([], []) for k in set(list(testsubjs))}
    tt.tick("loading rel test cans")
    for line in open(relsperentp):
        subj, relsout, relsin = line[:-1].split("\t")
        if subj in subjdic:
            relsperent[subjdic[subj]] = (
                    [reldic[x] for x in relsout.split(" ")] if relsout != "" else [],
                    [reldic[x] for x in relsin.split(" ")] if relsin != "" else []
            )
        #if subj in subjdic and subjdic[subj] in relsoftestsubjs:
        #    relsoftestsubjs[subjdic[subj]] = (
        #        [reldic[x] for x in relsout.split(" ")] if relsout != "" else [],
        #        [reldic[x] for x in relsin.split(" ")] if relsin != "" else []
        #    )
    tt.tock("test cans loaded")
    relsoftestexamples = [(relsperent[x][0], relsperent[x][1])
                          for x in testsubjs]
    return relsoftestexamples, relsperent


def loadsubjtestcans(p="../../../../data/simplequestions/clean/testcans{}.pkl", numcans=None):
    tt = ticktock("test subjects candidate loader")
    tt.tick("loading candidates")
    p = p.format("{}c".format(numcans)) if numcans is not None else p.format("")
    ret = pickle.load(open(p))
    tt.tock("canddiates loaded")
    return ret


def loadsubjinfo(entinfp, entdic, cachep=None):#"subjinfo.cache.pkl"):
    tt = ticktock("subjinfoloader")
    def make():
        tt.tick("making subject info from file")
        subjinf = {}
        c = 0
        for line in open(entinfp):
            subjuri, subjc, objc, subjname, typuri, typname = line[:-1].split("\t")
            subjinf[entdic[subjuri]] = (subjname, typname.lower().split(), typuri, subjc, objc)
            if c % 1000 == 0:
                tt.live(str(c))
            c += 1
        tt.tock("made subject info from file")
        return subjinf
    if cachep is not None:
        if os.path.isfile(cachep):      # load
            tt.tick("loading cached subject info")
            subjinfo = pickle.load(open(cachep))
            tt.tock("loaded cached subject info")
        else:                           # make  and dump
            subjinfo = make()
            tt.tick("dumping subject info in cache")
            pickle.dump(subjinfo, open(cachep, "w"))
            tt.tock("dumped subject info in cache")
    else:       # just make
        subjinfo = make()
    return subjinfo


def buildrelsamplespace(entmat, wd, maskid=-1):
    tt = ticktock("samplespace")
    tt.tick("making sample space")
    #rwd = {v: k for k, v in wd.items()}
    entmatm = sparse.dok_matrix((entmat.shape[0], np.max(entmat) + 1))
    posblacklist = {0: {wd["base"], wd["user"]}}
    blacklist = set([wd[x] for x in "default domain of by the in at s this for with type".split()])
    #revin = {k: set() for k in np.unique(entmat)}
    #revinm = sparse.dok_matrix((np.max(entmat), entmat.shape[0]))
    samdic = {k: set() for k in range(entmat.shape[0])}     # from ent ids to sets of ent ids
    #samdic = np.zeros((entmat.shape[0], entmat.shape[0]))
    for i in range(entmat.shape[0]):
        for j in range(entmat.shape[1]):
            w = entmat[i, j]
            if w == -1:     # beginning of padding
                break
            if j in posblacklist:
                if w in posblacklist[j]:
                    continue
            if w in blacklist:
                continue
            entmatm[i, w] = 1
            #for oe in revin[w]:     # other entities already in revind
            #    samdic[oe].add(i)
            #    samdic[i].add(oe)
            #revin[w].add(i)
            #revinm[w, i] = 1
    samdicm = entmatm.dot(entmatm.T)
    for i in range(samdicm.shape[0]):
        samdic[i] = list(np.argwhere(samdicm[i, :])[:, 1])
    tt.tock("made rel sample space")
    return samdic, entmatm.T


def loadsubjsamplespace(p="../../../../data/simplequestions/clean/subjclose.dic.pkl"):
    d = pickle.load(open(p))
    return d


def loadsubjtraincans(p="../../../../data/simplequestions/clean/traincans10c.pkl"):
    t = pickle.load(open(p))
    return t


def buildtraincanspace(traindata, maskid=-1):
    cans = loadsubjtraincans()
    out = {}
    for i in range(traindata.shape[0]):
        key = list(traindata[i, :, 0])  # only words
        key = " ".join([str(x) for x in key if x != maskid])
        if key not in out:
            out[key] = set()
        out[key].update(set(cans[i]))
    return out


def buildtypmat(subjmat, subjinfo, worddic, maxlen=6, maskid=-1):
    ret = maskid * np.ones((subjmat.shape[0], maxlen), dtype="int32")
    import re
    splitterregex = re.compile("[\s/]")
    for i in range(ret.shape[0]):
        typstring = " ".join(subjinfo[i][1] if i in subjinfo else "<unk>")
        typwords = splitterregex.split(typstring)
        typwordids = [worddic[typword] if typword in worddic else 0 for typword in typwords]
        typwordids = typwordids[:min(len(typwords), maxlen)]
        ret[i, :len(typwordids)] = typwordids
    return ret


class SeqLeftBlock(Block):
    def __init__(self, inner, **kw):
        super(SeqLeftBlock, self).__init__(**kw)
        self.inner = inner

    def apply(self, x):
        # idxs^(batsize, seqlen, ...) --> (batsize, seqlen, 2, encdim)
        res = self.inner(x).dimshuffle(0, "x", 1)
        ret = T.concatenate([res, res], axis=1)
        return ret      # (batsize, 2, decdim)


class ConcatLeftBlock(Block):
    def __init__(self, inner, mid, **kw):
        super(ConcatLeftBlock, self).__init__(**kw)
        self.inner = inner
        self.mid = mid

    def apply(self, x):
        res = self.inner(x).dimshuffle(0, "x", 1) # (batsize, 1, q_enc_dim)
        mid = self.mid #res.shape[2]/2
        ret = T.concatenate([res[:, :, :mid], res[:, :, mid:]], axis=1)
        return ret      # (batsize, 2, decdim)


class MultiLeftBlock(Block):
    def __init__(self, inner, mid, mode, **kw):
        super(MultiLeftBlock, self).__init__(**kw)
        self.inner = inner
        self.mode = mode
        self.mid = mid

    def apply(self, x):
        res = self.inner(x)             # (batsize, 2, encdim)
        if self.mode == "multic":   # take top half of first and bottom half of second
            mid = self.mid
            ret = T.concatenate([res[:, 0:1, :mid], res[:, 1:2, mid:]], axis=1)
        else:                       # return as is
            ret = res
        print "!!!!!!!!!!!!!!!!!!!!!{}".format(ret.ndim)
        return ret      # (batsize, 2, decdim)


class RightBlock(Block):
    def __init__(self, a, b, **kw):
        super(RightBlock, self).__init__(**kw)
        self.subjenc = a
        self.predenc = b

    def apply(self, subjslice, relslice):  # idxs^(batsize, len)
        aret = self.subjenc(subjslice).dimshuffle(0, "x", 1)
        bret = self.predenc(relslice).dimshuffle(0, "x", 1)
        ret = T.concatenate([aret, bret], axis=1)
        return ret  # (batsize, 2, decdim)


class TypedSubjBlock(Block):
    def __init__(self, typelen, subjenc, typenc, **kw):
        super(TypedSubjBlock, self).__init__(**kw)
        self.typelen = typelen
        self.typenc = typenc
        self.subjenc = subjenc

    def apply(self, x):
        typewords = x[:, :self.typelen]
        subjchars = x[:, self.typelen:]
        typemb = self.typenc(typewords)
        subemb = self.subjenc(subjchars)
        ret = T.concatenate([subemb, typemb], axis=1)
        return ret


class CustomPredictor(object):
    def __init__(self, questionencoder=None, entityencoder=None,
                 relationencoder=None, mode=None, maskid=-1,
                 enttrans=None, reltrans=None, debug=False,
                 subjinfo=None, silent=False, subjmat=None, testregions=None):
        self.qenc = questionencoder
        self.eenc = entityencoder
        self.renc = relationencoder
        self.mode = mode
        self.enttrans = enttrans
        self.reltrans = reltrans
        self.debug = debug
        self.subjinfo = subjinfo
        self.qencodings = None
        self.silent = silent
        self.subjmat = subjmat
        self.maskid = maskid
        if testregions is not None:
            print "Using TestRegions"
        self.testregions = testregions      # 92.8% seq accuracy 2layer bigru
        self.tt = ticktock("predictor", verbose=not silent)

    # stateful API
    def encodequestions(self, data):
        self.tt.tick("encoding questions")
        self.qencodings = self.qenc.predict(data)
        self.tt.tock("encoded questions")

    def ranksubjects(self, entcans, data=None):
        assert(self.qencodings is not None)
        if self.mode == "concat":
            qencforent = self.qencodings[:, :(self.qencodings.shape[1] / 2)]
        elif self.mode == "seq":
            qencforent = self.qencodings[:, :]
        elif self.mode == "multi":
            qencforent = self.qencodings[:, 0, :]
        elif self.mode == "multic":
            qencforent = self.qencodings[:, 0, :(self.qencodings.shape[2] / 2)]
        else:
            raise Exception("unrecognized mode in prediction")
        self.tt.tick("rank subjects")
        ret = []    # list of lists of (subj, score) tuples, sorted
        for i in range(self.qencodings.shape[0]):       # for every question
            entcansi = entcans[i]
            if self.testregions is not None:    # filter by regions
                question = data[i, :, 0]
                region = np.argmax(self.testregions[i], axis=-1)
                region[question == self.maskid] = 0
                regionwordpos = np.argwhere(region)[:, 0]
                regionwordids = list(question[regionwordpos])
                exact_retcans = []
                overlap_retcans = []
                for entcansii in entcansi:
                    entcansi_words = filter(lambda x: x != self.maskid,
                                            list(self.subjmat[entcansii]))
                    if regionwordids == entcansi_words:
                        exact_retcans.append(entcansii)
                    elif len(set(regionwordids).intersection(set(entcansi_words))) > 0:
                        # at least one word in common
                        overlap_retcans.append(entcansii)
                if len(exact_retcans) > 0:        # if region yields something
                    entcansi = exact_retcans      # return region-filtered set
                elif len(overlap_retcans) > 0:
                    entcansi = overlap_retcans
                #embed()
            if len(entcansi) == 0:
                scoredentcans = [(-1, 0)]
            elif len(entcansi) == 1:
                scoredentcans = [(entcansi[0], 1)]
            else:   # nontrivial
                entembs = self.eenc.predict.transform(self.enttrans)(entcansi)
                #embed()
                entscoresi = np.tensordot(qencforent[i], entembs, axes=(0, 1))
                entscoresi /= np.linalg.norm(qencforent[i])
                entscoresi /= np.linalg.norm(entembs, axis=1)
                scoredentcans = sorted(zip(entcansi, entscoresi), key=lambda (x, y): y, reverse=True)
            ret.append(scoredentcans)
            self.tt.progress(i, self.qencodings.shape[0], live=True)
        self.tt.tock("ranked subjects")
        self.subjranks = ret
        return ret

    def rankrelations(self, relcans):
        assert(self.qencodings is not None)
        if self.mode == "concat":
            qencforrel = self.qencodings[:, (self.qencodings.shape[1] / 2):]
        elif self.mode == "seq":
            qencforrel = self.qencodings[:, :]
        elif self.mode == "multi":
            qencforrel = self.qencodings[:, 1, :]
        elif self.mode == "multic":
            qencforrel = self.qencodings[:, 1, (self.qencodings.shape[2] / 2):]
        else:
            raise Exception("unrecognized mode in prediction")
        self.tt.tick("rank relations")
        ret = []
        for i in range(self.qencodings.shape[0]):
            if len(relcans[i]) == 0:
                scoredrelcans = [(-1, 0)]
            elif len(relcans[i]) == 1:
                scoredrelcans = [(relcans[i][0], 1)]
            else:
                relembs = self.renc.predict.transform(self.reltrans)(relcans[i])
                relscoresi = np.tensordot(qencforrel[i], relembs, axes=(0, 1))
                relscoresi /= np.linalg.norm(qencforrel[i])
                relscoresi /= np.linalg.norm(relembs, axis=1)
                scoredrelcans = sorted(zip(relcans[i], relscoresi), key=lambda (x, y): y, reverse=True)
            ret.append(scoredrelcans)
            self.tt.progress(i, self.qencodings.shape[0], live=True)
        self.tt.tock("ranked relations")
        self.relranks = ret
        return ret

    def rankrelationsfroments(self, bestsubjs, relsperent):
        relcans = [relsperent[bestsubj][0] if bestsubj in relsperent else [] for bestsubj in bestsubjs]
        return self.rankrelations(relcans)

    def predict(self, data, entcans=None, relsperent=None,
                relcans=None, multiprune=-1,
                goldsubjects=None):
        #print multiprune
        assert(relsperent is None or relcans is None)
        assert(relsperent is not None or relcans is not None)
        assert(entcans is not None)
        self.encodequestions(data)
        if goldsubjects is None:
            rankedsubjs = self.ranksubjects(entcans, data=data)
        else:
            print "USING GOLD SUBJECTS!!! WRONG !!!"
            rankedsubjs = [[[x]] for x in goldsubjects]
        bestsubjs = [x[0][0] for x in rankedsubjs]
        if relcans is not None:
            rankedrels = self.rankrelations(relcans)
            bestrels = [x[0][0] for x in rankedrels]
        else:
            if multiprune <= 0:
                relcans = [relsperent[bestsubj][0] if bestsubj in relsperent else [] for bestsubj in bestsubjs]
                rankedrels = self.rankrelations(relcans)
                bestrels = [x[0][0] for x in rankedrels]
            else:
                if not self.silent:
                    print "multipruning !!!!!!!!!!!!!!!!!"
                topk = multiprune        # TOP K !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # get relcans
                relcans = []
                for subjranking in rankedsubjs:
                    toplabel = None
                    relcanse = []
                    i = 0
                    for subj, score in subjranking:
                        subjlabel = " ".join(tokenize(self.subjinfo[subj][0]) if subj in self.subjinfo else [])
                        topcan = None
                        if toplabel is None:
                            toplabel = subjlabel
                            topcan = subj
                        elif subjlabel == toplabel:
                            topcan = subj
                        elif i < topk:
                            topcan = subj
                        else:
                            pass
                        toadd = relsperent[topcan][0] if topcan in relsperent else []
                        relcanse.extend(toadd)
                        i += 1
                    relcans.append(relcanse)
                # rank relations
                rankedrels = self.rankrelations(relcans)
                bestrels = [x[0][0] for x in rankedrels]
                # build ents per relation
                entsperrel = {}
                for ent, rels in relsperent.items():
                    for rel in rels[0]:
                        if rel not in entsperrel:
                            entsperrel[rel] = set()
                        entsperrel[rel].add(ent)
                # filter rankedsubjs
                filteredrankedsubjs = []
                for i in range(len(rankedsubjs)):
                    filteredrankedsubjs.append([])
                    for subj, score in rankedsubjs[i]:
                        if bestrels[i] in entsperrel and \
                                        subj in entsperrel[bestrels[i]]:
                                filteredrankedsubjs[i].append((subj, score))
                    if len(filteredrankedsubjs[i]) == 0:
                        filteredrankedsubjs[i].append((-1, -1.))
                bestsubjs = [x[0][0] for x in filteredrankedsubjs]






        ret = np.concatenate([
            np.expand_dims(np.asarray(bestsubjs, dtype="int32"), axis=1),
            np.expand_dims(np.asarray(bestrels, dtype="int32"), axis=1)
        ], axis=1)
        return ret

    def oldpredict(self, data, entcans, relsperent):
        tt = ticktock("predictor")
        tt.tick("computing question encodings")
        qencodings = self.qenc.predict(data)    # (numsam, encdim)
        tt.tock("computed question encodings")
        tt.tick("predicting")
        ret = np.zeros((data.shape[0], 2), dtype="int32")
        if self.mode == "concat":
            mid = qencodings.shape[1] / 2
            qencforent = qencodings[:, :mid]
            qencforrel = qencodings[:, mid:]
        elif self.mode == "seq":
            qencforent = qencodings[:, :]
            qencforrel = qencodings[:, :]
        else:
            raise Exception("unrecognized mode")
        for i in range(qencodings.shape[0]):
            # predict subject
            if len(entcans[i]) == 0:
                bestsubj = -1
            elif len(entcans[i]) == 1:
                bestsubj = entcans[i][0]
            else:
                entembs = self.eenc.predict.transform(self.enttrans)(entcans[i])
                entscoresi = np.tensordot(qencforent[i], entembs, axes=(0, 1))
                scoredentcans = sorted(zip(entcans[i], entscoresi), key=lambda (x, y): y, reverse=True)
                bestsubj = scoredentcans[0][0]
                if self.debug:
                    embed()
            ret[i, 0] = bestsubj
            # predict relation
            relcans = relsperent[ret[i, 0]][0] if ret[i, 0] in relsperent else []
            if len(relcans) == 0:
                bestrel = -1
            elif len(relcans) == 1:
                bestrel = relcans[0]
            else:
                if self.debug:
                    embed()
                relembs = self.renc.predict.transform(self.reltrans)(relcans)
                relscoresi = np.tensordot(qencforrel[i], relembs, axes=(0, 1))
                scoredrelcans = sorted(zip(relcans, relscoresi), key=lambda (x, y): y, reverse=True)
                bestrel = scoredrelcans[0][0]
            ret[i, 1] = bestrel
            if self.debug:
                embed()
            tt.progress(i, qencodings.shape[0], live=True)
        tt.tock("predicted")
        return ret


class NegIdxGen(object):
    def __init__(self, maxentid, maxrelid, relclose=None, subjclose=None,
                 relsperent=None, usefive=False, surrogates=None,
                 traincanspace=None, maskid=-1):
        self.maxentid = maxentid
        self.maxrelid = maxrelid
        self.maskid = maskid
        print "using relclose" if relclose is not None else "no relclose"
        print "using subjclose" if subjclose is not None else "no subjclose"
        print "using relsperent" if relsperent is not None else "no relsperent"
        print "using surrogates" if surrogates is not None else "no surrogates"
        print "using traincanspace" if traincanspace is not None else "no traincanspace"
        self.relclose = {k: set(v) for k, v in relclose.items()} if relclose is not None else None
        self.subjclose = {k: set(v) for k, v in subjclose.items()} if subjclose is not None else None
        self.relsperent = {k: set(v[0]) for k, v in relsperent.items()} if relsperent is not None else None
        self.surrogates = surrogates
        self.samprobf = lambda x: np.tanh(np.log(x + 1)/3)
        self.usefive = usefive
        self.minimal = 15
        self.traincanspace = traincanspace
        print "use five: {}".format(self.usefive)

    def __call__(self, datas, gold, negrate=0):
        nrate = 1 if negrate < 1 else negrate
        if self.traincanspace is not None:
            subjrand = self.sample_traincans(datas, gold[:, 0], self.maxentid)
        elif self.surrogates is not None:
            subjrand = self.sample_surrogates(datas, gold[:, 0], self.maxentid)
        else:
            subjrand = self.sample(gold[:, 0], self.subjclose, self.maxentid, negrate=nrate)
        if self.relsperent is not None:     # sample uber-close
            relrand = self.samplereluberclose(gold[:, 1], gold[:, 0], negrate=nrate)
        else:
            relrand = self.sample(gold[:, 1], self.relclose, self.maxrelid, negrate=nrate)
        ret = np.concatenate([subjrand, relrand], axis=-1)
        # embed()
        # TODO NEGATIVE SAMPLING OF RELATIONS FROM GOLD ENTITY'S RELATIONS
        return datas, ret.astype("int32")

    def new_samplereluberclose(self, relgold, entgold, negrate=1):
        if negrate > 1:
            return self.samplereluberclose_multi(relgold, entgold, negrate)
        ret = np.zeros_like(relgold, dtype="int32")
        for i in range(relgold.shape[0]):
            sampleset = (self.relsperent[entgold[i]] if entgold[i] in self.relsperent else set()) \
                .difference({relgold[i]})
            closesampleset = (self.relclose[relgold[i]] if relgold[i] in self.relclose else set())\
                .difference({relgold[i]}).difference(sampleset)
            addset = set(random.sample(closesampleset, min(len(closesampleset), max(0, self.minimal - len(sampleset)))))
            sampleset.update(addset)
            addset = set(random.sample(xrange(self.maxrelid + 1), max(1, self.minimal - len(sampleset))))
            sampleset.update(addset)
            sampleset = sampleset.difference({relgold[i]})
            ret[i] = random.sample(sampleset, 1)[0]
        ret = ret[:, np.newaxis]
        return ret

    def samplereluberclose(self, relgold, entgold, negrate=1):
        if negrate > 1:
            return self.samplereluberclose_multi(relgold, entgold, negrate)
        ret = np.zeros_like(relgold, dtype="int32")
        for i in range(relgold.shape[0]):
            uberclosesampleset = (self.relsperent[entgold[i]] if entgold[i] in self.relsperent else set())\
                .difference({relgold[i]})
            if np.random.random() < self.samprobf(len(uberclosesampleset)):
                ret[i] = random.sample(uberclosesampleset, 1)[0]
            else:
                completerandom = False
                if self.relclose is not None:
                    closesampleset = (self.relclose[relgold[i]] if relgold[i] in self.relclose else set())\
                        .difference({relgold[i]})
                    if np.random.random() < self.samprobf(len(closesampleset)):
                        ret[i] = random.sample(closesampleset, 1)[0]
                    else:
                        completerandom = True
                else:
                    completerandom = True
                if completerandom:
                    ret[i] = np.random.randint(0, self.maxrelid + 1)
        ret = np.expand_dims(ret, axis=1)
        return ret

    def samplereluberclose_multi(self, relgold, entgold, negrate):
        ret = np.zeros((relgold.shape[0], negrate), dtype="int32")
        for i in range(relgold.shape[0]):
            sampleset = self.relsperent[entgold[i]].difference({relgold[i]}) if entgold[i] in self.relsperent else set()
            closesampleset = self.relclose[relgold[i]].difference(sampleset).difference({relgold[i]}) if relgold[i] in self.relclose else set()
            addset = set(random.sample(closesampleset, min(len(closesampleset), max(0, negrate - len(sampleset)))))
            sampleset.update(addset)
            addset = set(random.sample(xrange(self.maxrelid + 1), max(1, negrate - len(sampleset))))
            sampleset.update(addset)
            sampleset = sampleset.difference({relgold[i]})
            ret[i, :] = random.sample(sampleset, negrate)
        return ret[:, :, np.newaxis]

    def old_samplereluberclose_multi(self, relgold, entgold, negrate):
        ret = np.zeros((relgold.shape[0], negrate), dtype="int32")
        for i in range(relgold.shape[0]):
            try:
                uberclosesampleset = self.relsperent[entgold[i]].difference({relgold[i]}) if entgold[i] in self.relsperent else set()
                closesampleset = self.relclose[relgold[i]].difference(uberclosesampleset).difference({relgold[i]}) if relgold[i] in self.relclose else set()
                randomsampleset = set()
                while len(randomsampleset) < negrate:
                    randomsampleset.update(set(random.sample(xrange(self.maxrelid + 1), 1))
                                               .difference({relgold[i]})
                                               .difference(uberclosesampleset)
                                               .difference(closesampleset))
                #randomsampleset = set(random.sample(xrange(self.maxrelid + 1), negrate * 2)).difference({relgold[i]}).difference(uberclosesampleset).difference(closesampleset)
                #randomsampleset = set(random.sample(randomsampleset, negrate))
                mergedsampleset = closesampleset.union(randomsampleset)
                numsam = negrate - len(uberclosesampleset) if len(uberclosesampleset) < negrate \
                                                           else min(len(mergedsampleset), int(round(len(uberclosesampleset) / 2.)))
                sampleset = set(random.sample(mergedsampleset, numsam))
                mergedsampleset = uberclosesampleset.union(sampleset).difference({relgold[i]})
                sampleset = set(random.sample(mergedsampleset, negrate))
                ret[i, :] = list(sampleset)
            except Exception, e:
                embed()
        return ret[:, :, np.newaxis]

    def new_sample(self, gold, closeset, maxid, negrate=1):
        if negrate > 1:
            return self.sample_multi(gold, closeset, maxid, negrate)
        # assert(gold.ndim == 2 and gold.shape[1] == 1)
        if closeset is None:
            return np.random.randint(0, maxid + 1, (gold.shape[0], 1))
        else:
            ret = np.zeros_like(gold)
            for i in range(gold.shape[0]):
                sampleset = closeset[gold[i]].difference({gold[i]}) if gold[i] in closeset else set()
                addset = set(random.sample(xrange(maxid + 1), max(1, self.minimal - len(sampleset))))
                sampleset.update(addset)
                sampleset = sampleset.difference({gold[i]})
                ret[i] = random.sample(sampleset, 1)[0]
            ret = ret[:, np.newaxis]
            return ret

    def sample(self, gold, closeset, maxid, negrate=1):
        if negrate > 1:
            return self.sample_multi(gold, closeset, maxid, negrate)
        # assert(gold.ndim == 2 and gold.shape[1] == 1)
        if closeset is None:
            return np.random.randint(0, maxid + 1, (gold.shape[0], 1))
        else:
            ret = np.zeros_like(gold)
            for i in range(gold.shape[0]):
                sample = gold[i]
                while sample == gold[i]:
                    sampleset = closeset[gold[i]] if gold[i] in closeset else []
                    samplefromset = len(sampleset) >= 4
                    if samplefromset:
                        sample = random.sample(sampleset, 1)[0]
                    else:
                        sample = np.random.randint(0, maxid + 1)
                ret[i] = sample
            ret = np.expand_dims(ret, axis=1)
            return ret.astype("int32")

    def sample_surrogates(self, data, gold, maxid):
        ret = np.zeros_like(gold)
        for i in range(data.shape[0]):
            surset = get_surrogates(data[i], self.surrogates, maskid=self.maskid)
            if len(surset) < 5:
                ret[i] = np.random.randint(0, maxid + 1)
            else:
                ret[i] = random.sample(surset, 1)[0]
        ret = np.expand_dims(ret, axis=1)
        return ret

    def sample_traincans(self, data, gold, maxid):
        ret = np.zeros_like(gold)
        for i in range(data.shape[0]):
            key = filter(lambda x: x != self.maskid, list(data[i, :, 0]))
            key = " ".join([str(x) for x in key])
            canset = self.traincanspace[key].difference({gold[i]}) if key in self.traincanspace else set()
            if len(canset) >= 4:
                ret[i] = random.sample(canset, 1)[0]
            elif self.surrogates is not None:
                surset = get_surrogates(data[i], self.surrogates, maskid=self.maskid)
                canset = canset.union(surset).difference({gold[i]})
                if len(canset) >= 5:
                    ret[i] = random.sample(canset, 1)[0]
                else:
                    ret[i] = np.random.randint(0, maxid + 1)
            else:
                ret[i] = np.random.randint(0, maxid + 1)
        ret = np.expand_dims(ret, axis=1)
        return ret

    def oldsample(self, gold, closeset, maxid, negrate=1):
        if negrate > 1:
            return self.sample_multi(gold, closeset, maxid, negrate)
        # assert(gold.ndim == 2 and gold.shape[1] == 1)
        if closeset is None:
            return np.random.randint(0, maxid + 1, (gold.shape[0], 1))
        else:
            ret = np.zeros_like(gold)
            for i in range(gold.shape[0]):
                sample = gold[i]
                while sample == gold[i]:
                    sampleset = closeset[gold[i]] if gold[i] in closeset else []
                    if self.usefive:
                        samplefromset = len(sampleset) >= 4
                    else:
                        samplefromset = np.random.random() < self.samprobf(len(sampleset))
                    if samplefromset:
                        sample = random.sample(sampleset, 1)[0]
                    else:
                        sample = np.random.randint(0, maxid + 1)
                ret[i] = sample
            ret = np.expand_dims(ret, axis=1)
            return ret.astype("int32")

    def sample_multi(self, gold, closeset, maxid, negrate):
        ret = np.zeros((gold.shape[0], negrate), dtype="int32")
        for i in range(gold.shape[0]):
            sampleset = closeset[gold[i]].difference({gold[i]}) if gold[i] in closeset else set()
            randomset = set(random.sample(xrange(maxid), max(1, negrate - len(sampleset) + 1)))
            sampleset = sampleset.union(randomset)
            sampleset = sampleset.difference({gold[i]})
            ret[i, :] = random.sample(sampleset, negrate)
        return ret[:, :, np.newaxis]

# python fullrank.py -numtestcans 400 -loadmodel ? -multiprune ?(1) -mode ?
# margin is still default (0.5), no need to specify
# mode must be matching mode of model --> will fail
# actual model hyperparams and training hyperparams need not specified for eval
# forcesubjincl must be false !!!
def run(negsammode="closest",   # "close" or "random"
        usetypes=True,
        mode="concat",      # "seq" or "concat" or "multi" or "multic"
        glove=True,
        embdim=100,
        charencdim=100,
        charembdim=50,
        encdim=400,
        bidir=False,
        charenc="rnn",  # "cnn" or "rnn"
        margin=0.5,
        lr=0.1,
        numbats=700,
        epochs=50,
        gradnorm=5.0,
        wreg=0.000000001,
        loadmodel="no",
        debug=False,
        debugtest=False,
        forcesubjincl=False,
        randsameval=0,          # random sampling experiment, not using the candidates pregenerated
        numtestcans=400,
        multiprune=1,
        checkdata=False,
        testnegsam=False,
        testmodel=False,        # just embed
        debugvalid=False,
        loss="margin",          # or "ce" or "multice"
        negrate=10,             # if loss is "multice"
        usefive=False,
        inspectloadedmodel=False,
        inspectpredictions=False,
        evalwithgoldsubjs=False,
        usesurrogates=True,
        usetraincans=True,
        numlayers=1,
        usetestregions=False,
        validontest=False,
        dropoutin=0.1,
        dropouth=0.0,
        dropoutd=0.0,
        zoneout=0.05,
        ):
    maskid = -1
    """
    # TODO remove
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    (subjmat, relmat), (subjdic, reldic), worddic, \
    subjinfo, (testsubjcans, relsperent) = readdata(wordlevel=True,
                                                    numtestcans=5)
    embed()
    """
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    (subjmat, relmat), (subjdic, reldic), worddic, \
    subjinfo, (testsubjcans, relsperent) = readdata(debug=debug,
                                                    numtestcans=numtestcans if numtestcans > 0 else None)

    rwd = {v: k for k, v in worddic.items()}

    if usetypes:
        print "building type matrix"
        typmat = buildtypmat(subjmat, subjinfo, worddic)
        subjmat = np.concatenate([typmat, subjmat], axis=1)
        typlen = typmat.shape[1]

    surrogates = None
    if usesurrogates:
        tt.tick("making surrogates")
        if usetypes:
            surmat, surtypmat, datatosur = generate_surrogate_entities(traindata, traingold, typmat=typmat, maskid=maskid)
        else:
            surmat, surtypmat, datatosur = generate_surrogate_entities(traindata, traingold,
                                                             maskid=maskid)

        surmat = wordmat2charmat(surmat, rwd=rwd, maskid=maskid, raretoken="<RARE>", maxlen=75)
        if usetypes:
            surmat = np.concatenate([surtypmat, surmat], axis=1)
        tt.tock("made surrogates")
        # udpate datatosur id's
        surrogateid = max(subjdic.values()) + 1
        datatosur = {k: {ve + surrogateid for ve in v} for k, v in datatosur.items()}
        # update subjdic
        for i in range(surmat.shape[0]):
            subjdic["surrogate_{}".format(surrogateid)] = surrogateid
            surrogateid += 1
        # update subjmat
        surmat = np.concatenate([surmat,
                                 maskid * np.ones((surmat.shape[0], subjmat.shape[1] - surmat.shape[1]))],
                                axis=1)
        subjmat = np.concatenate([subjmat, surmat], axis=0).astype("int32")

        surrogates = datatosur      # use surrogates in negidxgen
        #embed()

    relsamplespace = None
    subjsamplespace = None
    if negsammode == "closest" or negsammode == "close":
        relsamplespace, revind = buildrelsamplespace(relmat, worddic)
        subjsamplespace = loadsubjsamplespace()
    traincanspace = None
    if usetraincans:
        traincanspace = buildtraincanspace(traindata, maskid=maskid)
    tt.tock("data loaded")


    numwords = max(worddic.values()) + 1
    numsubjs = max(subjdic.values()) + 1
    numrels = max(reldic.values()) + 1
    numchars = 256

    nsrelsperent = relsperent if negsammode == "closest" else None

    nig = NegIdxGen(numsubjs-1, numrels-1,
                    relclose=relsamplespace,
                    subjclose=subjsamplespace,
                    relsperent=nsrelsperent,
                    usefive=usefive,
                    surrogates=surrogates,
                    maskid=maskid,
                    traincanspace=traincanspace,
                    )

    # negative matrices for multi ce training with negrate
    if loss == "multice":
        tt.tick("generating neg matrix for multi CE")
        _, negatives = nig(traindata, traingold, negrate=negrate)  # (batsize, negrate, 2)
        traintargets = np.concatenate([traingold[:, np.newaxis, :],
                                       negatives], axis=1)
        tt.tock("generated neg matrix")

    if testnegsam or checkdata:
        embed()

    if mode == "seq" or mode == "multi":
        decdim = encdim
    elif mode == "concat" or mode == "multic":
        decdim = encdim / 2
    else:
        raise Exception("unrecognized mode")

    print "{} mode: {} decdim".format(mode, decdim)

    # defining model
    if glove:
        wordemb = Glove(embdim).adapt(worddic)
    else:
        wordemb = WordEmb(dim=embdim, indim=numwords)

    charemb = VectorEmbed(indim=numchars, dim=charembdim)
    if charenc == "cnn":
        print "using CNN char encoder"
        charenc = CNNSeqEncoder(inpemb=charemb,
                                innerdim=[charencdim]*2, maskid=maskid,
                                stride=1)
    elif charenc == "rnn":
        print "using RNN char encoder"
        charenc = RNNSeqEncoder(inpemb=charemb, innerdim=charencdim,
                                zoneout=zoneout, dropout_in=dropoutin, dropout_h=dropouth) \
            .maskoptions(maskid, MaskMode.AUTO)
    else:
        raise Exception("no other character encoding modes available")

    if mode == "multi" or mode == "multic":
        wordenc = \
            SimpleSeq2MultiVec(inpemb=False, inpembdim=wordemb.outdim + charencdim,
                               innerdim=encdim, bidir=bidir, numouts=2, mode="seq")
    else:

        wordenc = RNNSeqEncoder(inpemb=False, inpembdim=wordemb.outdim + charencdim,
                                innerdim=[encdim]*numlayers, bidir=bidir,
                                zoneout=zoneout, dropout_in=dropoutin, dropout_h=dropouth).maskoptions(MaskMode.NONE)

    question_encoder = TwoLevelEncoder(l1enc=charenc, l2emb=wordemb,
                                       l2enc=wordenc, maskid=maskid)
    # encode predicate on word level
    predemb = SimpleSeq2Vec(inpemb=wordemb,
                            innerdim=decdim,
                            maskid=maskid,
                            bidir=bidir,
                            layers=1,
                            zoneout=zoneout, dropout_in=dropoutin, dropout_h=dropouth)
    #predemb.load(relmat)

    if usetypes:
        # encode subj type on word level
        subjtypemb = SimpleSeq2Vec(inpemb=wordemb,
                                   innerdim=int(np.ceil(decdim * 1./3)),
                                   maskid=maskid,
                                   bidir=bidir,
                                   layers=1,
                                   zoneout=zoneout, dropout_h=dropouth, dropout_in=dropoutin)
        # encode subject on character level
        subjemb = SimpleSeq2Vec(inpemb=charemb,
                                innerdim=int(np.floor(decdim * 2./3)),
                                maskid=maskid,
                                bidir=bidir,
                                layers=1,
                                zoneout=zoneout, dropout_h=dropouth, dropout_in=dropoutin)
        subjemb = TypedSubjBlock(typlen, subjemb, subjtypemb)
    else:
        # encode subject on character level
        subjemb = SimpleSeq2Vec(inpemb=charemb,
                                innerdim=decdim,
                                maskid=maskid,
                                bidir=bidir,
                                layers=1,
                                zoneout=zoneout, dropout_h=dropouth, dropout_in=dropoutin)
    #subjemb.load(subjmat)
    if testmodel:
        embed()
    # package
    if mode == "seq":
        lb = SeqLeftBlock(question_encoder)
        rb = RightBlock(subjemb, predemb)
    elif mode == "concat":
        lb = ConcatLeftBlock(question_encoder, decdim)
        rb = RightBlock(subjemb, predemb)
    elif mode == "multi" or mode == "multic":
        lb = MultiLeftBlock(question_encoder, decdim, mode)
        rb = RightBlock(subjemb, predemb)
    else:
        raise Exception("unrecognized mode")

    if loss == "margin":
        scoref = CosineDistance(dropout=dropoutd)
    elif loss == "ce":
        scoref = DotDistance()      # GenDotDistance() ??
    elif loss == "multice":
        scoref = CosineDistance()
    else:
        raise Exception("unknown option for loss argument")
    scorer = SeqMatchScore(lb, rb, scorer=scoref,
                           aggregator=lambda x: x,
                           argproc=lambda x, y, z: ((x,), (y, z)))
        # returns (batsize, 2) scores, one for subj and one for pred

    if loss == "margin":
        obj = lambda p, n: T.sum((n - p + margin).clip(0, np.infty), axis=1)
    elif loss == "ce":
        def ce_loss(p, n):
            sx = T.concatenate([p[:, 0:1], n[:, 0:1]], axis=1)
            px = T.concatenate([p[:, 1:2], n[:, 1:2]], axis=1)
            sxsm = T.nnet.softmax(sx)
            pxsm = T.nnet.softmax(px)
            ret = -T.log(sxsm[:, 0]+1e-6) - T.log(pxsm[:, 0]+1e-6)
            #ret = sxsm[:, 0] * pxsm[:, 0]
            #embed()
            return ret
        obj = ce_loss

    class PreProc(object):
        def __init__(self, subjmat, relmat):
            self.ef = PreProcEnt(subjmat)
            self.rf = PreProcEnt(relmat)

        def __call__(self, data, gold):     # gold: idxs-(batsize, 2)
            st = self.ef(gold[:, 0])[0][0]
            rt = self.rf(gold[:, 1])[0][0]
            return (data, st, rt), {}

    class PreProcE(object):
        def __init__(self, subjmat, relmat):
            self.ef = PreProcEnt(subjmat)
            self.rf = PreProcEnt(relmat)

        def __call__(self, x):
            subjslice = self.ef(x[:, 0])[0][0]
            relslice = self.rf(x[:, 1])[0][0]
            return (subjslice, relslice), {}

    class PreProcEnt(object):
        def __init__(self, mat):
            self.entmat = Val(mat)

        def __call__(self, x):
            return (self.entmat[x],), {}

    class PreProcMultiCE(object):
        def __init__(self, subjmat, relmat):
            self.ef = PreProcEnt(subjmat)
            self.rf = PreProcEnt(relmat)

        def __call__(self, data, targets):  # targets: (batsize, negrate+1, 2)
            st = self.ef(targets[:, :, 0])[0][0]
            rt = self.rf(targets[:, :, 1])[0][0]
            return (data, st, rt), {}

    transf = PreProc(subjmat, relmat)

    if debug:
        embed()

    # BEWARE: extremely ugly
    def get_validate_acc(savepath):
        predictor = {0: None}
        offset = {0: 0}
        if validontest:
            print "VALIDONTEST !!!!!!!!!!! WRONG !"
            subjcans = testsubjcans     #pickle.load(open("../../../../data/simplequestions/clean/testcans{}c.pkl".format(numtestcans), "r"))
            vdata = testdata
        else:
            subjcans = pickle.load(open("../../../../data/simplequestions/clean/validcans{}c.pkl".format(numtestcans), "r"))
            vdata = validdata
        assert(len(subjcans) == vdata.shape[0])

        def validate_acc(*sampleinps):
            if predictor[0] is None:    # reload model for a whole iter
                if os.path.isfile(savepath):
                    m = SeqMatchScore.load(savepath)
                    predictor[0] = CustomPredictor(questionencoder=m.l.inner,
                                                entityencoder=m.r.subjenc,
                                                relationencoder=m.r.predenc,
                                                mode=mode,
                                                enttrans=transf.ef,
                                                reltrans=transf.rf,
                                                debug=debugtest,
                                                subjinfo=subjinfo,
                                                silent=True)
                else:
                    raise Exception("could not load model")
            multipru = multiprune
            relspere = relsperent
            #tt = ticktock("External Accuracy Validator")
            qmat = sampleinps[0]
            cans = subjcans[offset[0]: offset[0] + qmat.shape[0]]
            amat = sampleinps[1]
            #tt.tick("predicting")
            pred = predictor[0].predict(qmat, entcans=cans, relsperent=relspere, multiprune=multipru)
            #tt.tock("predicted")
            #tt.tick("evaluating")
            evalmat = amat == pred
            subjacc = np.sum(evalmat[:, 0]) * 1. / evalmat.shape[0]
            predacc = np.sum(evalmat[:, 1]) * 1. / evalmat.shape[0]
            totalacc = np.sum(np.sum(evalmat, axis=1) == 2) * 1. / evalmat.shape[0]
            #tt.tock("evaluated")
            if offset[0] == 0:
                #embed()
                pass
            offset[0] += qmat.shape[0]
            if offset[0] == len(subjcans):
                #embed()
                print "RESET OFFSET IN VALIDATOR"
                offset[0] = 0
                predictor[0] = None
            return totalacc, subjacc, predacc
        return validate_acc

    savep = None

    if epochs > 0 and loadmodel == "no":
        tt.tick("training")
        pathexists = True
        while pathexists:
            saveid = "".join([str(np.random.randint(0, 10)) for i in range(4)])
            print("CHECKPOINTING AS: {}".format(saveid))
            savep = "fullrank{}.model".format(saveid)
            pathexists = os.path.isfile(savep)
        extvalidf = get_validate_acc(savep)

        if loss == "multice":  # normal softmax with fixed negatives
            class ScorerMultiCEWrap(Block):
                def __init__(self, scorer, subjmat, relmat, **kw):
                    super(ScorerMultiCEWrap, self).__init__(**kw)
                    self.scorer = scorer
                    self.transf = PreProcMultiCE(subjmat, relmat)
                    #self.transedblock = TransWrapBlock(scorer, transf)

                def apply(self, data, targets): # targets: idx~(batsize, negrate+1, 2)
                    transedinps, _ = self.transf(data, targets)
                    d, s, r = transedinps[0], transedinps[1], transedinps[2]
                    s = s.dimswap(1, 0)
                    r = r.dimswap(1, 0)     # (negrate+1, batsize, 2, ?)
                    qencs = self.scorer.l(d)
                    scores, _ = T.scan(self.rec, sequences=[s, r],
                                     non_sequences=qencs,
                                     outputs_info=[None])
                    scores = scores.dimshuffle(1, 2, 0)   # (batsize, 2, negrate+1)
                    probs = T.softmax(scores)        # (batsize, 2, negrate+1)
                    return probs

                def rec(self, s, r, qenc):    # target: idx~(batsize, 2)
                    subjpredenc = self.scorer.r(s, r)
                    score = self.scorer.innerapply(qenc, subjpredenc)
                    return score

            scorerMultiCeWrap = ScorerMultiCEWrap(scorer, subjmat, relmat)

            cegold = np.zeros((traindata.shape[0], 2)).astype("int32")
            tt.msg("doing multi CE")
            nscorer = scorerMultiCeWrap.train([traindata, traintargets], cegold) \
                .seq_cross_entropy().adadelta(lr=lr).l2(wreg).grad_total_norm(gradnorm) \
                .validate_on([validdata, validgold], np.ones_like(validgold)).extvalid(extvalidf) \
                .autosavethis(scorer, savep).writeresultstofile(savep + ".progress.tsv") \
                .takebest(lambda x: x[1], save=True, smallerbetter=False) \
                .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)
            tt.tock("trained").tick()
            print("SAVED AS: {}".format(saveid))
        else:
            if validontest:
                print "VALIDATING ON TEST SET"
                vd = testdata
                vg = testgold
            else:
                vd = validdata
                vg = validgold
            nscorer = scorer.nstrain([traindata, traingold]).transform(transf) \
                .negsamplegen(nig) \
                .objective(obj).adagrad(lr=lr).l2(wreg).grad_total_norm(gradnorm) \
                .validate_on([vd, vg]).extvalid(extvalidf) \
                .autosavethis(scorer, savep).writeresultstofile(savep+".progress.tsv") \
                .takebest(lambda x: x[2], save=True, smallerbetter=False) \
                .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)
            tt.tock("trained").tick()

            # saving
            #scorer.save("fullrank{}.model".format(saveid))
            #embed()
            print("SAVED AS: {}".format(saveid))

    # LOAD MODEL FOR EVAL
    loadp = savep
    if loadmodel != "no":
        loadp = "fullrank{}.model".format(loadmodel)
    if os.path.isfile(loadp + ".best"):
        loadp = loadp + ".best"

    tt.tick("loading model")
    m = SeqMatchScore.load(loadp)

    if inspectloadedmodel:
        embed()
    question_encoder = m.l.inner
    subjemb = m.r.subjenc
    predemb = m.r.predenc
    tt.tock("loaded model")

    # EVALUATION
    testregions = None
    subjwordmat = None
    if usetestregions:
        _, _, _, \
        (subjwordmat, _), (_, _), _, \
        _, (_, _) = readdata(wordlevel=True, numtestcans=5)
        testregions = pickle.load(open("testregions.smout.pkl"))
    predictor = CustomPredictor(questionencoder=question_encoder,
                                entityencoder=subjemb,
                                relationencoder=predemb,
                                mode=mode,
                                enttrans=transf.ef,
                                reltrans=transf.rf,
                                debug=debugtest,
                                subjinfo=subjinfo,
                                testregions=testregions,
                                subjmat=subjwordmat)

    tt.tick("predicting")
    if forcesubjincl:       # forces the intended subject entity to be among candidates
        print "FORCED SUBJ INCL !!! WRONG !!!!!!"
        for i in range(len(testsubjcans)):
            if testgold[i, 0] not in testsubjcans[i]:
                testsubjcans[i].append(testgold[i, 0])

    if randsameval > 0:     # generate random sampling eval data
        print "RAND SAM EVAL!!! WRONG !!!!!!"
        testsubjcans = np.random.randint(0, numsubjs, (testgold.shape[0], randsameval))
        testrelcans = np.random.randint(0, numrels, (testgold.shape[0], randsameval))
        testsubjcans = np.concatenate([testgold[:, 0:1], testsubjcans], axis=1)
        testrelcans = np.concatenate([testgold[:, 1:2], testrelcans], axis=1)
        testsubjcans = testsubjcans.tolist()
        testrelcans = testrelcans.tolist()
        prediction = predictor.predict(testdata, entcans=testsubjcans,
                                       relcans=testrelcans)
    else:
        if not evalwithgoldsubjs:
            prediction = predictor.predict(testdata, entcans=testsubjcans,
                            relsperent=relsperent, multiprune=multiprune)
        else:
            print "GOLD SUBJECTS !!!! WRONG !!!!"
            prediction = predictor.predict(testdata, entcans=testsubjcans,
                            relsperent=relsperent, multiprune=0,
                            goldsubjects=list(testgold[:, 0]))
    tt.tock("predicted")
    tt.tick("evaluating")
    evalmat = prediction == testgold
    subjacc = np.sum(evalmat[:, 0]) * 1. / evalmat.shape[0]
    predacc = np.sum(evalmat[:, 1]) * 1. / evalmat.shape[0]
    totalacc = np.sum(np.sum(evalmat, axis=1) == 2) * 1. / evalmat.shape[0]
    print "Test results ::::::::::::::::"
    print "Total Acc: \t {}".format(totalacc)
    print "Subj Acc: \t {}".format(subjacc)
    print "Pred Acc: \t {}".format(predacc)
    tt.tock("evaluated")

    def subjinspect(subjrank, gold):
        ret = [(("GOLD - " if gold == x else "       ") +
                subjinfo[x][0] + " (" + " ".join(subjinfo[x][1]) + ")" +
                str(subjinfo[x][3]) + " rels",
                y) if x in subjinfo else (x, y)
               for x, y in subjrank]
        return ret

    def inspectsubjs(hidecorrect=False, hidenotincan=False, shownotincan=False):
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.subjranks)):
            subjx = testgold[i, 0]
            subjrank = predictor.subjranks[i]
            if subjx == subjrank[0][0] and hidecorrect:     # only look for errors
                continue
            if subjx not in [k for k, v in subjrank]:
                if hidenotincan:
                    continue
            if shownotincan and subjx in [k for k, v in subjrank]:
                continue
            print "test question {}: {} \t GOLD: {}".format(i,
                                                            wordids2string(testdata[i, :, 0], rwd),
                                                            "{} ({}) - {} rels --- {}".format(
                                                                *([subjinfo[subjx][0],
                                                                   subjinfo[subjx][1],
                                                                   subjinfo[subjx][3],
                                                                   subjinfo[subjx][2]]
                                                                  if subjx in subjinfo
                                                                  else ["<UNK>", "<UNK>", "<UNK>", "<UNK>"])
                                                            ))
            inspres = subjinspect(subjrank, subjx)
            i = 1
            for inspre in inspres:
                print "{}:\t{}\t{}".format(i, inspre[1], inspre[0])
                if i % 50 == 0:
                    inp()
                i += 1
            inp()

    def inspectpreds(hidecorrect=False):
        rwd = {v: k for k, v in worddic.items()}
        for i in range(len(predictor.relranks)):
            relx = testgold[i, 1]
            subjx = testgold[i, 0]
            relrank = predictor.relranks[i]
            if relx == relrank[0][0] and hidecorrect:
                continue
            print "test question {}: {} \t GOLD: {}".format(i,
                                                            wordids2string(testdata[i, :, 0], rwd),
                                                            wordids2string(relmat[relx, :], rwd))
            inspres = [(("GOLD - " if relx == x else "        ") +
                        wordids2string(relmat[x], rwd), y) for x, y in relrank]
            i = 1
            for inspre in inspres:
                print "{}:\t{}\t{}".format(i, inspre[1], inspre[0])
                if i % 50 == 0:
                    inp()
                i += 1
            inp()

    print "##### DONE #####"
    if inspectpredictions:
        embed()


def generate_surrogate_entities(data, gold, typmat=None, topngram=4, maskid=-1):
    # data: (numsam, seqlen, wordlen+1),
    # gold: (numsam, 2)
    # typmat: (numsubjs, len) - words
    from collections import OrderedDict
    data = data[:, :, 0]        # only words
    gold = list(gold[:, 0])
    ngramids = OrderedDict()
    ngramtypes = {}
    maxlen = 0
    ngramaddr = 0
    datatosur = {}
    for i in range(data.shape[0]):      # for each question
        question = list(data[i])
        qs = " ".join([str(qwid) for qwid in question if qwid != maskid])
        if not qs in datatosur:
            datatosur[qs] = set()
        j = 0
        stop = False
        while not stop:
            if j+1 >= len(question) or question[j] == maskid:
                break
            for k in range(1, topngram):
                try:
                    if j+k+1 > len(question) or question[j+k-1] == maskid:
                        break
                    # TODO: check n-gram overlap with gold words
                    ngram = tuple(question[j:j+k])
                    if ngram not in ngramids:
                        ngramids[ngram] = ngramaddr
                        ngramaddr += 1
                    # add ngramaddr to qs
                    datatosur[qs].add(ngramids[ngram])
                    # types
                    if typmat is not None:
                        ngramtypes[ngram] = typmat[gold[i]]
                except Exception, e:
                    embed()
                maxlen = max(maxlen, len(ngram))
            j += 1
    surrogatetypmat = None
    surrogatewordmat = np.ones((len(ngramids), maxlen), dtype="int32") * maskid
    if typmat is not None:
        surrogatetypmat = np.ones((len(ngramids), typmat.shape[1]), dtype="int32") * maskid
    i = 0
    for ngram in ngramids.keys():
        surrogatewordmat[i, :len(ngram)] = ngram
        if typmat is not None:
            surrogatetypmat[i, :] = ngramtypes[ngram]
        i += 1
    randomtypmat = True
    if randomtypmat and typmat is not None:
        typmatidx = np.random.randint(0, typmat.shape[0], (surrogatewordmat.shape[0],))
        surrogatetypmat = typmat[typmatidx]
    return surrogatewordmat, surrogatetypmat, datatosur


def get_surrogates(datarow, datatosur, maskid=-1):
    # return set of surrogate integer ids
    key = list(datarow[:, 0])   # only words
    key = " ".join([str(x) for x in key if x != maskid])
    ret = datatosur[key] if key in datatosur else set()
    return ret



if __name__ == "__main__":
    data = np.random.randint(0, 1000, (5,6,7))
    gold = np.random.randint(0, 10, (5, 2))
    generate_surrogate_entities(data, gold)
    argprun(run)