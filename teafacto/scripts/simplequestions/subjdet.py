from contextlib import closing
from multiprocessing import Pool, cpu_count

import numpy as np
import pickle
from IPython import embed

from teafacto.blocks.seq.oldseqproc import SimpleVec2Idx, MemVec2Idx
from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.seq.enc import Seq2Idx
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.memory import LinearGateMemAddr, DotMemAddr
from teafacto.util import argprun, tokenize


def _readdata(p):
    x = pickle.load(open(p))
    def preprocessforsubjdet(x):
        goldmat = x[1]
        return x[0], goldmat[:, 0]
    entdic = x["entdic"]
    numents = x["numents"]
    newdic = {}
    for k, v in entdic.items():
        if v < numents:
            newdic[k] = v
    train = preprocessforsubjdet(x["train"])
    valid = preprocessforsubjdet(x["valid"])
    test  = preprocessforsubjdet(x["test"])
    return train, valid, test, x, newdic


def readdata(p):
    train, valid, test, x, newdic = _readdata(p)
    return train, valid, test, x["worddic"], newdic, x


def loadlabels(labelp="../../../data/simplequestions/labels.map"):
    ret = {}
    for line in open(labelp):
        ns = line[:-1].split("\t")
        ret[ns[0]] = ns[1]
    return ret


def ents2labels(labelp, entdic, maxwords=50, parallel=True):
    labeldic = loadlabels(labelp)
    wolabels = set()
    ents = sorted(entdic.items(), key=lambda (x, y): y)
    if parallel:
        with closing(Pool(cpu_count() - 1)) as p:
            ents = p.map(MapEnts2labels(labeldic, wolabels=wolabels, maxlen=maxwords), ents)
    else:
        ents = map(MapEnts2labels(labeldic, wolabels=wolabels, maxlen=maxwords), ents)
        print "%d entities have no labels" % len(wolabels)
    return ents


class MapEnts2labels():
    def __init__(self, labeldic, wolabels=set(), maxlen=50):
        self.labeldic = labeldic
        self.wolabels = wolabels
        self.maxlen = maxlen

    def __call__(self, x):
        ret = tokenize(self.labelfy(x[0]))
        ret = ret[:min(self.maxlen, len(ret))]
        return ret, x[1]

    def labelfy(self, x):
        if x in self.labeldic:
            return self.labeldic[x]
        else:
            self.wolabels.add(x)
            return x


def getmemdata(entdic, worddic,
               labelp="../../../data/simplequestions/labels.map",
               maxwords=30):    # updates worddic with words found in entity labels
    ents = ents2labels(labelp, entdic, maxwords=maxwords)
    allentwords = set()
    maxlen = 0
    prevc = -1
    for ent, c in ents:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(ent))
        for entw in ent:
            allentwords.add(entw)
    maxlen = min(maxlen, maxwords)
    entwordsnotinworddic = allentwords.difference(set(worddic.keys()))
    for ewniw in entwordsnotinworddic:
        worddic[ewniw] = len(worddic)
    ret = [[worddic[w] for w in ent] for (ent, _) in ents]
    retmat = np.zeros((len(ents), maxlen)).astype("int32") - 1
    i = 0
    for r in ret:
        retmat[i, :len(r)] = r
        i += 1
    return retmat


def toglove(wordmat, worddic, dim=50):
    g = Glove(dim)
    gws = set(g.D.keys())
    wdws = set(worddic.keys())
    diff = wdws.difference(gws)
    # gather states about diff
    diffcounts = {worddic[k]: 0 for k in diff}
    total = 0
    moretal = 0
    for i in range(wordmat.shape[0]):
        for j in range(wordmat.shape[1]):
            if wordmat[i, j] >= 0:
                total += 1
                if wordmat[i, j] in diffcounts:
                    diffcounts[wordmat[i, j]] += 1
                    moretal += 1
    diffcounts = sorted(diffcounts.items(), key=lambda (k, v): v, reverse=True)
    print "%d words unknown by Glove of %d total words" % (moretal, total)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    newmat = np.vectorize(d2g)(wordmat)
    revgdic = {v: k for k, v in g.D.items()}
    embed()

def getdic2glove(worddic, dim=50):
    g = Glove(dim)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    return d2g, newdic


def getcharmemdata(reldic):
    rels = sorted(reldic.items(), key=lambda (x, y): y)
    maxlen = 0
    prevc = -1
    for rel, c in rels:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(rel))
    retmat = np.zeros((len(rels), maxlen)).astype("int32") - 1
    for rel, c in rels:
        retmat[c, :len(rel)] = map(ord, rel)
    return retmat


def evaluate(pred, gold):
    return np.sum(gold == pred) * 100. / gold.shape[0]


def run(
        epochs=10,
        numbats=100,
        numsam=10000,
        lr=0.1,
        datap="../../../data/simplequestions/datamat.word.pkl",
        embdim=100,
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        mem=False,
        dynmem=False,
        sameenc=False,
        memaddr="dot",
        memattdim=100,
        membidir=False,
        memlayers=1,
        memmaxwords=5,
        layers=1,
        ):

    (traindata, traingold), (validdata, validgold), (testdata, testgold), worddic, entdic, x\
        = readdata(datap)

    if mem:
        if not dynmem:
            memdata = getmemdata(entdic, worddic, maxwords=memmaxwords)
        else:
            pass    #TODO

    print traindata.shape, testdata.shape

    numwords = max(worddic.values()) + 1
    numrels = max(entdic.values()) + 1
    print numwords, numrels

    if bidir:
        encinnerdim = [innerdim/2]*layers
    else:
        encinnerdim = [innerdim]*layers

    enc = SimpleSeq2Vec(indim=numwords,
                        inpembdim=embdim,
                        innerdim=encinnerdim,
                        maskid=-1,
                        bidir=bidir)

    if mem:
        memembdim = embdim
        if membidir:
            innerdim = [innerdim/2]*memlayers
        else:
            innerdim = [innerdim]*memlayers
        memindim = numwords
        memembdim = embdim
        memenc = enc if sameenc \
                     else SimpleSeq2Vec(indim=memindim,
                                        inpembdim=memembdim,
                                        innerdim=innerdim,
                                        maskid=-1,
                                        bidir=membidir)
        if memaddr is None or memaddr == "dot":
            memaddr = DotMemAddr
        elif memaddr == "lin":
            memaddr = LinearGateMemAddr
        dec = MemVec2Idx(memenc, memdata, memdim=innerdim,
                         memaddr=memaddr, memattdim=memattdim)
    else:
        dec = SimpleVec2Idx(indim=innerdim, outdim=numrels)

    m = Seq2Idx(enc, dec)

    m = m.train([traindata], traingold).adagrad(lr=lr).l2(wreg).grad_total_norm(1.0).cross_entropy()\
        .validate_on([validdata], validgold).accuracy().cross_entropy().takebest()\
        .train(numbats=numbats, epochs=epochs)

    pred = m.predict(testdata)
    print pred.shape
    evalres = evaluate(np.argmax(pred, axis=1), testgold)
    print str(evalres) + "%"


if __name__ == "__main__":
    argprun(run)