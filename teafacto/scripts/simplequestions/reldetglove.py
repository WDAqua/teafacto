import re

import numpy as np
import pickle
from IPython import embed

from teafacto.blocks.seq.oldseqproc import SimpleVec2Idx, MemVec2Idx
from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.seq.enc import Seq2Idx, Seq2Vec
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.memory import LinearGateMemAddr, DotMemAddr
from teafacto.util import argprun


def readdata(p):
    x = pickle.load(open(p))
    def preprocessforreldet(x, numents):
        goldmat = x[1]
        return x[0], goldmat[:, 1] - numents
    entdic = x["entdic"]
    numents = x["numents"]
    newdic = {}
    for k, v in entdic.items():
        if v >= numents:
            newdic[k] = v - numents
    train = preprocessforreldet(x["train"], numents)
    valid = preprocessforreldet(x["valid"], numents)
    test  = preprocessforreldet(x["test"],  numents)
    return train, valid, test, x["worddic"], newdic


def getmemdata(reldic, worddic):    # updates worddic with words found in relation
    rels = sorted(reldic.items(), key=lambda (x, y): y)
    rels = map(lambda (x, y): (filter(lambda x: len(x) > 0, re.split("[\W_]", x)), y), rels)
    allrelwords = set()
    maxlen = 0
    prevc = -1
    for rel, c in rels:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(rel))
        for relw in rel:
            allrelwords.add(relw)
    relwordsnotinworddic = allrelwords.difference(set(worddic.keys()))
    for rwniw in relwordsnotinworddic:
        worddic[rwniw] = len(worddic)
    ret = [[worddic[w] for w in rel] for (rel, _) in rels]
    retmat = np.zeros((len(rels), maxlen)).astype("int32") - 1
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


def getdic2glove(worddic, dim=50, trainfrac=0.0):
    g = Glove(dim, trainfrac=trainfrac)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    return d2g, newdic, g


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
        embdim=50,
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        sameenc=False,
        memaddr="dot",
        memattdim=100,
        layers=1,
        embtrainfrac=0.0,
        mem=False,
        membidir=False,
        memlayers=1,
        ):
    """ Memory match-based glove-based word-level relation classification """

    (traindata, traingold), (validdata, validgold), (testdata, testgold), worddic, entdic\
        = readdata(datap)

    # get words from relation names, update word dic
    memdata = getmemdata(entdic, worddic)

    # get glove and transform word mats to glove index space
    d2g, newdic, glove = getdic2glove(worddic, dim=embdim, trainfrac=embtrainfrac)
    traindata, validdata, testdata, memdata = \
        [np.vectorize(d2g)(x) for x in [traindata, validdata, testdata, memdata]]

    print traindata.shape, testdata.shape

    numwords = max(worddic.values()) + 1    # don't use this, use glove
    numrels = max(entdic.values()) + 1

    if bidir:
        encinnerdim = [innerdim/2]*layers
    else:
        encinnerdim = [innerdim]*layers

    wordemb = glove.block
    rnn, lastdim = SimpleSeq2Vec.makernu(embdim, encinnerdim, bidir=bidir)
    enc = Seq2Vec(wordemb, rnn, maskid=-1)

    if mem:
        memembdim = embdim
        if membidir:
            innerdim = [innerdim/2]*memlayers
        else:
            innerdim = [innerdim]*memlayers
        memrnn, memlastdim = SimpleSeq2Vec.makernu(memembdim, innerdim, bidir=membidir)
        memenc = Seq2Vec(wordemb, memrnn, maskid=-1)
        if memaddr is None or memaddr == "dot":
            memaddr = DotMemAddr
        elif memaddr == "lin":
            memaddr = LinearGateMemAddr
        dec = MemVec2Idx(memenc, memdata, memdim=innerdim, memaddr=memaddr, memattdim=memattdim)
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