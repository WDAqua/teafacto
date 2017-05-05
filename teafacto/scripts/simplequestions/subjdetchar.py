import numpy as np

from teafacto.blocks.seq.oldseqproc import SimpleVec2Idx, MemVec2Idx
from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.seq.enc import Seq2Idx
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.memory import LinearGateMemAddr, DotMemAddr
from teafacto.scripts.simplequestions.subjdet import _readdata, ents2labels
from teafacto.util import argprun


def readdata(p):
    train, valid, test, x, newdic = _readdata(p)
    return train, valid, test, x["chardic"], newdic


def getmemdata(reldic, worddic, labelp="../../../data/simplequestions/labels.map"):    # updates worddic with words found in relation
    rels = ents2labels(labelp, reldic)
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
    #embed()

def getdic2glove(worddic, dim=50):
    g = Glove(dim)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    return d2g, newdic


def getcharmemdata(reldic, chardic, maxchar=70, maxwords=30,
                   labelp="../../../data/simplequestions/labels.map"):
    rels = ents2labels(labelp, reldic, maxwords=maxwords)
    rels = map(lambda (x, y): (" ".join(x), y), rels)
    maxlen = 0
    prevc = -1
    allrelchars = set()
    for rel, c in rels:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(rel))
        for relchar in rel:
            allrelchars.add(relchar)
    charsnotinchardic = allrelchars.difference(set(chardic.keys()))
    nextid = 0
    for cnic in charsnotinchardic:
        while nextid in chardic.values():
            nextid += 1
        chardic[cnic] = nextid
    maxlen = min(maxlen, maxchar)
    retmat = np.zeros((len(rels), maxlen)).astype("int32") - 1
    for rel, k in rels:
        rel = [chardic[c] for c in rel[:min(len(rel), maxchar)]]
        retmat[k, :len(rel)] = rel
    return retmat


def evaluate(pred, gold):
    return np.sum(gold == pred) * 100. / gold.shape[0]


def run(
        epochs=10,
        numbats=100,
        numsam=10000,
        lr=0.1,
        datap="../../../data/simplequestions/datamat.char.pkl",
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        mem=False,
        sameenc=False,
        memaddr="dot",
        memattdim=100,
        membidir=False,
        memlayers=1,
        memmaxwords=5,
        memmaxchars=20,
        layers=1,
        ):

    (traindata, traingold), (validdata, validgold), (testdata, testgold), chardic, entdic\
        = readdata(datap)

    if mem:
        memdata = getcharmemdata(entdic, chardic, maxwords=memmaxwords, maxchar=memmaxchars)

    print traindata.shape, testdata.shape

    numchars = max(chardic.values()) + 1
    numrels = max(entdic.values()) + 1
    print numchars, numrels

    if bidir:
        encinnerdim = [innerdim/2]*layers
    else:
        encinnerdim = [innerdim]*layers

    enc = SimpleSeq2Vec(indim=numchars, inpembdim=None, innerdim=encinnerdim, maskid=-1, bidir=bidir)

    if mem:
        if membidir:
            innerdim = [innerdim/2]*memlayers
        else:
            innerdim = [innerdim]*memlayers
        memindim = numchars
        memenc = SimpleSeq2Vec(indim=memindim, inpembdim=None, innerdim=innerdim, maskid=-1, bidir=membidir)
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