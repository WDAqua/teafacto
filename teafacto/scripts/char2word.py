import pickle

import numpy as np

from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.match import MatchScore, CosineDistance, EuclideanDistance
from teafacto.util import argprun, ticktock


def run(
        epochs=10,
        numbats=100,
        negrate=1,
        lr=0.1,
        embdim=50,
        encdim=50,
        wreg=0.00005,
        marginloss=False,
        margin=1.,
        cosine=False,
        bidir=False,
    ):
    tt = ticktock("script")
    # get glove words
    g = Glove(encdim)
    words = g.D.keys()
    maxwordlen = 0
    for word in words:
        maxwordlen = max(maxwordlen, len(word))
    chars = set("".join(words))
    chars.add(" ")
    print "{} words, maxlen {}, {} characters in words".format(len(words), maxwordlen, len(chars))
    # get char word matrix
    chardic = dict(zip(chars, range(len(chars))))
    pickle.dump(chardic, open("glove2c2w.chardic.pkl", "w"))
    charwordmat = -np.ones((len(words)+1, maxwordlen), dtype="int32")
    charwordmat[0, 0] = chardic[" "]
    for i in range(0, len(words)):
        word = words[i]
        charwordmat[i+1, :len(word)] = [chardic[x] for x in word]
    print charwordmat[0]
    # encode characters
    cwenc = SimpleSeq2Vec(indim=len(chars),
                          inpembdim=embdim,
                          innerdim=encdim/2 if bidir else encdim,
                          maskid=-1,
                          bidir=bidir)
    dist = CosineDistance() if cosine else EuclideanDistance() #DotDistance()
    print "using " + str(dist)
    scorer = MatchScore(cwenc, g.block, scorer=dist)

    '''
    scorer.train([charwordmat, np.arange(len(words)+1)], np.ones((charwordmat.shape[0],), dtype="int32") * (-1 if cosine else 1))\
        .linear_objective().adagrad(lr=lr).l2(wreg)\
        .train(numbats=numbats, epochs=epochs)

    #embed()
    '''
    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):
            return datas, np.random.randint(self.min, self.max, gold.shape).astype("int32")

    if marginloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)
    else:
        obj = lambda p, n: n - p

    nscorer = scorer.nstrain([charwordmat, np.arange(len(words)+1)])\
        .negsamplegen(NegIdxGen(len(words))).negrate(negrate)\
        .objective(obj).adagrad(lr=lr).l2(wreg)\
        .train(numbats=numbats, epochs=epochs)

    cwenc.save("glove2c2w.block")



if __name__ == "__main__":
    argprun(run)