import numpy as np
import pandas as pd
import re
from unittest import TestCase

from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.lang.wordembed import Glove
from teafacto.search import SeqEncDecSearch


def word2int(word):
    return [ord(letter)-96 if letter is not " " else 0 for letter in word]


def words2ints(words):
    wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    data = wldf.values.astype("int32")
    del wldf
    return data


def int2word(ints):
    chars = [chr(i+96) if i > 0 else " " for i in ints]
    return "".join(chars)


def ints2words(ints):
    return [int2word(x) for x in ints]


def shiftdata(x):
    return np.concatenate([np.zeros_like(x[:, 0:1]), x[:, :-1]], axis=1)


class TestSeqDecSearch(TestCase):
    def test_seqdecatt(  # seems to work
            wreg=0.00001,  # TODO: regularization other than 0.0001 first stagnates, then goes down
            epochs=50,
            numbats=20,
            lr=0.1,
            statedim=50,
            encdim=50,
            attdim=50,
            startsym=0,
    ):
        # get words
        vocsize = 27
        embdim = 50
        lm = Glove(embdim, 2000)
        allwords = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
        words = allwords[1000:]
        vwords = allwords[:1000]
        data = words2ints(words)
        sdata = shiftdata(data)
        vdata = words2ints(vwords)
        svdata = shiftdata(vdata)
        testneglogprob = 17

        testpred = ["the", "alias", "mock", "test", "stalin", "allahuakbar", "python", "pythonista",
                    " " * (data.shape[1])]
        testpred = words2ints(testpred)
        print testpred

        block = SimpleSeqEncDecAtt(inpvocsize=vocsize, outvocsize=vocsize, encdim=encdim, decdim=statedim,
                                   attdim=attdim, inconcat=False)

        s = SeqEncDecSearch(block)
        pred, probs = s.decode(testpred, startsym, testpred.shape[1])
        print ints2words(pred), probs
