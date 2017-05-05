from collections import OrderedDict

import numpy as np
import os
from IPython import embed

from teafacto.core.base import Block, Val, tensorops as T
from teafacto.blocks.basic import VectorEmbed, Embedder
from teafacto.util import ticktock as TT, isnumber, isstring
from teafacto.blocks.seq.enc import SimpleSeqStar2Vec


class WordEmbBase(object):
    def __init__(self, worddic, raretoken="<RARE>", **kw):
        super(WordEmbBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic
        self._raretoken = raretoken

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else 0

    def __mul__(self, other):
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        try:
            if isstring(word):
                return self.w[self.D[word]]
            elif isnumber(word):
                return self.w[word, :]
        except Exception:
            return None

    def __getitem__(self, word):
        v = self.getvector(word)
        return v if v is not None else self.w[0, :]

    @property
    def w(self):
        return self.W.d.get_value()

    @property
    def shape(self):
        return self.w.shape

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def __mod__(self, other):
        if isinstance(other, (tuple, list)):  # distance
            assert len(other) > 1
            if len(other) == 2:
                return self.getdistance(other[0], other[1])
            else:
                y = other[0]
                return map(lambda x: self.getdistance(y, x), other[1:])
        else:  # embed
            return self.__getitem__(other)
    # endregion

    @property
    def block(self):
        return self

    @property
    def raretoken(self):
        return self._raretoken


class WordEmb(WordEmbBase, VectorEmbed): # unknown words are mapped to index 0, their embedding is a zero vector
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, indim=None, value=None, worddic=None,
                 normalize=False, trainfrac=1.0, init=None, raretoken="<RARE>",
                 **kw):
        if isstring(value):     # path
            assert(init is None and worddic is None)
            value, worddic = self.loadvalue(value, dim, indim=indim)
            indim = max(worddic.values()) + 1
        if worddic is not None:
            wdvals = worddic.values()
            #embed()
            if raretoken is not None:
                if raretoken not in worddic:
                    assert(0 not in wdvals)     # make sure index zero is free
                    worddic[raretoken] = 0
                assert(raretoken in worddic)        # raretoken must be in dic
            else:
                pass        # no rare tokens
            assert(min(wdvals) >= 0)     # word ids must be positive non-zero
            assert(indim == max(wdvals)+1 or indim is None)
            if indim is None:
                indim = max(worddic.values())+1        # to init from worddic
        super(WordEmb, self).__init__(indim=indim, dim=dim, value=value,
                                      normalize=normalize, worddic=worddic,
                                      trainfrac=trainfrac, init=init, raretoken=raretoken,
                                      **kw)

    def adapt(self, wdic):
        return AdaptedWordEmb(self, wdic, maskid=self.maskid)

    def override(self, wordemb):
        return OverriddenWordEmb(self, wordemb, maskid=self.maskid)

    def augment(self, wordemb):
        return AugmentedWordEmb(self, wordemb, maskid=self.maskid)

    def loadvalue(self, path, dim, indim=None):
        tt = TT(self.__class__.__name__)
        tt.tick()
        W = [np.zeros((1, dim))]
        D = OrderedDict()
        i = 1
        for line in open(path):
            if indim is not None and i >= (indim+1):
                break
            ls = line.split(" ")
            word = ls[0]
            D[word] = i
            W.append(np.asarray([map(lambda x: float(x), ls[1:])]))
            i += 1
        W = np.concatenate(W, axis=0)
        tt.tock("loaded")
        return W, D


class AdaptedWordEmb(WordEmb):
    def __init__(self, wordemb, wdic, **kw):
        D = wordemb.D
        nativeraretoken = wordemb.raretoken
        super(AdaptedWordEmb, self).__init__(worddic=wdic, value=False,
                dim=wordemb.outdim, normalize=wordemb.normalize,
                trainfrac=wordemb.trainfrac, raretoken=nativeraretoken, **kw)
        self.inner = wordemb

        self.ad = {v: D[k] if k in D else D[nativeraretoken]
                   for k, v in wdic.items()}

        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    @property
    def w(self):
        return self.inner.W.d.get_value()[self.adb.d.get_value()]

    def apply(self, inp):
        x = self.adb[inp]
        ret = self.inner(x)
        self._maskfrom(ret, inp)
        return ret


class OverriddenWordEmb(WordEmb): # TODO: RARE TOKEN MGMT
    def __init__(self, base, override, **kw):
        assert(base.outdim == override.outdim)
        super(OverriddenWordEmb, self).__init__(worddic=base.D, value=False,
                dim=base.outdim, normalize=base.normalize,
                trainfrac=base.trainfrac, **kw)

        self.base = base
        self.override = override
        self.ad = {v: override.D[k] if k in override.D else 0 for k, v in base.D.items()}
        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    def apply(self, x):
        overx = self.adb[x]
        mask = overx > 0
        mask = T.outer(mask, T.ones((self.outdim,)))
        ret = T.switch(mask, self.override(overx), self.base(x))
        self._maskfrom(ret, x)
        return ret

    @property
    def w(self):
        return None         # TODO


class AugmentedWordEmb(WordEmb):    # TODO: RARE TOKEN MGMT
    def __init__(self, base, augment, **kw):
        assert(base.outdim == augment.outdim)
        super(AugmentedWordEmb, self).__init__(worddic=base.D, value=False,
                dim=base.outdim, normalize=base.normalize,
                trainfrac=base.trainfrac, **kw)
        self.base = base
        self.augment = augment
        self.ad = {v: augment.D[k] if k in augment.D else 0 for k, v in base.D.items()}
        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    def apply(self, x):
        ret = T.concatenate([self.base(x), self.augment(self.adb[x])], axis=1)
        self._maskfrom(ret, x)
        return ret

    @property
    def w(self):
        return None         # TODO


class Glove(WordEmb):
    defaultpath = "../../../data/glove/glove.6B.%dd.txt"

    def __init__(self, dim, vocabsize=None, path=None, trainfrac=0.0, **kw):     # if dim=None, load all
        path = self.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        super(Glove, self).__init__(dim=dim, indim=vocabsize, value=path, trainfrac=trainfrac, **kw)
