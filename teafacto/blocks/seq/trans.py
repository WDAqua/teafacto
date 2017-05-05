from teafacto.core.base import Block
from teafacto.blocks.seq.rnn import SeqEncoder, MakeRNU, MaskMode
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import VectorEmbed, Linear as Lin, Softmax
from teafacto.util import issequence


class SeqTrans(Block):
    def __init__(self, embedder, *layers, **kw):
        super(SeqTrans, self).__init__(**kw)
        self.enc = SeqEncoder(embedder, *layers)
        self.enc.all_outputs().maskoption(MaskMode.NONE)

    def apply(self, x):
        return self.enc(x)


class SimpleSeqTrans(SeqTrans):
    def __init__(self, indim=400, embdim=50, inpemb=None,
                 innerdim=100, outdim=50, rnu=GRU, **kw):
        if inpemb is None:
            emb = VectorEmbed(indim=indim, dim=embdim)
        else:
            emb = inpemb
            embdim = emb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [embdim] + innerdim
        rnn, _ = MakeRNU.fromdims(innerdim, rnu=rnu)
        smo = Lin(indim=innerdim[-1], dim=outdim)
        super(SimpleSeqTrans, self).__init__(emb, *(rnn + [smo, Softmax()]), **kw)
