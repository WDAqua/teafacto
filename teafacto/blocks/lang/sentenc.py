from teafacto.blocks.seq.enc import SimpleSeqStar2Vec, SeqEncoder, MaskMode, EncLastDim
from teafacto.blocks.seq.rnn import MakeRNU
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import Block, tensorops as T
from teafacto.util import issequence

# sentence encoders

class CharWordSentEnc(SimpleSeqStar2Vec):
    def __init__(self, numchars=256, charembdim=50, wordembdim=100, innerdim=200, maskid=None, **kw):
        super(CharWordSentEnc, self).__init__(indim=numchars, inpembdim=charembdim, innerdim=[wordembdim, innerdim], maskid=maskid)


class TwoLevelEncoder(Block):
    def __init__(self, l1enc=None, l2emb=None, l2enc=None,
                 maskid=None, **kw):
        super(TwoLevelEncoder, self).__init__(**kw)
        self.l2emb = l2emb
        self.l1enc = l1enc
        self.l2enc = l2enc
        self.maskid = maskid

    def all_outputs(self):
        self.l2enc = self.l2enc.all_outputs()
        return self

    def apply(self, x):
        if self.l2emb is not None:
            l1tensor = x[:, :, 1:]
            l1encs = EncLastDim(self.l1enc)(l1tensor)
            l2mat = x[:, :, 0]
            assert(l2mat.ndim == 2)
            l2embs = self.l2emb(l2mat)
            l2vecs = T.concatenate([l1encs, l2embs], axis=2)
            wmask = T.neq(l2mat, self.maskid) if self.maskid is not None else None
        else:
            l2vecs = EncLastDim(self.l1enc)(x)
            wmask = T.gt(T.sum(T.eq(x, self.maskid), axis=2), 0)
        l2vecs.mask = wmask
        fenc = self.l2enc(l2vecs)
        return fenc         #, wmask #mask for debug


class WordCharSentEnc(TwoLevelEncoder):
    def __init__(self, numchars=256, charembdim=50, charemb=None, charinnerdim=100,
                 numwords=1000, wordembdim=100, wordemb=None, wordinnerdim=200,
                 maskid=None, bidir=False, returnall=False, **kw):
        # char level inits
        if charemb is None:
            charemb = VectorEmbed(indim=numchars, dim=charembdim)
        else:
            charemb = charemb
            charembdim = charemb.outdim
        if not issequence(charinnerdim):
            charinnerdim = [charinnerdim]
        charlayers, lastchardim = MakeRNU.make(charembdim, charinnerdim, bidir=bidir)
        charenc = SeqEncoder(charemb, *charlayers).maskoptions(maskid, MaskMode.AUTO)
        # word level inits
        if wordemb is None:
            wordemb = VectorEmbed(indim=numwords, dim=wordembdim)
        elif wordemb is False:
            wordemb = None
            wordembdim = 0
        else:
            wordemb = wordemb
            wordembdim = wordemb.outdim
        if not issequence(wordinnerdim):
            wordinnerdim = [wordinnerdim]
        wordlayers, outdim = MakeRNU.make(wordembdim + lastchardim, wordinnerdim, bidir=bidir)
        wordenc = SeqEncoder(None, *wordlayers).maskoptions(MaskMode.NONE)
        if returnall:
            wordenc.all_outputs()
        self.outdim = outdim
        super(WordCharSentEnc, self).__init__(l1enc=charenc,
                l2emb=wordemb, l2enc=wordenc, maskid=maskid)