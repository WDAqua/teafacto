from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, MatDot, Softmax, Linear
from teafacto.blocks.pool import Pool
from teafacto.blocks.seq.rnn import MakeRNU, EncLastDim
from teafacto.blocks.seq.oldseqproc import Vec2Idx, SimpleVec2Idx
from teafacto.blocks.seq.rnn import SeqEncoder, MaskMode
from teafacto.core.base import Block, tensorops as T, param
from teafacto.util import issequence


class SeqUnroll(Block):
    def __init__(self, block, **kw):
        self.inner = block
        super(SeqUnroll, self).__init__(**kw)

    def apply(self, seq):   # (batsize, seqlen, ...)
        x = seq.dimswap(1, 0)
        ret, _ = T.scan(self.rec, sequences=x)
        return ret.dimswap(1, 0)

    def rec(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class Seq2Idx(Block):
    def __init__(self, seq2vec, vec2idx, **kw):
        self.enc = seq2vec
        self.out = vec2idx
        super(Seq2Idx, self).__init__(**kw)

    def apply(self, x, mask=None):         # x: idx^(batsize, seqlen)
        enco = self.enc(x, mask=mask)      # (batsize, innerdim)
        out = self.out(enco)    # (batsize, probs)
        return out


class LayerSeq2Idx(Seq2Idx):
    def __init__(self, inpemb, enclayers, outlayers, maskid=0, **kw):
        enc = Seq2Vec(inpemb, enclayers, maskid)
        out = Vec2Idx(outlayers)
        super(LayerSeq2Idx, self).__init__(enc, out, **kw)


class SimpleSeq2Idx(Seq2Idx):
    def __init__(self, indim=400, outdim=100, inpembdim=50, innerdim=100, maskid=0, bidir=False, **kw):
        enc = SimpleSeq2Vec(indim=indim, inpembdim=inpembdim, innerdim=innerdim, maskid=0, bidir=bidir)
        out = SimpleVec2Idx(indim=enc.outdim, outdim=outdim)
        super(SimpleSeq2Idx, self).__init__(enc, out, **kw)


class Seq2Vec(Block):
    def __init__(self, inpemb, enclayers, maskid=0, pool=None, **kw):
        super(Seq2Vec, self).__init__(**kw)
        self.maskid = maskid
        self.inpemb = inpemb
        if not issequence(enclayers):
            enclayers = [enclayers]
        self.pool = pool
        self.enc = SeqEncoder(inpemb, *enclayers).maskoptions(maskid, MaskMode.AUTO)
        if self.pool is not None:
            self.enc = self.enc.all_outputs.with_mask

    def all_outputs(self):
        self.enc = self.enc.all_outputs()
        return self

    def apply(self, x, mask=None, weights=None):
        if self.pool is not None:
            ret, mask = self.enc(x, mask=mask, weights=weights)
            ret = self.pool(ret, mask)
        else:
            ret = self.enc(x, mask=mask, weights=weights)
        return ret


class SimpleSeq2Vec(Seq2Vec):
    def __init__(self, indim=400, inpembdim=50, inpemb=None,
                 innerdim=100, maskid=0, bidir=False, pool=False,
                 zoneout=False, dropout_in=False, dropout_h=False, **kw):
        if inpemb is None:
            if inpembdim is None:
                inpemb = IdxToOneHot(indim)
                inpembdim = indim
            else:
                inpemb = VectorEmbed(indim=indim, dim=inpembdim)
        else:
            inpembdim = inpemb.outdim
        rnn, lastdim = self.makernu(inpembdim, innerdim, bidir=bidir,
                                    zoneout=zoneout, dropout_h=dropout_h, dropout_in=dropout_in)
        self.outdim = lastdim
        poolblock = None if pool is False else Pool((None,), axis=(1,), mode="max")
        super(SimpleSeq2Vec, self).__init__(inpemb, rnn, maskid=maskid, pool=poolblock, **kw)

    @staticmethod
    def makernu(inpembdim, innerdim, bidir=False, zoneout=False, dropout_h=False, dropout_in=False):
        return MakeRNU.make(inpembdim, innerdim, bidir=bidir,
                            zoneout=zoneout, dropout_h=dropout_h, dropout_in=dropout_in)


class SimpleSeq2MultiVec(Block):
    def __init__(self, indim=400, inpembdim=50, inpemb=None, mode="concat",
                 innerdim=100, numouts=1, maskid=0, bidir=False,
                 maskmode=MaskMode.NONE, **kw):
        super(SimpleSeq2MultiVec, self).__init__(**kw)
        if inpemb is None:
            if inpembdim is None:
                inpemb = IdxToOneHot(indim)
                inpembdim = indim
            else:
                inpemb = VectorEmbed(indim=indim, dim=inpembdim)
        elif inpemb is False:
            inpemb = None
        else:
            inpembdim = inpemb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim[-1] += numouts
        rnn, lastdim = self.makernu(inpembdim, innerdim, bidir=bidir)
        self.outdim = lastdim*numouts
        self.maskid = maskid
        self.inpemb = inpemb
        self.numouts = numouts
        self.mode = mode
        if not issequence(rnn):
            rnn = [rnn]
        self.enc = SeqEncoder(inpemb, *rnn).maskoptions(maskid, maskmode)
        self.enc.all_outputs()

    @staticmethod
    def makernu(inpembdim, innerdim, bidir=False):
        return MakeRNU.make(inpembdim, innerdim, bidir=bidir)

    def apply(self, x, mask=None, weights=None):
        ret = self.enc(x, mask=mask, weights=weights)   # (batsize, seqlen, lastdim)
        outs = []
        # apply mask    (SeqEncoder should attach mask to outvar if all_outputs()
        mask = ret.mask
        for i in range(self.numouts):
            selfweights = Softmax()(ret[:, :, i])   # (batsize, seqlen)
            selfweights *= mask     # apply mask
            selfweights = selfweights / T.sum(selfweights, axis=1).dimshuffle(0, "x")   # renormalize
            weightedstates = ret[:, :, self.numouts:] * selfweights.dimshuffle(0, 1, "x")
            out = T.sum(weightedstates, axis=1)     # (batsize, lastdim)
            outs.append(out)
        if self.mode == "concat":
            ret = T.concatenate(outs, axis=1)
        elif self.mode == "seq":
            outs = [out.dimshuffle(0, "x", 1) for out in outs]
            ret = T.concatenate(outs, axis=1)
        return ret


class SeqStar2Vec(Block):
    def __init__(self, baseemb, *layersforencs, **kw):
        super(SeqStar2Vec, self).__init__(**kw)
        self.maskid = None if "maskid" not in kw else kw["maskid"]
        self.encoders = []
        atbase = True
        for layers in layersforencs:
            if not issequence(layers):
                layers = [layers]
            if atbase:
                enc = SeqEncoder(baseemb, *layers).maskoptions(MaskMode.NONE)
                atbase = False
            else:
                enc = SeqEncoder(None, *layers).maskoptions(MaskMode.NONE)
            self.encoders.append(enc)

    def apply(self, x):     # (batsize, outerseqlen, innerseqlen)
        y = x
        xm = T.neq(x, self.maskid) if self.maskid is not None else None
        for enc in self.encoders:
            y = EncLastDim(enc)(y, mask=xm)
            xm = T.sum(xm, axis=-1) > 0 if self.maskid is not None else None
        return y


class SimpleSeqStar2Vec(SeqStar2Vec):
    def __init__(self, indim=400, inpembdim=50, inpemb=None, innerdim=100, maskid=None, bidir=False, **kw):
        if inpemb is None:
            if inpembdim is None:
                inpemb = IdxToOneHot(indim)
                inpembdim = indim
            else:
                inpemb = VectorEmbed(indim=indim, dim=inpembdim)
        else:
            inpembdim = inpemb.outdim
        lastdim = inpembdim
        if not issequence(innerdim):    # single encoder
            innerdim = [innerdim]
        rnns = []
        for innerdimi in innerdim:
            if not issequence(innerdimi):    # one layer in encoder
                innerdimi = [innerdimi]
            rnn, lastdim = MakeRNU.make(lastdim, innerdimi, bidir=bidir)
            rnns.append(rnn)
        self.outdim = lastdim
        super(SimpleSeqStar2Vec, self).__init__(inpemb, *rnns, maskid=maskid, **kw)


class SimpleSeq2Bool(SimpleSeq2Vec):
    def __init__(self, **kwargs):
        super(SimpleSeq2Bool, self).__init__(**kwargs)
        self.summ = param((self.outdim,), name="summarizer").uniform()

    def apply(self, x, mask=None, weights=None):
        ret = super(SimpleSeq2Bool, self).apply(x, mask=mask, weights=weights)
        return T.tanh(T.dot(ret, self.summ))


class SimpleSeq2Sca(SimpleSeq2Vec):
    def __init__(self, **kw):
        super(SimpleSeq2Sca, self).__init__(**kw)
        self.enc.all_outputs().with_mask()
        if "innerdim" in kw:
            kwindim = kw["innerdim"]
            if issequence(kwindim):
                summdim = kwindim[-1]
            else:
                summdim = kwindim
        else:
            summdim = 100
        self.summ = param((summdim,), name="summarize").uniform()

    def apply(self, x, mask=None):
        enco, mask = self.enc(x, mask=mask)
        return T.nnet.sigmoid(T.dot(enco, self.summ)), mask





