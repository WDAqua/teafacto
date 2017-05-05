from teafacto.blocks.basic import Linear as Lin, Softmax
from teafacto.blocks.basic import VectorEmbed, IdxToOneHot, MatDot
from teafacto.blocks.memory import MemoryStack, MemoryBlock, DotMemAddr
from teafacto.blocks.seq.attention import Attention, LinearGateAttentionGenerator, WeightedSumAttCon
from teafacto.blocks.seq.rnn import MakeRNU
from teafacto.blocks.seq.rnn import RecStack, SeqDecoder, BiRNU, SeqEncoder, MaskSetMode, MaskMode
from teafacto.blocks.seq.rnu import GRU
from teafacto.core.base import Block, tensorops as T, Val, asblock
from teafacto.core.stack import stack
from teafacto.util import issequence

"""
class SeqEncDec(Block):
    def __init__(self, enc, dec, statetrans=None, **kw):
        super(SeqEncDec, self).__init__(**kw)
        self.enc = enc
        self.dec = dec
        if isinstance(statetrans, Block):
            self.statetrans = asblock(lambda x, y: statetrans(x))
        elif statetrans is True:
            self.statetrans = asblock(lambda x, y: x)
        else:
            self.statetrans = statetrans

    def apply(self, inpseq, outseq, maskseq=None):
        if maskseq is None:
            mask = "auto"
        else:
            mask = maskseq
        enco, allenco, encmask = self.enc(inpseq, mask=mask)
        mask = None
        if self.statetrans is not None:
            topstate = self.statetrans(enco, allenco)
            deco = self.dec(allenco, outseq, initstates=[topstate], mask=mask, encmask=encmask)
        else:
            deco = self.dec(allenco, outseq, mask=mask, encmask=encmask)      # no state transfer
        return deco

    def get_init_info(self, inpseq, batsize, maskseq=None):     # TODO: must evaluate enc here, in place, without any side effects
        enco, allenco, encmask = self.enc.predict(inpseq, mask=maskseq)

        if self.statetrans is not None:
            topstate = self.statetrans.predict(enco, allenco)   # this gives unused input warning in theano - it's normal
            initstates = [topstate]
        else:
            initstates = batsize
        return self.dec.get_init_info(Val(allenco),
                                      None,
                                      [Val(x) for x in initstates]
                                            if issequence(initstates)
                                            else initstates,
                                      encmask=Val(encmask))

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SeqEncDecAtt(SeqEncDec):
    def __init__(self, enclayers, declayers, attgen, attcon,
                 decinnerdim, inconcat, outconcat,
                 statetrans=None, vecout=False, **kw):
        enc = SeqEncoder(*enclayers)\
            .with_outputs\
            .with_mask\
            .maskoptions(-1, MaskMode.AUTO, MaskSetMode.ZERO)
        smo = False if vecout else None
        dec = SeqDecoder(
            declayers,
            attention=Attention(attgen, attcon),
            innerdim=decinnerdim,
            outconcat=outconcat,
            inconcat=inconcat,
            softmaxoutblock=smo,
        )
        super(SeqEncDecAtt, self).__init__(enc, dec, statetrans=statetrans, **kw)


class SimpleSeqEncDecAtt(SeqEncDecAtt):
    def __init__(self,
                 inpvocsize=400,
                 inpembdim=None,
                 outvocsize=100,
                 outembdim=None,
                 encdim=100,
                 decdim=100,
                 attdim=100,
                 bidir=False,
                 rnu=GRU,
                 outconcat=True,
                 inconcat=False,
                 statetrans=None,
                 vecout=False,
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        self.enclayers, lastencinnerdim = self.getenclayers(inpembdim, inpvocsize, encinnerdim, bidir, rnu)

        # attention
        lastdecinnerdim = decinnerdim[-1]
        attgen = LinearGateAttentionGenerator(indim=lastencinnerdim + lastdecinnerdim, attdim=attdim)
        attcon = WeightedSumAttCon()

        self.declayers = self.getdeclayers(outembdim, outvocsize, encinnerdim, decinnerdim, inconcat, rnu)
        argdecinnerdim = lastdecinnerdim if outconcat is False else lastencinnerdim + lastdecinnerdim

        if statetrans is True:
            if lastencinnerdim != lastdecinnerdim:  # state shape mismatch
                statetrans = MatDot(lastencinnerdim, lastdecinnerdim)

        super(SimpleSeqEncDecAtt, self).__init__(self.enclayers, self.declayers,
            attgen, attcon, argdecinnerdim, inconcat, outconcat,
            statetrans=statetrans, vecout=vecout, **kw)

    def getenclayers(self, inpembdim, inpvocsize, encinnerdim, bidir, rnu):
        if inpembdim is None:
            inpemb = IdxToOneHot(inpvocsize)
            inpembdim = inpvocsize
        elif isinstance(inpembdim, Block):
            inpemb = inpembdim
            inpembdim = inpemb.outdim
        else:
            inpemb = VectorEmbed(indim=inpvocsize, dim=inpembdim)
        encrnus = []
        dims = [inpembdim] + encinnerdim
        #print dims
        i = 1
        lastencinnerdim = dims[-1] if not bidir else dims[-1] * 2
        while i < len(dims):
            if bidir:
                newrnu = BiRNU.fromrnu(rnu, dim=dims[i - 1], innerdim=dims[i])
            else:
                newrnu = rnu(dim=dims[i - 1], innerdim=dims[i])
            encrnus.append(newrnu)
            i += 1
        enclayers = [inpemb] + encrnus
        return enclayers, lastencinnerdim

    def getdeclayers(self, outembdim, outvocsize, encinnerdim,
                     decinnerdim, inconcat, rnu):
        if outembdim is None:
            outemb = IdxToOneHot(outvocsize)
            outembdim = outvocsize
        elif isinstance(outembdim, Block):
            outemb = outembdim
            outembdim = outemb.outdim
        else:
            outemb = VectorEmbed(indim=outvocsize, dim=outembdim)
        decrnus = []
        firstdecdim = outembdim if inconcat is False else outembdim + encinnerdim
        dims = [firstdecdim] + decinnerdim
        i = 1
        while i < len(dims):
            decrnus.append(rnu(dim=dims[i - 1], innerdim=dims[i]))
            i += 1
        declayers = [outemb] + decrnus
        return declayers
"""

class SeqTransducer(Block):
    def __init__(self, embedder, *layers, **kw):
        """ layers must have an embedding layers first, final softmax layer is added automatically"""
        assert("smodim" in kw and "outdim" in kw)
        self.embedder = embedder
        smodim = kw["smodim"]
        outdim = kw["outdim"]
        del kw["smodim"]; del kw["outdim"]
        super(SeqTransducer, self).__init__(**kw)
        self.block = RecStack(*(layers +
                                (Lin(indim=smodim, dim=outdim),
                                 Softmax())))

    def apply(self, inpseq, maskseq=None):    # inpseq: idx^(batsize, seqlen), maskseq: f32^(batsize, seqlen)
        embseq = self.embedder(inpseq)
        res = self.block(embseq, mask=maskseq)            # f32^(batsize, seqlen, outdim)
        ret = self.applymask(res, maskseq=maskseq)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate([T.ones((xseq.shape[0], xseq.shape[1], 1)), T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))], axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
        return ret


class SimpleSeqTransducer(SeqTransducer):
    def __init__(self, indim=400, embdim=50, inpemb=None, innerdim=100, outdim=50, rnu=GRU, **kw):
        if inpemb is None:
            self.emb = VectorEmbed(indim=indim, dim=embdim)
        else:
            self.emb = inpemb
            embdim = self.emb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [embdim] + innerdim
        self.rnn = MakeRNU.fromdims(innerdim, rnu=rnu)[0]
        super(SimpleSeqTransducer, self).__init__(self.emb, *self.rnn, smodim=innerdim[-1], outdim=outdim, **kw)


class SeqTransDec(Block):
    def __init__(self, *layers, **kw):
        """ first two layers must be embedding layers. Final softmax is added automatically"""
        assert("smodim" in kw and "outdim" in kw)
        smodim = kw["smodim"]
        outdim = kw["outdim"]
        del kw["smodim"]; del kw["outdim"]
        super(SeqTransDec, self).__init__(**kw)
        self.inpemb = layers[0]
        self.outemb = layers[1]
        self.block = RecStack(*(layers[2:] + (Lin(indim=smodim, dim=outdim), Softmax())))

    def apply(self, inpseq, outseq, maskseq=None):
        # embed with the two embedding layers
        emb = self._get_emb(inpseq, outseq)
        res = self.block(emb)
        ret = SeqTransducer.applymask(res, maskseq=maskseq)
        return ret

    def _get_emb(self, inpseq, outseq):
        iemb = self.inpemb(inpseq)     # (batsize, seqlen, inpembdim)
        oemb = self.outemb(outseq)     # (batsize, seqlen, outembdim)
        emb = T.concatenate([iemb, oemb], axis=iemb.ndim-1)                       # (batsize, seqlen, inpembdim+outembdim)
        return emb

    def rec(self, inpa, inpb, *states):
        emb = self._get_emb(inpa, inpb)
        return self.block.rec(emb, *states)

    def get_init_info(self, initstates):
        return self.block.get_init_info(initstates)


class SimpleSeqTransDec(SeqTransDec):
    def __init__(self, indim=400, outdim=50, inpembdim=50, outembdim=50, innerdim=100, **kw):
        self.inpemb = VectorEmbed(indim=indim, dim=inpembdim)
        self.outemb = VectorEmbed(indim=outdim, dim=outembdim)
        self.rnn = []
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [inpembdim+outembdim] + innerdim
        self.rnn = MakeRNU.fromdims(innerdim)[0]
        super(SimpleSeqTransDec, self).__init__(self.inpemb, self.outemb, *self.rnn, smodim=innerdim[-1], outdim=outdim, **kw)


# BASIC SEQ TO IDX
# specify by  enc and out
# specify by layers
# specify by dims
# components:
# seq2vec
# specify by layers
# specify by dims
# vec2idx:
# specify by layers
class Vec2Idx(Block):
    def __init__(self, outlayers, **kw):
        super(Vec2Idx, self).__init__(**kw)
        if isinstance(outlayers, MemoryStack):
            out = outlayers
        else:
            if not issequence(outlayers):
                outlayers = [outlayers]
            if type(outlayers[-1]) is not Softmax:
                outlayers.append(Softmax())
            out = stack(*outlayers)
        self.out = out

    def apply(self, x, *args):
        return self.out(x, *args)


# specify by dims
class SimpleVec2Idx(Vec2Idx):
    def __init__(self, indim=100, outdim=100, **kw):
        outl = MatDot(indim=indim, dim=outdim)
        super(SimpleVec2Idx, self).__init__(outl, **kw)


class MemVec2Idx(Vec2Idx):
    def __init__(self, memenc, memdata, memaddr=DotMemAddr, memdim=None, memattdim=100, **kw):
        assert(memenc is not None)
        memblock = MemoryBlock(memenc, memdata, indim=memdata.shape[0], outdim=memdim)
        memstack = MemoryStack(memblock, memaddr, memattdim=memattdim)
        super(MemVec2Idx, self).__init__(memstack, **kw)


class DynMemVec2Idx(MemVec2Idx):
    def __init__(self, memenc, memaddr=DotMemAddr, memdim=None, memattdim=100, **kw):
        super(self, DynMemVec2Idx).__init__(memenc, None, memaddr=memaddr, memdim=memdim, memattdim=memattdim, **kw)
