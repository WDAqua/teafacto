from teafacto.core.base import Block, asblock, Val, issequence
from teafacto.blocks.seq.rnn import SeqEncoder, MaskMode, MaskSetMode, SeqDecoder, BiRNU
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, GenDotProdAttGen, ForwardAttGen
from teafacto.blocks.basic import VectorEmbed, IdxToOneHot, MatDot


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
        """
        VERY DIFFERENT FROM THE PURELY SYMBOLIC GET_INIT_INFO IN REAL REC BLOCKS !!!
        This one is used in decoder/prediction
        """
        enco, allenco, encmask = self.enc.predict(inpseq, mask=maskseq)

        if self.statetrans is not None:
            topstate = self.statetrans.predict(enco, allenco)   # this gives unused input warning in theano - it's normal
            initstates = [topstate]
        else:
            initstates = batsize
        return self.dec.get_init_info(Val(allenco),
                                      [Val(x) for x in initstates]
                                            if issequence(initstates)
                                            else initstates,
                                      encmask=Val(encmask))

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SeqEncDecAtt(SeqEncDec):
    def __init__(self, enclayers, declayers, attgen, attcon,
                 decinnerdim, statetrans=None, vecout=False,
                 inconcat=True, outconcat=False, **kw):
        enc = SeqEncoder(*enclayers)\
            .with_outputs()\
            .with_mask()\
            .maskoptions(-1, MaskMode.AUTO, MaskSetMode.ZERO)
        smo = False if vecout else None
        dec = SeqDecoder(
            declayers,
            attention=Attention(attgen, attcon),
            innerdim=decinnerdim, inconcat=inconcat,
            softmaxoutblock=smo, outconcat=outconcat
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
                 statetrans=None,
                 vecout=False,
                 inconcat=True,
                 outconcat=False,
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        self.enclayers, lastencinnerdim = \
            self.getenclayers(inpembdim, inpvocsize, encinnerdim, bidir, rnu)

        self.declayers = \
            self.getdeclayers(outembdim, outvocsize, lastencinnerdim,
                              decinnerdim, rnu, inconcat)

        # attention
        lastdecinnerdim = decinnerdim[-1]
        argdecinnerdim = lastdecinnerdim if outconcat is False else lastencinnerdim + lastdecinnerdim
        attgen = GenDotProdAttGen(indim=lastencinnerdim, memdim=lastdecinnerdim)
        attcon = WeightedSumAttCon()

        if statetrans is True:
            if lastencinnerdim != lastdecinnerdim:  # state shape mismatch
                statetrans = MatDot(lastencinnerdim, lastdecinnerdim)
        elif statetrans == "matdot":
            statetrans = MatDot(lastencinnerdim, lastdecinnerdim)

        super(SimpleSeqEncDecAtt, self).__init__(self.enclayers, self.declayers,
            attgen, attcon, argdecinnerdim, statetrans=statetrans, vecout=vecout,
            inconcat=inconcat, outconcat=outconcat, **kw)

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

    def getdeclayers(self, outembdim, outvocsize, lastencinnerdim,
                     decinnerdim, rnu, inconcat):
        if outembdim is None:
            outemb = IdxToOneHot(outvocsize)
            outembdim = outvocsize
        elif isinstance(outembdim, Block):
            outemb = outembdim
            outembdim = outemb.outdim
        else:
            outemb = VectorEmbed(indim=outvocsize, dim=outembdim)
        decrnus = []
        firstdecdim = outembdim + lastencinnerdim if inconcat else outembdim
        dims = [firstdecdim] + decinnerdim
        i = 1
        while i < len(dims):
            decrnus.append(rnu(dim=dims[i - 1], innerdim=dims[i]))
            i += 1
        declayers = [outemb] + decrnus
        return declayers