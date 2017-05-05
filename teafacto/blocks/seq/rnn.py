from enum import Enum
import numpy as np

from teafacto.blocks.seq.attention import WeightedSumAttCon, Attention, AttentionConsumer, LinearGateAttentionGenerator
from teafacto.blocks.seq.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase
from teafacto.blocks.basic import IdxToOneHot, Softmax, MatDot, Eye, VectorEmbed
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.util import issequence, isnumber


class RecStack(ReccableBlock):
    # must handle RecurrentBlocks ==> can not recappl, if all ReccableBlocks ==> can do recappl
    # must give access to final states of internal layers
    # must give access to all outputs of top layer
    # must handle masks
    def __init__(self, *layers, **kw):
        super(RecStack, self).__init__(**kw)
        self.layers = layers

    @property
    def numstates(self):
        return reduce(lambda x, y: x + y, [x.numstates for x in self.layers if isinstance(x, RecurrentBlock)], 0)

    # FWD API. initial states can be set, mask is accepted, everything is returned. Works for all RecurrentBlocks
    # FWD API IMPLEMENTED USING FWD API
    def innerapply(self, seq, mask=None, initstates=None):
        states = []
        for layer in self.layers:
            if isinstance(layer, RecurrentBlock):
                if initstates is not None:
                    layerinpstates = initstates[:layer.numstates]
                    initstates = initstates[layer.numstates:]
                else:
                    layerinpstates = None
                final, seq, layerstates = layer.innerapply(seq, mask=mask, initstates=layerinpstates)
                states.extend(layerstates)
            elif isinstance(layer, Block):
                seq = self.recurnonreclayer(seq, layer)
                final = seq[:, -1, :]
            else:
                raise Exception("can not apply this layer: " + str(layer) + " in RecStack")
        return final, seq, states           # full history of final output and all states (ordered from bottom layer to top)

    @classmethod
    def apply_mask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            ret = mask * xseq
        return ret

    @classmethod
    def dummyrec(cls, layer):
        def innerrec(x_t):
            return layer(x_t)

        return innerrec

    @classmethod
    def recurnonreclayer(cls, x, layer):
        y, _ = T.scan(fn=cls.dummyrec(layer),
                      sequences=x.dimswap(1, 0),
                      outputs_info=None)
        return y.dimswap(1, 0)

    # REC API: only works with ReccableBlocks
    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert (len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock),
                           self.layers)) == 0)  # no non-reccable blocks allowed
        if issequence(initstates):  # fill up init state args so that layers for which no init state is specified get default arguments that lets them specify a default init state
                                    # if is a sequence, expecting a value, not batsize
            if len(initstates) < self.numstates:    # top layers are being given the given init states, bottoms make their own default
                initstates = [initstates.shape[0]] * (self.numstates - len(initstates)) + initstates
        else:   # expecting a batsize as initstate arg
            initstates = [initstates] * self.numstates
        init_infos = []
        for recurrentlayer in recurrentlayers:  # from bottom layers to top
            arg = initstates[:recurrentlayer.numstates]
            initstates = initstates[recurrentlayer.numstates:]
            initinfo = recurrentlayer.get_init_info(arg)
            init_infos.extend(initinfo)
        return init_infos

    def rec(self, x_t, *states):
        # apply each block on x_t to get next-level input, consume states in the process
        nextinp = x_t
        nextstates = []
        for block in self.layers:
            if isinstance(block, ReccableBlock):
                numstates = block.numstates
                recstates = states[:numstates]
                states = states[numstates:]
                rnuret = block.rec(nextinp, *recstates)
                nextstates.extend(rnuret[1:])
                nextinp = rnuret[0]
            elif isinstance(block, Block): # block is a function
                nextinp = block(nextinp)
        return [nextinp] + nextstates


class BiRNU(RecurrentBlock): # TODO: optimizer can't process this
    def __init__(self, fwd=None, rew=None, **kw):
        super(BiRNU, self).__init__(**kw)
        assert(isinstance(fwd, RNUBase) and isinstance(rew, RNUBase))
        self.fwd = fwd
        self.rew = rew
        assert(self.fwd.indim == self.rew.indim)

    @classmethod
    def fromrnu(cls, rnucls, *args, **kw):
        assert(issubclass(rnucls, RNUBase))
        kw["reverse"] = False
        fwd = rnucls(*args, **kw)
        kw["reverse"] = True
        rew = rnucls(*args, **kw)
        return cls(fwd=fwd, rew=rew)

    @property
    def numstates(self):
        return self.fwd.numstates + self.rew.numstates

    def innerapply(self, seq, mask=None, initstates=None):
        initstatesfwd = initstates[:self.fwd.numstates] if initstates is not None else initstates
        initstates = initstates[self.fwd.numstates:] if initstates is not None else initstates
        assert(initstates is None or len(initstates) == self.rew.numstates)
        initstatesrew = initstates
        fwdfinal, fwdout, fwdstates = self.fwd.innerapply(seq, mask=mask, initstates=initstatesfwd)   # (batsize, seqlen, innerdim)
        rewfinal, rewout, rewstates = self.rew.innerapply(seq, mask=mask, initstates=initstatesrew) # TODO: reverse?
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        finalout = T.concatenate([fwdfinal, rewfinal], axis=1)
        out = T.concatenate([fwdout, rewout.reverse(1)], axis=2)
        return finalout, out, fwdstates+rewstates



class MaskMode(Enum):
    NONE = 0
    AUTO = 1
    AUTO_FORCE = 2

class MaskSetMode(Enum):
    NONE = 0
    ZERO = 1
    MASKID = 2


class MaskConfig(object):
    def __init__(self, maskmode=MaskMode.NONE, maskid=0, maskset=MaskSetMode.NONE):
        self.maskmode = maskmode
        self.maskid = maskid
        self.maskset = maskset

    def option(self, o):
        if isinstance(o, MaskSetMode):
            self.maskset = o
        elif isinstance(o, MaskMode):
            self.maskmode = o
        elif isinstance(o, int):
            self.maskid = o
        else:
            raise NotImplementedError("unrecognized mask configuration option")


class EncLastDim(Block):
    def __init__(self, enc, **kw):
        super(EncLastDim, self).__init__(**kw)
        self.enc = enc

    def apply(self, x, mask=None):
        if self.enc.embedder is None:
            mindim = 3
            maskdim = x.ndim - 1
        else:
            mindim = 2
            maskdim = x.ndim
        if mask is not None:
            assert(mask.ndim == maskdim)
        else:
            mask = T.ones(x.shape[:maskdim])
        if x.ndim == mindim:
            return self.enc(x, mask=mask)
        elif x.ndim > mindim:
            ret, _ = T.scan(fn=self.outerrec, sequences=[x, mask], outputs_info=None)
            return ret
        else:
            raise Exception("cannot have less than {} dims".format(mindim))

    def outerrec(self, xred, mask):  # x: ndim-1
        ret = self.apply(xred, mask=mask)
        return ret

    @property
    def outdim(self):
        return self.enc.outdim



class SeqEncoder(AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''

    def __init__(self, embedder, *layers, **kw):
        super(SeqEncoder, self).__init__(**kw)
        self._returnings = {"enc"}
        self._nomask = False
        self._maskconfig = kw["maskcfg"] if "maskcfg" in kw else MaskConfig(MaskMode.AUTO, 0, MaskSetMode.NONE)
        self.embedder = embedder
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, RecurrentBlock))
            else:
                self.block = RecStack(*layers)
        else:
            self.block = None

    def apply(self, seq, weights=None, mask=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen) OR (batsize, seqlen, seqlen*, dim) ==> reduce the innermost seqlen
        mask = seq.mask if mask is None else mask
        # embed
        if self.embedder is None:
            seqemb = seq
        else:
            seqemb = self.embedder(seq)     # maybe this way of embedding is not so nice for memory
            mask = seqemb.mask if mask is None else mask
            # auto mask
            if self._maskconfig.maskmode == MaskMode.AUTO_FORCE or \
                    (mask is None and self._maskconfig.maskmode == MaskMode.AUTO) or \
                    mask == "auto":
                mask = self._autogenerate_mask(seq, seqemb)

        # full mask
        fullmask = None
        if mask is not None:
            fullmask = mask
        if weights is not None:
            fullmask = weights if fullmask is None else weights * fullmask
        final, outputs, states = self.block.innerapply(seqemb, mask=fullmask)
        if mask is not None:
            outputs.mask = mask
        return self._get_apply_outputs(final, outputs, states, mask)

    def _autogenerate_mask(self, seq, seqemb):
        assert(seqemb.ndim == 3)
        print "automasking in SeqEncoder (%s)" % __file__
        axes = range(2, seq.ndim)       # mask must be 2D
        if "int" in seq.dtype:       # ==> indexes  # mask must be 2D
            if seq.ndim < 2:
                raise AttributeError("CAN NOT GENERATE MASK FOR NON-SEQUENCE")
            elif seq.ndim == 2:
                seqformask = seq
            else:
                print "generating default mask for non-standard seq shape (SeqEncoder, %s)" % __file__
                seqformask = seq[(slice(None, None, None),) * 2 + (0,) * (seq.ndim-2)]
                #if self._maskconfig.maskid != 0:
                #    raise AttributeError("CAN NOT CREATE MASK USING CUSTOM MASKID %d BECAUSE OF NON-STANDARD SEQ (%d dims, %s)" % (self._maskconfig.maskid, seq.ndim, str(seq.dtype)))
                #mask = T.gt(seq.sum(axis=axes), 0)      # 0 is TERMINUS
            assert(seqformask.ndim == 2)
            mask = T.neq(seqformask, self._maskconfig.maskid)
        else:
            #TODO raise AttributeError("CAN NOT GENERATE MASK FOR NON-INT SEQ")
            mask = T.gt(seq.norm(2, axis=axes), 0)
        return mask

    def _get_apply_outputs(self, final, outputs, states, mask):
        ret = []
        if "enc" in self._returnings:       # final states of topmost layer
            ret.append(final)
        if "all" in self._returnings:       # states (over all time) of topmost layer
            rete = outputs       # (batsize, seqlen, dim) --> zero-fy according to mask
            if self._maskconfig.maskset == MaskSetMode.ZERO and mask is not None:
                fmask = T.tensordot(mask, T.ones((outputs.shape[2],)), 0)
                rete = rete * fmask
            rete.mask = mask
            ret.append(rete)
        if "states" in self._returnings:    # final states (over all layers)???
            pass # TODO: do we need to support this?
        if "mask" in self._returnings:
            ret.append(mask)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _get_apply_outputs_old(self, outputs):
        output = outputs[0]
        if self._all_states:    # get the last values of each of the states
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:                   # get the last state of the final state
            ret = [output[:, -1, :]] #output is (batsize, innerdim)
        if self._alloutput:    # include stack output
            return ret + [output]
        else:
            if len(ret) == 1:   # if only topmost last state or only one stateful layer --> one output
                return ret[0]
            else:               # else: a list of outputs
                return ret

    ### FLUENT MASK SETTINGS
    def mask(self, maskcfg):
        assert(isinstance(maskcfg, MaskConfig))
        self._maskconfig = maskcfg
        return self

    def maskoptions(self, *opt):
        for o in opt:
            self.maskoption(o)
        return self

    def maskoption(self, maskoption):
        self._maskconfig.option(maskoption)
        return self

    ### FLUENT OUTPUT SETTINGS
    def reset_return(self):
        self._returnings.clear()
        return self

    def with_states(self):       # TODO
        '''Call this switch to get the final states of all recurrent layers'''
        self._returnings.add("states")
        return self

    def all_outputs(self):
        '''Call this switch to get the actual output of top layer as the last outputs'''
        self.reset_return()
        self._returnings.add("all")
        return self

    def with_outputs(self):
        self._returnings.add("all")
        return self

    def with_mask(self):
        ''' Calling this switch also returns the mask on original idx input sequence'''
        self._returnings.add("mask")
        return self

    def setreturn(self, *args):
        self.reset_return()
        for arg in args:
            self._returnings.add(arg)
        return self


class RNNSeqEncoder(SeqEncoder):
    def __init__(self, indim=500, inpembdim=100, inpemb=None,
                 innerdim=200, bidir=False, maskid=None,
                 zoneout=False, dropout_in=False, dropout_h=False, **kw):
        if inpemb is None:
            inpemb = VectorEmbed(indim=indim, dim=inpembdim, maskid=maskid)
        elif inpemb is False:
            inpemb = None
        else:
            inpembdim = inpemb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        layers, _ = MakeRNU.make(inpembdim, innerdim, bidir=bidir,
                                 zoneout=zoneout, dropout_in=dropout_in, dropout_h=dropout_h)
        super(RNNSeqEncoder, self).__init__(inpemb, *layers, **kw)


class RNNSeqEncoderMulti(RNNSeqEncoder):
    def __init__(self, indim=500, inpembdim=100, inpemb=None, mode="concat",
                 innerdim=200, numouts=1, bidir=False, maskid=None,**kw):
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim[-1] += numouts
        super(RNNSeqEncoderMulti, self).__init__(indim=indim, inpembdim=inpembdim,
             inpemb=inpemb, innerdim=innerdim, bidir=bidir, maskid=maskid, **kw)
        self.numouts = numouts
        self.mode = mode
        self.all_outputs()


class SeqDecoder(Block):
    """ seq decoder with attention with new inconcat implementation """
    def __init__(self, layers, softmaxoutblock=None, innerdim=None,
                 attention=None, inconcat=True, outconcat=False, **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.embedder = layers[0]
        self.block = RecStack(*layers[1:])
        self.outdim = innerdim
        self.attention = attention
        self.inconcat = inconcat
        self.outconcat = outconcat
        self._mask = False
        self._attention = None
        assert(isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None: # default softmax out block
            sm = Softmax()
            self.lin = MatDot(indim=self.outdim, dim=self.embedder.indim)
            self.softmaxoutblock = asblock(lambda x: sm(self.lin(x)))
        elif softmaxoutblock is False:
            self.softmaxoutblock = asblock(lambda x: x)
        else:
            self.softmaxoutblock = softmaxoutblock

    @property
    def numstates(self):
        return self.block.numstates

    def _get_seq_emb_t0(self, num, startsymemb=None):
        # seq_emb = self.embedder(seq[:, 1:])    # (batsize, seqlen-1, embdim)
        dim = self.embedder.outdim
        seq_emb_t0_sym = T.zeros((dim,), dtype="float32") if startsymemb is None else startsymemb
        seq_emb_t0 = T.repeat(seq_emb_t0_sym[np.newaxis, :], num, axis=0)
        return seq_emb_t0

    def apply(self, context, seq, context_0=None, initstates=None, mask=None, encmask=None, startsymemb=None, **kw):  # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        if initstates is None:
            initstates = seq.shape[0]
        elif issequence(initstates):
            if len(initstates) < self.numstates:  # fill up with batsizes for lower layers
                initstates = [seq.shape[0]] * (self.numstates - len(initstates)) + initstates
        init_info, nonseq = self.get_init_info(context, initstates,
                                    ctx_0=context_0, encmask=encmask)  # sets init states to provided ones
        embedder = self.embedder
        def recemb(x):
            return embedder(x)
        seq_emb, _ = T.scan(fn=recemb, sequences=seq[:, 1:].dimswap(1, 0))
        seq_emb = seq_emb.dimswap(1, 0)
        seq_emb_t0 = self._get_seq_emb_t0(seq_emb.shape[0], startsymemb=startsymemb)
        seq_emb = T.concatenate([seq_emb_t0.dimshuffle(0, "x", 1), seq_emb], axis=1)
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq_emb.dimswap(1, 0),
                            outputs_info=[None] + init_info,
                            non_sequences=nonseq)
        ret = outputs[0].dimswap(1, 0)  # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        if mask == "auto":
            mask = (seq > 0).astype("int32")
        ret = self.applymask(ret, mask)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq):
        if maskseq is None:
            return xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate(
                [T.ones((xseq.shape[0], xseq.shape[1], 1)),
                 T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))],
                axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
            return ret

    def get_init_info(self, context, initstates, ctx_0=None, encmask=None):
        initstates = self.block.get_init_info(initstates)
        ctx_0 = self._get_ctx_t(context, initstates, encmask) if ctx_0 is None else ctx_0
        if encmask is None:
            encmask = T.ones(context.shape[:2], dtype="float32")
        return [ctx_0, 0] + initstates, [encmask, context]

    def _get_ctx_t(self, ctx, states_tm1, encmask):
        if ctx.d.ndim == 2:     # static context
            ctx_t = ctx
        elif ctx.d.ndim > 2:
            # ctx is 3D, always dynamic context
            assert(self.attention is not None)
            h_tm1 = states_tm1[0]   # ??? --> will it also work with multi-state RNUs?
            ctx_t = self.attention(h_tm1, ctx, mask=encmask)
        return ctx_t

    def rec(self, x_t_emb, ctx_tm1, t, *args):  # x_t_emb: (batsize, embdim), context: (batsize, enc.innerdim)
        states_tm1 = args[:-2]
        ctx = args[-1]
        encmask = args[-2]
        #x_t_emb = self.embedder(x_t)  # i_t: (batsize, embdim)
        # do inconcat
        i_t = T.concatenate([x_t_emb, ctx_tm1], axis=1) if self.inconcat else x_t_emb
        rnuret = self.block.rec(i_t, *states_tm1)
        t += 1
        h_t = rnuret[0]
        states_t = rnuret[1:]
        ctx_t = self._get_ctx_t(ctx, states_t, encmask)  # get context with attention
        _y_t = T.concatenate([h_t, ctx_t], axis=1) if self.outconcat else h_t
        y_t = self.softmaxoutblock(_y_t)
        return [y_t, ctx_t, t] + states_t

# ----------------------------------------------------------------------------------------------------------------------

# TODO: travis error messages about theano optimization and shapes only involve things below
class SimpleEncoderDecoder(Block):  # gets two sequences of indexes for training
    def __init__(self, innerdim=50, input_vocsize=100, output_vocsize=100, **kw):
        super(SimpleEncoderDecoder, self).__init__(**kw)
        input_embedder = IdxToOneHot(input_vocsize)
        output_embedder = IdxToOneHot(output_vocsize)
        encrec = GRU(dim=input_vocsize, innerdim=innerdim)
        decrecrnu = GRU(dim=output_vocsize, innerdim=innerdim)
        self.enc = SeqEncoder(input_embedder, encrec)
        self.dec = SeqDecoder([output_embedder, decrecrnu], outconcat=True, inconcat=False, innerdim=innerdim+innerdim)

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco


class RNNAutoEncoder(Block):    # tries to decode original sequence
    def __init__(self, vocsize=25, encdim=200, innerdim=200, seqlen=50, **kw):
        super(RNNAutoEncoder, self).__init__(**kw)
        self.seqlen = seqlen
        self.encoder = SeqEncoder(
            IdxToOneHot(vocsize=vocsize),
            GRU(dim=vocsize, innerdim=encdim))
        self.decoder = SeqDecoder([IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
                                  innerdim=innerdim,
                                  inconcat=True
                                  )

    def apply(self, inpseq, outseq):    # inpseq: (batsize, seqlen), indexes
        enc = self.encoder(inpseq)
        dec = self.decoder(enc, outseq)
        return dec


class RewAttRNNEncDecoder(Block):
    '''
    Take the input index sequence as-is, transform to one-hot, feed to gate AttentionGenerator, encode with weighted SeqEncoder,
    put everything as attention inside SeqDecoder
    '''
    def __init__(self, vocsize=25, outvocsize=20, encdim=200, innerdim=200, attdim=50, **kw):
        super(RewAttRNNEncDecoder, self).__init__(**kw)
        self.emb = IdxToOneHot(vocsize)
        attgen = LinearGateAttentionGenerator(indim=innerdim+vocsize, innerdim=attdim)
        attcon = SeqEncoder(None,
            GRU(dim=vocsize, innerdim=encdim))
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim, innerdim=innerdim)],
                              attention=Attention(attgen, attcon),
                              inconcat=True,
                              innerdim=innerdim)

    def apply(self, inpseq, outseq):    # inpseq: indexes~(batsize, seqlen)
        inp = self.emb(inpseq)  # inp:    floats~(batsize, seqlen, vocsize)
        return self.dec(inp, outseq)


class FwdAttRNNEncDecoder(Block):
    '''
    Take the input index sequence as-is, transform to one-hot, feed to gate AttentionGenerator, encode with weighted SeqEncoder,
    put everything as attention inside SeqDecoder
    '''
    def __init__(self, vocsize=25, outvocsize=20, encdim=200, innerdim=200, attdim=50, **kw):
        super(FwdAttRNNEncDecoder, self).__init__(**kw)
        self.emb = IdxToOneHot(vocsize)
        attgen = LinearGateAttentionGenerator(indim=innerdim+vocsize, innerdim=attdim)
        attcon = SeqEncoder(None,
            GRU(dim=vocsize, innerdim=encdim))
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True, inconcat=False,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim)

    def apply(self, inpseq, outseq):    # inpseq: indexes~(batsize, seqlen)
        inp = self.emb(inpseq)  # inp:    floats~(batsize, seqlen, vocsize)
        return self.dec(inp, outseq)


class RewAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=200, innerdim=200, attdim=50, **kw):
        super(RewAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim)).all_outputs()
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim, innerdim=innerdim)],
                              inconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim)

    def apply(self, inpseq, outseq):        # inpseq: indexes~(batsize, seqlen)
        rnnout = self.rnn(inpseq)   # (batsize, seqlen, encdim)
        return self.dec(rnnout, outseq)     # (batsize, seqlen, vocsize)


class FwdAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(FwdAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim)).all_outputs()
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True, inconcat=False,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)


class BiFwdAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(BiFwdAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim)).all_outputs()
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim*2, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True, inconcat=False,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim*2
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)


class BiRewAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(BiRewAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim)).all_outputs()
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim*2, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim*2, innerdim=innerdim)],
                              inconcat=True, outconcat=False,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)


class MakeRNU(object):
    ''' generates a list of RNU's'''
    @staticmethod
    def make(initdim, specs, rnu=GRU, bidir=False,
             zoneout=False, dropout_in=False, dropout_h=False):
        if not issequence(specs):
            specs = [specs]
        rnns = []
        prevdim = initdim
        for spec in specs:
            fspec = {"dim": None, "bidir": bidir, "rnu": rnu}
            if isinstance(spec, int):
                fspec["dim"] = spec
            elif isinstance(spec, dict):
                assert(hasattr(spec, "dim")
                       and
                       set(spec.keys()).union(set(fspec.keys()))
                            == set(fspec.keys()))
                fspec.update(spec)
            if fspec["bidir"] == True:
                rnn = BiRNU.fromrnu(fspec["rnu"], dim=prevdim, innerdim=fspec["dim"])
                prevdim = fspec["dim"] * 2
            else:
                rnn = fspec["rnu"](dim=prevdim, innerdim=fspec["dim"],
                                   zoneout=zoneout, dropout_h=dropout_h, dropout_in=dropout_in)
                prevdim = fspec["dim"]
            rnns.append(rnn)
        return rnns, prevdim

    @staticmethod
    def fromdims(innerdim, rnu=GRU):
        assert(len(innerdim) >= 2)
        initdim = innerdim[0]
        otherdim = innerdim[1:]
        return MakeRNU.make(initdim, otherdim, rnu=rnu)

