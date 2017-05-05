from teafacto.blocks.seq.attention import LinearGateAttentionGenerator, WeightedSumAttCon, Attention
from teafacto.blocks.seq.rnn import SeqEncoder, SeqDecoder, RecStack
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import Softmax, MatDot as Lin, VectorEmbed, ConcatBlock
from teafacto.blocks.lang.wordembed import WordEncoderPlusGlove, WordEmbed
from teafacto.blocks.memory import MemoryBlock, GeneralDotMemAddr, DotMemAddr, TransDotMemAddr
from teafacto.core.base import Block
from teafacto.core.stack import stack


class FBBasicCompositeEncoder(Block):    # SeqEncoder of WordEncoderPlusGlove, fed to single-layer Softmax output
    def __init__(self, wordembdim=50, wordencdim=100, innerdim=200, outdim=1e4, numwords=4e5, numchars=128, glovepath=None, **kw):
        super(FBBasicCompositeEncoder, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.innerdim = innerdim

        self.enc = SeqEncoder(
            WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath),
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.innerdim)
        )

        self.out = Lin(indim=self.innerdim, dim=self.outdim)

    def apply(self, inp):
        enco = self.enc(inp)
        ret = Softmax()(self.out(enco))
        return ret


class FBSeqCompositeEncDec(Block):
    '''
    The input sequence is encoded into a vector with a GRU.
    Each input sequence element is mapped to a vector with the composite Glove + character encoding block.
    The encoding is passed to the decoder, as part of the decoder RNN's input.
    No attention in this model.

    '''
    def __init__(self, wordembdim=50, wordencdim=100, entembdim=200, innerdim=200, outdim=1e4, numwords=4e5, numchars=128, glovepath=None, **kw):
        super(FBSeqCompositeEncDec, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.encinnerdim = innerdim
        self.entembdim = entembdim
        self.decinnerdim = innerdim

        self.enc = SeqEncoder(
            WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath),
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        self.dec = SeqDecoder(
            [VectorEmbed(indim=self.outdim, dim=self.entembdim), GRU(dim=self.entembdim+self.encinnerdim, innerdim=self.decinnerdim)],
            inconcat=True,
            innerdim=self.decinnerdim,
        )

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco

    def fixO(self, lr=0.0):
        self.dec.lin.W.lrmul = lr


class FBSeqSimpEncDecAtt(Block):
    def __init__(self, wordembdim=50, entembdim=200, innerdim=200, attdim=100, outdim=1e4, numwords=4e5, **kw):
        super(FBSeqSimpEncDecAtt, self).__init__(**kw)
        self.indim = wordembdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.encinnerdim = innerdim
        self.decinnerdim = innerdim
        self.entembdim = entembdim

        self.wordencoder = WordEmbed(indim=numwords, outdim=self.wordembdim, trainfrac=1.0)
        self.rnn = RecStack(self.wordencoder, GRU(dim=self.wordembdim, innerdim=self.encinnerdim))

        attgen = LinearGateAttentionGenerator(indim=self.encinnerdim + self.decinnerdim, attdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder(
            [VectorEmbed(indim=self.outdim, dim=self.entembdim), GRU(dim=self.entembdim, innerdim=self.decinnerdim)],
            attention=Attention(attgen, attcon),
            outconcat=True, inconcat=False,
            innerdim=self.encinnerdim + self.decinnerdim
        )

    def apply(self, inpseq, outseq):
        enco = self.rnn(inpseq)
        deco = self.dec(enco, outseq, initstates=[enco[:, -1, :]])
        return deco



class FBSeqCompEncDecAtt(Block):
    def __init__(self, wordembdim=50, wordencdim=50, entembdim=200, innerdim=200, attdim=100, outdim=1e4, numwords=4e5, numchars=128, glovepath=None, **kw):
        super(FBSeqCompEncDecAtt, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.encinnerdim = innerdim
        self.entembdim = entembdim
        self.decinnerdim = innerdim

        self.wordencoder = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath)

        self.rnn = RecStack(self.wordencoder, GRU(dim=wordembdim+wordencdim, innerdim=self.encinnerdim))
        attgen = LinearGateAttentionGenerator(indim=self.encinnerdim + self.decinnerdim, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder(
            [VectorEmbed(indim=self.outdim, dim=self.entembdim), GRU(dim=self.entembdim, innerdim=self.decinnerdim)],
            attention=Attention(attgen, attcon),
            outconcat=True, inconcat=False,
            innerdim=self.encinnerdim + self.decinnerdim)

    def apply(self, inpseq, outseq):
        enco = self.rnn(inpseq)         # (batsize, inpseqlen, inpencdim)# set init state of decoder to last encoding
        deco = self.dec(enco, outseq, initstates=[enco[:, -1, :]])
        return deco

    def fixO(self, lr=0.0):
        self.dec.lin.W.lrmul = lr


class FBMemMatch(Block):
    def __init__(self,  entembdim=50,
                        wordembdim=50,
                        wordencdim=100,
                        memdata=None,
                        attdim=100,
                        numchars=128,       # number of different chars
                        numwords=4e5,       # number of different words
                        glovepath=None,
                        innerdim=100,       # dim of memory payload encoder output
                        outdim=1e4,         # number of entities
                        memaddr=DotMemAddr, **kw):
        super(FBMemMatch, self).__init__(**kw)
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.entembdim = entembdim
        self.attdim = attdim
        self.encinnerdim = innerdim
        self.outdim = outdim

        memaddr = TransDotMemAddr

        # memory encoder per word
        #wencpg = WordEmbed(indim=numwords, outdim=self.wordembdim, trainfrac=1.0)
        wordencoder = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim,
                                      embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath)

        # memory encoder for one cell
        self.phraseencoder = SeqEncoder(
            wordencoder,
            GRU(dim=self.wordembdim + self.wordencdim,
                innerdim=self.encinnerdim)
        )
        # entity embedder
        entemb = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.entembs = entemb(memdata[0]) #Val(np.arange(0, self.outdim, dtype="int32")))
        # memory block
        self.mempayload = self.phraseencoder #ConcatBlock(entemb, self.phraseencoder)
        self.memblock = MemoryBlock(self.mempayload, memdata[1], indim=self.outdim,
                                    outdim=self.encinnerdim)# + self.entembdim)
        # memory addressing
        self.mema = memaddr(self.memblock,
                       memdim=self.memblock.outdim, attdim=attdim, indim=self.encinnerdim)
        #mema = Lin(indim=self.encinnerdim, dim=self.outdim)

        # for testing purposes
        #self.W = param((self.encinnerdim, self.entembdim), name="dummy_W").uniform()

    def apply(self, inpseq):    # (batsize, amwords, amchars+1)
        inpenc = self.phraseencoder(inpseq)   # (batsize, encdim)
        #scores = T.dot(inpenc, self.W)          # (batsize, memdim)
        #scores = T.nnet.sigmoid(T.dot(scores, self.memblock.innervar.T))      # (batsize, memsize)
        #return Softmax()(scores)                    #
        return Softmax()(self.mema(inpenc))


class FBSeqCompositeEncMemDec(Block):
    def __init__(self,  wordembdim=50,
                        wordencdim=100,
                        entembdim=200,
                        innerdim=200,
                        outdim=1e4,
                        numwords=4e5,
                        numchars=128,
                        glovepath=None,
                        memdata=None,
                        attdim=100,
                        memaddr=GeneralDotMemAddr, **kw):
        super(FBSeqCompositeEncMemDec, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.encinnerdim = innerdim
        self.entembdim = entembdim
        self.decinnerdim = innerdim
        self.numchars = numchars
        self.numwords = numwords
        self.glovepath = glovepath
        self.memdata = memdata
        self.attdim = attdim
        self.memaddr = memaddr

        self.init()

    def init(self):
        #memory
        wencpg = WordEncoderPlusGlove(numchars=self.numchars, numwords=self.numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=self.glovepath)
        self.memenco = SeqEncoder(
            wencpg,
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        entemb = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.mempayload = ConcatBlock(entemb, self.memenco)
        self.memblock = MemoryBlock(self.mempayload, self.memdata, indim=self.outdim, outdim=self.encinnerdim+self.entembdim)

        #encoder
        wencpg2 = WordEncoderPlusGlove(numchars=self.numchars, numwords=self.numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=self.glovepath)
        self.enc = SeqEncoder(
            wencpg2,
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        #decoder
        entemb2 = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.softmaxoutblock = stack(self.memaddr(self.memblock, indim=self.decinnerdim, memdim=self.memblock.outdim, attdim=self.attdim), Softmax())
        self.dec = SeqDecoder(
            [entemb2,  #self.memblock,
             GRU(dim=entemb.outdim + self.encinnerdim, innerdim=self.decinnerdim),             # GRU(dim=self.memblock.outdim + self.encinnerdim, innerdim=self.decinnerdim),
             ],
            inconcat=True,
            innerdim=self.decinnerdim,
            softmaxoutblock=self.softmaxoutblock
        )

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco


class FBSeqCompEncMemDecAtt(FBSeqCompositeEncMemDec):
    def init(self):
        #MEMORY: encodes how entity is written + custom entity embeddings
        wencpg = WordEncoderPlusGlove(numchars=self.numchars, numwords=self.numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=self.glovepath)
        self.memenco = SeqEncoder(
            wencpg,
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        entemb = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.mempayload = ConcatBlock(entemb, self.memenco)
        self.memblock = MemoryBlock(self.mempayload, self.memdata, indim=self.outdim, outdim=self.encinnerdim+self.entembdim)

        #ENCODER: uses the same language encoder as memory
        #wencpg2 = WordEncoderPlusGlove(numchars=self.numchars, numwords=self.numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath)
        self.enc = RecStack(wencpg, GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim))

        #ATTENTION
        attgen = LinearGateAttentionGenerator(indim=self.encinnerdim + self.decinnerdim, innerdim=self.attdim)
        attcon = WeightedSumAttCon()

        #DECODER
        #entemb2 = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.softmaxoutblock = stack(
            self.memaddr(
                self.memblock,
                indim=self.decinnerdim + self.encinnerdim,
                memdim=self.memblock.outdim,
                attdim=self.attdim),
            Softmax())

        self.dec = SeqDecoder(
            [self.memblock, GRU(dim=self.entembdim + self.encinnerdim, innerdim=self.decinnerdim)],
            outconcat=True, inconcat=False,
            attention=Attention(attgen, attcon),
            innerdim=self.decinnerdim + self.encinnerdim,
            softmaxoutblock=self.softmaxoutblock
        )


