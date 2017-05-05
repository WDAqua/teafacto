from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.enc import SimpleSeq2Vec
from teafacto.blocks.seq.enc import Seq2Vec
from teafacto.blocks.lang.wordembed import WordEncoder, WordEncoderPlusGlove, WordEmbedPlusGlove
from teafacto.blocks.lang.wordvec import Glove


class TestWordEncoder(TestCase):
    def test_word_encoder_output_shape(self):
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        data = np.random.randint(0, numchars, (batsize, wordlen))
        encdim = 100
        block = WordEncoder(indim=numchars, outdim=encdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim))


class TestWordEncoderPlusGlove(TestCase):
    def test_word_encoder_output_shape(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        numwords = 1000
        worddata = np.random.randint(0, numwords, (batsize, 1))
        chardata = np.random.randint(0, numchars, (batsize, wordlen))
        data = np.concatenate([worddata, chardata], axis=1)
        encdim = 100
        embdim = 50
        block = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=encdim, embdim=embdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim+embdim))

    def test_char_auto_mask(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 11
        seqlen = 13
        wordlen = 3
        blank = 2
        numchars = 20
        numwords = 100
        worddata = np.random.randint(0, numwords, (batsize, 1))
        chardata = np.random.randint(0, numchars, (batsize, wordlen))
        blank = np.zeros((batsize, blank)).astype("int32") - 1
        data = np.concatenate([worddata, chardata, blank], axis=1)
        encdim = 4
        embdim = 50
        block = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=encdim, embdim=embdim, maskid=-1)
        block.enc.enc.all_outputs()
        pred = block.enc.enc.predict(data[:, 1:])
        i = 1
        while i < pred.shape[1]:
            self.assertEqual(np.allclose(pred[:, i-1, :], pred[:, i, :]), i >= wordlen)
            i += 1

    def test_auto_mask_within_seq2vec(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 11
        seqlen = 3
        seqblank = 2
        wordlen = 3
        wordblank = 2
        numchars = 20
        numwords = 100
        encdim = 4
        embdim = 50
        innerdim = 2

        worddata = np.random.randint(0, numwords, (batsize, seqlen, 1))
        worddatablank = np.zeros((batsize, seqblank, 1)).astype("int32") - 1
        worddata = np.concatenate([worddata, worddatablank], axis=1)
        chardata = np.random.randint(0, numchars, (batsize, seqlen, wordlen))
        charblank = np.zeros((batsize, seqlen, wordblank)).astype("int32") - 1
        chardata = np.concatenate([chardata, charblank], axis=2)
        charblankblank = np.zeros((batsize, seqblank, wordlen+wordblank)).astype("int32") - 1
        chardata = np.concatenate([chardata, charblankblank], axis=1)
        data = np.concatenate([worddata, chardata], axis=2)

        wordemb = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=encdim, embdim=embdim,
                                       maskid=-1,
                                       embtrainfrac=0)
        rnn, lastdim = SimpleSeq2Vec.makernu(embdim + encdim, innerdim, bidir=False)
        enc = Seq2Vec(wordemb, rnn, maskid=-1)
        enc.enc.with_outputs()
        finalpred, pred = enc.predict(data)
        #print pred.shape, finalpred.shape
        #print pred[0], finalpred[0]
        i = 1
        while i < pred.shape[1]:
            self.assertEqual(np.allclose(pred[:, i-1, :], pred[:, i, :]), i >= seqlen)
            i += 1


class TestWordEmbedPlusGlove(TestCase):
    def test_word_embed_plus_glove_output_shape(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        numwords = 1000
        data = np.random.randint(0, numwords, (batsize,))
        encdim = 100
        embdim = 50
        block = WordEmbedPlusGlove(indim=numwords, outdim=encdim, embdim=embdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim+embdim))

