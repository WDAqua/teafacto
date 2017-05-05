from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot
from teafacto.blocks.lang.wordembed import WordEmbedGlove
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.memory import MemoryBlock


class TestMemoryBlock(TestCase):
    def setUp(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"

    def test_memory_block_with_seq_encoder_dynamic_fail(self):
        invocabsize = 5
        encdim = 13
        gru = GRU(dim=invocabsize, innerdim=encdim)
        payload = SeqEncoder(
            IdxToOneHot(vocsize=invocabsize),
            gru
        )
        dynmemb = MemoryBlock(payload, outdim=encdim)
        idxs = [0, 2, 5]
        #dynmemb.predict(idxs)
        self.assertRaises(AssertionError, lambda: dynmemb.predict(idxs))

    def test_memory_block_with_glove_embedder(self):
        payload = WordEmbedGlove()
        original_embedding = payload.predict([6])
        memb = MemoryBlock(payload, np.asarray([1, 2, 6]), outdim=payload.outdim)
        memory_embedding = memb.predict([2])
        self.assertTrue(np.allclose(original_embedding, memory_embedding))

    def test_memory_block_with_glove_embedder_params(self):
        payload = WordEmbedGlove()
        original_embedding = payload.predict([6])
        memb = MemoryBlock(payload, np.asarray([1, 2, 6]), outdim=payload.outdim)
        memory_embedding = memb.predict([2])
        self.assertTrue(np.allclose(original_embedding, memory_embedding))
        print memb.output.allparams

    def test_memory_block_with_seq_encoder(self):
        invocabsize = 5
        memsize = 10
        seqlen = 3
        encdim = 13
        data = np.random.randint(0, invocabsize, (memsize, seqlen))
        gru = GRU(dim=invocabsize, innerdim=encdim)
        payload = SeqEncoder(
            IdxToOneHot(vocsize=invocabsize),
            gru
        )
        memb = MemoryBlock(payload, data, indim=invocabsize, outdim=encdim)
        idxs = [0, 2, 5]
        memory_element = memb.predict(idxs)
        self.assertEqual(memory_element.shape, (len(idxs), encdim))
        gruparams = set([getattr(gru, pname) for pname in gru.paramnames])
        allparams = set(memb.output.allparams)
        self.assertEqual(gruparams.intersection(allparams), allparams)

    def test_memory_block_with_seq_encoder_dynamic(self):
        invocabsize = 5
        memsize = 10
        seqlen = 3
        encdim = 13
        data = np.random.randint(0, invocabsize, (memsize, seqlen))
        gru = GRU(dim=invocabsize, innerdim=encdim)
        payload = SeqEncoder(
            IdxToOneHot(vocsize=invocabsize),
            gru
        )
        dynmemb = MemoryBlock(payload, outdim=encdim)
        idxs = [0, 2, 5]
        memory_element = dynmemb.predict(idxs, data)
        self.assertEqual(memory_element.shape, (len(idxs), encdim))
        gruparams = set([getattr(gru, pname) for pname in gru.paramnames])
        allparams = set(dynmemb.output.allparams)
        self.assertEqual(gruparams.intersection(allparams), allparams)

        statmemb = MemoryBlock(payload, data, outdim=encdim)
        statpred = statmemb.predict(idxs)
        self.assertTrue(np.allclose(statpred, memory_element))

    def test_memory_block_with_seq_encoder_static_fail(self):
        invocabsize = 5
        memsize = 10
        seqlen = 3
        encdim = 13
        data = np.random.randint(0, invocabsize, (memsize, seqlen))
        gru = GRU(dim=invocabsize, innerdim=encdim)
        payload = SeqEncoder(
            IdxToOneHot(vocsize=invocabsize),
            gru
        )
        memb = MemoryBlock(payload, data, outdim=encdim)
        idxs = [0, 2, 5]
        self.assertRaises(AssertionError, lambda: memb.predict(idxs, data))


