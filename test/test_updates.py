from unittest import TestCase
from teafacto.core.base import Var, Val, Input, tensorops as T
from teafacto.blocks.basic import Linear
from teafacto.blocks.seq.rnn import RNNSeqEncoder
import numpy as np


class TestUpdateProp(TestCase):
    def test_update_propagation_through_operations(self):
        x = Input(ndim=1, dtype="float32")
        x.push_updates({"a": "b"})
        y = x + 1
        self.assertEqual(y.allupdates, x.allupdates)

    def test_update_propagation_through_basic_blocks(self):
        x = Input(ndim=2, dtype="float32")
        x.push_updates({"a":"b"})
        y = Linear(5, 6)(x)
        self.assertEqual(y.allupdates, x.allupdates)

    def test_update_propagation_through_scan(self):
        x = Input(ndim=2, dtype="float32")
        x.push_updates({"a": "b"})
        def f(ix):
            return ix + 1
        y, _ = T.scan(f, x)
        self.assertEqual(y.allupdates, x.allupdates)

    def test_update_propagation_through_rec_blocks(self):
        x = Input(ndim=3, dtype="float32")
        x.push_updates({"a": "b"})
        enc = RNNSeqEncoder(inpemb=False, innerdim=10)
        y = enc(x)
        self.assertEqual(x.allupdates, y.allupdates)

    def test_update_propagation_and_addition_through_dropout_rec_blocks(self):
        x = Input(ndim=3, dtype="float32")
        x.push_updates({"a": "b"})
        enc = RNNSeqEncoder(inpemb=False, innerdim=10, dropout_in=0.3, dropout_h=0.3)
        y = enc(x, _trainmode=True)
        for oldupdatesrc in x.allupdates:
            self.assertIn(oldupdatesrc, y.allupdates)
            self.assertEqual(x.allupdates[oldupdatesrc], y.allupdates[oldupdatesrc])
            del y.allupdates[oldupdatesrc]
        self.assertEqual(len(y.allupdates), 2)