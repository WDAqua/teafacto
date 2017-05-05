from base import Block
from teafacto.blocks.seq.rnn import RecStack
from teafacto.blocks.seq.rnu import RNUBase


def stack(*layers, **kw):
    rec = False
    for layer in layers:
        if isinstance(layer, RNUBase):
            rec = True
            break
    if rec is True:
        return RecStack(*layers, **kw)
    else:
        return BlockStack(*layers, **kw)


class BlockStack(Block):
    def __init__(self, *layers, **kw):
        super(BlockStack, self).__init__(**kw)
        self.layers = layers

    def apply(self, *vars):
        ret = vars
        for layer in self.layers:
            ret = [layer(ret[0])]
            assert(len(ret) == 1)
        return ret[0]

    def __getitem__(self, idx):
        return self.layers[idx]