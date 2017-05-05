from teafacto.core.base import Block, param, tensorops as T, Val
import numpy as np
from teafacto.util import issequence
from teafacto.blocks.activations import Tanh, ReLU
from teafacto.blocks.basic import VectorEmbed


class CNNEnc(Block):
    def __init__(self, indim=100, innerdim=200, window=5,
                 poolmode="max", activation=Tanh, stride=1, **kw):
        super(CNNEnc, self).__init__(**kw)
        self.layers = []
        if not issequence(innerdim):
            innerdim = [innerdim]
        if not issequence(window):
            window = [window] * len(innerdim)
        if not issequence(activation):
            activation = [activation()] * len(innerdim)
        else:
            activation = [act() for act in activation]
        if not issequence(stride):
            stride = [stride] * len(innerdim)
        assert(len(window) == len(innerdim))
        innerdim = [indim] + innerdim
        for i in range(1, len(innerdim)):
            layer = Conv1D(indim=innerdim[i-1], outdim=innerdim[i],
                           window=window[i-1], stride=stride[i-1])
            self.layers.append(layer)
            self.layers.append(activation[i-1])
        self.layers.append(GlobalPool1D(mode=poolmode))

    def apply(self, x, mask=None):
        mask = x.mask if mask is None else mask
        acc = x
        acc.mask = mask
        for layer in self.layers:
            acc = layer(acc)
        return T.cast(acc, "float32")        # TODO: why is "float64" returned?


class CNNSeqEncoder(CNNEnc):
    def __init__(self, indim=500, inpembdim=100, inpemb=None, innerdim=200,
                 window=5, poolmode="max", activation=Tanh, maskid=None, **kw):
        if inpemb is None:
            self.embedder = VectorEmbed(indim, inpembdim, maskid=maskid)
        else:
            self.embedder = inpemb
            inpembdim = inpemb.outdim
        super(CNNSeqEncoder, self).__init__(indim=inpembdim, innerdim=innerdim,
                                            window=window, poolmode=poolmode,
                                            activation=activation, **kw)

    def apply(self, x, mask=None):
        acc = self.embedder(x)
        ret = super(CNNSeqEncoder, self).apply(acc, mask=mask)
        return ret


class Conv1D(Block):
    def __init__(self, indim=50, outdim=50, window=5,
                 border_mode="half", stride=1, filter_flip=True, **kw):
        super(Conv1D, self).__init__(**kw)
        if isinstance(border_mode, tuple):
            (border_mode,) = border_mode
        if isinstance(border_mode, int):
            border_mode = (border_mode, 0)
        self.border_mode = border_mode
        self.stride = stride
        self.filter_flip = filter_flip
        self.filter_shape = (outdim, indim, window, 1)
        self.filter = param(self.filter_shape, name="conv_w").glorotuniform()
        self.maskfilter_shape = (1, 1, window, 1)
        self.maskfilter = Val(np.ones(self.maskfilter_shape, dtype="float32"))

    def apply(self, x, mask=None):     # (batsize, seqlen, dim)
        mask = x.mask if mask is None else mask
        if mask is not None:
            assert(mask.ndim == x.ndim - 1) # mask must be (batsize, seqlen)
            realm = T.cast(T.tensordot(mask, T.ones((x.shape[-1],), dtype="int32"), 0), "float32")
            x = x * realm
        input = x.dimshuffle(0, 2, 1, 'x')
        input_shape = None #input.shape
        convout = T.nnet.conv2d(input, self.filter, input_shape, self.filter_shape,
                            border_mode=self.border_mode, subsample=(self.stride, 1),
                            filter_flip=self.filter_flip)
        ret = convout[:, :, :, 0].dimshuffle(0, 2, 1)
        if mask is not None:    # compute new mask
            print "conving the mask"
            mask_shape = None
            maskout = T.nnet.conv2d(T.cast(mask.dimshuffle(0, "x", 1, "x"), "float32"),
                                    self.maskfilter, mask_shape, self.maskfilter_shape,
                                    border_mode=self.border_mode, subsample=(self.stride, 1),
                                    filter_flip=self.filter_flip)
            mask = T.cast(maskout[:, 0, :, 0] > 0, "int32")
        ret.mask = mask
        return ret


class GlobalPool1D(Block):
    def __init__(self, mode="max", **kw):
        super(GlobalPool1D, self).__init__(**kw)
        self.mode = mode

    def apply(self, x, mask=None):  # (batsize, seqlen, dim)
        mask = x.mask if mask is None else mask
        if mask is not None:
            assert(mask.ndim == x.ndim - 1)
            realm = T.tensordot(mask, T.ones((x.shape[-1],)), 0)
            if self.mode == "max":
                x = T.switch(realm, x, np.infty * (realm - 1))
            else:
                x = x * realm
        if self.mode == "max":
            ret = T.max(x, axis=-2)
        elif self.mode == "sum":
            ret = T.sum(x, axis=-2)
        elif self.mode == "avg":
            ret = T.sum(x, axis=-2) / x.shape[-2]
        else:
            raise Exception("unknown pooling mode: {:3s}".format(self.mode))
        # ret: (batsize, dim)
        if mask is not None:
            mask = 1 * (T.sum(mask, axis=-1) > 0)
            ret = T.switch(T.tensordot(mask, T.ones((x.shape[-1],)), 0),
                           ret, T.zeros_like(ret))
            ret.mask = mask
        return ret


