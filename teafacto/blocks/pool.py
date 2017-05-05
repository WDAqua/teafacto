from teafacto.core.base import Block
from teafacto.core.base import tensorops as T


class OneDPool(Block):
    def __init__(self, mode="max", **kw):
        """
        :param axis:
        :param stride:
        :param pad:
        :param mode:
        :param ignore_border:
        :param kw:
        """
        self.mode = mode

    def apply(self, x, mask=None):  # (batsize, seqlen, dim) and (batsize, seqlen)
        if mask is None:
            mask = T.zeros((x.shape[0], x.shape[1]))

        T.scan(fn=self.rec, sequences=[x.dimswap(1, 0), mask.dimswap(1, 0)])


class Pool(Block):
    def __init__(self, size, axis=None, stride=None, pad=None,
                 mode="max", ignore_border=True, **kw):
        """
        General pooling block.
        For 1D pooling, axis, size, stride and pad must be tuples of 1 element
        For 2D pooling, axis, size, stride and pad must be tuples of 2 elements
        :param axis:            along which axis of the N(>=2)-dim input to pool
                                By default, last axis is pooled over
        :param size:            size of the pooling region
                                (None,) or (None, None) ==> pooling over whole input
        :param stride:          how much stride
                                By default, no overlapping pooling regions
        :param pad:             how much to pad
                                By default, (0,) or (0,0) --> no padding
        :param mode:            {max, sum, average_inc_pad, average_exc_pad}
                                Default: max
        :param ignore_border:   partial pooling regions ignored
                                Default: True
        """
        # determine 1D or 2D, save and assure params compatible
        self.pdim = len(size)
        self.size = size
        self.axis = axis
        self.stride = stride
        self.pad = pad if pad is not None else tuple([0]*self.pdim)
        assert self.pdim >= 1
        assert self.axis is None or len(self.axis) == self.pdim
        assert self.stride is None or len(self.stride) == self.pdim
        assert len(self.pad) == self.pdim
        self.mode = mode
        self.ignore_border = ignore_border
        super(Pool, self).__init__(**kw)

    def apply(self, x, mask=None):
        assert x.ndim >= 2
        curaxes = range(x.ndim)
        if self.axis is not None:
            for axis in self.axis:
                for i in range(len(curaxes)):
                    if curaxes[i] == axis:
                        del curaxes[i]
                        break
                curaxes.append(axis)
            x = x.dimshuffle(*curaxes)

        ds = tuple(self.size)
        st = tuple(self.stride) if self.stride is not None else self.stride
        padding = tuple(self.pad)

        if set(self.size) == {None}:    # just do raw op over last dimensions
            axes = range(x.ndim)[-len(self.size):]     # last axes
            if self.mode == "max":
                pooled = x.max(axis=axes)
            elif self.mode == "sum":
                pooled = x.sum(axis=axes)
            elif "average" in self.mode:
                pooled = x.sum(axis=axes) / \
                         (1.* reduce(lambda x, y: x * y,
                                     [x.shape[i] for i in axes], 1.))
            else:
                raise Exception("mode not valid")
            if self.axis is not None:   # undimshuffle
                pass
            return pooled

        if self.pdim == 1: # make 1D pool through 2D pool
            x = T.shape_padright(x, 1)
            ds = (ds[0], 1)
            st = st if st is None else (st[0], 1)
            padding = (padding[0], 0)

        pooled = T.signal.pool.pool_2d(x,
                         ds=ds,
                         st=st,
                         padding=padding,
                         ignore_border=self.ignore_border,
                         mode=self.mode,
                         )
        ret = pooled
        if self.pdim == 1:
            ret = pooled.__getitem__(*[[slice(None, None, None)]*(pooled.ndim-1)+[0]])
        if self.axis is not None:   #undimshuffle
            argdic = dict(zip(curaxes, range(len(curaxes))))
            ret = ret.dimshuffle(*[argdic[d] for d in range(len(curaxes))])
        return ret


class MaxPool(Pool):
    def __init__(self, size, axis=None, stride=None, pad=None,
                 ignore_border=True, **kw):
        kw["mode"] = "max"
        super(MaxPool, self).__init__(size, axis=axis, stride=stride, pad=pad, ignore_border=ignore_border, **kw)


class SumPool(Pool):
    def __init__(self, size, axis=None, stride=None, pad=None,
                 ignore_border=True, **kw):
        kw["mode"] = "sum"
        super(SumPool, self).__init__(size, axis=axis, stride=stride, pad=pad, ignore_border=ignore_border, **kw)


class AvgPool(Pool):
    def __init__(self, size, axis=None, stride=None, pad=None,
                 ignore_border=True, inc_pad=False, **kw):
        kw["mode"] = "average_exc_pad" if inc_pad is False else "average_inc_pad"
        super(AvgPool, self).__init__(size, axis=axis, stride=stride, pad=pad, ignore_border=ignore_border, **kw)


class DeadPool(Pool):
    def __init__(self, **kw):
        pass
        super(DeadPool, self).__init__(**kw)