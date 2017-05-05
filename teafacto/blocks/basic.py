from teafacto.core.base import Block, tensorops as T, param, Val, Var, Parameter
from teafacto.util import issequence, isfunction
import numpy as np


class ConcatBlock(Block):
    def __init__(self, *blocks, **lw):
        super(ConcatBlock, self).__init__(**lw)
        self.blocks = blocks
        self.axis = lw["axis"] if "axis" in lw else 1
        self.argfun = lw["argfun"] if "argfun" in lw else None

    def apply(self, *args):  # args is a tuple of tuples of *args and **kwargs for each of the blocks in the concatenation
        res = []
        for block, arg in zip(self.blocks, args):
            if self.argfun is not None:
                arglist, argdic = self.argfun(arg)
            elif issequence(arg):
                assert(len(arg) < 3 and len(arg) > 0)
                arglist = arg[0]
                argdic = arg[1] if len(arg) > 1 else {}
            elif isinstance(arg, (Var, Val)):
                arglist = [arg]
                argdic = {}
            else:
                raise Exception("something wrong with concat's arguments: " + str(args))
            res.append(block(*arglist, **argdic))
        return T.concatenate(res, axis=self.axis)


class Softmax(Block):
    def apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class MatDot(Block):
    def __init__(self, indim, dim, init="uniform", **kw):
        super(MatDot, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.W = param((self.indim, self.dim), name="matdot").init(init)

    def apply(self, inptensor):
        return T.dot(inptensor, self.W)


class Linear(Block):
    def __init__(self, indim, dim, w_init="uniform", b_init="uniform", **kw):
        super(Linear, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.W = param((self.indim, self.dim)).init(w_init)
        self.b = param((self.dim,)).init(b_init)

    def apply(self, inp):
        return T.dot(inp, self.W) + self.b


class Embedder(Block):
    def __init__(self, indim=None, outdim=None, normalize=False, trainfrac=1., **kw):
        super(Embedder, self).__init__(**kw)
        assert(indim is not None and outdim is not None)
        self.indim = indim
        self.outdim = outdim
        self.normalize = normalize
        self.trainfrac = trainfrac

    def apply(self, idxs):
        raise NotImplementedError("use subclass")


class IdxToOneHot(Embedder):
    def __init__(self, vocsize, **kw):
        super(IdxToOneHot, self).__init__(vocsize, vocsize, **kw)
        self.W = Val(np.eye(vocsize, vocsize))

    def apply(self, inp):
        return self.W[inp, :]

class Eye(Block):
    def __init__(self, dim=None, **kw):
        super(Eye, self).__init__(**kw)
        self.outdim = dim

    def apply(self, inp):
        return inp


class Masker(Block):
    def __init__(self, maskid=None, **kw):
        self.maskid = maskid
        super(Masker, self).__init__(**kw)

    def apply(self, x):
        if self.maskid is not None:
            return T.eq(x, self.maskid)
        else:
            return x

from teafacto.core.base import RVal


class Dropout(Block):
    def __init__(self, p=0.3, seed=None, rescale=True, _alwaysrandom=False, **kw):
        super(Dropout, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.p = 0.0 if (p is False or p is None) else 0.3 if p is True else p
        self.rescale = rescale
        self.seed = seed
        self._debug = _alwaysrandom
        self.rval = RVal(self.seed)

    def apply(self, x, _trainmode=False):
        if (_trainmode or self._debug) and self.p > 0:
            #print "Dropout: YES"
            xmask = x.mask
            if self.rescale:
                one = T.constant(1)
                x /= one - self.p
            #rng = RVal(self.seed)
            rv = self.rval.binomial(x.shape, p=1-self.p, dtype=x.dtype)
            x = x * rv
            #print "done dropout"
            x.mask = xmask
            # x.push_extra_outs({"dropout{}".format(np.random.randint(100, 199)): rv})
            return x
        else:
            #print "Dropout: NO"
            return x


class VectorEmbed(Embedder):
    def __init__(self, indim=None, dim=None, value=None,
                 normalize=False, trainfrac=1.0, init=None, maskid=None, **kw):
        super(VectorEmbed, self).__init__(indim, dim, normalize=normalize,
                                          trainfrac=trainfrac, **kw)
        self.maskid = maskid
        if value is None:
            self.W = param((indim, dim), lrmul=self.trainfrac, name="embedder")
            if init == "zero":
                self.W = self.W.constant(0.0)
            elif init in ["glorot", None]:
                self.W = self.W.glorotuniform()
            elif init == "uniform":
                self.W = self.W.uniform()
        elif value is False:
            self.W = None       # no initialization
        else:
            self.setvalue(value)
            self.indim, self.outdim = value.shape
        if self.normalize:
            self.W = self.W.normalize(axis=1)
        # assertions
        assert(self.W is None or self.W.d.get_value().shape == (self.indim, self.outdim))

    def setvalue(self, v):
        if self.trainfrac == 0.0:
            self.W = Val(v, name="embedder_val")
        else:
            self.W = Parameter(v, lrmul=self.trainfrac, name="embedder")

    def apply(self, inptensor):
        ret = self.W[inptensor]
        self._maskfrom(ret, inptensor)
        return ret

    def _maskfrom(self, ret, x):
        if self.maskid is not None:
            mask = T.neq(x, self.maskid)
        else:
            mask = None
        ret.mask = mask


class SMO(Block):   # softmax output layer
    def __init__(self, inner, outdim=2, nobias=False, **kw):
        super(SMO, self).__init__(**kw)
        self.inner = inner
        self.outdim = outdim
        self.outl = Linear(inner.outdim, outdim) if not nobias else MatDot(inner.outdim, outdim)

    def apply(self, *args, **kwargs):
        vec = self.inner(*args, **kwargs)
        ret = self.outl(vec)
        return Softmax()(ret)
