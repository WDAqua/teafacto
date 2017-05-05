from types import ModuleType
from collections import OrderedDict
import inspect
from IPython import embed
import theano
from lasagne.init import *
from lasagne.updates import norm_constraint
from theano import tensor
from theano.tensor.var import _tensor_py_operators

from teafacto.core.trainer import ModelTrainer, NSModelTrainer
from teafacto.util import isstring, issequence, isfunction, Saveable, isnumber
from teafacto.core.datafeed import DataFeed


_TRAINMODE = False
_DEBUGMODE = False


def recurmap(fun, data):
    if isinstance(data, dict):
        return type(data)(dict([(recurmap(fun, item[0]), recurmap(fun, item[1])) for item in data.items()]))
    elif isinstance(data, (tuple, list, set)):
        return type(data)([recurmap(fun, elem) for elem in data])
    else:
        return fun(data)

def recurfilter(fun, data):
    acc = []
    if isinstance(data, dict):
        data = data.items()
    if isinstance(data, (tuple, list, set)):
        for elem in data:
            ret = recurfilter(fun, elem)
            acc.extend(ret)
    else:
        if fun(data):
            acc.append(data)
        else:
            acc.append(None)
    return filter(lambda x: x is not None, acc)


### DON'T WORRY ABOUT THIS
class TWrapper(type):
    def __getattr__(cls, item):
        top = getattr(tensor, item)
        return wrapf(top)

    @property
    def config(cls):
        return theano.config

    def scan(cls, fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False,mode=None, name=None, profile=False, allow_gc=None, strict=False):
        return scan()(fn, sequences=sequences, outputs_info=outputs_info, non_sequences=non_sequences, n_steps=n_steps,
                      truncate_gradient=truncate_gradient, go_backwards=go_backwards,mode=mode, name=name, profile=profile,
                      allow_gc=allow_gc, strict=strict)

    def until(cls, expr):
        return until(expr)

    def as_long_as(cls, expr):
        return until(cls.xor(expr, 1))     # xor?

    def softmax(cls, x, mask=None, temperature=1.):     # masked, multidim softmax
        xndim = x.ndim
        s = x.shape
        if xndim > 2:
            x = x.dimmove(xndim - 1, 0).flatten(2).T
            if mask is not None:
                mask = mask.dimmove(xndim - 1, 0).flatten(2).T
        x = x / temperature
        if mask is None:
            z = tensorops.nnet.softmax(x)
        else:
            o_exp = tensorops.exp(x - tensorops.max(x, axis=1).dimshuffle(0, 'x'))
            o_exp *= mask
            o_exp_sum = tensorops.sum(o_exp, axis=1)
            o_exp_sum = o_exp_sum.dimshuffle(0, 'x')
            z = o_exp / o_exp_sum
        if xndim > 2:
            z = z.reshape(s)
        return z


def wrapf(attr, root=None):
    if isfunction(attr): # real function
        innerwrap = OpBlock(attr, root)
    elif isnumber(attr) or isstring(attr): # or other literals/non-syms/modules/properties/...
        return attr
    elif isinstance(attr, ModuleType):
        innerwrap = WrappedAttr(attr)
    elif isinstance(attr, theano.Variable):
        innerwrap = Var(attr)
        if isinstance(root, Var):       # propagate params
            innerwrap.push_params(root._params)
    else:
        innerwrap = attr
    return innerwrap

"""
def prefwrap(attr, root):
    def innerprefwrap(*args, **kwargs):
        return fwrap(attr, root, *args, **kwargs)
    return innerprefwrap


def fwrap(attr, root, *args, **kwargs):
    params = recurfilter(lambda x: isinstance(x, Parameter), args)
    kwparams = recurfilter(lambda x: isinstance(x, Parameter), kwargs)
    wrapper = wrap(lambda *args, **kwargs: attr(*args, **kwargs), *(params+kwparams))
    ret = wrapper(*args, **kwargs)
    if root is not None:
        if isinstance(root, (Var, Val)):
            wrapper.add_parent(root)
        elif isinstance(root, Parameter):
            wrapper.add_param(root)
    return ret
"""

class WrappedAttr():
    def __init__(self, attr):
        self.attr = attr

    def __getattr__(self, item):
        return wrapf(getattr(self.attr, item))


class tensorops:
    __metaclass__ = TWrapper


class TensorWrapper(type):
    """Wrapper class that provides proxy access to an instance of some
       internal instance."""

    __ignore__ = "class mro new init setattr getattr getattribute getstate setstate dict"

    def __init__(cls, name, bases, dct):

        def make_proxy(name):   # only called for magic methods?
            def proxy(self, *args):
                attr = getattr(self.d, name)
                return wrapf(attr, root=self)
            return proxy

        ignore = set("__%s__" % n for n in cls.__ignore__.split())
        for name in _tensor_py_operators.__dict__:      #dir(_tensor_py_operators):
            if name.startswith("__"):
                if name not in ignore and name not in dct:
                    setattr(cls, name, property(make_proxy(name)))
        type.__init__(cls, name, bases, dct)


class TensorWrapped(object):
    __metaclass__ = TensorWrapper

    def __getattr__(self, item):
        if item in ["__%s__" % a for a in self.__metaclass__.__ignore__.split(" ")]:
            raise AttributeError()
        if item == "allparams":
            print self._name if hasattr(self, "_name") else "- - nameless - -"
            print self.dtype, type(self), dir(self)

        ret = getattr(self.d, item)

        return wrapf(ret, root=self)

    def dimswap(self, a, b):
        def tinner(v, a, b):
            dims = range(v.ndim)
            dims[a] = b
            dims[b] = a
            return v.dimshuffle(*dims)
        return OpBlock(tinner, name="dimswap")(self, a, b)

    def dimmove(self, dim, index):
        def tinner(v, val, idx):
            dims = range(v.ndim)
            del dims[val]
            dims.insert(idx, val)
            return v.dimshuffle(*dims)
        return OpBlock(tinner, name="dimmove")(self, dim, index)

    def reverse(self, *axes):
        """ axis can be an *int* or a sequence of *int*s"""
        if len(axes) == 0:
            axes = [0]
        for a in axes:
            assert(isinstance(a, int))
        def rinner(v, a):
            slices = []
            for i in range(v.ndim):
                if i in a:
                    slices.append(slice(None, None, -1))
                else:
                    slices.append(slice(None, None, None))
            return v[tuple(slices)]
        return OpBlock(rinner, name="reverse")(self, axes)



### WORRY ABOUT THIS
class Parameter(TensorWrapped):
    '''
    A parameter wraps a shared variable and can optionally have a different learning rate and regularization multiplier
    '''
    def __init__(self, value, name=None, lrmul=1., regmul=1., shape=None):
        self.initializer = None
        if isinstance(value, theano.compile.sharedvalue.SharedVariable):
            self.value = value
            self.shape = value.get_value().shape
            self.initializer = lambda: value.get_values()
        elif isinstance(value, Initializer):
            self.shape = shape
            self.initializer = lambda: value.sample(shape).astype(theano.config.floatX)
            self.value = theano.shared(np.zeros(shape).astype(theano.config.floatX))
            self.reset()
        elif isinstance(value, Val):
            self.value = value.d.astype(theano.config.floatX)
            self.shape = value.d.get_value().shape
            self.initializer = lambda: value.d.get_value()
        else:
            self.value = theano.shared(value.astype(theano.config.floatX))
            self.initializer = lambda: value.astype(theano.config.floatX)
            self.shape = value.shape
        self.lrmul = lrmul
        self.regmul = regmul
        self.name = str(name) if name is not None else "auto" + str(np.random.randint(0, 10000))
        self.value.name = self.name
        self.constraints = []

    def applyonval(self, f):
        self.value.set_value(f(self.value.get_value()))
        return self

    def reset(self):
        #print "resetting param %s \n\t\t (in %s)" % (str(self), self.__class__.__name__)
        self.value.set_value(self.initializer())

    @property
    def d(self):
        return self.value

    def __repr__(self):
        return "param::'%s':%s%s" % (str(self.name), str(self.value.dtype), str(self.value.get_value().shape))

    ############## VALUE CONSTRAINTS ############### --> applied in the order that the were added
    def clip(self, a, b):
        self.constraints.append(lambda x: tensor.clip(x, a, b))
        return self

    def normalize(self, axis=0, norm=2, epsilon=1e-7):
        self.constraints.append(lambda x: (x.T/(x.norm(norm, axis=axis)+epsilon)).T) # TODO
        return self

    def norm_constraint(self, max_norm, norm_axes=None, epsilon=1e-7):
        self.constraints.append(lambda x: norm_constraint(x, max_norm=max_norm, norm_axes=norm_axes, epsilon=epsilon))
        return self

    def constraintf(self):
        cs = self.constraints
        def innerconstraintf(x):
            ret = x
            for cf in cs:
                ret = cf(ret)
            return ret
        return innerconstraintf

    @property
    def allparams(self):
        return {self}


class param(object):
    def __init__(self, shape, lrmul=1., regmul=1., name=None):
        self.shape = shape
        self.lrmul = lrmul
        self.regmul = regmul
        self.value = None
        self.name = name

    def _init_helper(self, f):
        ret = Parameter(f(self.shape), lrmul=self.lrmul, regmul=self.regmul, name=self.name)
        ret.initializer = f
        return ret

    def init(self, arg, *args, **kwargs):
        if isstring(arg):
            assert hasattr(self, arg)
            return getattr(self, arg)(*args, **kwargs)
        elif isfunction(arg):
            return self._init_helper(arg)

    ############## OWN INITS ###################
    def random(self, offset=0.5, scale=0.1):
        return self._init_helper(lambda shape: (np.random.random(shape).astype("float32") - offset) * scale)

    def eye(self, offset=0):
        return self._init_helper(lambda shape: np.eye(shape[0], shape[1], k=offset, dtype="float32"))

    ############## LASAGE INITS ################
    def _lasagne_init(self, initializer):
        return Parameter(initializer, lrmul=self.lrmul, regmul=self.regmul, shape=self.shape, name=self.name)

    def uniform(self, range=0.01, std=None, mean=0.0):
        return self._lasagne_init(Uniform(range, std, mean))

    def normal(self, std=0.01, mean=0.0):
        return self._lasagne_init(Normal(std, mean))

    def glorotnormal(self, gain=1.0, c01b=False):
        return self._lasagne_init(GlorotNormal(gain, c01b))

    def glorotuniform(self, gain=1.0, c01b=False):
        return self._lasagne_init(GlorotUniform(gain, c01b))

    def henormal(self, gain=1.0, c01b=False):
        return self._lasagne_init(HeNormal(gain, c01b))

    def heuniform(self, gain=1.0, c01b=False):
        return self._lasagne_init(HeUniform(gain, c01b))

    def constant(self, val=0.0):
        return self._lasagne_init(Constant(val))

    def sparse(self, sparsity=0.1, std=0.01):
        return self._lasagne_init(Sparse(sparsity, std))

    def orthogonal(self, gain=1.0):
        return self._lasagne_init(Orthogonal(gain))


class Elem(object):    # carries name
    def __init__(self, name=None, **kw):
        super(Elem, self).__init__()
        self._name = name


class Masked(object):
    def __init__(self, mask=None, **kw):
        self._mask = mask
        super(Masked, self).__init__(**kw)

    @property
    def hasmask(self):
        return self._mask is not None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        self._mask = m

    @mask.deleter
    def mask(self):
        self._mask = None

    def makemask(self, arg=0):      # default: mask zero
        if isfunction(arg):
            self._mask = arg(self)
        elif isnumber(arg):
            self._mask = tensorops.eq(self, arg)
        else:
            raise Exception("wrong input argument type")
        return self


class Val(Elem, TensorWrapped, Masked):
    def __init__(self, value, name=None, **kw):
        super(Val, self).__init__(name=name, **kw)
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        dtype = value.dtype.kind
        if dtype == "i":
            dtype = str(value.dtype)
        elif dtype == "f":
            dtype = theano.config.floatX
        self.value = theano.shared(value.astype(dtype=dtype), name=name)

    @property
    def d(self):
        return self.value

    @property
    def v(self):
        return self.value.get_value()


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class RVal(Elem, TensorWrapped, Masked):  # random value
    def __init__(self, seed=None, **kw):
        super(RVal, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.rng = RandomStreams(seed=seed)
        self.value = None

    def binomial(self, shape, n=1, p=0.5, ndim=None, dtype="int32"):
        if isinstance(shape, Elem):
            shape = shape.d
        self.value = self.rng.binomial(shape, n, p, ndim, dtype)
        return self

    def normal(self, shape, avg=0.0, std=1.0, ndim=None, dtype=None):
        if isinstance(shape, Elem):
            shape = shape.d
        self.value = self.rng.normal(shape, avg, std, ndim, dtype)
        return self

    def multinomial(self, shape, n=1, pvals=None, without_replacement=False, ndim=None, dtype="int32"):
        if isinstance(shape, Elem):
            shape = shape.d
        if without_replacement:
            self.value = self.rng.multinomial_wo_replacement(shape, n, pvals, ndim, dtype)
        else:
            self.value = self.rng.multinomial(shape, n, pvals, ndim, dtype)
        return self

    def gumbel(self, shape, eps=1e-10):
        if isinstance(shape, Elem):
            shape = shape.d
        x = self.rng.uniform(shape, 0.0, 1.0)
        self.value = -theano.tensor.log(-theano.tensor.log(x + eps) + eps)
        return self

    @property
    def d(self):
        return self.value

    @property
    def v(self):
        return self.value.eval()

    @property
    def allparams(self):
        return set()

    @property
    def allupdates(self):
        return {}

    @property
    def all_extra_outs(self):
        return {}


### WORRY ABOUT THIS
class Var(Elem, TensorWrapped, Masked): # result of applying a block on theano variables
    """ Var has params propagated from all the blocks used to compute it """
    def __init__(self, value, name=None, **kw):
        nam = name if name is not None else value.name
        super(Var, self).__init__(name=nam, **kw)
        assert(isinstance(value, theano.Variable))
        self.value = value
        self._shape = None
        self._params = set()            # params this variable may depend on
        self._updates = OrderedDict()

    def push_params(self, setofparams):
        self._params.update(setofparams)

    def push_updates(self, updates):
        for updatesrc in updates:
            updatetgt = updates[updatesrc]
            if updatesrc in self._updates and updatetgt != self._updates[updatesrc]:
                raise Exception("update collision")
            self._updates[updatesrc] = updatetgt

    @property
    def allupdates(self):
        return self._updates

    @property
    def v(self):
        return self.eval()

    def eval(self, argdic={}):
        return self.d.eval(dict(map(lambda (x, y): (x.d, y), argdic.items())))

    @property
    def d(self):
        return self.value

    def __repr__(self):
        return "var::%s-%s:%s" % (self._name, self.value.dtype, str(self._shape))

    @property
    def allparams(self):
        return self._params

    @property
    def name(self):
        return self.d.name

    @name.setter
    def name(self, name):
        self._name = name
        self.d.name = name


class Input(Var): # generates feed + creates symbolic vars for input
    def __init__(self, ndim, dtype, name=None, **kw): # data source (numpy array)
        value = tensor.TensorType(dtype, (False,) * ndim)(name=name)
        super(Input, self).__init__(value, parent=None, **kw)
        self.ndim = ndim # store number of dimensions


class TrainModeContext(object):
    def __init__(self, x, b):
        self.p = x
        self.newtrainmode = b
        self.oldtrainmode = None

    def __enter__(self):
        print "ENTERING TRAINING MODE"
        global _TRAINMODE
        self.oldtrainmode = _TRAINMODE
        _TRAINMODE = self.newtrainmode
        return self.p

    def __exit__(self, e_type, e_value, traceback):
        global _TRAINMODE
        _TRAINMODE = self.oldtrainmode


class Block(Elem, Saveable): # block with parameters
    def __init__(self, **kw):
        super(Block, self).__init__(**kw)
        self._ownparams = set()
        self.inputs = []
        self.outputs = []
        self._predictf = None
        self._pristine = True

    @property
    def ownparams(self):
        return self._ownparams

    @ownparams.setter
    def ownparams(self, x):
        self._ownparams = x if isinstance(x, set) else set(x)

    @property
    def output(self):
        assert(len(self.outputs) == 1)
        return self.outputs[0]

    def trainmode(self, b):
        return TrainModeContext(self, b)

    def reset(self): # clear all non-param info in whole expression structure that ends in this block
        print "resetting block"
        self.inputs = []
        self.outputs = []
        super(Block, self).reset()

    def apply(self, *vars, **kwargs):
        trueargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, vars)
        truekwargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, kwargs)
        result = self._apply(*trueargs, **truekwargs)
        return Var(result)#, parent=self)

    # may override: -------------------------------------------------
    @property
    def predict(self): # returns callable object
        class BlockPredictor(object):
            def __init__(self, block):
                def ident(*args, **kwargs): return args, kwargs
                self.transf = ident
                self.block = block

            def transform(self, f):
                if f is not None:
                    assert(isfunction(f))
                    self.transf = f if f is not None and isfunction(f) else self.transf
                return self

            def __call__(self, *inputdata, **kwinputdata):    # do predict, take into account prediction settings set
                if self.block._predictf is None: # or block._predictf._transform != self.transfZ:
                    # if False or len(self.inputs) == 0 or self.output is None:
                    kwinpl = kwinputdata.items()
                    if self.transf is not None:
                        kwinpl.append(("transform", self.transf))
                    inps, outp = self.block.autobuild(*inputdata, **dict(kwinpl))
                    if hasattr(self.block, "_predict_postapply"):
                        outp = self.block._predict_postapply(outp)
                    self.block._predictf = theano.function(outputs=[o.d for o in outp],
                                                           inputs=[x.d for x in inps],
                                                           on_unused_input="warn")
                args = []

                def _inner(x):
                    if isinstance(x, DataFeed):
                        return x[:]
                    elif not isinstance(x, np.ndarray):
                        return np.asarray(x)
                    else:
                        return x

                kwn = []
                for k in sorted(kwinputdata.keys()):
                    kwn.append(kwinputdata[k])
                allinputdata = inputdata + tuple(kwn)
                allinputdata = filter(lambda x: x is not None, allinputdata)
                args = map(_inner, allinputdata)
                valret = self.block._predictf(*args)
                ret = valret[0] if len(valret) == 1 else tuple(valret)
                return ret
        return BlockPredictor(self)
    """
    def predict(self, transform=None, *inputdata, **kwinputdata):
        if self._predictf is None:
            #if False or len(self.inputs) == 0 or self.output is None:
            inps, outp = self.autobuild(*inputdata, **kwinputdata)
            self._predictf = theano.function(outputs=[o.d for o in outp], inputs=[x.d for x in inps])
        args = []
        def _inner(x):
            if isinstance(x, DataFeed):
                return x[:]
            elif not isinstance(x, np.ndarray):
                return np.asarray(x)
            else:
                return x
        kwn = []
        for k in sorted(kwinputdata.keys()):
            kwn.append(kwinputdata[k])
        allinputdata = inputdata + tuple(kwn)
        allinputdata = filter(lambda x: x is not None, allinputdata)
        args = map(_inner, allinputdata)
        valret = self._predictf(*args)
        ret = valret[0] if len(valret) == 1 else tuple(valret)
        return ret
    """

    def gettrainer(self, goldvar):
        return ModelTrainer(self, goldvar)

    # do not override ------------------------------------------------
    # TODO: what if wrapply gets params in args?
    # TODO: propagate _ownparams to output vars
    def wrapply(self, *args, **kwargs): # is this multi-output compatible?
        transform = None
        oldtrainmode = None
        olddebugmode = None
        global _TRAINMODE
        global _DEBUGMODE
        #print "{} train mode (in base.py)".format(_TRAINMODE)
        if "transform" in kwargs and kwargs["transform"] is not None:
            transform = kwargs.pop("transform")
        if "_trainmode" in kwargs:      # changes global _TRAINMODE
            oldtrainmode = _TRAINMODE
            _TRAINMODE = kwargs.pop("_trainmode")
        if "_debugmode" in kwargs:      # changes global _DEBUGMODE
            olddebugmode = _DEBUGMODE
            _DEBUGMODE = kwargs.pop("_debugmode")
        if "_batsize" in kwargs:
            batsize = kwargs.pop("_batsize")
        if "_trainmode" in inspect.getargspec(self.apply)[0]:
            kwargs["_trainmode"] = _TRAINMODE
        if "_debugmode" in inspect.getargspec(self.apply)[0]:
            kwargs["_debugmode"] = _DEBUGMODE

        if "transform" in kwargs and kwargs["transform"] is not None:
            transform = kwargs.pop("transform")
        paramstopush = set()        # params to transfer from input vars to output vars
        for var in recurfilter(lambda x: isinstance(x, Var), kwargs) + recurfilter(lambda x: isinstance(x, Var), args):
            paramstopush.update(var._params)
        if transform is not None and isfunction(transform):
            args, kwargs = transform(*args, **kwargs)
        updatestopush = _get_updates_from([args, kwargs])
        ret = self.apply(*args, **kwargs)   # ret carries params of its own --> these params have been added in this block
        possiblechildren = recurfilter(lambda x: isinstance(x, Var), ret)
        for p in possiblechildren:
            p.push_params(paramstopush)
            p.push_params(self.ownparams)
            p.push_updates(updatestopush)

        if oldtrainmode is not None:    # this was where we changed the global _TRAINMODE
            _TRAINMODE = oldtrainmode   # put it back
        if olddebugmode is not None:
            _DEBUGMODE = olddebugmode
        return ret

    def build(self): # stores block inputs and block output
        self.inputs = self.initinputs()
        self._build(*self.inputs)

    def _build(self, *inps, **kwinps):
        output = self.wrapply(*inps, **kwinps)
        return output

    def autobuild(self, *inputdata, **kwinputdata):
        transform = None
        trainmode = False
        if "transform" in kwinputdata:
            transform = kwinputdata.pop("transform")
        if "_trainmode" in kwinputdata:
            trainmode = kwinputdata.pop("_trainmode")
        inputdata = map(lambda x:
                        x if isinstance(x, (np.ndarray, DataFeed)) else (np.asarray(x) if x is not None else None),
                        inputdata)
        for k in kwinputdata:
            x = kwinputdata[k]
            kwinputdata[k] = x if isinstance(x, (np.ndarray, DataFeed)) else (np.asarray(x)
                                                  if x is not None else x)
        inputs = []
        kwinputs = {}
        inpnum = 1
        for td in inputdata:
            inputs.append(None if td is None else Input(ndim=td.ndim, dtype=td.dtype, name="inp:%d" % inpnum))
            inpnum += 1
        for k in kwinputdata:
            td = kwinputdata[k]
            kwinputs[k] = None if td is None else Input(ndim=td.ndim, dtype=td.dtype, name="kwinp:%s" % k)

        kwinputl = kwinputs.items()
        kwinputl.append(("_trainmode", trainmode))
        if transform is not None:
            kwinputl.append(("transform", transform))
        output = self._build(*inputs, **dict(kwinputl))

        kwn = []
        for k in sorted(kwinputs.keys()):
            kwn.append(kwinputs[k])

        outinputs = tuple(inputs) + tuple(kwn)
        outinputs = filter(lambda x: x is not None, outinputs)
        output = (output,) if not issequence(output) else output
        self.inputs = outinputs
        self.outputs = output
        return outinputs, output

    def __call__(self, *args, **kwargs):
        return self.wrapply(*args, **kwargs)

    # explicit parameter management
    def add_params(self, params):
        for param in params:
            self.add_param(param)

    def add_param(self, p): # always returns a Parameter
        if isinstance(p, Parameter):
            p = p
        elif isinstance(p, theano.compile.sharedvalue.SharedVariable): # if shared var --> wrap in a param
            p = Parameter(p)
        elif isinstance(p, np.ndarray): # numpy array
            p = Parameter(param(p))
        elif isinstance(p, tuple): # try to decode as a list of (param, lrmul, regmul) entries --> wrap in a param
            assert(isinstance(p[0], theano.compile.sharedvalue.SharedVariable))
            lrmul = 1.
            regmul = 1.
            p = p[0]
            if len(p) > 1:
                lrmul = p[1]
            if len(p) > 2:
                regmul = p[2]
            p = Parameter(p, lrmul=lrmul, regmul=regmul)
        self._ownparams.add(p)
        return p

    # training
        # wrap data in datafeeds, generate gold var
        goldvar = Input(gold.ndim, gold.dtype, name="gold")
        #inps, outp = self.autobuild(*inputdata)

        trainer = ModelTrainer(self, goldvar.d)
        trainer.traindata = inputdata
        trainer.traingold = gold
        if hasattr(self, "_trainer_cost"):  # sets cost in block
            trainer._set_objective(self._trainer_cost)
        return trainer

    def nstrain(self, datas):
        """ training with negative sampling"""
        return NSTrainConfig(self, datas)


class OpBlock(Block):
    def __init__(self, f, root=None, **kw):
        super(OpBlock, self).__init__(**kw)
        self.f = f
        self.root = root

    def __str__(self):
        return "OpBlock:" + str(self.f)

    def reset(self):
        print "resetting opblock"
        if "sigmoid" in str(self.f):
            print "sigmoid reset"
        super(OpBlock, self).reset()

    def wrapply(self, *args, **kwargs): # is this multi-output compatible?
        # push all params from args and root vars
        paramstopush = set()
        for var in recurfilter(lambda x: isinstance(x, Var), args) + recurfilter(lambda x: isinstance(x, Var), kwargs):
            paramstopush.update(var._params)
        if self.root is not None and isinstance(self.root, Var):
            paramstopush.update(self.root._params)
        # push all "own" params, params that are args or root
        params = recurfilter(lambda x: isinstance(x, Parameter), args)
        kwparams = recurfilter(lambda x: isinstance(x, Parameter), kwargs)
        ownparams = set(params + kwparams)
        if self.root is not None and isinstance(self.root, Parameter):
            ownparams.add(self.root)
        self.add_params(ownparams)
        updatestopush = _get_updates_from([args, kwargs] +
            ([self.root] if self.root is not None and isinstance(self.root, Var) else []))  # gather all updates
        # get theano vars for all args
        trueargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, args)
        truekwargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, kwargs)
        # apply theano-space Op
        result = self.f(*trueargs, **truekwargs)
        # wrap result in Var and return
        ret = Var(result)  # , parent=self)
        # do push params to output var
        possiblechildren = recurfilter(lambda x: isinstance(x, (Var, Val)), ret)
        for p in possiblechildren:
            p.push_params(paramstopush)
            p.push_params(ownparams)
            p.push_updates(updatestopush)
        return ret


class NSBlock(Block):
    """ To wrap around normal blocks for negative sampling training """
    def __init__(self, innerblock, obj, **kw):
        self.inner = innerblock
        self.obj = obj
        super(NSBlock, self).__init__(**kw)

    def apply(self, *vars):
        lvars = vars[:len(vars)/2]
        rvars = vars[len(vars)/2:]
        return self.obj(self.inner(*lvars), self.inner(*rvars))

    @property
    def predict(self):
        return self.inner.predict


class TransWrapBlock(Block):
    """ Wraps data transformation function """
    def __init__(self, block, transf, **kw):
        self.block = block
        self.transf = transf
        super(TransWrapBlock, self).__init__(**kw)

    def apply(self, *args, **kwargs):
        nargs, nkwargs = self.transf(*args, **kwargs)
        return self.block(*nargs, **nkwargs)


class NSTrainConfig():
    """ Intercepts fluent interface definition and stores settings"""
    def __init__(self, block, datas):
        self.datas = datas
        self.block = block
        self.obj = lambda p, n:  n - p
        def ident(*args, **kwargs): return args, kwargs
        self.trans = ident
        self.nrate = 1
        self.nsamgen = None
        self.trainerargs = []
        self.linear_objective()     # will be stored <-- default trainer loss for NS training
        self._validmodeflag = False
        self.nrate_valid = 1

    #region =========== OWN SETTINGS ===============
    def objective(self, f):
        self.obj = f
        return self.getret()

    def transform(self, f=None):
        if f is not None:
            self.trans = f
        return self.getret()

    def negrate(self, n):
        if not self._validmodeflag:
            self.nrate = n
        else:
            self.nrate_valid = n
        return self.getret()

    def negsamplegen(self, f):
        self.nsamgen = f
        return self.getret()
    #endregion

    def __getattr__(self, f):
        """ when a trainer config option is called """
        # TODO accept only what NSModelTrainer has
        return lambda *args, **kwargs: self._trainerconfigstorer(f, *args, **kwargs)

    def _trainerconfigstorer(self, f, *args, **kwargs):
        self.trainerargs.append((f, (args, kwargs)))
        return self

    def _ready(self):
        return self.nsamgen is not None and self.obj is not None

    def _makeblock(self):
        tb = TransWrapBlock(self.block, self.trans) # TODO: factor this TransWrap out
        return NSBlock(tb, self.obj)

    def _maketrainer(self):
        block = self._makeblock()
        gold = np.ones((self.datas[-1].shape[0],), dtype="float32")  # gold is a vector of ones of length batsize
        inputdata = self.datas + self.datas
        # wrap data in datafeeds, generate gold var
        goldvar = Input(gold.ndim, gold.dtype, name="gold")
        #inps, outp = block.autobuild(*inputdata)

        trainer = NSModelTrainer(block, goldvar.d, self.nrate, self.nsamgen)
        trainer.traindata = self.datas
        trainer.traingold = gold

        # apply settings on trainer
        #print self.trainerargs
        for k, v in self.trainerargs:
            kf = getattr(trainer, k)
            kf(*v[0], **v[1])
        return trainer

    def train(self, *args, **kwargs):
        if not self._ready():
            raise Exception("configuration not ready yet")
        t = self._maketrainer()
        #embed()
        return t.train(*args, **kwargs)

    def validate_on(self, data, splits=1, random=False):
        gold = np.ones((data[-1].shape[0],), dtype="float32")
        self._trainerconfigstorer("validate_on", data, gold, splits=splits, random=random)
        self.linear_objective()
        self._validmodeflag = True
        return self

    def getret(self):
        return self


def asblock(f):
    retblock = Block()
    retblock.apply = f
    return retblock


class scan(Block):
    def __init__(self, **kw):
        super(scan, self).__init__(**kw)
        # set params
        self._recparams = set()

    def fnwrap(self, fn): # enables writing fn in blocks level
        scanblock = self
        def fwrapper(*args): # theano vars
            trueargs = [Var(x, name="innerrecwrapvarwrap") for x in args]
            res = fn(*trueargs) # has the params from inner rec
            updates = OrderedDict()
            retupdates = None
            if isinstance(res[-1], dict):
                retupdates = res[-1]
                res = res[0]
            ret = recurmap(lambda x: x.d if hasattr(x, "d") else x, res)
            if issequence(ret):
                ret = tuple(ret)
            outvars = recurfilter(lambda x: isinstance(x, Var), res)
            for var in outvars:
                scanblock._recparams.update(var._params)
            for reswithupdates in recurfilter(lambda x: isinstance(x, Var), res):
                updates.update(reswithupdates.allupdates)
            if retupdates is not None:
                updatesupdate = {k.d: v.d for k, v in retupdates.items()}
                updates.update(updatesupdate)
            if len(updates) > 0:
                return ret, updates
            else:
                return ret
        return fwrapper

    def apply(self, fn, **kwargs):
        trueargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, kwargs)
        oldupdates = _get_updates_from(kwargs)
        o, newupdates = theano.scan(self.fnwrap(fn), **trueargs)
        ret = [Var(oe) for oe in o] if issequence(o) else Var(o)
        for var in recurfilter(lambda x: isinstance(x, Var), ret):
            var.push_params(self._recparams)
            var.push_updates(oldupdates)
            var.push_updates(newupdates)
        return ret, newupdates


def _get_updates_from(kwargs):
    updates = {}
    for var in recurfilter(lambda x: isinstance(x, Var), kwargs):
        for updatesrc in var.allupdates:
            updatetgt = var.allupdates[updatesrc]
            if updatesrc in updates and updatetgt != updates[updatesrc]:
                raise Exception("update overwriting same update source")
            updates[updatesrc] = updatetgt
    return updates


class until(Elem):
    def __init__(self, expr, **kw):
        super(until, self).__init__(**kw)
        self.add_parent(expr)
        self.expr = expr

    @property
    def d(self): # wrap theano.scan_module.until(cond)
        return theano.scan_module.until(self.expr.d)
