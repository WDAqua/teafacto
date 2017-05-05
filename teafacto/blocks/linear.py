from teafacto.core.base import Block, param, tensorops as T
import theano.tensor as TT
import numpy as np


#### BINARY LINEAR CLASSIFICATION ####################
class BinaryLinear(Block):
    def __init__(self, dim, **kw):
        self.dim = dim
        self.w = param((dim,), name="weights").uniform()
        self.b = param((1,), name="bias").uniform()
        super(BinaryLinear, self).__init__(**kw)

    def apply(self, datas):
        return T.dot(datas, self.w) + self.b[0]

    def _predict_postapply(self, x):
        print "postapply"
        return (T.switch(x[0]>0, +1, -1),)


class Perceptron(BinaryLinear):
    def _trainer_cost(self, x, gold):       # !!! theano space
        return (- x * gold).clip(0, np.infty)


class SVM(Perceptron):  # TODO: SUM TING WONG???
    def __init__(self, numparams, wreg=1., margin=0.1, **kw):
        self.wreg = wreg
        self.margin = margin
        super(SVM, self).__init__(numparams, **kw)

    def _trainer_cost(self, x, gold):
        loss = (self.margin - x * gold).clip(0, np.infty)
        reg = self.w.d.norm(2)**2 * self.wreg
        cost = loss + reg
        return cost


class LogisticRegression(Perceptron):
    def _trainer_cost(self, x, gold):
        return TT.nnet.sigmoid(- x * gold)
