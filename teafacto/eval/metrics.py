from teafacto.util import issequence
import numpy as np

class Metric(object):
    def __init__(self, **kw):
        self.reset()

    def reset(self):
        self.acc = 0.
        self.div = 0.

    @property
    def name(self):
        raise NotImplementedError("use subclass")

    def accumulate(self, label, ranking):
        raise NotImplementedError("use subclass")

    def compute(self):
        raise NotImplementedError("use subclass")

    def __call__(self, label=None, ranking=None): # df grouped by inputs, ranks ranks all entities
        if ranking is None or label is None:
            return self.compute()
        else:
            self.accumulate(label, ranking)


class ClassAccuracy(Metric):

    @property
    def name(self):
        return "Classification Accuracy"

    def accumulate(self, gold, pred):
        if issequence(pred):
            pred = pred[0][0]
        else:
            pred = [pred]
        if issequence(gold):
            assert(len(gold) == 1)
            gold = gold[0]
        else:
            gold = [gold]
        if np.array_equal(gold, pred):
            self.acc += 1
        self.div += 1

    def compute(self):
        return self.acc * 1.0 / (self.div + 1e-9)

    def __str__(self):
        return "Acc: %.3f%%" % (self.compute()*100.)



class RecallAt(Metric):
    def __init__(self, top, **kw):
        self.topn = top
        super(RecallAt, self).__init__(**kw)

    @property
    def name(self):
        return "Recall @ %d" % self.topn

    def accumulate(self, label, ranking):
        topnvals = map(lambda x: x[0], ranking[:self.topn])
        recall = len(set(label).intersection(set(topnvals))) * 1. / len(label)
        self.acc += recall
        self.div += 1

    def compute(self):
        return self.acc / self.div

    def __str__(self):
        return "R@%d: %.3f" % (self.topn, self.compute())


class MeanQuantile(Metric):

    @property
    def name(self):
        return "Mean Quantile"

    def accumulate(self, label, ranking):
        total = len(ranking)
        for correctone in label:
            pos = 0
            for (x, y) in ranking:
                if x == correctone:
                    pos -= 1
                    break
                pos += 1
            self.acc += 1. * (total - pos) / total
            self.div += 1

    def compute(self):
        return self.acc / self.div

    def __str__(self):
        return "MQ:%.3f" % self.compute()
