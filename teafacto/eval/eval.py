import numpy as np, pandas as pd, pickle
from teafacto.core.utils import ticktock as TT
from IPython import embed
from teafacto.core.trainutil import Saveable


class Evaluator(object):
    def __init__(self, *metrics, **kw):
        super(Evaluator, self).__init__(**kw)
        self.metrics = metrics

    def run(self, model, data, labels):
        self.results = self.evaluate(model, data, labels)
        return self.results

    def evaluate(self, model, data, labels):
        raise NotImplementedError("use subclass")

    def save(self, results, model):
        try:
            assert isinstance(model, Saveable)
            savepath = model._autosave_filepath
            assert savepath is not None
            pickle.dump(self.results, open(savepath+".eval.pkl", "w"))
        except AssertionError:
            print "could not save evaluation results, model might be incompatible"
            print "Result dump:"
            print self


class KBCEvaluator(Evaluator):

    def evaluate(self, model, data, labels):
        tt = TT("Evaluator")
        entidxs = set(data[:, 0]).union(set(labels))
        for n in range(data.shape[0]):
            ranking = self.makerank(data[n], model, entidxs)
            tt.progress(n, data.shape[0])
            for metric in self.metrics:
                metric([labels[n]], ranking)
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric()
        tt.tock("computed")
        return results

    def makerank(self, data, model, entidxs):
        entidxs = list(entidxs)
        svec = np.repeat(data[0], len(entidxs))
        rvec = np.expand_dims(np.repeat(data[1:], len(entidxs)), axis=1)
        ovec = np.asarray(entidxs)
        preds = model.predict(svec, rvec, ovec)
        ret = zip(entidxs, preds)
        ret = sorted(ret, key=lambda x: x[1], reverse=True)
        return ret


class ClasEvaluator(Evaluator):
    def evaluate(self, model, data, labels):
        tt = TT("Evaluator")
        preds = model.predict(data)
        for n in range(data.shape[0]): # iterate over all examples
            #tt.progress(n, data.shape[0])
            for metric in self.metrics:
                metric(labels[n], preds[n])
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric()
        tt.tock("computed")
        return results