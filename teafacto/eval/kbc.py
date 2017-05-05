import os
import pickle

import numpy as np
import pandas as pd

from teafacto.core.sptensor import SparseTensor
from teafacto.core.trainutil import Saveable
from teafacto.core.utils import ticktock as TT
from teafacto.eval.metrics import RecallAt, MeanQuantile
from teafacto.models.kb.kmm import KMM


class Evaluation(object):
    def __init__(self, model):
        self.model = model

    def run(self, data, *metrics):
        self.results = evaluate(data, self.model, *metrics)

    def __repr__(self):
        return str(self.results)

    def __str__(self):
        return str(self.results)

    def save(self):
        try:
            assert isinstance(self.model, Saveable)
            savepath = self.model._autosave_filepath
            assert savepath is not None
            pickle.dump(self.results, open(savepath+".eval.pkl", "w"))
        except AssertionError:
            print "could not save evaluation results, model might be incompatible"
            print "Result dump:"
            print self


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    st = SparseTensor.from_ssd(file)
    return st

def loadmodel(modelp):
    emodel = KMM.load(modelp)
    return emodel

def evaluate(data, model, *metrics):
    tt = TT("Evaluator")
    datadf = pd.DataFrame(data)
    datadf.columns = ["s", "r", "o"]
    datadfgb = datadf.groupby(by=["s", "r"]).apply(lambda x: set(x["o"].unique())).reset_index()
    datadf.describe()
    entidxs = set(datadf["s"].unique()).union(set(datadf["o"].unique()))
    results = {}
    tt.tock("initialized").tick()
    for n, row in datadfgb.iterrows():
        ranking = makerank(row, model, entidxs)
        tt.progress(n, datadfgb.shape[0])
        for metric in metrics:
            metric(row, ranking)
    for metric in metrics:
        results[metric.name] = metric()
    tt.tock("computed")
    return results


def makeranks(df, model, entidxs, relidxs):
    tt = TT("Ranker")
    tt.tick("making ranks")
    rankings = []
    for _, row in df.iterrows():
        rankings.append(makerank(row, model, entidxs))
    tt.tock("made ranks")
    return rankings

def makerank(row, model, entidxs):
    entidxs = list(entidxs)
    row = row.values
    svec = np.repeat(row[0], len(entidxs))
    rvec = np.expand_dims(np.repeat(row[1], len(entidxs)), axis=1)
    ovec = np.asarray(entidxs)
    preds = model.predict(svec, rvec, ovec)
    ret = zip(entidxs, preds)
    ret = sorted(ret, key=lambda x: x[1], reverse=True)
    return ret


def eval(tdp, mp, *metrics):
    ret = evaluate(loaddata(tdp).keys.lok, loadmodel(mp), *metrics)
    return ret


if __name__ == "__main__":
    testdatapath = "../../data/nycfilms/triples.flat/alltripletensor.test.flat.ssd"
    modelpath = "../../models/AddEKMM+SGD+n10+E20D.2016-01-12=20:14.auto"
    #modelpath = "../../models/MatMulEKMM+SGD+n10+E20D.2016-01-12=20:18.auto"
    modelpath = "../../models/RNNEKMM+SGD+n10+E20D+GRU.2016-01-12=22:17.auto"
    modelpath = "../../models/ERNNEKMM+SGD+n10+E20D+IEGRU.2016-01-13=00:01.auto"
    results = eval(testdatapath, modelpath, RecallAt(10), RecallAt(15), MeanQuantile())
    results.update({"modelfile": modelpath})
    pickle.dump(results, open(modelpath+".results", "w"))
    print results