from teafacto.scripts.simplequestions.fullrank.fullrank import loadsubjtestcans, loadsubjinfo, loadreltestcans, readdata
from teafacto.util import argprun, ticktock
import pickle
from IPython import embed
import numpy as np, random


def loaddata(numtestcans=5):
    tt = ticktock("dataloader")
    tt.tick("loading data")
    p = "../../../../data/simplequestions/clean/datamat.word.fb2m.pkl"
    entinfp = "../../../../data/simplequestions/clean/subjs-counts-labels-types.fb2m.tsv"
    x = pickle.load(open(p))
    tt.tock("datamat loaded")
    worddic = x["worddic"]
    entdic = x["entdic"]
    entmat = x["entmat"]
    numents = x["numents"]
    traindata, traingold = x["train"]
    validdata, validgold = x["valid"]
    testdata, testgold = x["test"]
    traingold[:, 1] -= numents
    validgold[:, 1] -= numents
    testgold[:, 1] -= numents

    rwd = {v: k for k, v in worddic.items()}

    subjdic = {k: v for k, v in entdic.items() if v < numents}
    reldic = {k: v - numents for k, v in entdic.items() if v >= numents}

    subjinfo = loadsubjinfo(entinfp, subjdic)
    testsubjcans = loadsubjtestcans(numcans=numtestcans)
    testrelcans, relspersubj = loadreltestcans(testgold, subjdic, reldic)
    return testgold, testsubjcans, relspersubj


def randgen(entcans, relsperent):
    tt = ticktock("randgen")
    tt.tick("generating")
    mat = np.zeros((len(entcans), 2), dtype="int32")
    for i in range(mat.shape[0]):
        cans = entcans[i]
        mat[i, 0] = random.sample(cans, 1)[0] if len(cans) > 0 else -1                   # only those appearing as subject
        cans = relsperent[mat[i, 0]][0] if mat[i, 0] >= 0 else []
        mat[i, 1] = random.sample(cans, 1)[0] if len(cans) > 0 else -1     # only outgoing relations of predicted subject
    tt.tock("generated")
    return mat


def main(numtestcans=10, numruns=100):
    testgold, testsubjcans, relspersubj = loaddata(numtestcans=numtestcans)
    totalaccs = []
    subjaccs = []
    predaccs = []
    for i in range(numruns):
        testrand = randgen(testsubjcans, relspersubj)
        evalmat = testrand == testgold
        subjacc = np.sum(evalmat[:, 0]) * 1. / evalmat.shape[0]
        predacc = np.sum(evalmat[:, 1]) * 1. / evalmat.shape[0]
        totalacc = np.sum(np.sum(evalmat, axis=1) == 2) * 1. / evalmat.shape[0]
        print "Test results ::::::::::::::::"
        print "Total Acc: \t {}".format(totalacc)
        print "Subj Acc: \t {}".format(subjacc)
        print "Pred Acc: \t {}".format(predacc)
        totalaccs.append(totalacc)
        subjaccs.append(subjacc)
        predaccs.append(predacc)
        print testrand.shape, testgold.shape
    meantotalacc, vartotalacc = np.mean(totalaccs), np.std(totalaccs)
    meansubjacc, varsubjacc = np.mean(subjaccs), np.std(subjaccs)
    meanpredacc, varpredacc = np.mean(predaccs), np.std(predaccs)
    print "MEANS over {} runs".format(numruns)
    print "Test results ::::::::::::::::"
    print "Total Acc: \t {}+-{}".format(meantotalacc, vartotalacc)
    print "Subj Acc: \t {}+-{}".format(meansubjacc, varsubjacc)
    print "Pred Acc: \t {}+-{}".format(meanpredacc, varpredacc)



if __name__ == "__main__":
    argprun(main)