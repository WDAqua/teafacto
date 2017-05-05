import numpy as np

from teafacto.modelusers import RecPredictor, SeqEncDecPredictor


class Searcher(object):
    def __init__(self, model, beamsize=1, *buildargs, **kw):
        super(Searcher, self).__init__(**kw)
        self.beamsize = beamsize
        self.model = model
        self.mu = RecPredictor(model, *buildargs)


class SeqTransDecSearch(Searcher):
    # responsible for generating recappl prediction function from recappl of decoder
    """ Default: greedy search strategy """
    def decode(self, inpseq):
        stop = False
        i = 0
        curout = np.zeros((inpseq.shape[0])).astype("int32")
        accprobs = np.ones((inpseq.shape[0]))
        outs = []
        while not stop:
            curinp = inpseq[:, i]
            curprobs = self.mu.feed(curinp, curout)
            accprobs *= np.max(curprobs, axis=1)
            curout = np.argmax(curprobs, axis=1).astype("int32")
            outs.append(curout)
            i += 1
            stop = i == inpseq.shape[1]
        #print accprobs
        ret = np.stack(outs).T
        assert (ret.shape == inpseq.shape)
        return ret, accprobs

