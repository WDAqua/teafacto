import theano
from theano import tensor as T
import numpy as np

from teafacto.core.base import Block, param, Val
from teafacto.blocks.basic import Softmax, MatDot as Lin, VectorEmbed
from teafacto.core.stack import stack

def run():
    e = T.eye(10, 10)
    w = theano.shared(np.eye(10, 10, 1))
    x = T.ivector()
    xval = np.random.randint(0, 10, (20,)).astype("int32")
    y = T.dot(e[x, :], w).argmax(axis=1)
    print xval
    print y.eval({x: xval})

    def rec(x_t):
        return T.dot(e[x_t, :], w).argmax(axis=1).astype("int32")

    o, _ = theano.scan(fn=rec,
                       outputs_info=xval,
                       n_steps=5)

    print o.eval()



class idx2seqStupid(Block):
    def __init__(self, encdim, invocsize, outvocsize, innerdim, seqlen, **kw):
        super(idx2seqStupid, self).__init__(**kw)
        self.encdim = encdim
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.seqlen = seqlen
        self.emb = VectorEmbed(indim=self.invocsize, dim=self.encdim, normalize=True)
        self.aletter = stack(Lin(indim=self.encdim, dim=self.outvocsize), Softmax())
        self.bletter = stack(Lin(indim=self.encdim, dim=self.outvocsize), Softmax())
        self.cletter = stack(Lin(indim=self.encdim, dim=self.outvocsize), Softmax())

    def apply(self, idxs):
        idxemb = self.emb(idxs)
        outp = T.stack(self.aletter(idxemb), self.bletter(idxemb), self.cletter(idxemb)).dimshuffle(1,0,2)
        return outp




class idx2seqTheano(Block):
    def __init__(self, encdim, invocsize, outvocsize, innerdim, seqlen, **kw):
        super(idx2seqTheano, self).__init__(**kw)
        self.encdim = encdim
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.seqlen = seqlen
        self.wordemb = param((invocsize, encdim)).uniform()
        self.idxtovec = Val(np.eye(outvocsize, outvocsize))
        self.rnu_w = param((encdim+outvocsize, innerdim)).uniform()
        self.rnu_u = param((innerdim, innerdim)).uniform()
        self.outpf = theano.tensor.tanh
        self.olin = param((innerdim, outvocsize)).uniform()
        self.ownparams = [self.wordemb, self.rnu_u, self.rnu_w, self.olin]

    def _apply(self, idxs):
        def rec(x_t, h_tm1, enc):
            chosen = x_t.argmax(axis=1, keepdims=False)
            chosenvec = self.idxtovec.d[chosen, :]
            blockarg = theano.tensor.concatenate([enc, chosenvec], axis=1)
            h_t = self.outpf(theano.tensor.dot(blockarg, self.rnu_w.d) + theano.tensor.dot(h_tm1, self.rnu_u.d))
            x_tp1 = theano.tensor.nnet.softmax(theano.tensor.dot(h_t, self.olin.d))
            return [x_tp1, h_t]
        enco = self.wordemb.d[idxs, :]
        x_0 = theano.tensor.eye(1, self.outvocsize).repeat(idxs.shape[0], axis=0)
        h_0 = theano.tensor.zeros((idxs.shape[0], self.innerdim))
        o, _ = theano.scan(fn=rec,
                           non_sequences=[enco],
                           outputs_info=[x_0, h_0],
                           n_steps=self.seqlen
                           )
        outprobs = o[0].dimshuffle(1, 0, 2)
        return outprobs




if __name__ == "__main__":
    run()