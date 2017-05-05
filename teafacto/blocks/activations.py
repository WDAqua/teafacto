from teafacto.core.base import Block, tensorops as T


class Activation(Block):
    def apply(self, x):
        mask = x.mask
        ret = self.innerapply(x)
        ret.mask = mask
        return ret

    def innerapply(self, x):
        raise NotImplementedError("use subclasses")


class Tanh(Activation):
    def innerapply(self, x):
        return T.tanh(x)


class Sigmoid(Activation):
    def innerapply(self, x):
        return T.nnet.sigmoid(x)


class Linear(Activation):
    def innerapply(self, x):
        return x


class ReLU(Activation):
    def innerapply(self, x):
        return T.nnet.relu(x)


class Softmax(Activation):
    def innerapply(self, x):
        return T.nnet.softmax(x)
