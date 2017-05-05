import numpy as np

from teafacto.core.datafeed import DataFeed
from teafacto.core.datafeed import FeedTransform


class WordSeqFeed(DataFeed):
    def __init__(self, data, transformer=FeedTransform(), **kw):
        super(WordSeqFeed, self).__init__(data, **kw)
        self.transformer = transformer

    @property
    def dtype(self):
        return np.dtype("int32")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self.transformer.getshapefor(self.data.shape)

    def __getitem__(self, item):
        ret = self.data.__getitem__(item)
        return self.transform(ret)

    def get(self, idxs): # should return datafeed of the same type TODO write test
        return self.__class__(self.data[idxs], self.transformer)

    def transform(self, x):
        return self.transformer.transform(x)

