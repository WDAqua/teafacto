import numpy as np
from math import ceil


class DataFeeder(object): # contains data feeds
    def __init__(self, *feeds, **kw): # feeds or numpy arrays
        self.feeds = feeds
        self.batsize = None
        feedlens = [x.shape[0] for x in self.feeds]
        assert(feedlens.count(feedlens[0]) == len(feedlens)) # all data feeds must have equal number of examples (axis zero)
        self.size = feedlens[0]
        self._random = kw["random"] if "random" in kw else True     # or False or number
        # iter state
        self.iteridxs = np.arange(self.size)
        self.offset = 0
        self.reset()
        self.autoreset = True

    # fluent settings
    def numbats(self, numbats):
        self.batsize = int(ceil(self.size * 1. / numbats))
        return self

    @property
    def _numbats(self):
        return self.getnumbats()

    def getnumbats(self):
        return int(ceil(self.size * 1. / self.batsize))

    def random(self, random):
        self._random = random
        return self

    # batching
    def reset(self):
        if self._random is not False:
            np.random.shuffle(self.iteridxs)
        self.offset = 0

    def hasnextbatch(self):
        ret = self.offset <= self.size-2
        if not ret and self.autoreset:
            self.reset()
        return ret

    def nextbatch(self, withbatchsize=False):
        if self.batsize is None:
            self.offset = self.size # ensure stop
            return [x[:] for x in self.feeds]
        start = self.offset
        end = min(self.offset+self.batsize, self.size)
        sampleidxs = self.iteridxs[start:end]
        self.offset = end
        if withbatchsize:
            return [x[sampleidxs] for x in self.feeds], sampleidxs.shape[0]
        else:
            return [x[sampleidxs] for x in self.feeds]

    def split(self, split=2, random=False, df_randoms=(True, True)): # creates two new datafeeders with disjoint splits
        splitidxs = np.arange(0, self.size)
        if random is not False:
            np.random.shuffle(splitidxs)
        start = 0
        middle = int(ceil(1.*self.size / split))
        end = self.size
        dfvalid = DataFeeder(*[self.splitfeed(feed, splitidxs[start:middle]) for feed in self.feeds], random=df_randoms[1])
        dftrain = DataFeeder(*[self.splitfeed(feed, splitidxs[middle:end])   for feed in self.feeds], random=df_randoms[0])
        return dftrain, dfvalid

    def osplit(self, split=2, random=False):
        splitidxs = np.arange(0, self.size)
        if random is not False:
            np.random.shuffle(splitidxs)
        start = 0
        end = int(ceil(1.*self.size / split))

        return DataFeeder(*[self.splitfeed(feed, splitidxs[start:end]) for feed in self.feeds])

    def isplit(self, splitidxs, df_randoms=(True, True)):
        nsplitidxs = np.setdiff1d(np.arange(0, self.size), splitidxs)
        dfvalid = DataFeeder(*[self.splitfeed(feed, splitidxs)  for feed in self.feeds], random=df_randoms[1])
        dftrain = DataFeeder(*[self.splitfeed(feed, nsplitidxs) for feed in self.feeds], random=df_randoms[0])
        return dftrain, dfvalid

    def splitfeed(self, feed, idxs):
        if isinstance(feed, DataFeed):
            return feed.get(idxs)
        else:
            return feed[idxs]


class DataFeed(object):
    '''
    Wraps data, custom data feed can be implemented for dynamic sampling
    '''
    def __init__(self, data, *args, **kw): # data: numpy array
        super(DataFeed, self).__init__(*args, **kw)
        self.data = data

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def get(self, idxs):    # should return DataFeed of the same type
        pass


class DynamicDataFeed(DataFeed): # a dynamic data generator (e.g. for random negative sampling)
    def __getitem__(self, item):
        pass # TODO

    def get(self, idxs): # create a new Dynamic Data Feed
        pass # TODO

if __name__ == "__main__":
    x = np.random.random((10, 10))
    dx = DataFeed(x)
    print x.ndim, x.shape, x.dtype


class SplitIdxIterator(object):
    def __init__(self, datalen, split=10, random=False, folds=1):
        self.folds = folds
        self.splits = self.buildsplits(datalen, random, split, folds)

    def buildsplits(self, datalen, random, split, folds):    # random: whether and how random, split: percentage in split, folds: how many times to produce a split
        dataidxs = np.arange(0, datalen, 1, dtype="int32")
        if random is not False:     # do random splitting but not Monte Carlo
            if isinstance(random, (int, long)):  # set seed
                np.random.seed(random)
            np.random.shuffle(dataidxs)
        # generate a list of vectors of data indexes
        offset = 0
        splitsize = int(ceil(1. * datalen / split))
        currentfold = 0
        splits = []
        while currentfold < folds:
            start = offset
            end = min(offset + splitsize, datalen)
            splitidxs = dataidxs[start:end]
            splits.append(splitidxs)
            if end == datalen:  # restart
                if random is not False:     # reshuffle
                    np.random.shuffle(dataidxs)
                offset = 0
            currentfold += 1
            offset += splitsize
        return splits

    def __iter__(self):
        self.currentfold = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.currentfold < self.folds:
            ret = self.splits[self.currentfold]       # get the indexes
            self.currentfold += 1
            return ret
        else:
            raise StopIteration


class FeedTransform(object):
    def __init__(self, **kw):
        super(FeedTransform, self).__init__(**kw)

    def transform(self, x):
        return x

    def getshapefor(self, datashape):
        return datashape