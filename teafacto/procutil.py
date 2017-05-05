import numpy as np
from teafacto.util import ticktock


def wordmatfromdic(worddic, maxwordlen=30):
    maskid = -1
    rwd = sorted(worddic.items(), key=lambda (x, y): y)
    realmaxlen = 0
    wordmat = np.ones((rwd[-1][1]+1, maxwordlen), dtype="int32") * maskid
    for i in range(len(rwd)):
        rwdichars, rwdiidx = rwd[i]
        realmaxlen = max(realmaxlen, len(rwdichars))
        wordmat[rwdiidx, :min(len(rwdichars), maxwordlen)] \
            = [ord(c) for c in rwdichars[:min(len(rwdichars), maxwordlen)]]
    allchars = set(list(np.unique(wordmat))).difference({maskid})
    chardic = {maskid: maskid}
    chardic.update(dict(zip(allchars, range(len(allchars)))))
    wordmat = np.vectorize(lambda x: chardic[x])(wordmat)
    del chardic[maskid]
    chardic = {chr(k): v for k, v in chardic.items()}
    return wordmat, chardic


def wordmat2wordchartensor(wordmat, worddic=None, rwd=None, maxchars=30, maskid=-1):
    chartensor = wordmat2chartensor(wordmat, worddic=worddic, rwd=rwd, maxchars=maxchars, maskid=maskid)
    out = np.concatenate([wordmat[:, :, np.newaxis], chartensor], axis=2)
    #embed()
    return out


def wordmat2chartensor(wordmat, worddic=None, rwd=None, maxchars=30, maskid=-1):
    assert(worddic is not None or rwd is not None)
    assert(not(worddic is not None and rwd is not None))
    if rwd is None:
        rwd = {v: k for k, v in worddic.items()}
    wordcharmat = maskid * np.ones((max(rwd.keys())+1, maxchars), dtype="int32")
    realmaxlen = 0
    for i in rwd.keys():
        word = rwd[i]
        word = word[:min(maxchars, len(word))]
        realmaxlen = max(realmaxlen, len(word))
        wordcharmat[i, :len(word)] = [ord(ch) for ch in word]
    chartensor = wordcharmat[wordmat, :]
    chartensor[wordmat == -1] = -1
    if realmaxlen < maxchars:
        chartensor = chartensor[:, :, :realmaxlen]
    return chartensor


def wordmat2charmat(wordmat, worddic=None, rwd=None, maxlen=100, raretoken="<RARE>", maskid=-1):
    assert(worddic is not None or rwd is not None)
    assert(not(worddic is not None and rwd is not None))
    tt = ticktock("wordmat2charmat")
    tt.tick("transforming word mat to char mat")
    toolong = 0
    charmat = maskid * np.ones((wordmat.shape[0], maxlen), dtype="int32")
    if rwd is None:
        rwd = {v: (k if k != raretoken else " ")
               for k, v in worddic.items()}
    else:
        rwd = dict([(k, (v if v != raretoken else " "))
                   for k, v in rwd.items()])
    realmaxlen = 0
    for i in range(wordmat.shape[0]):
        s = wordids2string(wordmat[i], rwd, maskid=maskid)
        s = s[:min(len(s), maxlen)]
        realmaxlen = max(len(s), realmaxlen)
        if len(s) > maxlen:
            toolong += 1
        charmat[i, :len(s)] = [ord(ch) for ch in s]
        tt.progress(i, wordmat.shape[0], live=True)
    if realmaxlen < maxlen:
        charmat = charmat[:, :realmaxlen]
    if toolong > 0:
        print "{} too long".format(toolong)
    tt.tock("transformed")
    return charmat


def wordids2string(inp, rwd, maskid=-1):
    ret = " ".join([rwd[x] if x in rwd else "<???>"
                    for x in inp if x != maskid])
    return ret


def charids2string(inp, rcd=None, maskid=-1):
    if rcd is not None:
        ret = "".join([rcd[ch] if ch in rcd else "<???>"
                       for ch in inp if ch != maskid])
    else:
        ret = "".join([chr(ch) if ch != maskid else "" for ch in inp])
    return ret


def wordcharmat2string(inp, rcd=None, maskid=-1):
    if rcd is not None:
        tochar = np.vectorize(lambda x: rcd[x] if x != maskid else "" if x in rcd else "<???>")
    else:
        tochar = np.vectorize(lambda x: chr(x) if x != maskid else "")
    x = tochar(inp)
    acc = []
    for i in range(x.shape[0]):
        w = "".join(list(x[i]))
        acc.append(w)
    ret = " ".join([w for w in acc if len(w) > 0])
    return ret


def getmatrixvaluecounts(*x):
    x = np.concatenate([xe.flatten() for xe in x], axis=0)
    from pandas import Series
    pdx = Series(x)
    ret = pdx.value_counts()
    return dict(zip(ret.index, ret.values))
