import elasticsearch, re, sys
from teafacto.util import tokenize, argprun


class SimpleQuestionsLabelIndex(object):
    def __init__(self, host="drogon", index="simplequestions_subjects"):
        self.host = host
        self.indexp = index

    def index(self, labelp="labels.map"):
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        try:
            es.indices.delete(index=self.indexp)
        except Exception, e:
            print "could not delete index %s" % self.indexp
        es.indices.create(index=self.indexp,
                          body={
                              "settings": {
                                  "index": {
                                      "analysis": {
                                          "filter": {
                                            "pstemmer": {
                                              "type": "porter_stem",
                                              "language": "_english_"
                                            }
                                          },
                                          "char_filter": {
                                            "punctfil": {
                                              "type": "pattern_replace",
                                              "pattern": "[\W]",
                                              "replacement": ""
                                            }
                                          },
                                          "analyzer": {
                                              "myana": {
                                                  "tokenizer": "whitespace",
                                                  "filter": [
                                                      "lowercase",
                                                      "pstemmer",
                                                  ]
                                                  #,"char_filter": ["punctfil"]
                                              }
                                          }
                                      }
                                  }
                              },
                              "mappings": {
                                  "labelmap": {
                                      "properties": {
                                          "label": {
                                              "type": "string",
                                              "analyzer": "myana",
                                              "fields": {
                                                "len": {
                                                    "type": "token_count",
                                                    "store": "yes",
                                                    "analyzer": "myana"
                                                }
                                              }
                                          }
                                      }
                                  }
                              }
                          })
        i = 1
        for line in open(labelp):
            k, v = line[:-1].split("\t")
            vt = tokenize(v)
            es.index(index=self.indexp, doc_type="labelmap", id=i,
                     body={"label": " ".join(vt), "fbid": k})
            if i % 1000 == 0:
                print i
            i += 1
        print "indexed labels"

    def search(self, query, top=10):
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        res = es.search(index=self.indexp, q="label:%s" % query, size=top)
        acc = {}
        for r in res["hits"]["hits"]:
            self._merge(acc, {r["_source"]["fbid"]: (r["_score"], r["_source"]["label"])})
        return acc

    @staticmethod
    def _merge(acc, d):
        for k, v in d.items():
            if k in acc:
                if acc[k][0] < v[0]:
                    acc[k] = v
            else:
                acc[k] = v

    def searchsentence(self, s, top=None, topsize=None, exact=True):
        s = tokenize(s)
        ngrams = self.getallngrams(s, topsize)
        return self.searchallngrams(ngrams, top, exact=exact)

    def getallngrams(self, s, topsize=None):
        topsize = len(s) if topsize is None else topsize
        ngrams = set()
        i = 0
        while i < len(s):
            j = i + 1
            while j <= min(len(s), i + topsize):
                ngram = tuple(s[i:j])
                j += 1
                ngrams.add(ngram)
            i += 1
        return ngrams

    def searchallngrams(self, ngrams, top=None, exact=True):
        #print ngrams
        es = elasticsearch.Elasticsearch(hosts=[self.host])
        searchbody = []
        header = {"index": self.indexp, "type": "labelmap"}
        for ngram in ngrams:
            if not exact:
                body = {
                            "query": {
                                "filtered": {
                                    "query": {
                                        "match_phrase": {
                                            "label":  " ".join(ngram)
                                        }
                                    }
                                }
                            }
                        }
            else:
                body = {
                            "query": {
                                "filtered": {
                                    "query": {
                                        "match_phrase": {
                                            "label":  {
                                                "query": " ".join(ngram)
                                            }
                                        }
                                    },
                                    "filter": {
                                        "term": {
                                            "label.len": len(ngram)
                                        }
                                    }
                                }
                            }
                        }
                """{
                    "query": {
                        "constant_score": {
                            "filter": {
                                "term": {
                                    "label": " ".join(ngram)
                                }
                            }
                        }
                    }
                }
                """

            if top is not None:
                body.update({"size": top, "from": 0})
            searchbody.append(header)
            searchbody.append(body)
        ngramres = es.msearch(body=searchbody)
        cans = {}
        for response in ngramres["responses"]:
            try:
                for r in response["hits"]["hits"]:
                    self._merge(cans, {r["_source"]["fbid"]: (r["_score"], r["_source"]["label"])})
            except KeyError, e:
                print response
        return cans


def run(index=False, indexp="labels.map", indexname="sq_subjnames_fb2m",
        search="e mc", host="localhost", exact=False, top=10):
    idx = SimpleQuestionsLabelIndex(host=host, index=indexname)
    if index is True and indexp is not None:
        idx.index(labelp=indexp)
        sys.exit()
    #res = idx.search("e", top=10)
    res = idx.searchsentence(search, exact=exact, top=top)
    sres = sorted(res.items(), key=lambda (x, y): y[0], reverse=True)
    for x in sres:
        print x
    print len(sres)



if __name__ == "__main__":
    argprun(run)