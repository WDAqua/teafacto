from unittest import TestCase

from teafacto.util import ticktock as TT, argparsify, loadlexidtsv
import os

class TestUtils(TestCase):
    def test_ticktock_duration_string(self):
        tt = TT()
        testdata = [
            (1, "1.000 second"),
            (0.5689, "0.569 second"),
            (0.9999, "1.000 second"),
            (59, "59.000 seconds"),
            (59.00001, "59.000 seconds"),
            (59.0005, "59.001 seconds"),
            (60, "1 minute"),
            (60.005, "1 minute"),
            (61, "1 minute, 1 second"),
            (62, "1 minute, 2 seconds"),
            (121, "2 minutes, 1 second"),
            (120, "2 minutes"),
            (3656, "1 hour, 56 seconds"),
            (2*3600, "2 hours"),
            (24*3600+125, "1 day, 2 minutes, 5 seconds"),
            (25*3600+126, "1 day, 1 hour, 2 minutes, 6 seconds"),
            (50*3600, "2 days, 2 hours")
        ]
        for seconds, text in testdata:
            self.assertEqual(text, tt._getdurationstr(seconds))

    def test_argparsify(self):
        def testf(a=1, b="str"):
            pass
        self.assertEqual(argparsify(testf, test="-a 1"), {"a": 1})


class TestIDTSVLoader(TestCase):
    def test_load(self):
        p = os.path.join(os.path.dirname(__file__), "../data/freebase/labelsrevlex.map.id.tsv.sample")
        print p
        gids, charten, fbids = loadlexidtsv(p)
        print gids.shape, charten.shape, fbids.shape
        self.assertEqual(gids.shape, (10000, 10))
        self.assertEqual(charten.shape, (10000, 10, 30))
        self.assertEqual(fbids.shape, (10000,))

