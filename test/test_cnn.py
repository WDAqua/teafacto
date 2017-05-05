from unittest import TestCase
import numpy as np
from teafacto.blocks.cnn import Conv1D, GlobalPool1D, CNNSeqEncoder
from teafacto.blocks.seq.rnn import EncLastDim
from teafacto.core.base import Val

class TestConv1D(TestCase):
    def test_output_shape(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=5)
        pred = conv.predict(xval)
        self.assertEqual(pred.shape[:2], xval.shape[:2])
        self.assertEqual(pred.shape[2], 40)

    def test_output_shape_strided(self):
        xval = np.random.random((100, 13, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=5, stride=2, border_mode="valid")
        pred = conv.predict(xval)
        print pred.shape
        self.assertEqual(pred.shape, (100, 5, 40))

    def test_output_shape_masked(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(3, 20, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        conv = Conv1D(indim=50, outdim=40, window=5)
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        predval = pred.eval()
        predvalmask = (predval != 0.0) * 1
        predvalexpmask = np.ones_like(predvalmask)
        for i in range(predvalexpmask.shape[0]):
            predvalexpmask[i, min(maskid[i]+2, predvalexpmask.shape[1]):, :] = 0
        self.assertTrue(np.sum(predvalexpmask - predvalmask) == 0)
        self.assertEqual(predval.shape[:2], xval.shape[:2])
        self.assertEqual(predval.shape[2], 40)

    def test_output_mask_strided(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(3, 20, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        conv = Conv1D(indim=50, outdim=40, window=5, stride=4, border_mode="valid")
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        print predmask.eval().shape
        print predmask.eval()[:5]
        print mask[:5]


class TestPool1D(TestCase):
    def test_max_pool(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        pool = GlobalPool1D(mode="max")
        pred = pool.predict(xval)
        nppred = np.max(xval, axis=1)
        self.assertTrue(np.allclose(nppred, pred))

    def test_avg_pool(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        pool = GlobalPool1D(mode="avg")
        pred = pool.predict(xval)
        nppred = np.sum(xval, axis=1) / xval.shape[1]
        self.assertTrue(np.allclose(nppred, pred))

    def test_max_pool_masked(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(1, 18, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        #xval[:, :, -1] = 100
        x = Val(xval)
        x.mask = Val(mask)
        pool = GlobalPool1D(mode="max")
        pred = pool(x)
        predval = pred.eval()
        xval = xval - 1e9 * np.tensordot(1-mask, np.ones((xval.shape[-1],)), 0)
        predvalexp = np.max(xval, axis=1)
        self.assertTrue(np.allclose(predval, predvalexp))


class TestCNNEnc(TestCase):
    def test_enc(self):
        xval = np.random.randint(0, 200, (100, 20)).astype("int32")
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40])
        pred = enc.predict(xval)
        self.assertEqual(pred.shape, (100, 40))

    def test_cnnenc_in_dimred(self):
        xval = np.random.randint(0, 200, (3, 100, 20)).astype("int32")
        enc = EncLastDim(CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40]))
        pred = enc.predict(xval)
        self.assertEqual(pred.shape, (3, 100, 40))

    def test_enc_mask(self):
        xval = np.random.randint(1, 200, (100, 20)).astype("int32")
        maskid = np.random.randint(0, 5, (100,))
        for i in range(xval.shape[0]):
            xval[i, maskid[i]:] = 0
        x = Val(xval)
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=5, maskid=0)
        pred = enc(x)
        #print pred.mask.eval().shape
        predval = pred.eval()
        print predval.shape





