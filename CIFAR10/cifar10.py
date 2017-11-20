#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import six.moves.urllib.request as request
import six
import os
import tarfile
import shutil
import hashlib
import sys

import cPickle

import numpy as np


def _defaultCacheDir():
    xdg_cache_path = os.getenv('XDG_CACHE_HOME', '~/.cache')
    return os.path.abspath(os.path.expanduser(xdg_cache_path))


_DEFAULT_DOWNLOAD_DIR = os.path.join(_defaultCacheDir(), 'adl', 'cifar-10')
_DEFAULT_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CIFAR_SHA256_SUMS = {
    'batches.meta': 'f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb',
    'data_batch_1': '54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd',
    'data_batch_2': '766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8',
    'data_batch_3': '0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9',
    'data_batch_4': '3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b',
    'data_batch_5': 'd91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154',
    'readme.html': '4d1c3fb199d6a183ae03f5162b469d7bc04edf2fad9547bd5f224271d52f98e5',
    'test_batch': 'f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831',
}


def log(*args, **kwargs):
    assert file not in kwargs
    print(*args, file=sys.stderr, **kwargs)


def _hexdigest(path, chunk_size=1024 * 4, hash_algorithm=hashlib.sha256):
    hash = hash_algorithm()
    with open(path, 'rb', buffering=0) as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash.update(chunk)
    return hash.hexdigest()


def _download_file(url, path, chunk_size=1024 * 1024):
    req = request.urlopen(url)
    log("Downloading file from {} to {}".format(url, path))
    with open(path, 'w') as file:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                log()
                break
            file.write(chunk)
            log(".", end="")


def _validate_cifar10(dir, sha256sums=_CIFAR_SHA256_SUMS):
    valid = True
    for filename, expected_shasum in six.iteritems(sha256sums):
        file_path = os.path.join(dir, filename)
        if not os.path.exists(file_path):
            return False
        actual_shasum = _hexdigest(file_path)
        file_valid = actual_shasum == expected_shasum
        if not file_valid:
            log("{} has sha256sum {} but expected {}".format(file_path, actual_shasum, expected_shasum))
        valid = valid and file_valid
    return valid


def _maybe_download_cifar10(download_dir=_DEFAULT_DOWNLOAD_DIR,
                            download_url=_DEFAULT_URL):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    targz_path = os.path.join(download_dir, 'cifar-10.tar.gz')
    data_path = os.path.join(download_dir, 'cifar-10-batches-py')

    if os.path.exists(data_path) and not _validate_cifar10(data_path):
        log("CIFAR10 data present, but corrupted, removing and redownloading")
        shutil.rmtree(data_path)

    if not os.path.exists(data_path) and not os.path.exists(targz_path):
        _download_file(download_url, targz_path)

    if not os.path.exists(data_path) and os.path.exists(targz_path):
        try:
            with tarfile.open(targz_path) as tar:
                for item in tar:
                    tar.extract(item, download_dir)
        except:
            os.remove(targz_path)
            return _maybe_download_cifar10(download_dir=download_dir, download_url=download_url)

    if os.path.exists(targz_path):
        os.remove(targz_path)

    return data_path


def unpickle(file):
    with open(file, 'rb') as fo:
        return cPickle.load(fo)


def oneHotVector(classIdx, numClasses):
    v = np.zeros((len(classIdx), numClasses), dtype=np.int)
    v[np.arange(0, len(v)), classIdx] = 1
    return v


class cifar10:
    IMG_WIDTH = 32
    IMG_HEIGHT = 32
    IMG_CHANNELS = 3
    CLASS_COUNT = 10

    dataPath = ''
    batchSize = 128
    trainData = np.array([])
    trainLabels = np.array([])
    testData = np.array([])
    testLabels = np.array([])
    currentIndexTrain = 0
    currentIndexTest = 0
    nTrainSamples = 0
    nTestSamples = 0
    classNames = []
    pTrain = []
    pTest = []

    def __init__(self, batchSize=128, downloadDir=_DEFAULT_DOWNLOAD_DIR, downloadUrl=_DEFAULT_URL):

        self.batchSize = batchSize
        self.dataPath = _maybe_download_cifar10(download_dir=downloadDir, download_url=downloadUrl)
        self.loadCIFAR10()

        self.trainData = self._swapChannelOrdering(self.trainData)
        self.testData = self._swapChannelOrdering(self.testData)

    def preprocess(self):
        """
        Convert pixel values to lie within [0, 1]
        """
        self.trainData = self._normaliseImages(self.trainData.astype(np.float32, copy=False))
        self.testData = self._normaliseImages(self.testData.astype(np.float32, copy=False))

    def _normaliseImages(self, imgs_flat):
        min = np.min(imgs_flat)
        max = np.max(imgs_flat)
        range = max - min
        return (imgs_flat - min) / range

    def _unflatten(self, imgs_flat):
        return imgs_flat.reshape(-1, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS)

    def _flatten(self, imgs):
        return imgs.reshape(-1, self.IMG_WIDTH * self.IMG_HEIGHT * self.IMG_CHANNELS)

    def _swapChannelOrdering(self, imgs_flat):
        return self._flatten(imgs_flat.reshape(-1, self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_HEIGHT)\
                             .transpose(0, 2, 3, 1))

    def loadCIFAR10(self):
        trainFilenames = [os.path.join(self.dataPath, 'data_batch_%d' % i) for i in range(1, 6)]
        testFileName = os.path.join(self.dataPath, 'test_batch')
        self.classNames = unpickle(os.path.join(self.dataPath, 'batches.meta'))['label_names']

        for filePath in trainFilenames:
            d = unpickle(filePath)

            if self.trainData.size == 0:
                self.trainData = d['data']
                self.trainLabels = oneHotVector(d['labels'], 10)
            else:
                self.trainData = np.concatenate((self.trainData, d['data']))
                self.trainLabels = np.concatenate((self.trainLabels, oneHotVector(d['labels'], 10)))

        d = unpickle(testFileName)
        self.testData = np.array(d['data'])
        self.testLabels = oneHotVector(d['labels'], 10)

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

    def getTrainBatch(self, allowSmallerBatches=False):
        return self._getBatch('train', allowSmallerBatches)

    def getTestBatch(self, allowSmallerBatches=False):
        return self._getBatch('test', allowSmallerBatches)

    def _getBatch(self, dataSet, allowSmallerBatches=False):
        D = np.array([])
        L = np.array([])

        if dataSet == 'train':
            train = True
            test = False
        elif dataSet == 'test':
            train = False
            test = True
        else:
            raise ValueError('_getBatch: Unrecognised set: ' + dataSet)

        while True:
            if train:
                r = range(self.currentIndexTrain,
                          min(self.currentIndexTrain + self.batchSize - L.shape[0], self.nTrainSamples))
                self.currentIndexTrain = r[-1] + 1 if r[-1] < self.nTrainSamples - 1 else 0
                (d, l) = (self.trainData[self.pTrain[r]][:], self.trainLabels[self.pTrain[r]][:])
            elif test:
                r = range(self.currentIndexTest,
                          min(self.currentIndexTest + self.batchSize - L.shape[0], self.nTestSamples))
                self.currentIndexTest = r[-1] + 1 if r[-1] < self.nTestSamples - 1 else 0
                (d, l) = (self.testData[self.pTest[r]][:], self.testLabels[self.pTest[r]][:])

            if D.size == 0:
                D = d
                L = l
            else:
                D = np.concatenate((D, d))
                L = np.concatenate((L, l))

            if D.shape[0] == self.batchSize or allowSmallerBatches:
                break

        return (D, L)

    def showImage(self, image):
        from matplotlib import pyplot as plt
        plt.imshow(image.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)), interpolation='nearest')
        plt.show()

    def reset(self):
        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)


if __name__ == '__main__':
    cifar = cifar10(batchSize=128)
    (trainImages, trainLabels) = cifar.getTrainBatch()
    (testImages, testLabels) = cifar.getTestBatch()

    (allTestImages, allTestLabels) = (cifar.testData, cifar.testLabels)
    cifar.showImage(allTestImages[50])
