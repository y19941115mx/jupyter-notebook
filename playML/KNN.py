import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier(object):

    def __init__(self, k):
        super(KNNClassifier, self).__init__()
        self.k = k
        self._X_train = None
        self._y_train = None

    def __repr__(self):
        return 'KNNClassifier(k=%s)' % (self.k,)

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, 'must fit before predict'
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x_predict):
        distance = [sqrt(np.sum((x_predict - x_train) ** 2)) for x_train in self._X_train]
        nearest_index = np.argsort(distance)[:self.k]
        nearest = [self._y_train[index] for index in nearest_index]
        votes = Counter(nearest)
        return votes.most_common(1)[0][0]



