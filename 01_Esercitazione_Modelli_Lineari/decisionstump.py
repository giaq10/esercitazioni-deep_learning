import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionStump(ClassifierMixin, BaseEstimator):

    def __init__(self, axis_=0, thres_=0.5, offset_=0.5):
        self.axis_ = axis_
        self.thres_ = thres_
        self.offset_ = offset_

    def get_separator_2D(self):
        """Segmento che rappresenta la soglia nel quadrato [0, 1] x [0, 1]."""
        if self.axis_ == 1:
            x = [0, 1]
            y = [self.thres_, self.thres_]
        else:
            x = [self.thres_, self.thres_]
            y = [0, 1]
        return x, y

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m, d = X.shape

        if sample_weight is None:
            sample_weight = np.ones(m)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.direction_ = 1

        # Caso degenere: una sola classe, lo stump predice sempre quella
        if self.n_classes_ == 1:
            self.thres_ = X[:, self.axis_].max() + self.offset_
            return self
        if self.n_classes_ > 2:
            raise ValueError("DecisionStump supporta solo problemi binari.")

        W = sample_weight.sum()
        W0 = sample_weight[y == self.classes_[0]].sum()

        # Soglia sotto il minimo con direction_ = +1: tutto predetto classes_[1],
        # errore = W0; con direction_ = -1 l'errore è W - W0.
        first = X[:, 0].min() - self.offset_
        if W0 <= W - W0:
            self.axis_, self.thres_, self.direction_, minerr = 0, first, 1, W0
        else:
            self.axis_, self.thres_, self.direction_, minerr = 0, first, -1, W - W0

        for j in range(d):
            sorted_id = np.argsort(X[:, j], kind="stable")
            xs = X[sorted_id, j]
            ys = y[sorted_id]
            ws = sample_weight[sorted_id]

            err = W0  # errore con direction_ = +1 e soglia sotto il minimo
            for i in range(m):
                # Il punto i passa nel lato "<= soglia" (predetto classes_[0])
                if ys[i] == self.classes_[0]:
                    err -= ws[i]
                else:
                    err += ws[i]

                if i < m - 1:
                    # Soglia non valida tra valori uguali: separerebbe punti identici
                    if xs[i] == xs[i + 1]:
                        continue
                    thr = (xs[i] + xs[i + 1]) / 2
                else:
                    thr = xs[i] + self.offset_

                if err < minerr:
                    self.axis_, self.thres_, self.direction_, minerr = j, thr, 1, err
                if W - err < minerr:
                    self.axis_, self.thres_, self.direction_, minerr = j, thr, -1, W - err

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.n_classes_ == 1:
            return np.full(len(X), self.classes_[0])

        below = X[:, self.axis_] <= self.thres_
        if self.direction_ == 1:
            return np.where(below, self.classes_[0], self.classes_[1])
        return np.where(below, self.classes_[1], self.classes_[0])