import numpy as np


class svm:
    def __init__(self):
        self.alpha = []
        self.weights = []
        self.b = 0

    def KKT_and_E(self, X, Y):
        K = np.dot(X, X.T)

        KKT = self.alpha * (Y * (np.dot(X, self.weights) + self.b) - 1)
        E = np.dot(self.alpha, Y * K) + self.b - Y

        return KKT, E

    def train(self, X, Y):
        epsilon = 1e-6

        self.alpha = np.random.rand(len(Y))
        self.alpha -= np.dot(self.alpha, Y) * Y / np.dot(Y, Y)

        done = False
        while not done:
            self.weights = np.dot(X.T, self.alpha * Y)
            KKT, E = self.KKT_and_E(X, Y)

            i1 = np.argmax(KKT)

            e = E[i1] - E
            i2 = np.argmax(e)

            K = np.dot(X, X.T)
            k = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2]

            alpha2 = self.alpha[i2] + Y[i2] * e[i2] / k
            self.alpha[i1] = self.alpha[i1] + Y[i1] * Y[i2] * (self.alpha[i2] - alpha2)
            self.alpha[i2] = alpha2

            self.alpha[self.alpha < epsilon] = 0

            sv_idx = np.random.choice(np.where(self.alpha > 0)[0])

            self.b = Y[sv_idx] - np.dot(self.alpha * Y, K[:, sv_idx])

            predictions = np.sign(np.matmul(X, self.weights) + self.b)
            predictions[predictions == -1] = 0

            done = np.sum(predictions - Y) == 0
