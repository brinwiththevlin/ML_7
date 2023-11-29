from svm import svm
import numpy as np

if __name__ == "__main__":
    data = np.loadtxt("SMO_Data.txt", delimiter=",")
    X = data[:, :2]
    Y = data[:, 2]

    model = svm()
    model.train(X, Y)
    print(model.weights, model.b)
