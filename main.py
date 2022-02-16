import json

import numpy as np
import random as rd
import math as mt
from matplotlib import pyplot as plt
from matplotlib import colors as cls
import scipy.optimize as scopt

# Press the green button in the gutter to run the script.

config = json.loads(open("config.json", mode="r").read())


class SVM:

    def __init__(self, dataPoints, targetClass):

        sigmaValue = mt.pow(config["sigma"], 2)

        kernels = dict(linear=lambda s, x_i: np.dot(x_i, s),
                       polynomial=lambda s, x_i: np.power(np.dot(x_i, s) + 1, config["pFactor"]),
                       rbf=lambda s, x_i: np.exp(-1 * (np.power(np.linalg.norm(s - x_i), 2) / (2 * sigmaValue))))

        self.dataPoints = dataPoints
        self.targetClass = targetClass
        self.nInputPoints = len(dataPoints)
        self.C = config["cValue"]
        self.kernelFunction = kernels[config["kernel"]]

        # These value we be calculated in trainSVM
        self.alpha = None
        self.bias = None
        self.dualProblemCache = None
        self.is_trained = False
        self.trainSVM()

    @staticmethod
    def kernelFunction(s, x_i):

        # TODO should be able to set this function dynamically should have some config file
        # return np.exp(-1 * (np.power(np.linalg.norm(s - x_i), 2) / (2 * 1)))
        # return np.power(np.dot(x_i, s) + 1, 1)
        return np.dot(x_i, s)

    def dualProblem(self, alphas):

        return 0.5 * (sum(sum(a_i * a_j * self.dualProblemCache[i, j] for j, a_j in enumerate(alphas))
                          for i, a_i in enumerate(alphas))) - sum(a for a in alphas)

    def indicator(self, s):

        if self.is_trained:
            return sum(a_i * self.targetClass[i] * self.kernelFunction(s, self.dataPoints[i]) for i, a_i in
                       enumerate(self.alpha)) - self.bias
        else:
            print("You need to train the SVM before you make an indication call")
            # TODO this is not the best architecture will change later.
            return None

    def trainSVM(self):
        self.is_trained = True
        self.calcAlpha()
        self.calcBias()

    def calcAlpha(self):

        self.dualProblemCache = np.array([[self.targetClass[i] *
                                           self.targetClass[j] *
                                           self.kernelFunction(self.dataPoints[i], self.dataPoints[j, :])
                                           for j in range(self.nInputPoints)]
                                          for i in range(self.nInputPoints)])
        ret = scopt.minimize(fun=self.dualProblem,
                             x0=np.zeros(self.nInputPoints),
                             bounds=self.nInputPoints * [(0, self.C)],
                             constraints=dict(type='eq', fun=lambda a: np.dot(a, self.targetClass)),
                             options={"disp": True})

        self.alpha = np.array(ret['x'])
        self.alpha[np.abs(self.alpha) < 10e-5] = 0

    def calcBias(self):

        try:
            sIndex = next(i for i, a_i in enumerate(self.alpha) if (0 < a_i < self.C))
        except Exception:
            # TODO this is ugly as hell. BUt this is a small lab not a industry project
            print("could found a bias term. Your plot is not accurate")
            sIndex = min(enumerate(self.alpha), key=lambda e: e[1])[0]

        s = self.dataPoints[sIndex, :]
        t_s = self.targetClass[sIndex]
        self.bias = sum(a_i * self.targetClass[i] * self.kernelFunction(s, self.dataPoints[i]) for i, a_i in
                        enumerate(self.alpha)) - t_s


def plotSVM(saveLabel, clA, clB, xValues, yValues, zValue):
    print("plotting....")
    plt.contour(xValues, yValues, zValue,
                (-1.0, 0.0, 1.0),
                colors=('blue', 'black', 'red',),
                linewidths=(1, 2.5, 1))

    zValueHigh = np.max(zValue)
    zValueLow = np.min(zValue)

    h = plt.contourf(xValues, yValues, zValue, levels=150, norm=cls.TwoSlopeNorm(vmin=zValueLow, vcenter=0, vmax=zValueHigh), cmap="coolwarm")
    plt.axis('scaled')
    plt.colorbar()
    plt.scatter(clA[:, 0], clA[:, 1], color="#bf1806")
    plt.scatter(clB[:, 0], clB[:, 1], color="#0e45ab")
    plt.axis('equal')
    plt.title("CValue: {}, Kernel: {}".format(round(config["cValue"], 4), config["kernel"]))
    plt.savefig(saveLabel)

    plt.show()


def question1():
    """
    Move the clusters around and change their sizes to make it easier or
    harder for the classifier to find a decent boundary. Pay attention
    to when the optimizer (minimize function) is not able to find a
    solution at all
    """

    config["kernel"] = "linear"
    np.random.seed(0)
    nSamples = 30
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [1.5, -3.75],
         np.random.randn(n2Samples, 2) * 0.2 + [-1.5, -3.75])
    )
    classB = np.random.randn(nSamples, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)

    xValues = np.linspace(-7, 7, 100)
    yValues = np.linspace(-7, 7, 100)
    zValue = np.array([[predictor.indicator(np.array([x, y])) for x in xValues] for y in yValues])

    plotSVM("output_img/question1/Easy_DATA_svmPlot.pdf", classA, classB, xValues, yValues, zValue)

    nSamples = 20
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [2, -0.95],
         np.random.randn(n2Samples, 2) * 0.2 + [-2, -0.95])
    )
    classB = np.random.randn(nSamples, 2) * 0.2 + [0.0, -0.33]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)

    xValues = np.linspace(-7, 7, 100)
    yValues = np.linspace(-7, 7, 100)
    zValue = np.array([[predictor.indicator(np.array([x, y])) for x in xValues] for y in yValues])

    plotSVM("output_img/question1/Hard_DATA_svmPlot.pdf", classA, classB, xValues, yValues, zValue)

    nSamples = 20
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [1, -0.5],
         np.random.randn(n2Samples, 2) * 0.2 + [-1, -0.5])
    )
    classB = np.random.randn(nSamples, 2) * 1.5 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)

    xValues = np.linspace(-7, 7, 100)
    yValues = np.linspace(-7, 7, 100)
    zValue = np.array([[predictor.indicator(np.array([x, y])) for x in xValues] for y in yValues])

    plotSVM("output_img/question1/SuperHard_DATA_svmPlot.pdf", classA, classB, xValues, yValues, zValue)
    print("Question 1 Done")


def question2():
    """
    Implement the two non-linear kernels.
    You should be able to classify very hard data sets with these.
    """

    np.random.seed(4)
    nSamples = 40
    n4Samples = nSamples // 4

    classA = np.concatenate(
        (np.random.randn(n4Samples, 2) * 0.2 + [-3, 0.5],
         np.random.randn(n4Samples, 2) * 0.2 + [3, 0.5],
         np.random.randn(n4Samples, 2) * 0.2 + [0.5, -2.5],
         np.random.randn(n4Samples, 2) * 0.2 + [0.5, 2.5],
        )
    )
    classB = np.random.randn(nSamples, 2) * 0.4 + [0.0, 0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    config["kernel"] = "polynomial"
    config["pFactor"] = 2

    predictor = SVM(inputs, targets)
    xValues = np.linspace(-5, 5, 100)
    yValues = np.linspace(-5, 5, 100)
    zValue = np.array([[predictor.indicator(np.array([x, y])) for x in xValues] for y in yValues])

    plotSVM("output_img/question2/poly_svmPlot.pdf", classA, classB, xValues, yValues, zValue)

    classA = np.concatenate(
        (np.random.randn(n4Samples, 2) * 0.2 + [-3, 0.5],
         np.random.randn(n4Samples, 2) * 0.2 + [3, 0.5],
         np.random.randn(n4Samples, 2) * 0.2 + [0.5, -2.5],
         np.random.randn(n4Samples, 2) * 0.2 + [0.5, 2.5],
         )
    )
    classB = np.random.randn(nSamples, 2) * 0.4 + [0.0, 0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    config["kernel"] = "rbf"

    predictor = SVM(inputs, targets)
    xValues = np.linspace(-5, 5, 100)
    yValues = np.linspace(-5, 5, 100)
    zValue = np.array([[predictor.indicator(np.array([x, y])) for x in xValues] for y in yValues])

    plotSVM("output_img/question2/rbf_svmPlot.pdf", classA, classB, xValues, yValues, zValue)




def question3():
    """
    The non-linear kernels have parameters; explore how they influence
    the decision boundary. Reason about this in terms of the bias-
    variance trade-off.
    """
    np.random.seed(0)
    nSamples = 20
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [1.5, -2.5],
         np.random.randn(n2Samples, 2) * 0.2 + [-1.5, -2.5])
    )
    classB = np.random.randn(nSamples, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)


def question4():
    """
    Explore the role of the slack parameter C. What happens for very large/small values?
    """
    np.random.seed(0)
    nSamples = 20
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [1.5, -2.5],
         np.random.randn(n2Samples, 2) * 0.2 + [-1.5, -2.5])
    )
    classB = np.random.randn(nSamples, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)


def question5():
    """
    Imagine that you are given data that is not easily separable. When
    should you opt for more slack rather than going for a more complex
    model (kernel) and vice versa?
    """
    np.random.seed(0)
    nSamples = 20
    n2Samples = nSamples // 2

    classA = np.concatenate(
        (np.random.randn(n2Samples, 2) * 0.2 + [1.5, -2.5],
         np.random.randn(n2Samples, 2) * 0.2 + [-1.5, -2.5])
    )
    classB = np.random.randn(nSamples, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = inputs.shape[0]
    permute = list(range(N))
    rd.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    predictor = SVM(inputs, targets)


if __name__ == '__main__':
    #question1()
    question2()
