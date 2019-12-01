import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math

def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)

    return X, y


def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    theta = np.transpose(theta)
    prod = theta.dot(x)
    h = 1 / (1 + math.exp(-1 * prod))

    return h


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    sum = 0
    r = y.shape[0]
    t = theta.shape[0]
    """
    for i in range(r):
        first = y[i] * math.log(h(X[i],theta))
        second = (1 - y[i]) * math.log(1 -(h(X[i],theta)))
        both = first + second
        sum += both
    """
    for i in range(r):
        part = math.log((y[i] * h(X[i],theta)) + ((1-y[i]) * (1 - h(X[i],theta))))
        sum += part

    res = -(1/r * sum)


    sumTheta = 0
    for j in range(t):
        sumTheta += theta[j] ** 2

    reg = (lambda_/(2*r)) * sumTheta

    res += reg

    return res


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    t = theta.shape[0]
    m = y.shape[0]
    theta_new = np.zeros(t)

    for j in range(t):
        sum = 0
        for i in range(m):
            part = (h(X[i],theta) - y[i]) * X[i][j]
            sum += part
        sum = sum / m
        theta_new[j] = sum

    return theta_new

def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    t = theta.shape[0]
    eps = 0.000000000000001
    theta_new = np.zeros(t)

    for j in range(t):

        theta_tmp1 = np.copy(theta)
        theta_tmp2 = np.copy(theta)

        theta_tmp1[j] += eps
        theta_tmp2[j] -= eps

        c1 = cost(theta_tmp1,X,y,lambda_)
        c2 = cost(theta_tmp2,X,y,lambda_)

        grad = (c1 - c2)/(2 * eps)
        theta_new[j] = grad



    return theta_new


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    pass


def CA(real, predictions):
    # ... dopolnite (naloga 3)
    pass


def AUC(real, predictions):
    # ... dopolnite (dodatna naloga)
    pass


if __name__ == "__main__":
    # Primer uporabe

    X, y = load('reg.data')


    """
    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y)  # dobimo model
    print(classifier)


    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)
    """