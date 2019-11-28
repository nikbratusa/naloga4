from matplotlib import pyplot
import numpy as np

from solution import load, LogRegLearner


def draw_decision(X, y, classifier, at1, at2, grid=50):

    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    pyplot.figure(figsize=(8,8))

    for c,(x,y) in zip(y,points):
        pyplot.text(x,y,str(c), ha="center", va="center")
        pyplot.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi,x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pyplot.imshow(prob, extent=(minx,maxx,maxy,miny), cmap="seismic")

    pyplot.xlim(minx, maxx)
    pyplot.ylim(miny, maxy)
    pyplot.xlabel(at1)
    pyplot.ylabel(at2)

    pyplot.show()


if __name__ == "__main__":
    X,y = load('reg.data')

    learner = LogRegLearner(lambda_=0.)
    classifier = learner(X,y)

    draw_decision(X, y, classifier, 0, 1)

    # dopisite ....

    # sem dodajte tudi testiranje ...
