# PARTIE 1 - Problème linéairement séparable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

def plot_SVM_probas(X, Y, classifier, nameFig, colormap='jet'):
    plt.figure()
    
    classPos = (Y == 1)
    classNeg = (Y == -1)
    
    minx1 = min(X[:, 0])
    maxx1 = max(X[:, 0])
    minx2 = min(X[:, 1])
    maxx2 = max(X[:, 1])
    
    xx = np.linspace(minx1, maxx1, 100)
    yy = np.linspace(minx2, maxx2, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    probas = classifier.predict_proba(Xfull)
    Z = classifier.decision_function(Xfull)    
    
    k = 1
    plt.title("Class %d" % k)
    imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                               extent=(minx1, maxx1, minx2, maxx2), origin='lower', cmap=colormap)
    plt.contour(xx, yy, Z.reshape((100, 100)), [-1, 0, 1], colors=['blue', 'black', 'red'])
    plt.scatter(X[classPos, 0], X[classPos, 1], marker='o', c='r', edgecolor='k')
    plt.scatter(X[classNeg, 0], X[classNeg, 1], marker='o', c='b', edgecolor='k')
        
    ax = plt.axes([0.8, 0.15, 0.05, 0.7])

    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='vertical')

    plt.savefig(nameFig+'.jpg', dpi=300)
    plt.close()


def genere_ex_1(n1=100, n2=50, mu1=[0,3], mu2=[3,0], sd1=0.15, sd2=0.2):
    X = np.concatenate((np.random.multivariate_normal(mu1, np.diagflat(sd1*np.ones(2)), n1),
                        np.random.multivariate_normal(mu2, np.diagflat(sd2*np.ones(2)), n2)))

    Y = np.concatenate((np.ones((n1,1)), -1*np.ones((n2,1))))[:,0]

    return X, Y

def main(X, Y):
    classifier = SVC(kernel='linear', probability=True)
    classifier = classifier.fit(X, Y)
    plot_SVM_probas(X, Y, classifier, 'Graph_SVM_linear')

if __name__ == "__main__":
    X, Y = genere_ex_1()
    main(X, Y)