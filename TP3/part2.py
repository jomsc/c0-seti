# PARTIE 2 - Problème non linéairement séparable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

def genere_ex_2(n=300, mu=[0,0], std=0.25, delta=0.2):
    X = np.random.multivariate_normal(mu, np.diagflat(std*np.ones(2)), n)
    Y = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        x = X[i,0]
        y = X[i,1]
        if y < x*(x-1)*(x+1):
            Y[i] = -1
            X[i,1] = X[i,1] - delta
        else:
            Y[i] = 1
            X[i,1] = X[i,1] + delta
    return X,Y

def main(X, Y):
    parameters = {'kernel':['poly'], 'C':[1, 5, 10], 'degree':[5, 6], 'coef0':[1]}
    svc = SVC(probability=True)

    classifier = GridSearchCV(svc, parameters)
    classifier.fit(X, Y)
    plot_SVM_probas(X, Y, classifier, 'Graph_SVM_nonlinear')

    print("Résultats de la grid search :\n", classifier.best_params_)
    print("Meilleur score (CV) :", classifier.best_score_)

if __name__ == "__main__":
    X, Y = genere_ex_2()
    main(X, Y)