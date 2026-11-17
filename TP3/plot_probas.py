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

    plt.savefig(nameFig+'.jpg', figsize=(8, 8), dpi=300)
    plt.close()
