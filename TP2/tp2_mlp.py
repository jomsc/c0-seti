import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.neural_network import MLPRegressor

def plot_surf(figName, regr=None):
    step_v = 0.005

    x1v = np.arange(0,1,step_v)
    x2v = np.arange(0,1,step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i,x1 in enumerate(x1v):
        for j,x2 in enumerate(x2v):
            if not regr:
                R[i,j] = sinus_cardinal(np.array([x1, x2]))
            else:
                R[i,j] = regr.predict(np.array([[x1, x2]]))[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(Xv, Yv, R, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figName, dpi=300)
    plt.show()

def sinus_cardinal(x):
    A = np.array([[1, 1], [-2, 1]])
    b = np.array([0.2, -0.3])

    x = -np.pi + 2*x*np.pi
    z = A.dot(x + b)
    h = np.sqrt(np.transpose(z).dot(z))
    if np.abs(h) < 0.001:
        y = 1
    else:
        y = np.sin(h)/h
    return y

def main():
    fileData = "sinc_dim2_input.csv"

    with open(fileData, 'r') as f:
        f.readline()  # skip the header
        X = np.loadtxt(f, delimiter = ';')

    xapp = X
    yapp = [sinus_cardinal(v) for v in xapp]

    xtest = np.random.uniform(0, 1, (1600, 2))
    ytest = [sinus_cardinal(v) for v in xtest]

    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10),
                       random_state=1, max_iter=2000, activation='tanh')

    clf.fit(xapp, yapp)

    print("Train R2 :", clf.score(xapp, yapp))
    print("Test  R2 :", clf.score(xtest, ytest))

    plot_surf('test_mlp', clf)

if __name__ == "__main__":
    main()