from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def genere_exemple_dim1(xmin, xmax, NbEx, sigma):
    x = np.arange(xmin, xmax, (xmax-xmin)/NbEx)
    y = np.sin(-np.pi + 2*x*np.pi) + np.random.normal(loc=0, scale=sigma, size=x.size)
    return x.reshape(-1,1), y


def getMSE(x, y, reg):
    return sum(pow(reg.predict(x)-y,2))/x.shape[0]


def plot_model(Xa, Ya, Xt, Yt, reg, nameFig):
    Ypred = reg.predict(Xt)
    plt.plot(Xa[:,1], Ya, '*r')
    plt.plot(Xt[:,1], Yt, '-b')
    plt.plot(Xt[:,1], Ypred, '-r')
    plt.grid()
    plt.savefig(nameFig+'.jpg', dpi=200)
    plt.close()


def plot_error_profile(L_error_app, L_error_test, nameFig):
    plt.plot(range(1, len(L_error_app)+1), L_error_app, '-r')
    plt.plot(range(1, len(L_error_test)+1), L_error_test, '-b')
    plt.grid()
    plt.savefig(nameFig+'.jpg', dpi=200)
    plt.close()


def plot_confusion(Xt, Yt, reg, nameFig):
    plt.plot(Yt, reg.predict(Xt), '.b')
    plt.plot(Yt, Yt, '-r')
    plt.savefig(nameFig+'.jpg', dpi=200)
    plt.close()

def main(degreMax=12, NbEx=20, sigma=0.2):
    xmin = 0
    xmax = 1.2

    xapp, yapp = genere_exemple_dim1(xmin, xmax, NbEx, sigma)
    xtest, ytest = genere_exemple_dim1(xmin, xmax, 200, 0)

    L_error_app = []
    L_error_test = []

    for i in range(1, degreMax+1):
        print("Degre = ", i)

        pol = PolynomialFeatures(i).fit(xapp, yapp)
        Xa = pol.transform(xapp)
        Xt = pol.transform(xtest)

        reg = LinearRegression().fit(Xa, yapp)

        L_error_app.append(getMSE(Xa, yapp, reg))
        L_error_test.append(getMSE(Xt, ytest, reg))

        plot_model(Xa, yapp, Xt, ytest, reg, "Model_%02d" % i)

    best = np.argmin(L_error_test)+1
    print("Meilleur modèle -> degré = ", best)
    plot_error_profile(L_error_app, L_error_test, 'Profile_Err_App_Test')

    pol = PolynomialFeatures(best).fit(xapp, yapp)
    Xa = pol.transform(xapp)
    Xt = pol.transform(xtest)

    reg = LinearRegression().fit(Xa, yapp)

    plot_confusion(Xt, ytest, reg, 'Confusion')

main()