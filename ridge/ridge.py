import mglearn as mglearn
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def execute():
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ridge = Ridge().fit(X_train, y_train)
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-25,25)
    plt.legend()
    plt.show()
