import mglearn.datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


def execute():
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lasso = Lasso().fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

    lasso00001 = Lasso(alpha=0.0001,max_iter=100000).fit(X_train,y_train)
    print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

