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
