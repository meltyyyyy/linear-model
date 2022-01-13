import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def execute():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)

    logreg = LogisticRegression().fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

    fig_l2 = plt.figure()
    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    fig_l2.savefig("clf/logreg_l2.png")

    fig_l1 = plt.figure()
    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
        print("Training accuracy of l1 logreg with = C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
        print("Test accuracy of l1 logreg with = C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
        plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend(loc=3)
    fig_l1.savefig("clf/logreg_l1.png")
