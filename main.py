"""Forest Type Classification

written by Nicholas Cannon
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import pandas as pd
import os
import pickle

from plotting import plot_cf_mat, plot_ovo_pr, plot_ovo_roc

# PIPELINES
feature_pipeline = Pipeline([
    ('drop_columns', FunctionTransformer(lambda X: X.iloc[:, 0:9])),
    ('normalize', StandardScaler())
])
label_pipeline = Pipeline([
    ('remove_whitespace', FunctionTransformer(
        lambda y: y.apply(lambda c: c.replace(' ', ''))))
])


def save_model(model):
    """Saves model in a models dir in the cwd.
    """
    # create models dir if it doesn't exist
    path = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f'{type(model.estimator).__name__}.pickle'), 'wb') as f:
        pickle.dump(model, f)

    print(f'Saved model in {os.path.join(path, type(model.estimator).__name__)}')


def train_svc(X_train, y_train, save=True):
    ovo_svc = OneVsOneClassifier(SVC())

    # NOTE: the SVC estimator is stored as estimator on the OneVsOneClassifier
    # use __ to access properties of nested estimator
    parameters = [
        {
            'estimator__kernel': ['linear', 'rbf'],
            'estimator__C': [0.5, 0.75, 0.85, 0.9, 1]
        },
        {
            'estimator__kernel': ['poly'],
            'estimator__C': [1, 2, 3, 4],
            'estimator__degree': [3, 4, 5]
        }
    ]
    grid_search = GridSearchCV(ovo_svc, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    if save:
        save_model(grid_search.best_estimator_)

    print(f'Trained SVC, parameters = {grid_search.best_params_}')
    return grid_search.best_estimator_


def train_sgd(X_train, y_train, save=True):
    ovo_sgd = OneVsOneClassifier(SGDClassifier())

    # NOTE: the SVC estimator is stored as estimator on the OneVsOneClassifier
    # use __ to access properties of nested estimator
    parameters = [
        {
            'estimator__loss': ['log', 'perceptron'],
            'estimator__penalty': ['l1', 'l2'],
            'estimator__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'estimator__n_jobs': [-1],
            'estimator__max_iter': [100000]
        }
    ]
    sgd_gridsearch = GridSearchCV(
        ovo_sgd, parameters, cv=5, scoring='accuracy')
    sgd_gridsearch.fit(X_train, y_train)

    if save:
        save_model(sgd_gridsearch.best_estimator_)

    print(f'Trained SGD, parameters = {sgd_gridsearch.best_params_}')
    return sgd_gridsearch.best_estimator_


def load_data(test=False):
    """Load and pre-process data.
    """
    path = f'data/{"test" if test else "training"}.csv'
    print(f'Loading data from {path}')
    df = pd.read_csv(path)
    X = df.drop('class', axis=1)
    y = df['class']

    if test:
        return feature_pipeline.transform(X), label_pipeline.transform(y)

    return feature_pipeline.fit_transform(X), label_pipeline.fit_transform(y)


def main():
    X_train, y_train = load_data()
    X_test, y_test = load_data(test=True)

    # TRAIN
    final_svc = train_svc(X_train, y_train)
    final_sgd = train_sgd(X_train, y_train)

    # EVALUATION
    plot_cf_mat(final_svc, X_train, y_train)
    plot_cf_mat(final_sgd, X_test, y_test)

    plot_ovo_roc(final_svc, X_test, y_test)
    plot_ovo_roc(final_sgd, X_test, y_test)

    plot_ovo_pr(final_svc, X_test, y_test)
    plot_ovo_pr(final_sgd, X_test, y_test)


if __name__ == "__main__":
    main()
