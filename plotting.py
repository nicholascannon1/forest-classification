"""Plotting functions

Written by Nicholas Cannon
"""
from sklearn.metrics import (
    plot_confusion_matrix, precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import os


def get_model_name(model):
    """Gets class name of predictor.
    """
    return type(model.estimator).__name__


def save_fig(plt_name, path=''):
    """Saves matplotlib figure in plotting directory in path.
    """
    path = path or os.getcwd()

    # create plots directory if it doesn't exist
    if not os.path.exists(os.path.join(path, 'plots')):
        os.makedirs(os.path.join(path, 'plots'))

    # create full path
    path = os.path.join(path, 'plots', plt_name)

    print(f'Saving figure to {path}')
    plt.savefig(path)


def plot_cf_mat(model, X, y, name=False, norm='true', path=''):
    """Plots confusion matrix
    """
    name = name or get_model_name(model)
    plot_confusion_matrix(model, X, y, cmap=plt.cm.Blues,
                          normalize=norm, include_values=True)
    plt.title('Confusion Matrix for ' + name)
    save_fig(f'{name}_cf_mat', path=path)


def plot_ovo_roc(model, X_test, y_test, name=False, path=''):
    """Plots ROC curve for OneVOneClassifier.
    """
    name = name or get_model_name(model)
    plt.figure()
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate (recall)')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve for ' + name)

    y_score = model.decision_function(X_test)
    for i in range(model.n_classes_):
        # get the labels for only this class
        y_test_i = (y_test == model.classes_[i])
        fpr, tpr, _ = roc_curve(y_test_i, y_score[:, i])

        plt.plot(fpr, tpr, lw=2,
                 label=f'{model.classes_[i]} (AUC={auc(fpr, tpr)})')

    plt.legend(loc="lower right")
    save_fig(f'{name}_roc', path=path)


def plot_ovo_pr(model, X_test, y_test, name=False, path=''):
    """Plots Precision Recall curve for OneVOneClassifier.
    """
    name = name or get_model_name(model)
    plt.figure()
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.title('PR Curve for ' + name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    y_scores = model.decision_function(X_test)
    for i in range(model.n_classes_):
        y_test_i = (y_test == model.classes_[i])
        precisions, recalls, _ = precision_recall_curve(
            y_test_i, y_scores[:, i])

        plt.plot(recalls, precisions, lw=2,
                 label=f'{model.classes_[i]} (AUC={auc(recalls, precisions)})')

    plt.legend(loc='best')
    save_fig(f'{name}_pr', path=path)
