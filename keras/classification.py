import sklearn as sk
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer, precision_score
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import calculate_accuracy, calculate_confusion_matrix, plot_confusion_matrix2, print_accuracy
import config as cfg


def prepare_data(datatype):
    workspace = os.path.join(os.path.expanduser('~'), "Downloads/dcase2018_task1-master")
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(1))
    if datatype == 'train':
        hf = h5py.File(os.path.join(truncation_dir, 'train_hpss_l+r_9100.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)
    elif datatype == 'validate':
        hf = h5py.File(os.path.join(truncation_dir, 'validate_hpss_l+r_9100.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)


def model_validate(classifier, class_wise_accuracy=False, plot_confusion_matrix=False):
    x_val, y_val = prepare_data(datatype='validate')
    if class_wise_accuracy:
        predict = classifier.predict(x_val)
        if plot_confusion_matrix:
            cm = calculate_confusion_matrix(y_val, predict, 10)
            plot_confusion_matrix2(cm, "svm", cfg.labels)
        class_wise_accuracy = calculate_accuracy(y_val, predict, 10)
        print_accuracy(class_wise_accuracy, cfg.labels)
        score = np.mean(class_wise_accuracy)
    else:

        score = classifier.score(x_val, y_val)
        print('The accuracy of validation: {:.4f}'.format(score))
    return score


def model():
    # 0.699 提升1%
    # classifier = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=4,
    #                                     max_features=1, min_samples_leaf=1, max_depth=20, )

    # 提升1.2%
    # classifier = SVC(C=1.0, gamma=2e-4, random_state=10)

    # SVC(C=0.6, gamma=1e-4, random_state=10) 8500 提升0.9 72.2
    pass


def train():
    c1 = [x * 0.1 for x in range(1, 11, 1)]
    c2 = [x * 1e-4 for x in range(1, 10, 1)]
    # cs = range(100, 400, 10)
    # print(cs)
    # classifier = SVC(C=0.7, gamma=0.0008000000000000001, random_state=10)
    # X_train, y_train = prepare_data(datatype='train')
    # classifier.fit(X_train, y_train)
    # model_fit(classifier)

    # print(X_train.shape)
    # print(y_train.shape)
    # classifier = XGBClassifier(learning_rate=0.01, objective="multi:softmax", n_jobs=4, subsample=0.8, seed=10)
    # classifier = SVC(C=0.7, gamma=8e-4, random_state=10)
    # param_grid = {'C': [x * 0.1 for x in range(1, 10, 1)], 'gamma': [x * 1e-4 for x in range(1, 10, 1)]}
    # gv_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
    #                          scoring=make_scorer(precision_score,average="macro"), n_jobs=4, cv=3,
    #                          iid=True)
    # X_train, y_train = prepare_data(datatype='train')
    # gv_search.fit(X_train, y_train)
    # print(gv_search.cv_results_)
    # print(gv_search.best_params_)
    # print(gv_search.best_score_)

    # classifier = SVC(C=0.9, gamma=9e-4, random_state=10)
    # model_fit(classifier)

    # X_val, y_val = prepare_data(datatype='validate')
    # print(classification_report(y_val, gv_search.predict(X_val)))

    # base 0.653
    # 0.4 9e-5 0.6 0.7
    max_acc = 0
    for c3 in c1:
        for c4 in c2:
            print(c3, c4)
            classifier = SVC(C=c3, gamma=c4, random_state=10, probability=True)
            X_train, y_train = prepare_data(datatype='train')
            classifier.fit(X_train, y_train)
            score = model_validate(classifier)
            max_acc = max(max_acc, score)

    print('max score：{:.4f}'.format(max_acc))


if __name__ == '__main__':
    train()
    # classifier = SVC(C=0.5, gamma=1e-4, random_state=10, probability=True)
    # X_train, y_train = prepare_data(datatype='train')
    # classifier.fit(X_train, y_train)
    # model_validate(classifier, class_wise_accuracy=True, plot_confusion_matrix=True)
    pass
