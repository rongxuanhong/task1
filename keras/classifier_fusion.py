import sklearn as sk
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import calculate_accuracy
import numpy as np


def prepare_data(datatype):
    workspace = "/home/ccyoung/Downloads/dcase2018_task1-master"
    truncation_dir = os.path.join(workspace, 'features', 'truncation',
                                  'holdout_fold={}'.format(1))
    if datatype == 'train':
        hf = h5py.File(os.path.join(truncation_dir, 'train_hpss_6800.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)
    elif datatype == 'validate':
        hf = h5py.File(os.path.join(truncation_dir, 'validate_hpss_6800.h5'), 'r')
        features = hf['feature'][:]
        targets = hf['target'][:]
        return features, np.argmax(targets, axis=-1)


def stack_one_stage():
    ## 1, 准备数据
    '''创建训练的数据集'''
    X, y = prepare_data('train')
    X_predict, y_predict = prepare_data('validate')
    print(y_predict.shape)
    # y = np.argmax(y, axis=-1)
    # y_predict = np.argmax(y_predict, axis=-1)

    ## 2.定义K个第一阶段的模型
    '''模型融合中使用到的各个单模型'''
    clfs = [
        RandomForestClassifier(n_estimators=390, random_state=10, n_jobs=4,
                               max_features=1, min_samples_leaf=1, max_depth=13, ),
        SVC(C=0.6, gamma=1e-4, random_state=10)]
    ## 3. 保存第一阶段模型所有输出 shape=(预测的样本数，多个模型的预测)作为第二阶段的特征输入
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))
    print(dataset_blend_train.shape)
    print(dataset_blend_test.shape)
    # ４. 第一阶段训练：对每个模型分别做交叉验证训练，并汇总所有预测结果作为新的特征
    '''5折stacking'''
    n_folds = 5
    skf = StratifiedKFold(n_folds)
    for j, clf in enumerate(clfs):

        dataset_blend_test_j = np.zeros((X_predict.shape[0], n_folds))  # 汇总交叉验证的结果

        ##交叉验证训练
        for i, (train, test) in enumerate(skf.split(X, y)):  # 遍历每一折数据 train,test均为索引
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            print("Fold", i)
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission  # 保存每个模型Ｋ折预测结果
            dataset_blend_test_j[:, i] = clf.predict(X_predict)
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)  # 平均每一折结果
        # print("auc Score: %f" % calculate_accuracy(np.argmax(y_predict, axis=-1), dataset_blend_test[:, j], classes_num=10,
        #                                            average='macro'))

    ##５. 第二阶段的训练 使用新的模型训练　得到第一阶段生成的新训练集再次训练，并使用新测试集再次预测
    np.save('data.npz', dataset_blend_train)
    np.save('label.npz', dataset_blend_test)


def stack_two_stage():
    X, y = prepare_data('train')
    X_predict, y_predict = prepare_data('validate')
    dataset_blend_train = np.load('data.npz.npy')
    dataset_blend_test = np.load('label.npz.npy')

    clf = GradientBoostingClassifier(learning_rate=0.005,subsample=0.8,random_state=10,n_estimators=600)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)

    # print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("blend result")
    print("auc Score: %f" % (
        calculate_accuracy(y_predict, y_submission, classes_num=10, average='macro')))


if __name__ == '__main__':
    stack_two_stage()
