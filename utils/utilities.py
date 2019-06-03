import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from datetime import datetime


def compute_time_consumed(start_time):
    """
    计算训练总耗时
    :param start_time:
    :return:
    """
    time_elapsed = datetime.now() - start_time
    seconds = time_elapsed.seconds
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 3600 % 60
    print("本次训练共耗时 {0} 时 {1} 分 {2} 秒".format(hour, minute, second))


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def read_audio(path, target_fs=None):
    # (audio, fs) = soundfile.read(path)

    # if audio.ndim > 1:
    #     audio = np.mean(audio, axis=1)

    # audio = np.swapaxes(audio, 0, 1)
    # if target_fs is not None and fs != target_fs:
    #     audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
    #     fs = target_fs

    return librosa.load(path, sr=target_fs, mono=False)


def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
    elif x.ndim == 4:
        axis = (0, 1, 2)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):
    return (x - mean) / std


def inverse_scale(x, mean, std):
    return x * std + mean


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:  # 预测正确的情况
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_accuracy(class_wise_accuracy, labels):
    print('{:<30}{}'.format('Scene label', 'accuracy'))
    print('------------------------------------------------')
    for (n, label) in enumerate(labels):
        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
    print('------------------------------------------------')
    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))


def plot_confusion_matrix(confusion_matrix, title, labels, values):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal

    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix2(confusion_matrix, title, labels):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels

    Ouputs:
      None
    """
    plt.rcParams.update({'font.size': 10.5})
    import itertools
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')
    if labels:
        ax.set_xticklabels([''] + labels, rotation=45)
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    row, column = confusion_matrix.shape
    confusion_matrix = np.asarray(confusion_matrix, np.int32)

    for i, j in itertools.product(range(row), range(column)):
        plt.text(j, i, confusion_matrix[i, j], horizontalalignment='center',
                 color='white' if i == j else "black")

    plt.title(title)
    ttl = ax.title
    ttl.set_position([.5, 1.01])
    plt.tight_layout()
    plt.show()


def write_leaderboard_submission(submission_path, audio_names, predictions):
    ix_to_lb = config.ix_to_lb

    f = open(submission_path, 'w')
    f.write('Id,Scene_label\n')

    for n in range(len(audio_names)):
        f.write('{}'.format(os.path.splitext(audio_names[n])[0]))
        f.write(',')
        f.write(ix_to_lb[predictions[n]])
        f.write('\n')

    f.close()

    logging.info('Write result to {}'.format(submission_path))


def write_evaluation_submission(submission_path, audio_names, predictions):
    ix_to_lb = config.ix_to_lb

    f = open(submission_path, 'w')

    for n in range(len(audio_names)):
        f.write('audio/{}'.format(audio_names[n]))
        f.write('\t')
        f.write(ix_to_lb[predictions[n]])
        f.write('\n')

    f.close()

    logging.info('Write result to {}'.format(submission_path))


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.
    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000  # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats


if __name__ == '__main__':
    cm = np.load('../data1.npy')
    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    plot_confusion_matrix2(cm, 'the confusion matrix of DA-MFCNN', labels)
