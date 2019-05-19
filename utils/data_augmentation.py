import numpy as np
import pandas as pd
import os
import librosa
from utilities import compute_time_consumed
import time
import soundfile
import tqdm
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
base_path = os.path.join(os.path.expanduser('~'), 'DCase/data/TUT-urban-acoustic-scenes-2018-development')


## random erasing
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_h=1 / 0.3, v_l=0, v_h=255, ):
    """

    :param p:擦除概率
    :param s_l:擦除矩形面积比下界
    :param s_h:擦除矩形面积比上界
    :param r_1:宽高比下界
    :param r_h:宽高比上界
    :param v_l: 替换的随机值下界
    :param v_h:  替换的随机值上界
    :return: eraser func
    reference paper:https://arxiv.org/pdf/1708.04896.pdf
    """

    def eraser(input_img):

        img_c, img_h, img_w = input_img.shape
        p_1 = np.random.rand()  # 随机概率

        if p_1 > p:  # 擦除概率大于0.5，不擦除
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w  # 确定随机的面积比例确定擦除的面积大小
            r = np.random.uniform(r_1, r_h)  # 确定随机的宽高比
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))  # 得到待擦除的矩形宽 w 高 h

            # 随机确定擦除矩阵的第一个坐标
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:  # 擦除的矩形保证在图片范围之内
                break

        c = np.random.uniform(v_l, v_h)

        input_img[:, top:top + h, left:left + w] = c  # 以c擦除

        return input_img

    return eraser


def generate_mix_audio():
    dev_train_csv = os.path.join(base_path, 'evaluation_setup', 'fold1_train.txt')

    data = pd.read_csv(dev_train_csv, sep='\t', names=['file', 'label'])
    data = data.groupby('label')
    for name, group in data:
        print('generate {} audios'.format(name))
        for index, row in tqdm.tqdm(group.iterrows(), total=6122):
            print(row['file'])
            file_path = os.path.join(base_path, row['file'])
            print(file_path)
            audio_name = os.path.splitext(file_path)[0].split('/')[-1]
            print('generate {} audios'.format(audio_name))

            y1, sr = librosa.load(file_path, mono=False, sr=48000)
            sample_audio_file = group.sample(n=1).iloc[0]['file']
            y2, sr = librosa.load(os.path.join(base_path, sample_audio_file), mono=False, sr=48000)
            y = 0.5 * y1 + 0.5 * y2
            new_audio_name = str(audio_name) + '_mix.wav'
            save_path = os.path.join(base_path, 'audio', new_audio_name)
            librosa.output.write_wav(save_path, y, sr)


def generate_fold1_train_mix():
    dev_train_csv = os.path.join(base_path, 'evaluation_setup', 'fold1_train.txt')

    data = pd.read_csv(dev_train_csv, sep='\t', names=['file', 'label'])
    file, label = [], []
    for index, row in data.iterrows():
        file.append(row['file'])
        file.append(row['file'].replace('.wav', '_mix.wav'))
        label.append(row['label'])
        label.append(row['label'])
    data = pd.DataFrame({'file': file, 'label': label})
    data.to_csv('fold1_train_mix.txt', header=False, sep='\t', index=False)
    print(data)


def generate_meta_mix():
    dev_train_csv = os.path.join(base_path, 'meta.csv')

    data = pd.read_csv(dev_train_csv, sep='\t')
    filename, scene_label, identifier, source_label = [], [], [], []
    for index, row in data.iterrows():
        if index > 6121:
            break
        filename.append(row['filename'])
        filename.append(row['filename'].replace('.wav', '_mix.wav'))
        scene_label.append(row['scene_label'])
        scene_label.append(row['scene_label'])
        identifier.append(row['identifier'])
        identifier.append(row['identifier'])
        source_label.append(row['source_label'])
        source_label.append(row['source_label'])
    new_data = pd.DataFrame(
        data={'filename': filename, 'scene_label': scene_label, 'identifier': identifier, 'source_label': source_label},
        columns=['filename', 'scene_label', 'identifier', 'source_label'])
    new_data.to_csv('meta_mix.csv', header=True, sep='\t', index=False)
    # print(new_data)


if __name__ == '__main__':
    generate_mix_audio()
    generate_fold1_train_mix()
    generate_meta_mix()
    pass
