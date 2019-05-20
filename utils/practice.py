import librosa

from scipy import signal
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import pandas as pd
import soundfile

import h5py


def load_audio():
    # path = '/home/ccyoung/Downloads/dcase2018_task1-master/features/logmel/TUT-urban-acoustic-scenes-2018-development/development_hpss_lrad.h5'
    # hf1 = h5py.File(path, 'r')
    # feature = hf1['feature'][6122:]
    # filename1 = hf1['filename'][6122:]
    # identifier1 = hf1['identifier'][6122:]
    # scene_label1 = hf1['scene_label'][6122:]
    # source_label1 = hf1['source_label'][6122:]

    path1 = '/home/ccyoung/Downloads/dcase2018_task1-master/features/logmel/TUT-urban-acoustic-scenes-2018-development/development_hpss_lrad_mix.h5'
    hf2 = h5py.File(path1, 'r')
    print(list(hf2['source_label']))
    print(list(hf2['identifier']))
    print(list(hf2['scene_label']))
    # hf2['feature'].resize(2519, axis=0)
    # for i in range(2518):
    #     hf2['feature'][12244 + i] = feature[i]
    #
    # filename = hf2['filename']
    # filename = np.hstack((hf2['filename'], filename1))
    # identifier = hf2['identifier']
    # identifier = np.hstack((hf2['identifier'], identifier1))
    # scene_label = hf2['scene_label']
    # scene_label = np.hstack((hf2['scene_label'], scene_label1))
    # source_label = hf2['source_label']
    # source_label = np.hstack((hf2['source_label'], source_label1))
    # hf1.close()
    hf2.close()


if __name__ == '__main__':
    from data_generator import DataGenerator

    workspace = "/home/r506/Downloads/dcase2018_task1-master"
    subdir='TUT-urban-acoustic-scenes-2018-development'
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development_hpss_lrad_mix.h5')
    dev_train_csv = os.path.join('/home/r506/DCase/data', subdir, 'evaluation_setup',
                                 'fold1_train_mix.txt')

    dev_validate_csv = os.path.join('/home/r506/DCase/data', subdir, 'evaluation_setup',
                                    'fold1_evaluate.txt')
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=8,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv,
                              )
    generator.train()
    pass
