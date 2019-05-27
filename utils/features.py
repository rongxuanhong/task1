import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random

from utilities import read_audio, create_folder
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)

        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=50.,
                                        fmax=sample_rate // 2).T

    def transform(self, audio):
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        [f, t, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=window_size,
            noverlap=overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T

        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)

        return x


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    feature = feature_extractor.transform(audio)

    return feature


def calculate_logdelta(audio_path, sample_rate, feature_extractor):
    # Read audio
    (audio, fs) = librosa.load(audio_path, sr=44100)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    feature = feature_extractor.transform(audio)
    # feature = feature.astype(np.float32)
    # feature1 = librosa.feature.delta(feature, width=3)
    feature = librosa.feature.delta(feature, width=11)
    feature = np.log(feature + 1e-8)
    # feature3 = librosa.feature.delta(feature, width=19)

    return feature


def calculate_multi_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    l_feature = feature_extractor.transform(audio[0])
    r_feature = feature_extractor.transform(audio[1])

    return np.stack([l_feature, r_feature], axis=0)


def calculate_hpss_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    h, p = librosa.effects.hpss(audio)
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)

    return np.stack([h_feature, p_feature], axis=0)


def calculate_three_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    # Extract feature
    l_r = audio[0] + audio[1]
    audio = np.mean(audio, axis=0)
    h, p = librosa.effects.hpss(audio)
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)
    l_r_feature = feature_extractor.transform(l_r)

    return np.stack([h_feature, p_feature, l_r_feature], axis=0)


def calculate_lrad_logmel(audio_path, sample_rate, feature_extractor):
    # Read stereo audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    # audio = audio.T
    h = audio[0] - audio[1]
    p = audio[0] + audio[1]
    h_feature = feature_extractor.transform(h)
    p_feature = feature_extractor.transform(p)

    return np.stack([h_feature, p_feature], axis=0)


def read_development_meta(meta_csv):
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)

    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []

    for row in df.iterrows():
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']

        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)

    return audio_names, scene_labels, identifiers, source_labels


def read_evaluation_meta(evaluation_csv):
    with open(evaluation_csv, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)

    audio_names = []

    for li in lis:
        audio_name = li[0].split('/')[1]
        audio_names.append(audio_name)

    return audio_names


def calculate_multi_features(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')

    if data_type == 'development':
        meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')

    elif data_type in ['leaderboard', 'evaluation']:
        evaluation_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                      'test.txt')

    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 '{}_hpss_lrad_2019.h5'.format(data_type))

    create_folder(os.path.dirname(hdf5_path))

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    # Read meta csv
    if data_type == 'development':
        [audio_names, scene_labels, identifiers, source_labels] = \
            read_development_meta(meta_csv)

    elif data_type in ['leaderboard', 'evaluation']:
        audio_names = read_evaluation_meta(evaluation_csv)

    # Only use partial data when set mini_data to True
    if mini_data:

        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0: audios_num]

        audio_names = [audio_names[idx] for idx in audio_indexes]

        if data_type == 'development':
            scene_labels = [scene_labels[idx] for idx in audio_indexes]
            identifiers = [identifiers[idx] for idx in audio_indexes]
            source_labels = [source_labels[idx] for idx in audio_indexes]

    print('Number of audios: {}'.format(len(audio_names)))

    # Create hdf5 file
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(
        name='feature',
        shape=(0, 3, seq_len, mel_bins),
        maxshape=(None, 3, seq_len, mel_bins),
        dtype=np.float32)

    calculate_time = time.time()

    for (n, audio_name) in enumerate(audio_names):
        print(n, audio_name)

        # Calculate feature
        audio_path = os.path.join(audio_dir, audio_name)

        # Extract feature
        feature = calculate_three_logmel(audio_path=audio_path,
                                         sample_rate=sample_rate,
                                         feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''

        # print(feature.shape)

        hf['feature'].resize((n + 1, 3, seq_len, mel_bins))
        hf['feature'][n] = feature

    # Write meta info to hdf5
    hf.create_dataset(name='filename',
                      data=[s.encode() for s in audio_names],
                      dtype='S50')
    if data_type == 'development':
        hf.create_dataset(name='scene_label',
                          data=[s.encode() for s in scene_labels],
                          dtype='S20')

        hf.create_dataset(name='identifier',
                          data=[s.encode() for s in identifiers],
                          dtype='S20')

        hf.create_dataset(name='source_label',
                          data=[s.encode() for s in source_labels],
                          dtype='S20')

    hf.close()

    print('Write out hdf5 file to {}'.format(hdf5_path))
    print('Time spent: {} s'.format(time.time() - calculate_time))


def calculate_features(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')

    if data_type == 'development':
        meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')

    elif data_type in ['leaderboard', 'evaluation']:
        evaluation_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                      'test.txt')

    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_{}.h5'.format(data_type))
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 '{}_delta11.h5'.format(data_type))

    create_folder(os.path.dirname(hdf5_path))

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    # Read meta csv
    if data_type == 'development':
        [audio_names, scene_labels, identifiers, source_labels] = \
            read_development_meta(meta_csv)

    elif data_type in ['leaderboard', 'evaluation']:
        audio_names = read_evaluation_meta(evaluation_csv)

    # Only use partial data when set mini_data to True
    if mini_data:

        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0: audios_num]

        audio_names = [audio_names[idx] for idx in audio_indexes]

        if data_type == 'development':
            scene_labels = [scene_labels[idx] for idx in audio_indexes]
            identifiers = [identifiers[idx] for idx in audio_indexes]
            source_labels = [source_labels[idx] for idx in audio_indexes]

    print('Number of audios: {}'.format(len(audio_names)))

    # Create hdf5 file
    hf = h5py.File(hdf5_path, 'w')

    hf.create_dataset(
        name='feature',
        shape=(0, seq_len, mel_bins),
        maxshape=(None, seq_len, mel_bins),
        dtype=np.float32)

    calculate_time = time.time()

    for (n, audio_name) in enumerate(audio_names):
        print(n, audio_name)

        # Calculate feature
        audio_path = os.path.join(audio_dir, audio_name)

        # Extract feature
        feature = calculate_logdelta(audio_path=audio_path,
                                     sample_rate=sample_rate,
                                     feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''

        print(feature.shape)

        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature

        # Plot log Mel for debug
        # if False:
        #     plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
        #     plt.show()

    # Write meta info to hdf5
    hf.create_dataset(name='filename',
                      data=[s.encode() for s in audio_names],
                      dtype='S50')

    if data_type == 'development':
        hf.create_dataset(name='scene_label',
                          data=[s.encode() for s in scene_labels],
                          dtype='S20')

        hf.create_dataset(name='identifier',
                          data=[s.encode() for s in identifiers],
                          dtype='S20')

        hf.create_dataset(name='source_label',
                          data=[s.encode() for s in source_labels],
                          dtype='S20')

    hf.close()

    print('Write out hdf5 file to {}'.format(hdf5_path))
    print('Time spent: {} s'.format(time.time() - calculate_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--subdir', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, required=True,
                               choices=['development', 'leaderboard', 'evaluation'])
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'logmel':

        calculate_multi_features(args)
        # calculate_features(args)

    else:
        raise Exception('Incorrect arguments!')
