import argparse
import matplotlib.pyplot as plt
import os

from features import LogMelExtractor, calculate_logmel, calculate_three_logmel
import config


def plot_logmel():
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments & parameters
    # audios_dir = args.audios_dir

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels

    # Paths
    audio_names = ['airport-barcelona-0-0-a.wav']

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    feature_list = []

    # Select one audio per class and extract feature
    for label in labels:

        for audio_name in audio_names:

            if label in audio_name:
                # audio_path = os.path.join(audios_dir, audio_name)

                feature = calculate_three_logmel(audio_path=audio_name,
                                                 sample_rate=sample_rate,
                                                 feature_extractor=feature_extractor)

                feature_list.append(feature)

                break

    # Plot
    rows_num = 2
    cols_num = 2
    n = 0

    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))

    classes_num = len(labels)

    data = [feature[i] for i in range(3)]
    # for da in data:
    axs[0, 0].matshow(data[0].T, origin='lower', aspect='auto', cmap='jet')
    axs[0, 1].matshow(data[1].T, origin='lower', aspect='auto', cmap='jet')
    axs[1, 0].matshow(data[2].T, origin='lower', aspect='auto', cmap='jet')

    # for n in range(classes_num):
    #     row = n // cols_num
    #     col = n % cols_num
    #     axs[row, col].matshow(feature_list[n].T, origin='lower', aspect='auto', cmap='jet')
    #     axs[row, col].set_title(labels[n])
    #     axs[row, col].set_ylabel('log mel')
    #     axs[row, col].yaxis.set_ticks([])
    #     axs[row, col].xaxis.set_ticks([0, seq_len])
    #     axs[row, col].xaxis.set_ticklabels(['0', '10 s'], fontsize='small')
    #     axs[row, col].xaxis.tick_bottom()
    #
    # for n in range(classes_num, rows_num * cols_num):
    #     row = n // cols_num
    #     col = n % cols_num
    #     axs[row, col].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot(feature):
    rows_num = 2
    cols_num = 2
    n = 0

    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))

    data = [feature[i] for i in range(3)]
    # for da in data:
    axs[0, 0].matshow(data[0].T, origin='lower', aspect='auto', cmap='jet')
    axs[0, 1].matshow(data[1].T, origin='lower', aspect='auto', cmap='jet')
    axs[1, 0].matshow(data[2].T, origin='lower', aspect='auto', cmap='jet')
    plt.tight_layout()
    plt.show()


def test():
    import h5py
    h5 = h5py.File('../features/logmel/TUT-urban-acoustic-scenes-2018-development/development_hpss_lrad.h5','r')
    feature = h5['feature'][0]
    print(feature.shape)
    plot(feature)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    # subparsers = parser.add_subparsers(dest='mode')
    #
    # parser_plot_logmel = subparsers.add_parser('plot_logmel')
    # parser_plot_logmel.add_argument('--audios_dir', type=str, required=True)
    #
    # args = parser.parse_args()
    #
    # if args.mode == 'plot_logmel':
    plot_logmel()

    # else:
    #     raise Exception("Incorrect arguments!")
    # test()