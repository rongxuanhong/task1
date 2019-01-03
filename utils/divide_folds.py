import pandas as pd
import os
import dcase_util
import random
import tempfile

WORKSPACE = "/home/ccyoung/Downloads/dcase2018_task1-master"


def generate_new_meta_csv():
    meta_csv_path = os.path.join(WORKSPACE, 'appendixes', 'meta.csv')
    df = pd.read_csv(meta_csv_path, sep='\t')

    data = df.groupby(['scene_label'])
    new_df = pd.DataFrame(data=[], columns=['filename', 'scene_label', 'identifier', 'source_label'])
    for k, v in data:
        groups = [df for _, df in v.groupby('identifier')]
        random.shuffle(groups)
        v = pd.concat(groups).reset_index(drop=True)
        new_df = pd.concat([new_df, v])
    new_df.to_csv(os.path.join(WORKSPACE, 'appendixes', 'meta1.csv'), index=False, sep='\t')


def plot_data():
    path = '/home/ccyoung/DCase/data/TUT-urban-acoustic-scenes-2018-development/meta.csv'
    data = pd.read_csv(path, sep='\t', names=['a', 'b', 'c', 'd'])
    cnt = data[:6123].groupby(['b', 'c']).count()
    print(cnt.loc['tram']['a'])
    # print(cnt)


def get_validate_data():
    db = dcase_util.datasets.TUTUrbanAcousticScenes_2018_DevelopmentSet(
        data_path='',
        included_content_types=['meta']
    )
    db.initialize()
    # db.show()
    training_files, validation_files = db.validation_split(
        fold=1,
        split_type='balanced',
        validation_amount=0.1
    )
    training_files = ['audio/' + x.split('/')[-1] for x in training_files]
    validation_files = ['audio/' + x.split('/')[-1] for x in validation_files]
    base_path = '/home/ccyoung/DCase/data/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/'
    path = base_path + 'fold1_train.txt'
    data = pd.read_csv(path, sep='\t', names=['filename', 'scene'])

    train_df = data.loc[data['filename'].isin(training_files)]
    validation_df = data.loc[data['filename'].isin(validation_files)]

    # print(validation_files)
    # print(len(train_df))
    # print(len(validation_df))
    train_path = base_path + 'fold1_train_new.txt'
    validate_path = base_path + 'fold1_validate.txt'
    train_df.to_csv(train_path, header=False, sep='\t', index=False)
    validation_df.to_csv(validate_path, header=False, sep='\t', index=False)


if __name__ == '__main__':
    # generate_new_meta_csv()
    get_validate_data()
