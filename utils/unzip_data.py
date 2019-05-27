import zipfile
import shutil
import os

#########  home下的Downloads文件夹和DCase文件夹 #############3
DEV_DATA_PATH = 'DCase/data/'
DEV_TUT_NAME = 'TAU-urban-acoustic-scenes-2019-development'


# LEADBOARD_TUT_NAME = 'TUT-urban-acoustic-scenes-2018-leaderboard'
# LEADBORAD_DATA_PATH = BASE_PATH + 'TUT-urban-acoustic-scenes-2018-leaderboard-data/'


def un_zip(filename, DATA_PATH):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)  # 递归构建目录
    zip_file = zipfile.ZipFile(filename)
    for file in zip_file.namelist():
        zip_file.extract(file, DATA_PATH)  # 抽取文件到指定目录下 audio/下
    # for file in zip_file.namelist():
    #     shutil.move(DATA_PATH + file, path)
    zip_file.close()


def start(total):
    file_list = []
    for index in range(total):
        file_list.append(os.path.join(os.path.expanduser('~'),
                                      'Downloads', DEV_TUT_NAME + '.audio.' + str(index + 1) + '.zip?download=1'))

    for i, filename in enumerate(file_list):
        if i>1:
            print(filename)
            un_zip(filename, os.path.join(os.path.expanduser('~'), DEV_DATA_PATH))


if __name__ == '__main__':
    start(21)
#   start(DEV_DATA_PATH, DEV_TUT_NAME, 21)
# start(LEADBORAD_DATA_PATH, LEADBOARD_TUT_NAME, 3)
