import zipfile
import shutil
import os

#########  home下的Downloads文件夹和DCase文件夹 #############3
BASE_PATH = 'Downloads/'
DEV_DATA_PATH = 'DCase/data/TUT-urban-acoustic-scenes-2018-development-data/audio/'
DEV_TUT_NAME = 'TUT-urban-acoustic-scenes-2018-development'


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


def start(DATA_PATH, TUT_NAME, total):
    file_list = []
    for index in range(total):
        file_list.append(os.path.join(os.path.expanduser('~'),
                                      BASE_PATH + TUT_NAME + '.audio.' + str(index + 1) + '.zip?download=1'))

    for filename in file_list:
        un_zip(filename, os.path.join(os.path.expanduser('~'), DATA_PATH))


if __name__ == '__main__':
    start(DEV_DATA_PATH, DEV_TUT_NAME, 13)
    # start(LEADBORAD_DATA_PATH, LEADBOARD_TUT_NAME, 3)
