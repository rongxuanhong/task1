import zipfile
import shutil

BASE_PATH = '/home/r506/Downloads/'
DEV_DATA_PATH = '/home/r506/DCase/data/TUT-urban-acoustic-scenes-2018-development-data/'
DEV_TUT_NAME = 'TUT-urban-acoustic-scenes-2018-development'


# LEADBOARD_TUT_NAME = 'TUT-urban-acoustic-scenes-2018-leaderboard'
# LEADBORAD_DATA_PATH = BASE_PATH + 'TUT-urban-acoustic-scenes-2018-leaderboard-data/'


def un_zip(filename, DATA_PATH):
    zip_file = zipfile.ZipFile(filename)
    for file in zip_file.namelist():
        zip_file.extract(file, DATA_PATH)

    for file in zip_file.namelist():
        shutil.move(DATA_PATH + file, DATA_PATH)
    zip_file.close()


def start(DATA_PATH, TUT_NAME, total):
    file_list = []
    for index in range(total):
        file_list.append(BASE_PATH + TUT_NAME + '.audio.' + str(index + 1) + '.zip?download=1')

    for filename in file_list:
        un_zip(filename, DATA_PATH)


if __name__ == '__main__':
    start(DEV_DATA_PATH, DEV_TUT_NAME, 13)
    # start(LEADBORAD_DATA_PATH, LEADBOARD_TUT_NAME, 3)
