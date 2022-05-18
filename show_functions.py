import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

subroot = os.getcwd() + '/data/Algo'
image = 'image0'

def read_input_dim(file_path):
    # Read CASToR header file to retrieve image dimension """
    with open(file_path) as f:
        for line in f:
            if 'matrix size [1]' in line.strip():
                dim1 = [int(s) for s in line.split() if s.isdigit()][-1]
            if 'matrix size [2]' in line.strip():
                dim2 = [int(s) for s in line.split() if s.isdigit()][-1]
            if 'matrix size [3]' in line.strip():
                dim3 = [int(s) for s in line.split() if s.isdigit()][-1]

    # Create variables to store dimensions
    PETImage_shape = (dim1, dim2, dim3)
    PETImage_shape_str = str(dim1) + ',' + str(dim2) + ',' + str(dim3)
    # print('image shape :', PETImage_shape)
    return PETImage_shape_str

def input_dim_str_to_list(PETImage_shape_str):
    return [int(e.strip()) for e in PETImage_shape_str.split(',')]  # [:-1]

def getShape():
    PETimage_shape_str = read_input_dim(subroot + '/Data/database_v2/' + image + '/' + image + '.hdr')
    PETimage_shape = input_dim_str_to_list(PETimage_shape_str)

    return PETimage_shape


# copy from vGeneral.py
def fijii_np(path, shape=getShape(), type='<f'):
    """"Transforming raw data to numpy array"""
    file_path = path
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    return image




def find_nan(image):
    """ find NaN values on the image"""
    idx = np.argwhere(np.isnan(image))
    # print('index with NaN value:',len(idx))
    for i in range(len(idx)):
        image[idx[i, 0], idx[i, 1]] = 0
    # print('index with NaN value:',len(np.argwhere(np.isnan(image))))
    return image


def getGT():
    # load ground truth image
    image_path = subroot + '/Data/database_v2/' + image + '/' + image + '.raw'
    image_gt = fijii_np(image_path, shape=getShape(), type='<f')

    return image_gt

def getPhantomROI():
    # select only phantom ROI, not whole reconstructed image
    path_phantom_ROI = subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw'
    my_file = Path(path_phantom_ROI)
    if my_file.is_file():
        phantom_ROI = fijii_np(path_phantom_ROI, shape=getShape(), type='<f')
    else:
        phantom_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw',
                               shape=getShape(), type='<f')

    return phantom_ROI


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def dldir(path):
    folder = os.path.exists(path)
    if folder:
        os.system('rm - fr ' + path)


def moveData(copies, databaseNum=1, outFolder='400+MLEM+CT,random+0123+*100'):
    src = '/home/liutie/STAGE-STING/DIP/data/Algo/image0/*'
    dtn = '/home/liutie/Documents/outputDatabase' + str(databaseNum) + '/' + outFolder + '/' + str(copies) + '/'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)
    os.system('rm -fr /home/liutie/STAGE-STING/DIP/data/Algo/debug/*')
    os.system('rm -fr /home/liutie/STAGE-STING/DIP/data/Algo/metrics/*')


def moveRuns(copies, databaseNum=1, outFolder='400+MLEM+CT,random+0123+*100'):
    src = '/home/liutie/STAGE-STING/DIP/runs/*'
    dtn = '/home/liutie/Documents/outputDatabase' + str(databaseNum) + '/' + outFolder + '/tb/tb' + str(copies) + '/'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)


def initialALL():
    os.system('rm -rf /home/liutie/STAGE-STING/DIP/data/Algo/debug/')
    os.system('rm -rf /home/liutie/STAGE-STING/DIP/data/Algo/metrics/')
    os.system('rm -rf /home/liutie/STAGE-STING/DIP/data/Algo/image0/')
    os.system('rm -rf /home/liutie/STAGE-STING/DIP/runs/*')


def moveALL():
    time = datetime.datetime.now()

    src = '/home/liutie/STAGE-STING/DIP/data/Algo/image0/*'
    dtn = '/home/liutie/Documents/outputDatabase#/' + str(time.year) + '-' + str(time.month) + '-' \
          + str(time.day) + '+' + str(time.hour) + '-' + str(time.minute) + '-' + str(time.second)
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)

    src = '/home/liutie/STAGE-STING/DIP/runs/*'
    dtn = '/home/liutie/Documents/outputDatabase#/'  + str(time.year) + '-' + str(time.month) + '-' \
          + str(time.day) + '+' + str(time.hour) + '-' + str(time.minute) + '-' + str(time.second) + '/tb'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)


def getDatabasePath(i):
    return '/home/liutie/Documents/outputDatabase' + str(i)

def getDataFolderPath(i,folder):
    return getDatabasePath(i) + '/' + folder
