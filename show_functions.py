import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

subroot = os.getcwd() + '/data/Algo'
baseroot = os.getcwd()
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
    src = baseroot + '/data/Algo/image0/*'
    dtn = '/home/liutie/Documents/outputDatabase' + str(databaseNum) + '/' + outFolder + '/' + str(copies) + '/'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)
    os.system('rm -fr ' + baseroot + '/data/Algo/debug/*')
    os.system('rm -fr ' + baseroot + '/data/Algo/metrics/*')


def moveRuns(copies, databaseNum=1, outFolder='400+MLEM+CT,random+0123+*100'):
    src = baseroot + '/runs/*'
    dtn = '/home/liutie/Documents/outputDatabase' + str(databaseNum) + '/' + outFolder + '/tb/tb' + str(copies) + '/'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)


def initialALL():
    os.system('rm -rf ' + baseroot + '/data/Algo/debug/')
    os.system('rm -rf ' + baseroot + '/data/Algo/metrics/')
    os.system('rm -rf ' + baseroot + '/data/Algo/image0/')
    os.system('rm -rf ' + baseroot + '/runs/*')


def moveALL(folderName=''):
    time = datetime.datetime.now()

    if time.month < 10:
        s_month = '0' + str(time.month)
    else:
        s_month = str(time.month)

    if time.day < 10:
        s_day = '0' + str(time.day)
    else:
        s_day = str(time.day)

    if time.hour < 10:
        s_hour = '0' + str(time.hour)
    else:
        s_hour = str(time.hour)

    if time.minute < 10:
        s_minute = '0' + str(time.minute)
    else:
        s_minute = str(time.minute)

    if time.second < 10:
        s_second = '0' + str(time.second)
    else:
        s_second = str(time.second)

    if folderName == '':
        folderName = str(time.year) + '-' + s_month + '-' + s_day + '+' + s_hour + '-' + s_minute + '-' + s_second

    src = baseroot + '/data/Algo/image0/*'
    dtn = '/home/liutie/Documents/outputDatabase#/' + folderName
    mkdir(dtn)
    os.system('mv -b ' + src + ' ' + dtn)

    src = baseroot + '/runs/*'
    dtn = '/home/liutie/Documents/outputDatabase#/' + folderName + '/tb'
    mkdir(dtn)
    os.system('mv -b ' + src + ' ' + dtn)


def getDatabasePath(i):
    return '/home/liutie/Documents/outputDatabase' + str(i)


def getDataFolderPath(i,folder):
    return getDatabasePath(i) + '/' + folder


def computeThose4(f):
    f = fijii_np(f, shape=getShape(), type='<d')
    f_metric = find_nan(f)
    bkg_ROI = getPhantomROI()
    bkg_ROI_act = f_metric[bkg_ROI == 1]
    # IR
    if np.mean(bkg_ROI_act) != 0:
        IR_bkg_recon = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
    else:
        IR_bkg_recon = 1e10

    # MSE
    MSE_recon = np.mean((getGT() * getPhantomROI() - f_metric * getPhantomROI()) ** 2)

    # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
    hot_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw',
                       shape=getShape())
    hot_ROI_act = f_metric[hot_ROI == 1]

    # CRC hot
    # CRC_hot_recon.append(np.mean(hot_ROI_act) / 400.)
    CRC_hot_recon = np.abs(np.mean(hot_ROI_act) - 400.)

    cold_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw',
                        shape=getShape())
    cold_ROI_act = f_metric[cold_ROI == 1]

    # MA cold
    MA_cold_recon = np.abs(np.mean(cold_ROI_act))

    return IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon

def PLOT(X,
         Y,
         tuners,
         nbTuner,
         figNum=1,
         Xlabel='X',
         Ylabel='Y',
         Title='',
         beginning=1,
         bestValue=-1,
         showLess=[0,1],
         replicate=0,
         imagePath='/home/liutie/Pictures',
         whichOptimizer=''):
    plt.figure(figNum)
    if tuners[nbTuner] == bestValue:
        plt.plot(X[beginning:-1], Y[beginning:-1], 'r-x', label=str(tuners[nbTuner]))
    elif nbTuner in showLess:
        plt.plot(X[beginning:-1], Y[beginning:-1], '--',label=str(tuners[nbTuner]))
    elif nbTuner < 10:
        plt.plot(X[beginning:-1], Y[beginning:-1], label=str(tuners[nbTuner]))
    elif 10 <= nbTuner < 20:
        plt.plot(X[beginning:-1], Y[beginning:-1], '.-', label=str(tuners[nbTuner]))
    else:
        plt.plot(X[beginning:-1], Y[beginning:-1], 'x-', label=str(tuners[nbTuner]))
    plt.legend(loc='best')
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title('(' + whichOptimizer + ')(replicate ' + str(replicate) + ') ' + Title)
    if replicate > 0 and tuners[-1]==tuners[nbTuner]:
        plt.savefig(imagePath + '/(' + whichOptimizer + ')' + Title + '_rep' + str(replicate) + '.png')

