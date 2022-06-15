import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

subroot = os.getcwd() + '/data/Algo'
baseroot = os.getcwd()

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


def getShape(image='image0'):
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


def getGT(image='image0'):
    # load ground truth image
    image_path = subroot + '/Data/database_v2/' + image + '/' + image + '.raw'
    image_gt = fijii_np(image_path, shape=getShape(), type='<f')

    return image_gt

def getPhantomROI(image='image0'):
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


def moveALL(folderName='', model='solo'):
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

    # if folderName == '':
    folderName = str(time.year) + '-' + s_month + '-' + s_day + '+' + s_hour + '-' + s_minute + '-' + s_second + folderName

    src = baseroot + '/data/Algo/image0/*'
    dtn = '/home/liutie/Documents/outputDatabase#/' + folderName
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)

    src = baseroot + '/runs/*'
    dtn = '/home/liutie/Documents/outputDatabase#/' + folderName + '/tb'
    mkdir(dtn)
    os.system('mv -u ' + src + ' ' + dtn)


def getDatabasePath(i):
    return '/home/liutie/Documents/outputDatabase' + str(i)


def getDataFolderPath(i,folder):
    return getDatabasePath(i) + '/' + folder


def computeThose4(f, image='image0'):
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
         beginning=0,
         bestValue=-1,
         showLess=[],
         replicate=0,
         imagePath='/home/liutie/Pictures',
         whichOptimizer='',
         Together=True):
    plt.figure(figNum)
    end = len(X)
    if tuners[nbTuner] == bestValue:
        plt.plot(X[beginning:end], Y[beginning:end], 'r-x', label=str(tuners[nbTuner]))
    elif nbTuner in showLess:
        plt.plot(X[beginning:end], Y[beginning:end], '--',label=str(tuners[nbTuner]))
    elif nbTuner < 10:
        plt.plot(X[beginning:end], Y[beginning:end], label=str(tuners[nbTuner]))
    elif 10 <= nbTuner < 20:
        plt.plot(X[beginning:end], Y[beginning:end], '.-', label=str(tuners[nbTuner]))
    else:
        plt.plot(X[beginning:end], Y[beginning:end], 'x-', label=str(tuners[nbTuner]))
    plt.legend(loc='best')
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title('(' + whichOptimizer + ')(replicate ' + str(replicate) + ') ' + Title)
    if Together:
        if replicate > 0 and tuners[-1] == tuners[nbTuner]:
            mkdir(imagePath)
            plt.savefig(imagePath + '/(' + whichOptimizer + ')' + Title + '_rep' + str(replicate) + '.png')
    elif not Together:
        mkdir(imagePath)
        plt.savefig(imagePath + '/(' + whichOptimizer + ')' + Title + '_rep' + str(replicate) + ' - ' + str(tuners[nbTuner]) + '.png')


def getImagePath(outputDatabaseNb, folder, optimizer, innerIteration, alpha, outerNb, innerNb, rho=0, threads = 128, REPLICATES=True, replicates=1, total=0, MLEM='False'):
    if REPLICATES:
        replicatesPath = '/replicate_' + str(replicates) + '/' + optimizer + '/Comparison/' \
                         + optimizer
    else:
        replicatesPath = ''

    imagePath = getDataFolderPath(outputDatabaseNb, folder) + replicatesPath \
                + '/config_rho=' + str(rho) + '_sub_i='+str(innerIteration)+'_alpha='+str(alpha)+'_mlem_=' + MLEM \
                + '/ADMM_' + str(threads) + '/'
    imageName = str(total) + '_' + str(outerNb) + '_it' + str(innerNb) + '.img'

    return imagePath + imageName


def calculateDiffCurve(inners, outers, outputDatabaseNb, folder, optimizer, innerIteration, alpha, model='norm', Together=True, rho=0, threads = 128, REPLICATES=True, replicates=1, total=0, MLEM='False'):
    if inners[-1] == innerIteration:
        inners.pop()
    if inners[0] == 0:
        inners.pop(0)
    for o in outers:
        imageDiffs = []
        for i in inners:
            img1 = fijii_np(getImagePath(outputDatabaseNb, folder, optimizer, innerIteration, alpha, o, i, rho, threads, REPLICATES, replicates, total, MLEM), type='<d')
            img2 = fijii_np(getImagePath(outputDatabaseNb, folder, optimizer, innerIteration, alpha, o, i+1 , rho, threads, REPLICATES, replicates, total, MLEM), type='<d')
            imageDiff = 0
            if model == 'norm' :
                imageDiff = np.linalg.norm(img1 - img2)
                title = 'Norm of image difference'
            elif model == 'max':
                imageDiff = np.amax(np.abs(img1 - img2))
                title = 'abs(Max) of image difference'
            imageDiffs.append(imageDiff)
        if Together:
            figNum = 1
        else:
            figNum = o
        PLOT(inners, imageDiffs, outers, o - 1,
             figNum=figNum,
             Xlabel='inner iterations',
             Ylabel='legends -- outer iterations',
             Title=title,
             replicate=replicates,
             imagePath=getDataFolderPath(outputDatabaseNb, folder)+'/imageDifference(' + model + ')/'+optimizer+'+a='+str(alpha),
             whichOptimizer=optimizer,
             Together=Together)


