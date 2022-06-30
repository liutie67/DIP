import math
import numpy as np
import matplotlib.pyplot as plt

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir

for lr in Tuners.lrs0:
    databaseNum = 15
    dataFolderPath = '2022-06-30+11-03-32+wx+px+MLEM1000+lr=lrs0+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t64'

    # lr = Tuners[0]
    opti = 'Adam'
    skip = 0
    scaling = 'standardization'
    INPUT = 'CT'

    inner_iter = 50
    alpha = 0.084
    sub_iter = 1000

    windowSize = 100
    patienceNum = 100
    VARmin = math.inf
    VARs = []
    MSEs = []
    PSNRs = []

    x0 = np.zeros(getShape()).flatten()
    queueQ = []
    '''
    for i in range(windowSize-1):
        queueQ.append(x0)
    '''
    x_gt = getGT()
    stagnate = 0
    success = False
    epochStar = -1
    for epoch in range(sub_iter):
        filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'
        path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_1/nested/Block2/out_cnn' + '/24/' + filename

        x_out = fijii_np(path_img, shape=getShape())
        MSE = np.mean((x_gt*getPhantomROI()-x_out*getPhantomROI())**2)
        MSEs.append(MSE)
        PSNR = 10*np.log((np.amax(np.abs(x_gt*getPhantomROI())))**2/MSE)
        PSNRs.append(PSNR)

        queueQ.append(x_out.flatten())
        if len(queueQ) == windowSize:
            mean = queueQ[0].copy()
            for x in queueQ[1:windowSize]:
                mean += x
            mean = mean/windowSize
            VAR = np.linalg.norm(queueQ[0]-mean)**2
            for x in queueQ[1:windowSize]:
                VAR += np.linalg.norm(x-mean)**2
            VAR = VAR/windowSize
            if VAR < VARmin:
                VARmin = VAR
                epochStar = epoch
                stagnate = 1
            else:
                stagnate += 1
            if stagnate == patienceNum:
                success = True
            queueQ.pop(0)
            VARs.append(VAR)

    plt.figure()
    var_x = np.arange(windowSize, windowSize+len(VARs))
    plt.plot(var_x, VARs, '.')
    plt.title('Window moving variance - epochStar=' + str(epochStar))
    plt.axvline(x=epochStar+1, c="r")
    plt.axhline(y=np.min(VARs), c="g", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                + '/VARs') + '/lr' + str(lr) + '+VARs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    # constantYs = np.linspace(np.min(VARs), np.max(VARs), 100)

    plt.figure()
    plt.plot(MSEs, '.')
    plt.title('Mean square error - epochStar=' + str(epochStar))
    plt.axvline(x=epochStar+1, c="r")
    plt.axhline(y=np.min(MSEs), c="g", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                + '/MSEs') + '/lr' + str(lr) + '+MSEs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')


    plt.figure()
    plt.plot(PSNRs, '.')
    plt.title('Peak signal-noise ratio - epochStar=' + str(epochStar))
    plt.axvline(x=epochStar+1, c="r")
    plt.axhline(y=np.max(PSNRs), c="g", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                + '/PSNRs') + '/lr' + str(lr) + '+PSNRs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')

# plt.show()







