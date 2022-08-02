import math
import numpy as np
import matplotlib.pyplot as plt
import time

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir

'''
Save images from DIP into png.
Hardcoded path to loop through all the images(decided by variable 'epoches').

pseudo codes of this file:
    1. initialise all the parameters used in the hardcoded path.
    2. loop on learning rates.
        2.1 loop on epoches.
            2.1.1 save image into png.
        
'''

# 1. initialise all the parameters used in the hardcoded path.
databaseNum = 'F'  # choose the database number
dataFolderPath = '2022-07-26+15-06-43+ADMMi100o100a0.005+lr=lr2+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t128+313s'  # choose the folder path

opti = 'Adam'
skip = 0
scaling = 'standardization'
INPUT = 'CT'

inner_iter = 50
alpha = 0.084
sub_iter = 1000

FULLCONTRAST = False

lrs = Tuners.lrs2[2:14]  # choose the lrs to be saved into png
epoches = range(0, 500)  # choose the epoch to be saved into png

processPercentage = len(epoches)*len(lrs)       # variable used to track the process
processNum = 0                                  # variable used to track the process
timeStart = time.perf_counter()
# 2. loop on learning rates.
for lr in lrs:
    # 2.1 loop on epoches.
    for epoch in epoches:
        filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'
        path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_1/nested/Block2/out_cnn' + '/24/' + filename

        x_out = fijii_np(path_img, shape=getShape())

        plt.figure(1)
        if FULLCONTRAST:
            plt.imshow(x_out, cmap='gray_r')
            plt.title('epoch ' + str(epoch) + '(Full Contrast)')
        else:
            plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
            plt.title('epoch ' + str(epoch))
        plt.colorbar()

        # 2.1.1 save image into png.
        if FULLCONTRAST:
            plt.savefig(
                mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImages/lr' + str(lr) + '(FC)')
                + '/epoch' + str(epoch) + '(FC).png')
        else:
            plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImages/lr' + str(lr))
                        + '/epoch' + str(epoch) + '.png')
        plt.clf()

        timeEnd = time.perf_counter()
        processNum += 1
        if processNum % 5 == 0:
            print('Processing: ' + str(format((processNum/processPercentage)*100, '.2f')) + '% finished\t', end='')
            v = 5/(timeEnd - timeStart)
            minute = int(((processPercentage - processNum)/v)/60)
            second = int(((processPercentage - processNum)/v) % 60)
            print('estimated time left: ' + str(minute) + ' min ' + str(second) + 's')
            timeStart = time.perf_counter()
