import math
import numpy as np
import matplotlib.pyplot as plt
import time

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir

databaseNum = 20
dataFolderPath = '2022-07-11+16-00-02+admm4+ADMMadpAT+i1+o3000+t128+a=1+mu1+tau2'

optimizer = 'ADMMLim_adaptiveRhoTau-mx'
alphas = [1]
thread = 128
innerIteration = 1
outerIteration = 3000
FULLCONTRAST = True

iterations = range(1, outerIteration)


processPercentage = len(alphas)*len(iterations)
processNum = 0
timeStart = time.perf_counter()
for alpha in alphas:
    for iteration in iterations:
        filename = '0_' + str(iteration) + '_it' + str(innerIteration) + '.img'
        path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_1/' + optimizer + '/Comparison/' \
                   + optimizer + '/' + 'config_rho=0_sub_i=1_alpha=' + str(alpha) + '_mlem_=False/ADMM_' + str(thread) \
                   + '/' + filename

        x_out = fijii_np(path_img, shape=getShape(), type='<d')

        plt.figure(1)
        if FULLCONTRAST:
            plt.imshow(x_out, cmap='gray_r')
            plt.title('outer iteration ' + str(iteration) + '(Full Contrast)')
        else:
            plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
            plt.title('outer iteration ' + str(iteration))
        plt.colorbar()

        if FULLCONTRAST:
            plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImagesADMM/alpha' + str(alpha)
                              + '(FC)') + '/outerIteration' + str(iteration) + '(FC).png')
        else:
            plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImagesADMM/alpha' + str(alpha))
                        + '/outerIteration' + str(iteration) + '.png')
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
