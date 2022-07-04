import math
import numpy as np
import matplotlib.pyplot as plt
import time

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir


databaseNum = 15
dataFolderPath = '2022-06-30+15-42-21+wx+px+ADMMadpATi1o100*100a1+lr=lrs1+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t128+257s'

# lr = 0.01
opti = 'Adam'
skip = 0
scaling = 'standardization'
INPUT = 'CT'

inner_iter = 50
alpha = 0.084
sub_iter = 1000

lrs = Tuners.lrs1[4:6+1]
epoches = range(250, 400)


processPercentage = len(epoches)*len(lrs)
processNum = 0
timeStart = time.perf_counter()
for lr in lrs:
    for epoch in epoches:
        filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'
        path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_1/nested/Block2/out_cnn' + '/24/' + filename

        x_out = fijii_np(path_img, shape=getShape())

        plt.figure()
        plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
        plt.title('epoch ' + str(epoch))
        plt.colorbar()

        plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImages/lr' + str(lr)) + '/epoch' + str(epoch) + '.png')
        plt.close()

        timeEnd = time.perf_counter()
        processNum += 1
        if processNum % 5 == 0:
            print('Processing: ' + str(format((processNum/processPercentage)*100, '.2f')) + '% finished\t', end='')
            minute = int(((processPercentage - processNum)/(5/(timeEnd - timeStart)))/60)
            second = int(((processPercentage - processNum)/(5/(timeEnd - timeStart))) % 60)
            print('estimated time left: ' + str(minute) + ' min ' + str(second) + 's')
            timeStart = time.perf_counter()


