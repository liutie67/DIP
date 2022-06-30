import math
import numpy as np
import matplotlib.pyplot as plt

from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir


databaseNum = 15
dataFolderPath = '2022-06-30+11-03-32+wx+px+MLEM1000+lr=lrs0+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t64'

lr = 0.01
opti = 'Adam'
skip = 0
scaling = 'standardization'
INPUT = 'CT'

inner_iter = 50
alpha = 0.084
sub_iter = 1000

epoches = range(250,350)
flag = 1
for epoch in epoches:
    filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
               + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
               + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
               + '_mlem_=False.img'
    path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_1/nested/Block2/out_cnn' + '/24/' + filename

    x_out = fijii_np(path_img, shape=getShape())

    plt.figure(1)
    plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
    plt.title('epoch ' + str(epoch))
    if flag:
        plt.colorbar()
        flag = 0
    mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/processImages')
    plt.savefig(getDataFolderPath(databaseNum, dataFolderPath) + '/processImages/epoch' + str(epoch) + '.png')





