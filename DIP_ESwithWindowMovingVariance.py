import math
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir

databaseNum = 'F'
dataFolderPath = '2022-07-26+15-06-43+ADMMi100o100a0.005+lr=lr2+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t128+313s'
additionalTitle = 'ADMMi100o100a0.005'

opti = 'Adam'
skip = 0
scaling = 'standardization'
INPUT = 'CT'

inner_iter = 50
alpha = 0.084

'''
inner_iter = 10
alpha = 0.005
'''
sub_iter = 1000

windowSize = 50
patienceNum = 200

lrs = Tuners.lrs2
# lrs = [0.006]
SHOW = (len(lrs) == 1)

processPercentage = sub_iter * len(lrs)
processNum = 0
for lr in lrs:
    VARmin = math.inf
    VARs = []
    MSEs = []
    PSNRs = []
    SSIMs = []

    # x0 = np.zeros(getShape()).flatten()
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
        '''
        filename = 'out_DIP_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'
        '''
        path_img = getDataFolderPath(databaseNum,
                                     dataFolderPath) + '/replicate_1/nested/Block2/out_cnn' + '/24/' + filename

        x_out = fijii_np(path_img, shape=getShape()) * getPhantomROI()
        MSE = np.mean((x_gt * getPhantomROI() - x_out) ** 2)
        MSEs.append(MSE)
        # PSNR = 10 * np.log((np.amax(np.abs(x_gt * getPhantomROI()))) ** 2 / MSE)
        PSNRs.append(psnr(x_gt * getPhantomROI(), x_out, data_range=x_gt.max() - x_gt.min()))
        #PSNRs.append(PSNR)
        SSIMs.append(ssim(np.squeeze(x_gt * getPhantomROI()), np.squeeze(x_out), data_range=x_gt.max() - x_gt.min()))

        queueQ.append(x_out.flatten())
        if len(queueQ) == windowSize:
            mean = queueQ[0].copy()
            for x in queueQ[1:windowSize]:
                mean += x
            mean = mean / windowSize
            VAR = np.linalg.norm(queueQ[0] - mean) ** 2
            for x in queueQ[1:windowSize]:
                VAR += np.linalg.norm(x - mean) ** 2
            VAR = VAR / windowSize
            if VAR < VARmin and not success:
                VARmin = VAR
                epochStar = epoch
                stagnate = 1
            else:
                stagnate += 1
            if stagnate == patienceNum:
                success = True
            queueQ.pop(0)
            VARs.append(VAR)

        processNum += 1
        if processNum % 50 == 0:
            print('Processing: ' + str(format((processNum / processPercentage) * 100, '.2f')) + '% finished')

    plt.figure(1)
    var_x = np.arange(windowSize, windowSize + len(VARs))
    plt.plot(var_x, VARs, 'r')
    plt.title('Window Moving Variance,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter-1], [epochStar, 0, sub_iter-1], color='green')
    plt.axhline(y=np.min(VARs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                      + '/VARs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+VARs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    plt.figure(2)
    plt.plot(MSEs, 'y')
    plt.title('MSE,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter-1], [epochStar, 0, sub_iter-1], color='green')
    plt.axhline(y=np.min(MSEs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                      + '/MSEs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+MSEs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    plt.figure(3)
    plt.plot(PSNRs)
    plt.title('PSNR,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter - 1], [epochStar, 0, sub_iter - 1], color='green')
    plt.axhline(y=np.max(PSNRs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                      + '/PSNRs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+PSNRs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    plt.figure(4)
    plt.plot(SSIMs, c='orange')
    plt.title('SSIM,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter - 1], [epochStar, 0, sub_iter - 1], color='green')
    plt.axhline(y=np.max(SSIMs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                      + '/SSIMs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+SSIMs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.8, left=0.1, bottom=0.12)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax2.spines.right.set_position(("axes", 1.18))
    p4, = ax4.plot(MSEs, "y", label="MSE")
    p1, = ax1.plot(PSNRs, label="PSNR")
    p2, = ax2.plot(var_x, VARs, "r", label="WMV")
    p3, = ax3.plot(SSIMs, "orange", label="SSIM")
    ax1.set_xlim(0, sub_iter-1)
    plt.title(additionalTitle + ' lr=' + str(lr))
    # ax1.set_xlabel("sub iteration", color='green')
    ax1.set_ylabel("Peak Signal-Noise ratio")
    ax2.set_ylabel("Window-Moving variance")
    ax3.set_ylabel("Structural similarity")
    ax4.yaxis.set_visible(False)
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    tkw = dict(size=3, width=1)
    ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax1.tick_params(axis='x', colors="green", **tkw)
    ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax1.tick_params(axis='x', **tkw)
    ax1.legend(handles=[p1, p3, p2, p4])
    ax1.axvline(epochStar, c='g', linewidth=1, ls='--')
    ax1.axvline(windowSize-1, c='g', linewidth=1, ls=':')
    ax1.axvline(epochStar+patienceNum, c='g', lw=1, ls=':')
    if epochStar+patienceNum > epochStar:
        plt.xticks([epochStar, windowSize-1, epochStar+patienceNum], ['\n' + str(epochStar) + '\nES point', str(windowSize), '+' + str(patienceNum)], color='green')
    else:
        plt.xticks([epochStar, windowSize-1], ['\n' + str(epochStar) + '\nES point', str(windowSize)], color='green')
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath)
                      + '/combined/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+combined-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

if SHOW:
    plt.show()
