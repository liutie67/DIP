import math
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import Tuners
from show_functions import getGT, getDataFolderPath, fijii_np, getShape, getPhantomROI, mkdir

'''
Separated realization of WMV strategy, after simple DIP.
Hardcoded path to loop on the images from DIP.

pseudo codes of this file:
    1. initialise all the parameters used in the hardcoded path.
    2. loop on learning rates.
        2.1 loop on images form DIP and implementation of WMV algorithm
        2.2 plot window moving variance
        2.3 plot MSE
        2.4 plot PSNR
        2.5 plot SSIM
        2.6 plot all the curves together
'''

# 1. initialise all the parameters used in the hardcoded path.
databaseNum = 'D'  # choose the database number
dataFolderPath = '2022-08-03+10-49-43+ADMMadpATi1o100*100rep1reps=5+lr=lr4+iter1000+skip0+inputCT+optiAdam+scalingstandardization+t128+1415s'  # choose the folder name
replicate = 1  #  set the replicate number


additionalTitle = 'ADMMadpATi1o100*100rep1'  # additional title of the combined figures

opti = 'Adam'
skip = 0
scaling = 'standardization'
INPUT = 'CT'
'''
inner_iter = 50
alpha = 0.084
'''
inner_iter = 10
alpha = 0.005

sub_iter = 1000

lrs = Tuners.lrs4
# lrs = [0.007]
SHOW = (len(lrs) == 1)
SHOW = False

# set window size and patience number
windowSize = 50
patienceNum = 200

# 2. loop on learning rates.
processPercentage = sub_iter * len(lrs)  # variable used to track the process
processNum = 0                           # variable used to track the process
for lr in lrs:
    VARmin = math.inf
    VARs = []
    MSEs = []
    PSNRs = []
    SSIMs = []

    queueQ = []

    x_gt = getGT()
    stagnate = 0
    success = False
    epochStar = -1

    # 2.1 loop on images form DIP and implementation of WMV algorithm
    for epoch in range(sub_iter):
        '''
        # old version before MERGED
        filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'
        '''
        filename = 'out_DIP_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                   + str(sub_iter) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                   + INPUT + '_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                   + '_mlem_=False.img'

        path_img = getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate) + '/nested/Block2/out_cnn' + '/24/' + filename

        x_out = fijii_np(path_img, shape=getShape()) * getPhantomROI()
        MSE = np.mean((x_gt * getPhantomROI() - x_out) ** 2)
        MSEs.append(MSE)
        PSNRs.append(psnr(x_gt * getPhantomROI(), x_out, data_range=x_gt.max() - x_gt.min()))
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
                epochStar = epoch  # detection point
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

    # 2.2 plot window moving variance
    plt.figure(1)
    var_x = np.arange(windowSize, windowSize + len(VARs))  # define x axis of WMV
    plt.plot(var_x, VARs, 'r')
    plt.title('Window Moving Variance,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')  # plot a vertical line at epochStar(detection point)
    plt.xticks([epochStar, 0, sub_iter-1], [epochStar, 0, sub_iter-1], color='green')
    plt.axhline(y=np.min(VARs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate)
                      + '/VARs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+VARs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    # 2.3 plot MSE
    plt.figure(2)
    plt.plot(MSEs, 'y')
    plt.title('MSE,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter-1], [epochStar, 0, sub_iter-1], color='green')
    plt.axhline(y=np.min(MSEs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate)
                      + '/MSEs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+MSEs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    # 2.4 plot PSNR
    plt.figure(3)
    plt.plot(PSNRs)
    plt.title('PSNR,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter - 1], [epochStar, 0, sub_iter - 1], color='green')
    plt.axhline(y=np.max(PSNRs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate)
                      + '/PSNRs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+PSNRs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    # 2.5 plot SSIM
    plt.figure(4)
    plt.plot(SSIMs, c='orange')
    plt.title('SSIM,epoch*=' + str(epochStar) + ',lr=' + str(lr))
    plt.axvline(epochStar, c='g')
    plt.xticks([epochStar, 0, sub_iter - 1], [epochStar, 0, sub_iter - 1], color='green')
    plt.axhline(y=np.max(SSIMs), c="black", linewidth=0.5)
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate)
                      + '/SSIMs' + '/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+SSIMs-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

    # 2.6 plot all the curves together
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.8, left=0.1, bottom=0.12)
    ax2 = ax1.twinx()  # creat other y-axis for different scale
    ax3 = ax1.twinx()  # creat other y-axis for different scale
    ax4 = ax1.twinx()  # creat other y-axis for different scale
    ax2.spines.right.set_position(("axes", 1.18))
    p4, = ax4.plot(MSEs, "y", label="MSE")
    p1, = ax1.plot(PSNRs, label="PSNR")
    p2, = ax2.plot(var_x, VARs, "r", label="WMV")
    p3, = ax3.plot(SSIMs, "orange", label="SSIM")
    ax1.set_xlim(0, sub_iter-1)
    plt.title(additionalTitle + ' lr=' + str(lr))
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
    plt.savefig(mkdir(getDataFolderPath(databaseNum, dataFolderPath) + '/replicate_' + str(replicate)
                      + '/combined/w' + str(windowSize) + 'p' + str(patienceNum)) + '/' + str(
        lrs.index(lr)) + '-lr' + str(lr) + '+combined-w' + str(windowSize) + 'p' + str(patienceNum) + '.png')
    if not SHOW:
        plt.clf()

if SHOW:
    plt.show()
