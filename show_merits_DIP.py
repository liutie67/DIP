import os
import numpy as np
import matplotlib.pyplot as plt


from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list, getGT, getPhantomROI, getShape
from show_functions import mkdir


# preparer path
path_24 = '24'
subroot = os.getcwd() + '/data/Algo'
image = 'image0'

databasePATH_root = '/home/liutie/Documents/outputDatabase15'


image_gt = getGT()
phantom_ROI = getPhantomROI()
'''
plt.figure()
plt.imshow(phantom_ROI)
plt.show()
'''

# dataFolderPath = '1000+admm+CT,random+skip=...*2+lr=...*13'
# saveImagePath = databasePATH_root + '/' + dataFolderPath + '/figures'
# iteration = 1000

# replicates = '1'
# initialisation variables for the loop
# INPUT = 'CT'  # 'random' or 'CT'
# skip = '0'  # '0' or '3'
# start = 1
# end = 1000
# step = 1

# alpha = 0.084
# inner_iter = 50
'''
lrs = [0.0001, 0.0004, 0.0007,
       0.001, 0.004, 0.007,
       0.01, 0.04, 0.07,
       0.1, 0.4, 0.7,
       1]
'''
'''
lrs = [0.0001, 0.0004, 0.0007,
       0.001, 0.004, 0.007,
       0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
       0.1, 0.2, 0.3, 0.4, 0.7, 0.5, 0.6, 0.8, 0.9,
       1, 1.4, 1.7, 2]
'''
'''
lrs = [0.0001, 0.0004, 0.0007,
       0.001, 0.004, 0.007,
       0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
       0.1, 0.2, 0.3, 0.4, 0.7, 0.5, 0.6, 0.8, 0.9,
       1, 1.4, 1.7]
'''

def computeThose4(f):
    # f = fijii_np(f, shape=getShape())
    f_metric = find_nan(f)
    bkg_ROI = getPhantomROI()
    bkg_ROI_act = f_metric[bkg_ROI == 1]
    # IR
    if np.mean(bkg_ROI_act) != 0:
        IR_bkg_recon = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
    else:
        IR_bkg_recon = 1e10

    # MSE
    MSE_recon = np.mean((image_gt * phantom_ROI - f_metric * phantom_ROI) ** 2)

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


def get_recons(dataFolderPath,
               replicates,
               INPUT,
               skip,
               lrs,
               alpha=0.084,
               inner_iter=50,
               epoch_star=1,
               epoch_end=1000,
               epoch_step=1,
               iteration=1000,
               opti='LBFGS',
               scaling='normalization'
               ):
    # initialisation the parameters needed
    IR_bkg_recon = []
    MSE_recon = []
    CRC_hot_recon = []
    MA_cold_recon = []

    epochs = range(epoch_star - 1, epoch_end, epoch_step)
    # start the loop
    for i in range(len(lrs)):
        for j in range(len(epochs)):
            filename = 'out_DIP_post_reco_epoch=' + str(epochs[j]) + 'config_rho=0_lr=' + str(lrs[i]) + '_sub_i=' \
                       + str(iteration) + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' \
                       + INPUT + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) \
                       + '_mlem_=False.img'
            path_img = databasePATH_root + '/' + dataFolderPath + '/replicate_1/nested/Block2/out_cnn' + '/24' + str(replicates) + '/' + filename

            f = fijii_np(path_img, shape=getShape())
            IR, MSE, CRC, MA = computeThose4(f)
            IR_bkg_recon.append(IR)
            MSE_recon.append(MSE)
            CRC_hot_recon.append(CRC)
            MA_cold_recon.append(MA)

    return IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon

def newNormalization(NNorm, max):
    # print(NNorm)
    # NNorm = NNorm.flatten()
    for i in range(len(NNorm)):
        if NNorm[i] > max:
            NNorm[i] = max
        elif NNorm[i] < -max:
            NNorm[i] = -max
    NNorm = NNorm / np.abs(np.amax(NNorm) - np.amin(NNorm))
    return NNorm

def normalisation_DIP(IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon,
                      NEW_normalization=False,
                      maxIR=1,
                      maxMSE=1,
                      maxCRC=1,
                      maxMA=1):
    # Normalisation
    IR_bkg_norm = np.array(IR_bkg_recon)
    if NEW_normalization:
        # IR_bkg_norm = IR_bkg_norm / maxIR
        IR_bkg_norm = newNormalization(IR_bkg_norm, maxIR)
    else:
        # IR_bkg_norm = IR_bkg_norm / np.amax(np.abs(IR_bkg_norm))
        IR_bkg_norm = IR_bkg_norm / np.abs(np.amax(IR_bkg_norm) - np.amin(IR_bkg_norm))

    MSE_norm = np.array(MSE_recon)
    if NEW_normalization:
        # MSE_norm = MSE_norm / maxMSE
        MSE_norm = newNormalization(MSE_norm, maxMSE)
    else:
        # MSE_norm = MSE_norm / np.amax(np.abs(MSE_norm))
        MSE_norm = MSE_norm / np.abs(np.amax(MSE_norm) - np.amin(MSE_norm))

    CRC_hot_norm = np.array(CRC_hot_recon)
    CRC_hot_norm = CRC_hot_norm - 1
    if NEW_normalization:
        # CRC_hot_norm = CRC_hot_norm / maxCRC
        CRC_hot_norm = newNormalization(CRC_hot_norm, maxCRC)
    else:
        # CRC_hot_norm = CRC_hot_norm / np.amax(np.abs(CRC_hot_norm))
        CRC_hot_norm = CRC_hot_norm / np.abs(np.amax(CRC_hot_norm) - np.amin(CRC_hot_norm))

    MA_cold_norm = np.array(MA_cold_recon)
    if NEW_normalization:
        # MA_cold_norm = MA_cold_norm / maxMA
        MA_cold_norm = newNormalization(MA_cold_norm, maxMA)
    else:
        # MA_cold_norm = MA_cold_norm / np.amax(np.abs(MA_cold_norm))
        MA_cold_norm = MA_cold_norm / np.abs(np.amax(MA_cold_norm) - np.amin(MA_cold_norm))

    # calculate cost function for one learning rate
    # cost_function = IR_bkg_norm ** 2 + MSE_norm ** 2 + CRC_hot_norm ** 2 + MA_cold_norm ** 2
    '''
    cost_function = np.log(np.abs(IR_bkg_norm)) + np.log(np.abs(MSE_norm)) + np.log(np.abs(CRC_hot_norm)) + np.log(
        np.abs(MA_cold_norm))
    '''
    cost_function = np.log(np.abs(IR_bkg_norm)**2 + np.abs(MSE_norm)**2 + np.abs(CRC_hot_norm)**2 + np.abs(MA_cold_norm)**2)

    return cost_function


def plotAndSave_figures(replicates,
                        INPUT,
                        skip,
                        lrs,
                        IR_tag=True,
                        MSE_tag=True,
                        CRC_tag=True,
                        MA_tag=True,
                        cost_tag=True,
                        IR_bkg_recon=None,
                        MSE_recon=None,
                        CRC_hot_recon=None,
                        MA_cold_recon=None,
                        cost_function=None,
                        saveIMAGE=False,
                        dataFolderPath=None,
                        epoch_star=1,
                        epoch_end=1000,
                        epoch_step=1,
                        showFigures=False,
                        additional=0):
    epochs = range(epoch_star - 1, epoch_end, epoch_step)

    if saveIMAGE:
        if dataFolderPath == None:
            return print('No Path for saving images')
        saveImagePath = databasePATH_root + '/' + dataFolderPath + '/figures'
        mkdir(saveImagePath)

    # plot all the curves
    for i in range(len(lrs)):
        iter_to_show = len(epochs)

        if IR_tag:
            # plot IR
            plt.figure(1+additional*5)
            if i < 10:
                plt.plot(epochs, IR_bkg_recon[i * iter_to_show:i * iter_to_show + iter_to_show], label=str(lrs[i]))
            elif 10 <= i < 20:
                plt.plot(epochs, IR_bkg_recon[i * iter_to_show:i * iter_to_show + iter_to_show], '.-', label=str(lrs[i]))
            else:
                plt.plot(epochs, IR_bkg_recon[i * iter_to_show:i * iter_to_show + iter_to_show], 'x-', label=str(lrs[i]))
            plt.title('Image Roughness (best 0)')
            plt.xlabel('epoch')
            plt.legend()
            if saveIMAGE:
                plt.savefig(saveImagePath + '/IR-' + INPUT + '-' + 'skip=' + str(skip) + '-' + str(replicates) + '-(' + str(
                    epoch_star) + ',' + str(epoch_end) + ',' + str(epoch_step) + ')' + '.png')

        if MSE_tag:
            # plot MSE
            plt.figure(2+additional*5)
            if i < 10:
                plt.plot(epochs, MSE_recon[i * iter_to_show:i * iter_to_show + iter_to_show], label=str(lrs[i]))
            elif 10 <= i < 20:
                plt.plot(epochs, MSE_recon[i * iter_to_show:i * iter_to_show + iter_to_show], '.-', label=str(lrs[i]))
            else:
                plt.plot(epochs, MSE_recon[i * iter_to_show:i * iter_to_show + iter_to_show], 'x-', label=str(lrs[i]))
            plt.title('MSE (best 0)')
            plt.xlabel('epoch')
            plt.legend()
            if saveIMAGE:
                plt.savefig(saveImagePath + '/MSE-' + INPUT + '-' + 'skip=' + str(skip) + '-' + str(replicates) + '-(' + str(
                    epoch_star) + ',' + str(epoch_end) + ',' + str(epoch_step) + ')' + '.png')

        if CRC_tag:
            # plot CRC hot
            plt.figure(3+additional*5)
            if i < 10:
                plt.plot(epochs, CRC_hot_recon[i * iter_to_show:i * iter_to_show + iter_to_show], label=str(lrs[i]))
            elif 10 <= i < 20:
                plt.plot(epochs, CRC_hot_recon[i * iter_to_show:i * iter_to_show + iter_to_show], '.-', label=str(lrs[i]))
            else:
                plt.plot(epochs, CRC_hot_recon[i * iter_to_show:i * iter_to_show + iter_to_show], 'x-', label=str(lrs[i]))
            plt.title('CRC hot (best 1)')
            plt.xlabel('epoch')
            plt.legend()
            if saveIMAGE:
                plt.savefig(saveImagePath + '/CRC hot-' + INPUT + '-' + 'skip=' + str(skip) + '-' + str(replicates) + '-(' + str(
                    epoch_star) + ',' + str(epoch_end) + ',' + str(epoch_step) + ')' + '.png')

        if MA_tag:
            # plot MA cold
            plt.figure(4+additional*5)
            if i < 10:
                plt.plot(epochs, MA_cold_recon[i * iter_to_show:i * iter_to_show + iter_to_show], label=str(lrs[i]))
            elif 10 <= i < 20:
                plt.plot(epochs, MA_cold_recon[i * iter_to_show:i * iter_to_show + iter_to_show], '.-', label=str(lrs[i]))
            else:
                plt.plot(epochs, MA_cold_recon[i * iter_to_show:i * iter_to_show + iter_to_show], 'x-', label=str(lrs[i]))
            plt.title('MA cold (best 0)')
            plt.xlabel('epoch')
            plt.legend()
            if saveIMAGE:
                plt.savefig(saveImagePath + '/MA cold-' + INPUT + '-' + 'skip=' + str(skip) + '-' + str(replicates) + '-(' + str(
                    epoch_star) + ',' + str(epoch_end) + ',' + str(epoch_step) + ')' + '.png')

        # plot cost
        if cost_tag:
            plt.figure(5+additional*5)
            if i < 10:
                plt.plot(epochs, cost_function[i * iter_to_show:i * iter_to_show + iter_to_show], label=str(lrs[i]))
            elif 10 <= i < 20:
                plt.plot(epochs, cost_function[i * iter_to_show:i * iter_to_show + iter_to_show], '.-', label=str(lrs[i]))
            else:
                plt.plot(epochs, cost_function[i * iter_to_show:i * iter_to_show + iter_to_show], 'x-', label=str(lrs[i]))
            plt.title('Normalised cost (best 0)')
            plt.xlabel('epoch')
            plt.legend()
            if saveIMAGE:
                plt.savefig(saveImagePath + '/cost-' + INPUT + '-' + 'skip=' + str(skip) + '-' + str(replicates) + '-(' + str(
                    epoch_star) + ',' + str(epoch_end) + ',' + str(epoch_step) + ')' + '.png')
    if showFigures:
        plt.show()
