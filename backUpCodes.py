import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list


# preparer path
path_24 = '24'
subroot = os.getcwd() + '/data/Algo'
image = 'image0'
databasePATH_root = '/home/liutie/Documents/outputDatabase0'


# load ground truth image
PETimage_shape_str = read_input_dim(subroot + '/Data/database_v2/' + image + '/' + image + '.hdr')
PETimage_shape = input_dim_str_to_list(PETimage_shape_str)
image_path = subroot + '/Data/database_v2/' + image + '/' + image + '.raw'
image_gt = fijii_np(image_path, shape=PETimage_shape, type='<f')

# select only phantom ROI, not whole reconstructed image
path_phantom_ROI = subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw'
my_file = Path(path_phantom_ROI)
if my_file.is_file():
    phantom_ROI = fijii_np(path_phantom_ROI, shape=PETimage_shape, type='<f')
else:
    phantom_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw',
                           shape=PETimage_shape, type='<f')
'''
plt.figure()
plt.imshow(phantom_ROI)
plt.show()
'''

dataFolderPath = '1000+admm+CT,random+skip=...*2+lr=...*13'
saveImagePath = databasePATH_root + '/' + dataFolderPath + '/figures'
saveIMAGE = True  # True ot False
iteration = '1000'

replicates = '1'
# initialisation variables for the loop
INPUT = 'CT'  # 'random' or 'CT'
skip = '0'  # '0' or '3'
start = 1
end = 1000
step = 10
NEW_normalization = True

alpha = 0.084
inner_iter = 50
# lrs = [1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
lrs = [0.0001, 0.0004, 0.0007,
       0.001, 0.004, 0.007,
       0.01, 0.04, 0.07,
       0.1, 0.4, 0.7,
       1]
epochs = range(start-1, end, step)

# start the loop
for i in range(len(lrs)):
    # initialisation the parameters needed
    IR_bkg_recon = []
    MSE_recon = []
    CRC_hot_recon = []
    MA_cold_recon = []
    # cost_function = []

    for j in range(len(epochs)):
        filename = 'out_DIP_post_reco_epoch=' + str(epochs[j]) + 'config_rho=0_lr=' + str(lrs[i]) + '_sub_i=' \
                   + iteration + '_opti_=LBFGS_skip_=' + skip + '_scali=normalization_input=' + INPUT \
                   + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) + '_mlem_=False.img'
        path_img = databasePATH_root + '/' + dataFolderPath + '/24' + str(replicates) + '/' + filename

        f = fijii_np(path_img, shape=PETimage_shape)
        f_metric = find_nan(f)

        bkg_ROI = phantom_ROI
        bkg_ROI_act = f_metric[bkg_ROI == 1]
        # IR
        IR_bkg_recon.append(np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))

        # MSE
        MSE_recon.append(np.mean((image_gt * phantom_ROI - f_metric * phantom_ROI) ** 2))

        # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw',
                           shape=PETimage_shape)
        hot_ROI_act = f_metric[hot_ROI == 1]

        # CRC hot
        CRC_hot_recon.append(np.mean(hot_ROI_act) / 400.)

        cold_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw',
                            shape=PETimage_shape)
        cold_ROI_act = f_metric[cold_ROI == 1]

        # MA cold
        MA_cold_recon.append(np.mean(cold_ROI_act))

    # Normalisation
    IR_bkg_norm = np.array(IR_bkg_recon)
    if NEW_normalization:
        IR_bkg_norm = IR_bkg_norm / np.amin(np.abs(IR_bkg_norm))
    else:
        IR_bkg_norm = IR_bkg_norm / np.amax(np.abs(IR_bkg_norm))

    MSE_norm = np.array(MSE_recon)
    if NEW_normalization:
        MSE_norm = MSE_norm / 1e6
    else:
        MSE_norm = MSE_norm / np.amax(np.abs(MSE_norm))

    CRC_hot_norm = np.array(CRC_hot_recon)
    CRC_hot_norm = CRC_hot_norm - 1
    if NEW_normalization:
        CRC_hot_norm = CRC_hot_norm / np.amin(np.abs(CRC_hot_norm))
    else:
        CRC_hot_norm = CRC_hot_norm / np.amax(np.abs(CRC_hot_norm))

    MA_cold_norm = np.array(MA_cold_recon)
    if NEW_normalization:
        MA_cold_norm = MA_cold_norm / np.amin(np.abs(MA_cold_norm))
    else:
        MA_cold_norm = MA_cold_norm / np.amax(np.abs(MA_cold_norm))

    # calculate cost function for one learning rate
    cost_function = IR_bkg_norm ** 2 + MSE_norm ** 2 + CRC_hot_norm ** 2 + MA_cold_norm ** 2

    # plot all the curves
    # plot IR
    plt.figure(1)
    # plt.plot(epochs, IR_bkg_norm, '-x', label=str(lrs[i]))
    if i % 10 == i:
        plt.plot(epochs, IR_bkg_recon, label=str(lrs[i]))
    else:
        plt.plot(epochs, IR_bkg_recon, '.-', label=str(lrs[i]))
    plt.title('Image Roughness (best 0)')
    plt.xlabel('epoch')
    plt.legend()
    if saveIMAGE:
        plt.savefig(saveImagePath + '/IR-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

    # plot MSE
    plt.figure(2)
    if i % 10 == i:
        plt.plot(epochs, MSE_recon, label=str(lrs[i]))
    else:
        plt.plot(epochs, MSE_recon, '.-', label=str(lrs[i]))
    plt.title('MSE (best 0)')
    plt.xlabel('epoch')
    plt.legend()
    if saveIMAGE:
        plt.savefig(saveImagePath + '/MSE-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

    # plot CRC hot
    plt.figure(3)
    if i % 10 == i:
        plt.plot(epochs, CRC_hot_recon, label=str(lrs[i]))
    else:
        plt.plot(epochs, CRC_hot_recon, '.-', label=str(lrs[i]))
    plt.title('CRC hot (best 1)')
    plt.xlabel('epoch')
    plt.legend()
    if saveIMAGE:
        plt.savefig(saveImagePath + '/CRC hot-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

    # plot MA cold
    plt.figure(4)
    if i % 10 == i:
        plt.plot(epochs, MA_cold_recon, label=str(lrs[i]))
    else:
        plt.plot(epochs, MA_cold_recon, '.-', label=str(lrs[i]))
    plt.title('MA cold (best 0)')
    plt.xlabel('epoch')
    plt.legend()
    if saveIMAGE:
        plt.savefig(saveImagePath + '/MA cold-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

    # plot cost
    plt.figure(5)
    if i % 10 == i:
        plt.plot(epochs, cost_function, label=str(lrs[i]))
    else:
        plt.plot(epochs, cost_function, '.-', label=str(lrs[i]))
    plt.title('Normalised cost (best 0)')
    plt.xlabel('epoch')
    plt.legend()
    if saveIMAGE:
        plt.savefig(saveImagePath + '/cost-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

plt.show()


###########################
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list

# preparer path
path_24 = '24'
subroot = os.getcwd() + '/data/Algo'
image = 'image0'
databasePATH_root = '/home/liutie/Documents/outputDatabase0'

# load ground truth image
PETimage_shape_str = read_input_dim(subroot + '/Data/database_v2/' + image + '/' + image + '.hdr')
PETimage_shape = input_dim_str_to_list(PETimage_shape_str)
image_path = subroot + '/Data/database_v2/' + image + '/' + image + '.raw'
image_gt = fijii_np(image_path, shape=PETimage_shape, type='<f')

# select only phantom ROI, not whole reconstructed image
path_phantom_ROI = subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw'
my_file = Path(path_phantom_ROI)
if my_file.is_file():
    phantom_ROI = fijii_np(path_phantom_ROI, shape=PETimage_shape, type='<f')
else:
    phantom_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + 'background_mask' + image[-1] + '.raw',
                           shape=PETimage_shape, type='<f')
'''
plt.figure()
plt.imshow(phantom_ROI)
plt.show()
'''

dataFolderPath = '1000+admm+CT,random+skip=...*2+lr=...*13'
saveImagePath = databasePATH_root + '/' + dataFolderPath + '/figures'
saveIMAGE = True  # True ot False
iteration = 1000

replicates = '1'
# initialisation variables for the loop
INPUT = 'CT'  # 'random' or 'CT'
skip = '0'  # '0' or '3'
start = 1
end = 1000
step = 1

alpha = 0.084
inner_iter = 50
lrs = [0.0001, 0.0004, 0.0007,
       0.001, 0.004, 0.007,
       0.01, 0.04, 0.07,
       0.1, 0.4, 0.7,
       1]
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
epochs = range(start - 1, end, step)


def get_recons():
    # initialisation the parameters needed
    IR_bkg_recon = []
    MSE_recon = []
    CRC_hot_recon = []
    MA_cold_recon = []

    # start the loop
    for i in range(len(lrs)):
        for j in range(len(epochs)):
            filename = 'out_DIP_post_reco_epoch=' + str(epochs[j]) + 'config_rho=0_lr=' + str(lrs[i]) + '_sub_i=' \
                       + str(iteration) + '_opti_=LBFGS_skip_=' + skip + '_scali=normalization_input=' + INPUT \
                       + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) + '_mlem_=False.img'
            path_img = databasePATH_root + '/' + dataFolderPath + '/24' + str(replicates) + '/' + filename

            f = fijii_np(path_img, shape=PETimage_shape)
            f_metric = find_nan(f)

            bkg_ROI = phantom_ROI
            bkg_ROI_act = f_metric[bkg_ROI == 1]
            # IR
            IR_bkg_recon.append(np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))

            # MSE
            MSE_recon.append(np.mean((image_gt * phantom_ROI - f_metric * phantom_ROI) ** 2))

            # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
            hot_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw',
                               shape=PETimage_shape)
            hot_ROI_act = f_metric[hot_ROI == 1]

            # CRC hot
            CRC_hot_recon.append(np.mean(hot_ROI_act) / 400.)

            cold_ROI = fijii_np(subroot + '/Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw',
                                shape=PETimage_shape)
            cold_ROI_act = f_metric[cold_ROI == 1]

            # MA cold
            MA_cold_recon.append(np.mean(cold_ROI_act))
    return IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon


def normalisation_DIP(IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon, NEW_normalization=False):
    # Normalisation
    IR_bkg_norm = np.array(IR_bkg_recon)
    if NEW_normalization:
        IR_bkg_norm = IR_bkg_norm / np.amin(np.abs(IR_bkg_norm))
    else:
        IR_bkg_norm = IR_bkg_norm / np.amax(np.abs(IR_bkg_norm))

    MSE_norm = np.array(MSE_recon)
    if NEW_normalization:
        MSE_norm = MSE_norm / 1e6
    else:
        MSE_norm = MSE_norm / np.amax(np.abs(MSE_norm))

    CRC_hot_norm = np.array(CRC_hot_recon)
    CRC_hot_norm = CRC_hot_norm - 1
    if NEW_normalization:
        CRC_hot_norm = CRC_hot_norm / np.amin(np.abs(CRC_hot_norm))
    else:
        CRC_hot_norm = CRC_hot_norm / np.amax(np.abs(CRC_hot_norm))

    MA_cold_norm = np.array(MA_cold_recon)
    if NEW_normalization:
        MA_cold_norm = MA_cold_norm / np.amin(np.abs(MA_cold_norm))
    else:
        MA_cold_norm = MA_cold_norm / np.amax(np.abs(MA_cold_norm))

    # calculate cost function for one learning rate
    # cost_function = IR_bkg_norm ** 2 + MSE_norm ** 2 + CRC_hot_norm ** 2 + MA_cold_norm ** 2
    cost_function = np.log(np.abs(IR_bkg_norm)) + np.log(np.abs(MSE_norm)) + np.log(np.abs(CRC_hot_norm)) + np.log(
        np.abs(MA_cold_norm))

    return cost_function


def plotAndSave_figures(IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon, cost_function):
    # plot all the curves
    for i in range(len(lrs)):
        iter_to_show = len(epochs)
        # plot IR
        plt.figure(1)
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
            plt.savefig(saveImagePath + '/IR-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(
                start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

        # plot MSE
        plt.figure(2)
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
            plt.savefig(saveImagePath + '/MSE-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(
                start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

        # plot CRC hot
        plt.figure(3)
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
            plt.savefig(saveImagePath + '/CRC hot-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(
                start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

        # plot MA cold
        plt.figure(4)
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
            plt.savefig(saveImagePath + '/MA cold-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(
                start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

        # plot cost
        plt.figure(5)
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
            plt.savefig(saveImagePath + '/cost-' + INPUT + '-' + 'skip=' + skip + '-' + str(replicates) + '-(' + str(
                start) + ',' + str(end) + ',' + str(step) + ')' + '.png')

    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
from cProfile import run
import os
from ray import tune

# Local files to import
from iNestedADMM import iNestedADMM
from iComparison import iComparison
from iPostReconstruction import iPostReconstruction
from iResults import iResults
from iResultsReplicates import iResultsReplicates
from iResultsAlreadyComputed import iResultsAlreadyComputed


def specialTask(method_special='nested',
                replicates_special=1,
                sub_iter_special=1000,
                opti_special='LBFGS',
                skip_special=0,
                scaling_special='normalization',
                input_special='CT',
                inner_special=50,
                outer_special=70,
                alpha_special=0.084,
                rho_special=0,
                DIP_special=False,):
    # Configuration dictionnary for general parameters (not hyperparameters)
    fixed_config = {
        "image": tune.grid_search(['image0']),  # Image from database
        "net": tune.grid_search(['DIP']),  # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "method": tune.grid_search([method_special]),
        # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit": tune.grid_search(['CPU']),  # CPU or GPU
        "nb_threads": tune.grid_search([64]),  # Number of desired threads. 0 means all the available threads
        "FLTNB": tune.grid_search(['double']),
        # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
        "debug": False,  # Debug mode = run without raytune and with one iteration
        "max_iter": tune.grid_search([max_iter]),
        # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested
        "nb_subsets": tune.grid_search([28]),
        # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
        "finetuning": tune.grid_search(['last']),
        "experiment": tune.grid_search([24]),
        "image_init_path_without_extension": tune.grid_search(['1_im_value_cropped']),
        # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        # "f_init" : tune.grid_search(['1_im_value_cropped']),
        "penalty": tune.grid_search(['MRF']),  # Penalty used in CASToR for PLL algorithms
        "post_smoothing": tune.grid_search([False]),  # Post smoothing by CASToR after reconstruction
        "replicates": tune.grid_search(list(range(1, replicates_special + 1))),
        # List of desired replicates. list(range(1,n+1)) means n replicates
        "average_replicates": tune.grid_search([False]),
        # List of desired replicates. list(range(1,n+1)) means n replicates
    }
    # Configuration dictionnary for hyperparameters to tune
    hyperparameters_config = {
        "rho": tune.grid_search([rho_special]),
        # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)

        ## network hyperparameters
        "lr": tune.grid_search([lr_special]),  # Learning rate in network optimization
        # "lr" : tune.grid_search([0.001,0.041,0.01]), # Learning rate in network optimization
        "sub_iter_DIP": tune.grid_search([sub_iter_special]),  # Number of epochs in network optimization
        "opti_DIP": tune.grid_search([opti_special]),  # Optimization algorithm in neural network training (Adam, LBFGS)
        "skip_connections": tune.grid_search([skip_special]),  # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling": tune.grid_search([scaling_special]),
        # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        "input": tune.grid_search([input_special]),  # Neural network input (random or CT)
        # "input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
        "d_DD": tune.grid_search([4]),
        # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD": tune.grid_search([32]),  # k for Deep Decoder

        ## ADMMLim - OPTITR hyperparameters
        "sub_iter_PLL": tune.grid_search([inner_special]),
        # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
        "nb_iter_second_admm": tune.grid_search([outer_special]),  # Number outer iterations in ADMMLim
        "alpha": tune.grid_search([alpha_special]),  # alpha (penalty parameter) in ADMMLim

        ## hyperparameters from CASToR algorithms
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence": tune.grid_search([True]),
        # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML hyperparameters
        "A_AML": tune.grid_search([-100]),  # AML lower bound A
        # NNEPPS post processing
        "NNEPPS": tune.grid_search([False]),  # NNEPPS post-processing. True or False
    }

    # Merge 2 dictionaries
    split_config = {
        "hyperparameters": list(hyperparameters_config.keys())
    }
    config = {**fixed_config, **hyperparameters_config, **split_config}

    root = os.getcwd()

    '''
    # Gong reconstruction
    if (config["method"]["grid_search"][0] == 'Gong'):
        #config = np.load(root + 'config_Gong.npy',allow_pickle='TRUE').item()
        from Gong_configuration import config_func
        config = config_func()
    '''

    # Choose task to do (move this after raytune !!!)
    if (config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested'):
        task = 'full_reco_with_network'

    elif (config["method"]["grid_search"][0] == 'ADMMLim' or config["method"]["grid_search"][0] == 'MLEM' or
          config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML'):
        task = 'castor_reco'

    # task = 'full_reco_with_network' # Run Gong or nested ADMM
    # task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    if DIP_special:
        task = 'post_reco'  # Run network denoising after a given reconstructed image im_corrupt
    # task = 'show_results'
    # task = 'show_results_replicates'
    # task = 'show_metrics_results_already_computed'

    if (task == 'full_reco_with_network'):  # Run Gong or nested ADMM
        classTask = iNestedADMM(hyperparameters_config)
    elif (task == 'castor_reco'):  # Run CASToR reconstruction with given optimizer
        classTask = iComparison(config)
    elif (task == 'post_reco'):  # Run network denoising after a given reconstructed image im_corrupt
        classTask = iPostReconstruction(config)
    elif (task == 'show_results'):  # Show already computed results over iterations
        classTask = iResults(config)
    elif (task == 'show_results_replicates'):  # Show already computed results averaging over replicates
        classTask = iResultsReplicates(config)
    elif (task == 'show_metrics_results_already_computed'):  # Show already computed results averaging over replicates
        classTask = iResultsAlreadyComputed(config)

    # Incompatible parameters (should be written in vGeneral I think)
    if (config["method"]["grid_search"][0] == 'ADMMLim' and config["rho"]["grid_search"][0] != 0):
        raise ValueError("ADMMLim must be launched with rho = 0 for now")
    elif (config["method"]["grid_search"][0] == 'nested' and config["rho"]["grid_search"][
        0] == 0 and task == "castor_reco"):
        raise ValueError("nested must be launched with rho > 0")
    elif (config["method"]["grid_search"][0] == 'ADMMLim' and task == "post_reco"):
        raise ValueError("ADMMLim cannot be launched in post_reco mode. Please comment this line.")

    # '''
    for method in config["method"]['grid_search']:
        os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')

    # Launch task
    classTask.runRayTune(config, root, task)
    # '''

    from csv import reader as reader_csv
    import numpy as np
    import matplotlib.pyplot as plt

    for ROI in ['hot', 'cold']:

        suffixes_legend = []

        if classTask.debug:
            method_list = [config["method"]]
        else:
            method_list = config["method"]['grid_search']
        for method in method_list:
            print("method", method)
            suffixes = []

            PSNR_recon = []
            PSNR_norm_recon = []
            MSE_recon = []
            MA_cold_recon = []
            AR_hot_recon = []
            AR_bkg_recon = []
            IR_bkg_recon = []

            if ROI == 'hot':
                metrics = AR_hot_recon
            else:
                metrics = MA_cold_recon

            with open(root + '/data/Algo' + '/suffixes_for_last_run_' + method + '.txt') as f:
                suffixes.append(f.readlines())

            print("suffixes = ", suffixes)
            # Load metrics from last runs to merge them in one figure

            for suffix in suffixes[0]:
                metrics_file = root + '/data/Algo' + '/metrics/' + method + '/' + suffix.rstrip(
                    "\n") + '/' + 'metrics.csv'
                with open(metrics_file, 'r') as myfile:
                    spamreader = reader_csv(myfile, delimiter=';')
                    rows_csv = list(spamreader)
                    rows_csv[0] = [float(i) for i in rows_csv[0]]
                    rows_csv[1] = [float(i) for i in rows_csv[1]]
                    rows_csv[2] = [float(i) for i in rows_csv[2]]
                    rows_csv[3] = [float(i) for i in rows_csv[3]]
                    rows_csv[4] = [float(i) for i in rows_csv[4]]
                    rows_csv[5] = [float(i) for i in rows_csv[5]]
                    rows_csv[6] = [float(i) for i in rows_csv[6]]

                    PSNR_recon.append(np.array(rows_csv[0]))
                    PSNR_norm_recon.append(np.array(rows_csv[1]))
                    MSE_recon.append(np.array(rows_csv[2]))
                    MA_cold_recon.append(np.array(rows_csv[3]))
                    AR_hot_recon.append(np.array(rows_csv[4]))
                    AR_bkg_recon.append(np.array(rows_csv[5]))
                    IR_bkg_recon.append(np.array(rows_csv[6]))

                    '''
                    print(PSNR_recon)
                    print(PSNR_norm_recon)
                    print(MSE_recon)
                    print(MA_cold_recon)
                    print(AR_hot_recon)
                    print(AR_bkg_recon)
                    print(IR_bkg_recon)
                    '''

            plt.figure()
            for run_id in range(len(PSNR_recon)):
                plt.plot(IR_bkg_recon[run_id], metrics[run_id], '-o')

            plt.xlabel('IR')
            if ROI == 'hot':
                plt.ylabel('AR')
            elif ROI == 'cold':
                plt.ylabel('MA')

            for i in range(len(suffixes[0])):

                l = suffixes[0][i].replace('=', '_')
                l = l.replace('\n', '_')
                l = l.split('_')
                legend = ''
                for p in range(len(l)):
                    if l[p] == "AML":
                        legend += "A : " + l[p + 1] + ' / '
                    if l[p] == "NNEPP":
                        legend += "NNEPPS : " + l[p + 1]
                suffixes_legend.append(legend)
        plt.legend(suffixes_legend)

        # Saving this figure locally
        if ROI == 'hot':
            plt.savefig(
                root + '/data/Algo/' + 'debug/' * classTask.debug + 'metrics/' + 'AR in ' + ROI + ' region vs IR in background' + '.png')
        elif ROI == 'cold':
            plt.savefig(
                root + '/data/Algo/' + 'debug/' * classTask.debug + 'metrics/' + 'MA in ' + ROI + ' region vs IR in background' + '.png')
        from textwrap import wrap
        wrapped_title = "\n".join(wrap(suffix, 50))
        plt.title(wrapped_title, fontsize=12)