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

from Tuners import ADMMoptimizerName

def specialTask(method_special='nested',
                max_iter=30,
                replicates_special=1,
                lr_special=0.04,
                sub_iter_special=100,
                opti_special='LBFGS',
                skip_special=[0],
                scaling_special='normalization',
                input_special=['CT'],
                inner_special=50,
                outer_special=70,
                alpha_special=[0.084],
                rho_special=0,
                DIP_special=False,
                nb_subsets=28,
                mlem_sequence=False,
                threads=128):

    # Configuration dictionnary for general parameters (not hyperparameters)
    fixed_config = {
        "image" : tune.grid_search(['image0']), # Image from database
        "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "method" : tune.grid_search([method_special]), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
        "nb_threads" : tune.grid_search([threads]), # Number of desired threads. 0 means all the available threads
        "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
        "debug" : False, # Debug mode = run without raytune and with one iteration
        "max_iter" : tune.grid_search([max_iter]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested
        "nb_subsets" : tune.grid_search([nb_subsets]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
        "finetuning" : tune.grid_search(['last']),
        "experiment" : tune.grid_search([24]),
        "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        #"f_init" : tune.grid_search(['1_im_value_cropped']),
        "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
        "post_smoothing" : tune.grid_search([False]), # Post smoothing by CASToR after reconstruction
        "replicates" : tune.grid_search(list(range(1,replicates_special+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
        "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    }
    # Configuration dictionnary for hyperparameters to tune
    hyperparameters_config = {
        "rho" : tune.grid_search([rho_special]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)

        ## network hyperparameters
        "lr" : tune.grid_search([lr_special]), # Learning rate in network optimization
        #"lr" : tune.grid_search([0.001,0.041,0.01]), # Learning rate in network optimization
        "sub_iter_DIP" : tune.grid_search([sub_iter_special]), # Number of epochs in network optimization
        "opti_DIP" : tune.grid_search([opti_special]), # Optimization algorithm in neural network training (Adam, LBFGS)
        "skip_connections" : tune.grid_search(skip_special), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling" : tune.grid_search([scaling_special]), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        "input" : tune.grid_search(input_special), # Neural network input (random or CT)
        #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
        "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD" : tune.grid_search([32]), # k for Deep Decoder

        ## ADMMLim - OPTITR hyperparameters
        "sub_iter_PLL" : tune.grid_search([inner_special]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
        "nb_iter_second_admm": tune.grid_search([outer_special]), # Number outer iterations in ADMMLim
        "alpha" : tune.grid_search(alpha_special), # alpha (penalty parameter) in ADMMLim

        ## hyperparameters from CASToR algorithms
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence" : tune.grid_search([mlem_sequence]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML hyperparameters
        "A_AML" : tune.grid_search([-100]), # AML lower bound A
        # NNEPPS post processing
        "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
    }

    # Merge 2 dictionaries
    split_config = {
        "hyperparameters" : list(hyperparameters_config.keys())
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

    elif (config["method"]["grid_search"][0] in ADMMoptimizerName or config["method"]["grid_search"][0] == 'MLEM' or config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML'):
        task = 'castor_reco'

    #task = 'full_reco_with_network' # Run Gong or nested ADMM
    #task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    if DIP_special:
        task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    #task = 'show_results'
    #task = 'show_results_replicates'
    #task = 'show_metrics_results_already_computed'

    if (task == 'full_reco_with_network'): # Run Gong or nested ADMM
        classTask = iNestedADMM(hyperparameters_config)
    elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
        classTask = iComparison(config)
    elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
        classTask = iPostReconstruction(config)
    elif (task == 'show_results'): # Show already computed results over iterations
        classTask = iResults(config)
    elif (task == 'show_results_replicates'): # Show already computed results averaging over replicates
        classTask = iResultsReplicates(config)
    elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
        classTask = iResultsAlreadyComputed(config)


    # Incompatible parameters (should be written in vGeneral I think)
    if (config["method"]["grid_search"][0] == 'ADMMLim' and config["rho"]["grid_search"][0] != 0):
        raise ValueError("ADMMLim must be launched with rho = 0 for now")
    elif (config["method"]["grid_search"][0] == 'nested' and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("nested must be launched with rho > 0")
    elif (config["method"]["grid_search"][0] == 'ADMMLim' and task == "post_reco"):
        raise ValueError("ADMMLim cannot be launched in post_reco mode. Please comment this line.")

    #'''
    for method in config["method"]['grid_search']:
        os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')

    # Launch task
    classTask.runRayTune(config, root, task)
    #'''

    from csv import reader as reader_csv
    import numpy as np
    import matplotlib.pyplot as plt

    for ROI in ['hot','cold']:

        suffixes_legend = []

        if classTask.debug:
            method_list = [config["method"]]
        else:
            method_list = config["method"]['grid_search']
        for method in method_list:
            print("method",method)
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
                metrics_file = root + '/data/Algo' + '/metrics/' + method + '/' + suffix.rstrip("\n") + '/' + 'metrics.csv'
                with open(metrics_file, 'r') as myfile:
                    spamreader = reader_csv(myfile,delimiter=';')
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
                plt.plot(IR_bkg_recon[run_id],metrics[run_id],'-o')

            plt.xlabel('IR')
            if ROI == 'hot':
                plt.ylabel('AR')
            elif ROI == 'cold':
                plt.ylabel('MA')

            for i in range(len(suffixes[0])):

                l = suffixes[0][i].replace('=','_')
                l = l.replace('\n','_')
                l = l.split('_')
                legend = ''
                for p in range(len(l)):
                    if l[p] == "AML":
                        legend += "A : " + l[p+1] + ' / '
                    if l[p] == "NNEPP":
                        legend += "NNEPPS : " + l[p+1]
                suffixes_legend.append(legend)
        plt.legend(suffixes_legend)

        # Saving this figure locally
        if ROI == 'hot':
            plt.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + 'AR in ' + ROI + ' region vs IR in background' + '.png')
        elif ROI == 'cold':
            plt.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + 'MA in ' + ROI + ' region vs IR in background' + '.png')
        from textwrap import wrap
        wrapped_title = "\n".join(wrap(suffix, 50))
        plt.title(wrapped_title,fontsize=12)