import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from show_functions import getDatabasePath, getDataFolderPath, dldir, computeThose4, PLOT
import tuners


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
databasePath = getDatabasePath(10) + '/'
dataFolderPath = 'test17t3-1'
whichADMMoptimizer = tuners.ADMMoptimizerName[3]
NEWoptimzer = False

REPLICATES = True
replicates = 1
ALPHAS = [0.005]

outerIteration = 10
innerIteration = 10

vb = 1
threads = 3

option = 0

OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters']
tuners_tag = OPTION[option]  # tuners = 'alphas' or 'inner_iters' or 'outer_iters' or 'adaptiveRho'
if tuners_tag == 'alphas':
    # outerIteration = 1000
    # innerIteration = 50
    bestAlpha = 0

    alphas = ALPHAS

    inner_iters = range(innerIteration)
    outer_iters = range(outerIteration)
    # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
    tuners = alphas

elif tuners_tag == 'adaptiveRho':
    # outerIteration = 1000
    # innerIteration = 50
    alpha0s = ALPHAS
    duplicate = ''
    if REPLICATES:
        duplicate += '_rep' + str(replicates)
    inner_iters = range(innerIteration)
    outer_iters = range(outerIteration)
    # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
    tuners = alpha0s
    fp = open(databasePath + dataFolderPath + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')

elif tuners_tag == 'inner_iters':
    bestInner = 0

    bestAlpha = 0.005

    innerIteration = 70
    min_innerIteration = 15
    outerIteration = 100

    inner_iters = range(min_innerIteration, innerIteration+1, 5)
    # inner_iters = [65, 70]

    outer_iters = range(outerIteration)
    # dataFolderPath = '/ADMMLim+alpha=' + str(bestAlpha) + '+i...+o' + str(outerIteration)
    tuners = inner_iters

else:
    bestAlpha = 0.005
    bestInner = 65
    outerIteration = 100

    outer_iters = range(1, outerIteration+1)

    # dataFolderPath = '/ADMMLim+alpha=' + str(bestAlpha) + '+i...+o' + str(outerIteration)
    tuners = outer_iters

likelihoods_alpha = []
likelihoods_inner = []
likelihoods_outer = []

for i in range(len(tuners)):
    # tune alpha
    if tuners_tag == 'alphas':
        likelihoods = []
        for outer_iter in range(1, outerIteration+1):
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer + '/Comparison/' \
                                 + whichADMMoptimizer
            else:
                replicatesPath = ''
            logfolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
            logfile_name = '0_' + str(outer_iter) + '.log'
            path_log = logfolder + logfile_name
            theLog = pd.read_table(path_log)
            if vb == 3:
                theLikelihoodRow = theLog.loc[[theLog.shape[0] - 26]]
                theLikelihoodRowArray = np.array(theLikelihoodRow)
                theLikelihoodRowString = theLikelihoodRowArray[0, 0]
                theLikelihoodRowString = theLikelihoodRowString[22:44]
            elif vb == 1:
                theLikelihoodRow = theLog.loc[[theLog.shape[0] - 11]]
                theLikelihoodRowArray = np.array(theLikelihoodRow)
                theLikelihoodRowString = theLikelihoodRowArray[0, 0]
                theLikelihoodRowString = theLikelihoodRowString[22:44]
            else:
                print('************************* verbose(vb) is wrongly set! *************************')
                break
            if theLikelihoodRowString[0] == '-':
                theLikelihoodRowString = '0'
            likelihood = float(theLikelihoodRowString)
            if outer_iter == outerIteration:
                likelihoods_alpha.append(likelihood)
            likelihoods.append(likelihood)

        PLOT(outer_iters, likelihoods, tuners, i, figNum=1,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Likelihood',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        IR_bkgs = []
        MSEs = []
        CRC_hots = []
        MA_colds = []
        for outer_iter in range(1, outerIteration + 1):
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer + '/Comparison/' \
                                 + whichADMMoptimizer
            else:
                replicatesPath = ''
            imageFolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
            imageName = '0_' + str(outer_iter) + '_it' + str(innerIteration) + '.img'
            imagePath = imageFolder + imageName
            IR, MSE, CRC, MA = computeThose4(imagePath)
            IR_bkgs.append(IR)
            MSEs.append(MSE)
            CRC_hots.append(CRC)
            MA_colds.append(MA)

        PLOT(outer_iters, IR_bkgs, tuners, i, figNum=2,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Image Roughness in the background',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, MSEs, tuners, i, figNum=3,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Mean Square Error',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, CRC_hots, tuners, i, figNum=4,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='CRC hot',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, MA_colds, tuners, i, figNum=5,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='MA cold',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

    # adaptive alphas
    elif tuners_tag == 'adaptiveRho':
        adaptiveAlphas = []
        adaptiveTaus = []
        relPrimals = []
        relDuals = []
        xis = []
        for outer_iter in range(1,outerIteration+1):
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer + '/Comparison/' \
                                 + whichADMMoptimizer
            else:
                replicatesPath = ''
            logfolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
            logfile_name = '0_' + str(outer_iter) + '_adaptive.log'
            path_txt = logfolder + logfile_name
            theLog = pd.read_table(path_txt)

            # get adaptive alpha
            theAlphaRow = theLog.loc[[0]]
            theAlphaRowArray = np.array(theAlphaRow)
            theAlphaRowString = theAlphaRowArray[0, 0]
            adaptiveAlpha = float(theAlphaRowString)
            adaptiveAlphas.append(adaptiveAlpha)

            # get adaptive tau
            theTauRow = theLog.loc[[2]]
            theTauRowArray = np.array(theTauRow)
            theTauRowString = theTauRowArray[0, 0]
            adaptiveTau = float(theTauRowString)
            adaptiveTaus.append(adaptiveTau)

            # get relative primal residual
            theRelPrimalRow = theLog.loc[[6]]
            theRelPrimalRowArray = np.array(theRelPrimalRow)
            theRelPrimalRowString = theRelPrimalRowArray[0, 0]
            relPrimal = float(theRelPrimalRowString)
            relPrimals.append(relPrimal)

            # get adaptive tau
            theRelDualRow = theLog.loc[[8]]
            theRelDualRowArray = np.array(theRelDualRow)
            theRelDualRowString = theRelDualRowArray[0, 0]
            relDual = float(theRelDualRowString)
            relDuals.append(relDual)

            # get xi
            # xis.append(relPrimal/(relDual*2))

        PLOT(outer_iters, adaptiveAlphas, tuners, i, figNum=1,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Adaptive alphas',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, adaptiveTaus, tuners, i, figNum=2,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Adaptive taus',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, relPrimals, tuners, i, figNum=3,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Relative primal residuals',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)

        PLOT(outer_iters, relDuals, tuners, i, figNum=4,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Relative dual residuals',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)
        '''
        PLOT(outer_iters, xis, tuners, i, figNum=5,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Xis',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=databasePath + dataFolderPath)
        '''
        print('No.' + str(i), '  initial alpha =', tuners[i], '\trel primal', '\trel dual', file=fp)
        print(file=fp)
        for k in range(1, len(adaptiveAlphas)+1):
            if k < 10:
                print('           --(   ' + str(k) + ')-->', adaptiveAlphas[k-1], '\t', relPrimals[k-1], '\t', relDuals[k-1], file=fp)
            elif k < 100:
                print('           --(  ' + str(k) + ')-->', adaptiveAlphas[k-1], '\t', relPrimals[k-1], '\t', relDuals[k-1], file=fp)
            elif k < 1000:
                print('           --( ' + str(k) + ')-->', adaptiveAlphas[k-1], '\t', relPrimals[k-1], '\t', relDuals[k-1], file=fp)
            else:
                print('           --(' + str(k) + ')-->', adaptiveAlphas[k-1], '\t', relPrimals[k-1], '\t', relDuals[k-1], file=fp)
        print(file=fp)
        print(file=fp)

    # tune inner iterations
    elif tuners_tag == 'inner_iters':
        likelihoods = []
        for outer_iter in range(1, outerIteration+1):
            logfolder = databasePath + dataFolderPath + '/config_rho=0_sub_i=' + str(tuners[i]) + '_alpha=' \
                        + str(bestAlpha) + '_mlem_=False/ADMM_64/'
            logfile_name = '0_' + str(outer_iter) + '.log'
            path_log = logfolder + logfile_name
            theLog = pd.read_table(path_log)
            theLikelihoodRow = theLog.loc[[theLog.shape[0] - 26]]
            theLikelihoodRowArray = np.array(theLikelihoodRow)
            theLikelihoodRowString = theLikelihoodRowArray[0, 0]
            theLikelihoodRowString = theLikelihoodRowString[22:44]
            if theLikelihoodRowString[0] == '-':
                theLikelihoodRowString = '0'
            likelihood = float(theLikelihoodRowString)
            if outer_iter == outerIteration:
                likelihoods_inner.append(likelihood)
            likelihoods.append(likelihood)

        beginning = 0
        if tuners[i] == bestInner:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], 'r-x', label=str(tuners[i]))
        elif i < 10:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], label=str(tuners[i]))
        elif 10 < i < 20:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], '.-',label=str(tuners[i]))
        else:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], 'x-', label=str(tuners[i]))
        plt.legend()
        plt.xlabel('outer iterations')
        plt.ylabel('likelihood')
        plt.title('The legend shows different inner iterations')

    # tune outer iterations
    elif tuners_tag == 'outer_iters':
        logfolder = databasePath + dataFolderPath + '/config_rho=0_sub_i=' + str(bestInner) + '_alpha=' \
                    + str(bestAlpha) + '_mlem_=False/ADMM_64/'
        logfile_name = '0_' + str(tuners[i]) + '.log'
        path_log = logfolder + logfile_name
        theLog = pd.read_table(path_log)
        theLikelihoodRow = theLog.loc[[theLog.shape[0] - 26]]
        theLikelihoodRowArray = np.array(theLikelihoodRow)
        theLikelihoodRowString = theLikelihoodRowArray[0, 0]
        theLikelihoodRowString = theLikelihoodRowString[22:44]
        if theLikelihoodRowString[0] == '-':
            theLikelihoodRowString = '0'
        likelihood = float(theLikelihoodRowString)
        likelihoods_outer.append(likelihood)

    else:
        print('Wrong tuners!')
        break
if tuners_tag == 'adaptiveRho':
    fp.close()


if tuners_tag == 'inner_iters':
    plt.figure()
    plt.plot(inner_iters, likelihoods_inner, '-x')
    plt.xlabel('inner iterations')
    plt.title('likelihood')
elif tuners_tag == 'outer_iters':
    plt.figure()
    plt.plot(outer_iters, likelihoods_outer, '-x')
    plt.xlabel('outer iterations')
    plt.title('likelihood')
elif tuners_tag == 'alphas':
    plt.figure()
    plt.plot(alphas, likelihoods_alpha, '-x')
    plt.xlabel('alpha')
    plt.title('likelihood')

plt.show()
