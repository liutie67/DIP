import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from show_functions import getDatabasePath, getDataFolderPath, dldir
import tuners

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
databasePath = getDatabasePath(4) + '/'
dataFolderPath = 'ADMM-merged(double)+i50+o70+a=*9'
vb = 1
threads = 128

tuners_tag = 'alphas'  # tuners = 'alphas' or 'inner_iters' or 'outer_iters' or 'adaptiveRho'
if tuners_tag == 'alphas':
    outerIteration = 70
    innerIteration = 50*2
    bestAlpha = 0

    alphas = tuners.alphas0

    inner_iters = range(innerIteration)
    outer_iters = range(outerIteration)
    # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
    tuners = alphas

elif tuners_tag == 'adaptiveRho':
    outerIteration = 70
    innerIteration = 50
    alpha0s = [1]
    duplicate = ''
    inner_iters = range(innerIteration)
    outer_iters = range(outerIteration)
    # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
    tuners = alpha0s
    fp = open(databasePath + dataFolderPath + '/adaptiveAlphaProcess' + str(duplicate) + '.log', mode='w+')

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
            logfolder = databasePath + dataFolderPath + '/config_rho=0_sub_i=' + str(innerIteration) + '_alpha=' \
                       + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
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

        beginning = 1
        if tuners[i] == bestAlpha:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], 'r-x', label=str(alphas[i]))
        elif i < 10:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], label=str(alphas[i]))
        elif 10 <= i < 20:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], '.-',label=str(alphas[i]))
        else:
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], 'x-', label=str(alphas[i]))
        plt.legend(loc='best')
        plt.xlabel('outer iterations')
        plt.ylabel('likelihood')
        plt.title('The legend shows different alpha')

    # adaptive alphas
    elif tuners_tag == 'adaptiveRho':
        adaptiveAlphas = []
        adaptiveTaus = []
        relPrimals = []
        relDuals = []
        for outer_iter in range(1,outerIteration+1):
            logfolder = databasePath + dataFolderPath + '/config_rho=0_sub_i=' + str(innerIteration) + '_alpha=' \
                        + str(tuners[i]) + '_mlem_=False/ADMM_64/'
            logfile_name = '0_' + str(outer_iter) + '_adaptive_alpha.log'
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

        plt.figure(1)
        beginning = 0
        if i < 10:
            plt.plot(outer_iters[beginning:-1], adaptiveAlphas[beginning:-1], label=str(tuners[i]))
        elif 10 <= i < 20:
            plt.plot(outer_iters[beginning:-1], adaptiveAlphas[beginning:-1], '.-', label=str(tuners[i]))
        else:
            plt.plot(outer_iters[beginning:-1], adaptiveAlphas[beginning:-1], 'x-', label=str(tuners[i]))
        plt.legend(loc='best')
        plt.xlabel('outer iterations')
        plt.ylabel('adaptive alphas')
        plt.title('The legend shows different initial alpha')

        plt.figure(2)
        beginning = 0
        if i < 10:
            plt.plot(outer_iters[beginning:-1], adaptiveTaus[beginning:-1], label=str(tuners[i]))
        elif 10 <= i < 20:
            plt.plot(outer_iters[beginning:-1], adaptiveTaus[beginning:-1], '.-', label=str(tuners[i]))
        else:
            plt.plot(outer_iters[beginning:-1], adaptiveTaus[beginning:-1], 'x-', label=str(tuners[i]))
        plt.legend(loc='best')
        plt.xlabel('outer iterations')
        plt.ylabel('adaptive taus')
        plt.title('The legend shows different initial alpha')

        plt.figure(3)
        beginning = 0
        if i < 8:
            plt.plot(outer_iters[beginning:-1], relPrimals[beginning:-1], colors[i], label='primal ' + str(tuners[i]))
            plt.plot(outer_iters[beginning:-1], relDuals[beginning:-1], colors[i]+'-x', label='dual ' + str(tuners[i]))
        elif 8 <= i < 16:
            plt.plot(outer_iters[beginning:-1], relPrimals[beginning:-1], colors[i % 8]+'-*', label='primal ' + str(tuners[i]))
            plt.plot(outer_iters[beginning:-1], relDuals[beginning:-1], colors[i % 8]+'-.', label='dual ' + str(tuners[i]))
        else:
            plt.plot(outer_iters[beginning:-1], relPrimals[beginning:-1], colors[i % 8]+'.', label='primal ' + str(tuners[i]))
            plt.plot(outer_iters[beginning:-1], relDuals[beginning:-1], colors[i % 8]+',', label='dual ' + str(tuners[i]))
        plt.legend(loc='best')
        plt.xlabel('outer iterations')
        plt.ylabel('residuals')
        plt.title('The legend shows different initial alpha')

        print('No.' + str(i), '  initial alpha =', tuners[i], file=fp)
        print(file=fp)
        for k in range(1, len(adaptiveAlphas)+1):
            if k < 10:
                print('             --( ' + str(k) + ')-->', adaptiveAlphas[k-1], file=fp)
            else:
                print('             --(' + str(k) + ')-->', adaptiveAlphas[k-1], file=fp)
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
