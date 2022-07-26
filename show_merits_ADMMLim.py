import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from show_functions import getDatabasePath, getDataFolderPath, dldir, computeThose4, PLOT, calculateDiffCurve
from show_functions import getValueFromLogRow, computeNorm
import Tuners

from panel_show_merits_ADMMLim import colors, outputDatabaseNb, dataFolderPath, whichADMMoptimizer, REPLICATES
from panel_show_merits_ADMMLim import replicates, ALPHAS, outerIteration, innerIteration
from panel_show_merits_ADMMLim import vb, threads, SHOW, tuners_tag
from panel_show_merits_ADMMLim import inners, outers, alpha, MODEL, TOGETHER
from panel_show_merits_ADMMLim import _3NORMS, _2R, _squreNorm

databasePath = getDatabasePath(outputDatabaseNb) + '/'
fomSavingPath = databasePath + dataFolderPath +'/replicate_' + str(replicates)

if tuners_tag == 'alphas':
    # outerIteration = 1000
    # innerIteration = 50
    bestAlpha = 0

    alphas = ALPHAS

    inner_iters = range(innerIteration)
    outer_iters = outers
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
    outer_iters = outers
    # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
    tuners = alpha0s
    fp = open(databasePath + dataFolderPath + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')

elif tuners_tag == 'calculateDiffCurve':
    tuners = outers

elif tuners_tag == 'inner_iters':
    bestInner = 0

    bestAlpha = 0.005

    innerIteration = 70
    min_innerIteration = 15
    outerIteration = 100

    inner_iters = range(min_innerIteration, innerIteration + 1, 5)
    # inner_iters = [65, 70]

    outer_iters = range(outerIteration)
    # dataFolderPath = '/ADMMLim+alpha=' + str(bestAlpha) + '+i...+o' + str(outerIteration)
    tuners = inner_iters

elif tuners_tag == 'outer_iters':
    bestAlpha = 0.005
    bestInner = 65
    outerIteration = 100

    outer_iters = range(1, outerIteration + 1)

    # dataFolderPath = '/ADMMLim+alpha=' + str(bestAlpha) + '+i...+o' + str(outerIteration)
    tuners = outer_iters

likelihoods_alpha = []
likelihoods_inner = []
likelihoods_outer = []

for i in range(len(tuners)):
    if tuners_tag == 'calculateDiffCurve':
        break
    # tune alpha
    if tuners_tag == 'alphas':
        likelihoods = []
        for outer_iter in outers:
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

        PLOT(outer_iters, likelihoods, tuners, i, figNum=6,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Likelihood(same scale - 1)',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)
        plt.ylim([2.904e6, 2.922e6])

        PLOT(outer_iters, likelihoods, tuners, i, figNum=1,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Likelihood',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        IR_bkgs = []
        MSEs = []
        CRC_hots = []
        MA_colds = []
        Xnorms = []
        Vnorms = []
        Unorms = []
        for outer_iter in outers:
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer + '/Comparison/' \
                                 + whichADMMoptimizer
            else:
                replicatesPath = ''
            imageFolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' \
                          + str(innerIteration) + '_alpha=' + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
            imageName = '0_' + str(outer_iter) + '_it' + str(innerIteration) + '.img'
            vName = '0_' + str(outer_iter) + '_v.img'
            uName = '0_' + str(outer_iter) + '_u.img'

            imagePath = imageFolder + imageName
            IR, MSE, CRC, MA = computeThose4(imagePath)
            IR_bkgs.append(IR)
            MSEs.append(MSE)
            CRC_hots.append(CRC)
            MA_colds.append(MA)

            Xnorms.append(computeNorm(imagePath))
            Vnorms.append(computeNorm(imageFolder+vName))
            Unorms.append(computeNorm(imageFolder+uName))

        PLOT(outer_iters, IR_bkgs, tuners, i, figNum=2,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Image Roughness in the background',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, MSEs, tuners, i, figNum=3,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Mean Square Error',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, CRC_hots, tuners, i, figNum=4,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='CRC hot',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, MA_colds, tuners, i, figNum=5,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='MA cold',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, Xnorms, tuners, i, figNum=7,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='norm of x',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, Vnorms, tuners, i, figNum=8,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='norm of v',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, Unorms, tuners, i, figNum=9,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='norm of u',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

    # adaptive alphas
    elif tuners_tag == 'adaptiveRho':
        adaptiveAlphas = []
        adaptiveTaus = []
        relPrimals = []
        relDuals = []
        xis = []
        normAxvs = []
        normAxvus = []
        normAxv1us = []
        primals = []
        duals = []
        for outer_iter in outers:
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer + '/Comparison/' \
                                 + whichADMMoptimizer
            else:
                replicatesPath = ''
            logfolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False/ADMM_' + str(threads) + '/'
            logfile_name = '0_' + str(outer_iter) + '_adaptive.log'
            path_txt = logfolder + logfile_name

            # get adaptive alpha
            adaptiveAlphas.append(getValueFromLogRow(path_txt, 0))

            # get adaptive tau
            adaptiveTaus.append(getValueFromLogRow(path_txt, 2))

            # get relative primal residual
            relPrimals.append(getValueFromLogRow(path_txt, 6))

            # get relative dual residual
            relDuals.append(getValueFromLogRow(path_txt, 8))

            # get xi
            xis.append(getValueFromLogRow(path_txt, 6) / (getValueFromLogRow(path_txt, 8) * 2))

            if _3NORMS:
                # get norm of Ax(n+1) - v(n+1)
                normAxvs.append(getValueFromLogRow(path_txt, 10))

                # get norm of Ax(n+1) - v(n) + u(n)
                normAxvus.append(getValueFromLogRow(path_txt, 12))

                # get norm of Ax(n+1) - v(n+1) + u(n)
                normAxv1us.append(getValueFromLogRow(path_txt, 14))

            if _2R:
                # get norm of primal residual
                primals.append(getValueFromLogRow(path_txt, 16))

                # get norm of dual residual
                duals.append(getValueFromLogRow(path_txt, 18))

        PLOT(outer_iters, adaptiveAlphas, tuners, i, figNum=1,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Adaptive alphas',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, adaptiveTaus, tuners, i, figNum=2,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Adaptive taus',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, relPrimals, tuners, i, figNum=3,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Relative primal residuals',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, relDuals, tuners, i, figNum=4,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Relative dual residuals',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, xis, tuners, i, figNum=5,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='Xis',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        if _3NORMS:
            if _squreNorm:
                normAxvs = np.sqrt(normAxvs)
                normAxvus = np.sqrt(normAxvus)
                normAxv1us = np.sqrt(normAxv1us)
            PLOT(outer_iters, normAxvs, tuners, i, figNum=6,
                 Xlabel='Outer iteration',
                 Ylabel='The legend shows different alpha',
                 Title='norm of Ax(n+1) - v(n+1)',
                 replicate=replicates,
                 whichOptimizer=whichADMMoptimizer,
                 imagePath=fomSavingPath)

            PLOT(outer_iters, normAxvus, tuners, i, figNum=7,
                 Xlabel='Outer iteration',
                 Ylabel='The legend shows different alpha',
                 Title='norm of Ax(n+1) - v(n) + u(n)',
                 replicate=replicates,
                 whichOptimizer=whichADMMoptimizer,
                 imagePath=fomSavingPath)

            PLOT(outer_iters, normAxv1us, tuners, i, figNum=8,
                 Xlabel='Outer iteration',
                 Ylabel='The legend shows different alpha',
                 Title='norm of Ax(n+1) - v(n+1) + u(n)',
                 replicate=replicates,
                 whichOptimizer=whichADMMoptimizer,
                 imagePath=fomSavingPath)

        if _2R:
            if _squreNorm:
                primals = np.sqrt(primals)
                duals = np.sqrt(duals)
            PLOT(outer_iters, primals, tuners, i, figNum=9,
                 Xlabel='Outer iteration',
                 Ylabel='The legend shows different alpha',
                 Title='primal residual',
                 replicate=replicates,
                 whichOptimizer=whichADMMoptimizer,
                 imagePath=fomSavingPath)

            PLOT(outer_iters, duals, tuners, i, figNum=10,
                 Xlabel='Outer iteration',
                 Ylabel='The legend shows different alpha',
                 Title='dual residual',
                 replicate=replicates,
                 whichOptimizer=whichADMMoptimizer,
                 imagePath=fomSavingPath)

        print('No.' + str(i), '  initial alpha =', tuners[i], '\trel primal', '\trel dual', file=fp)
        print(file=fp)
        for k in range(1, len(adaptiveAlphas) + 1):
            if k < 10:
                print('           --(   ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                      relDuals[k - 1], file=fp)
            elif k < 100:
                print('           --(  ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                      relDuals[k - 1], file=fp)
            elif k < 1000:
                print('           --( ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                      relDuals[k - 1], file=fp)
            else:
                print('           --(' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                      relDuals[k - 1], file=fp)
        print(file=fp)
        print(file=fp)

    # tune inner iterations
    elif tuners_tag == 'inner_iters':
        likelihoods = []
        for outer_iter in outers:
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
            plt.plot(outer_iters[beginning:-1], likelihoods[beginning:-1], '.-', label=str(tuners[i]))
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

# calculate difference curves
if tuners_tag == 'calculateDiffCurve':
    calculateDiffCurve(inners, outers, outputDatabaseNb, dataFolderPath,
                       optimizer=whichADMMoptimizer,
                       innerIteration=innerIteration,
                       alpha=alpha,
                       threads=threads,
                       REPLICATES=REPLICATES,
                       replicates=replicates,
                       Together=TOGETHER,
                       model=MODEL)

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
elif tuners_tag == 'alphas' and len(alphas)==len(likelihoods_alpha):
    plt.figure()
    plt.plot(alphas, likelihoods_alpha, '-x')
    plt.xlabel('alpha')
    plt.title('likelihood')

if SHOW:
    plt.show()
