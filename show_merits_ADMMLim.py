import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from show_functions import getDatabasePath, getDataFolderPath, dldir, computeThose4, PLOT, calculateDiffCurve
from show_functions import getValueFromLogRow, computeNorm, computeAverage
import Tuners

from panel_show_merits_ADMMLim import outputDatabaseNb, dataFolderPath, whichADMMoptimizer, REPLICATES
from panel_show_merits_ADMMLim import replicates, ALPHAS, outerIteration, innerIteration
from panel_show_merits_ADMMLim import vb, threads, SHOW, tuners_tag
from panel_show_merits_ADMMLim import inners, outers, alpha, MODEL, TOGETHER
from panel_show_merits_ADMMLim import _3NORMS, _2R, _squreNorm

databasePath = getDatabasePath(outputDatabaseNb) + '/'
fomSavingPath = databasePath + dataFolderPath +'/replicate_' + str(replicates)

threads_folder = False
if threads_folder == True:
    ADMM_threads = 'ADMM_' + str(threads)
else:
    ADMM_threads = ''


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

likelihoods_alpha = []
likelihoods_inner = []
likelihoods_outer = []

for i in range(len(tuners)):
    if tuners_tag == 'alphas':
        likelihoods = []
        for outer_iter in outers:
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer \
                                 # + '/Comparison/' + whichADMMoptimizer
            else:
                replicatesPath = ''
            logfolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False_post_=0/' + ADMM_threads
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
             Title='Likelihood(same scale)',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)
        plt.ylim([2.904e6, 2.919e6])

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
        U_unscaled_norms = []
        coeff_alphas = []
        averageUs = []
        for outer_iter in outers:
            if REPLICATES:
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer \
                                 #+ '/Comparison/' + whichADMMoptimizer
            else:
                replicatesPath = ''
            imageFolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' \
                          + str(innerIteration) + '_alpha=' + str(tuners[i]) + '_mlem_=False_post_=0/' + ADMM_threads
            imageName = '0_' + str(outer_iter) + '_it' + str(innerIteration) + '.img'
            vName = '0_' + str(outer_iter) + '_v.img'
            uName = '0_' + str(outer_iter) + '_u.img'

            logfolder = imageFolder
            logfile_name = '0_' + str(outer_iter) + '_adaptive.log'
            path_txt = logfolder + logfile_name
            coeff_alpha = getValueFromLogRow(path_txt, 0)/getValueFromLogRow(path_txt, 4)



            imagePath = imageFolder + imageName
            IR, MSE, CRC, MA = computeThose4(imagePath)
            IR_bkgs.append(IR)
            MSEs.append(MSE)
            CRC_hots.append(CRC)
            MA_colds.append(MA)

            Xnorms.append(computeNorm(imagePath))
            Vnorms.append(computeNorm(imageFolder+vName))
            u_norm = computeNorm(imageFolder+uName)
            Unorms.append(u_norm)
            U_unscaled_norms.append(u_norm * coeff_alpha)
            coeff_alphas.append(coeff_alpha)
            averageUs.append(computeAverage(imageFolder+uName))

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

        PLOT(outer_iters, U_unscaled_norms, tuners, i, figNum=10,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='norm of UNSCALED u',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, coeff_alphas, tuners, i, figNum=11,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='coeff_alphas',
             replicate=replicates,
             whichOptimizer=whichADMMoptimizer,
             imagePath=fomSavingPath)

        PLOT(outer_iters, averageUs, tuners, i, figNum=12,
             Xlabel='Outer iteration',
             Ylabel='The legend shows different alpha',
             Title='average of u',
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
                replicatesPath = '/replicate_' + str(replicates) + '/' + whichADMMoptimizer \
                                 #+ '/Comparison/' + whichADMMoptimizer
            else:
                replicatesPath = ''
            logfolder = databasePath + dataFolderPath + replicatesPath + '/config_rho=0_sub_i=' + str(innerIteration) \
                        + '_alpha=' + str(tuners[i]) + '_mlem_=False_post_=0/' + ADMM_threads
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

if tuners_tag == 'adaptiveRho':
    fp.close()

elif tuners_tag == 'alphas' and len(alphas)==len(likelihoods_alpha):
    plt.figure()
    plt.plot(alphas, likelihoods_alpha, '-x')
    plt.xlabel('alpha')
    plt.title('likelihood')

if SHOW:
    plt.show()
