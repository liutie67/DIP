import matplotlib.pyplot as plt
import numpy as np

from show_merits_DIP import get_recons, normalisation_DIP, plotAndSave_figures
# from lrsDatabase import lrs0, lrs1, lrs2, lrs3, lrs4
import Tuners

dataFolderPath = '1000+MLEM1000-1+CT0+lr...*5+24'


end = 1000
step = 1

lrs = tuners.lrs4 + tuners.lrs6

# for only one case
IR, MSE, CRC, MA = get_recons(dataFolderPath=dataFolderPath,
                              replicates=1,
                              INPUT='CT',
                              skip=0,
                              lrs=lrs,
                              epoch_star=1,
                              epoch_end=end,
                              epoch_step=step,
                              iteration=1000,
                              scaling='nothing',
                              opti='LBFGS'
                              )

cost = normalisation_DIP(IR_bkg_recon=IR,
                         MSE_recon=MSE,
                         CRC_hot_recon=CRC,
                         MA_cold_recon=MA,
                         NEW_normalization=True,
                         maxIR=2,
                         maxMSE=0.2*1e6,
                         maxCRC=400,
                         maxMA=300)

plotAndSave_figures(IR_bkg_recon=IR,
                    MSE_recon=MSE,
                    CRC_hot_recon=CRC,
                    MA_cold_recon=MA,
                    cost_function=cost,
                    replicates=1,
                    INPUT='CT',
                    skip=0,
                    lrs=lrs,
                    saveIMAGE=False,
                    dataFolderPath=dataFolderPath,
                    epoch_star=1,
                    epoch_end=end,
                    epoch_step=step,
                    showFigures=True
                    )

'''
# find the maximum
maxIR = -1e10
maxMSE = -1e10
maxCRC = -1e10
maxMA = -1e10
for replicates in ['1', '2', '3']:
    for INPUT in ['CT', 'random']:
        for skip in [0, 3]:
            IR, MSE, CRC, MA = get_recons(dataFolderPath=dataFolderPath,
                                           replicates=replicates,
                                           INPUT=INPUT,
                                           skip=skip,
                                           lrs=lrs,
                                           epoch_star=1,
                                           epoch_end=end,
                                           epoch_step=step)

            if np.amax(IR) > maxIR:
                maxIR = np.amax(IR)
            if np.amax(MSE) > maxMSE:
                maxMSE = np.amax(MSE)
            if np.amax(CRC) > maxCRC:
                maxCRC = np.amax(CRC)
            if np.amax(MA) > maxMA:
                maxMA = np.amax(MA)
'''

# only plot cost function
'''
k = 0
for replicates in ['1', '2', '3']:
    for INPUT in ['CT', 'random']:
        for skip in [0, 3]:
            IR, MSE, CRC, MA = get_recons(dataFolderPath=dataFolderPath,
                                           replicates=replicates,
                                           INPUT=INPUT,
                                           skip=skip,
                                           lrs=lrs,
                                           epoch_star=1,
                                           epoch_end=end,
                                           epoch_step=1)

            cost = normalisation_DIP(IR_bkg_recon=IR,
                                     MSE_recon=MSE,
                                     CRC_hot_recon=CRC,
                                     MA_cold_recon=MA,
                                     NEW_normalization=False)

            plotAndSave_figures(IR_tag=False, MSE_tag=False, CRC_tag=False, MA_tag=False,
                                cost_function=cost,
                                replicates=replicates,
                                INPUT=INPUT,
                                skip=skip,
                                lrs=lrs,
                                saveIMAGE=True,
                                dataFolderPath=dataFolderPath,
                                epoch_star=1,
                                epoch_end=end,
                                epoch_step=step,
                                showFigures=False,
                                additional=k)
            k = k + 1
'''

