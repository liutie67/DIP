import numpy as np
import matplotlib.pyplot as plt


from show_merits_DIP import getGT, getPhantomROI, getShape, computeThose4
from show_functions import fijii_np, find_nan


databasePath = '/home/liutie/Documents/outputDatabase1'

dataFolderPath = 'nested+rho=...*5'
dataSubrootPath = '/Block2/out_cnn/24/'

def filePathNested(dataFolderPath,
                   DIPiter,
                   rho,
                   lr,
                   subIter,
                   opti,
                   scaling,
                   skip,
                   input,
                   inner,
                   alpha):
    filename = 'out_DIP' + str(DIPiter) + 'config_rho=' + str(rho) + '_lr=' + str(lr) + '_sub_i=' + str(subIter) \
               + '_opti_=' + opti + '_skip_=' + str(skip) + '_scali=' + scaling + '_input=' + input \
               + '_d_DD=4_k_DD=32_sub_i=' + str(inner) + '_alpha=' + str(alpha) + '_mlem_=False.img'
    filePath = databasePath + '/' + dataFolderPath + dataSubrootPath + filename

    return filePath

def getReconsNested(rho,
                    start=0,
                    end=30,
                    step=1):
    IRs = []
    MSEs = []
    CRCs = []
    MAs = []
    for i in range(start, end, step):
        filePath = filePathNested(dataFolderPath=dataFolderPath,
                                  DIPiter=i,
                                  rho=rho,
                                  lr=0.07,
                                  subIter=200,
                                  opti='LBFGS',
                                  scaling='normalization',
                                  skip=0,
                                  input='CT',
                                  inner=65,
                                  alpha=0.005)
        f = fijii_np(filePath, shape=getShape())
        IR, MSE, CRC, MA = computeThose4(f)
        IRs.append(IR)
        MSEs.append(MSE)
        CRCs.append(CRC)
        MAs.append(MA)
    return IRs, MSEs, CRCs, MAs


def plotOneFigure(numFigure, i, Xs, Ys, label, title='', xlabel='epochs'):
    # iter_to_show = len(Xs)
    plt.figure(numFigure)
    if i < 10:
        plt.plot(Xs, Ys, label=str(label))
    elif 10 <= i < 20:
        plt.plot(Xs, Ys, '.-', label=str(label))
    else:
        plt.plot(Xs, Ys, 'x-', label=str(label))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()

rhos = [0.0001, 0.001, 0.01, 0.1, 1]
epochs = range(30)
for i in range(len(rhos)):
    IRs, MSEs, CRCs, MAs = getReconsNested(rho=rhos[i])
    plotOneFigure(1, i, epochs, IRs, label=rhos[i],title='IR')
    plotOneFigure(2, i, epochs, MSEs, label=rhos[i], title='MSE')
    plotOneFigure(3, i, epochs, CRCs, label=rhos[i], title='CRC')
    plotOneFigure(4, i, epochs, MAs, label=rhos[i], title='MA')

plt.show()













