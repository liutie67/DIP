from show_functions import getDatabasePath, fijii_np
from show_merits_DIP import computeThose4
import matplotlib.pyplot as plt
import numpy as np

from show_merits_DIP import normalisation_DIP

skip = 1
input = 'CT'

def oneCase(skip=1, input='CT'):
    IRs = []
    MSEs = []
    CRCs = []
    MAs = []
    N = 80
    for i in range(N):
        filePath = getDatabasePath(1) + '/400+MLEM+CT,random+0123+*100/' + str(i) + '/'
        middlePath = 'replicate_1/Gong/Block2/out_cnn/24/'
        filename = 'out_DIP_post_reco_epoch=399config_rho=0_lr=0.05_sub_i=400_opti_=Adam_skip_=' + str(skip) \
                   + '_scali=nothing_input=' + input + '_d_DD=4_k_DD=32_sub_i=50_alpha=0.084_mlem_=False.img'
        fPath = filePath + middlePath + filename
        f = fijii_np(fPath)
        IR, MSE, CRC, MA = computeThose4(f)
        IRs.append(IR)
        MSEs.append(MSE)
        CRCs.append(CRC)
        MAs.append(MA)

    cost = normalisation_DIP(IR_bkg_recon=IRs,
                             MSE_recon=MSEs,
                             CRC_hot_recon=CRCs,
                             MA_cold_recon=MAs,
                             NEW_normalization=True,
                             maxIR=2,
                             maxMSE=0.2*1e6,
                             maxCRC=400,
                             maxMA=300)

    plt.figure(1+5)
    plt.plot(range(N), IRs, label=input+str(skip))
    plt.title('IR')
    plt.legend()

    plt.figure(2+5)
    plt.plot(range(N), MSEs, label=input+str(skip))
    plt.title('MSE')
    plt.legend()

    plt.figure(3+5)
    plt.plot(range(N), CRCs, label=input+str(skip))
    plt.title('CRC')
    plt.legend()

    plt.figure(4+5)
    plt.plot(range(N), MAs, label=input+str(skip))
    plt.title('MA')
    plt.legend()

    plt.figure(5+5)
    plt.plot(range(N), cost, label=input+str(skip))
    plt.title('cost')
    plt.legend()


oneCase(0, 'CT')
oneCase(1, 'CT')
oneCase(2, 'CT')
oneCase(3, 'CT')
oneCase(0, 'random')
oneCase(1, 'random')
oneCase(2, 'random')
oneCase(3, 'random')
plt.show()


