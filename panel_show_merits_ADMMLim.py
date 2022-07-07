from show_functions import getDatabasePath
import Tuners


# ----------------------------------------------------------------------------------------------------------------------
SHOW = True  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------------------------------------------------
outputDatabaseNb = 14  # remember to change _squreNorm
dataFolderPath = '->-1-2022-06-22+17-20-49+2r+new3norms+adpAT+i1+o100*100+t128+a=alphas0+mu1+tau2'
# ----------------------------------------------------------------------------------------------------------------------
whichADMMoptimizer = Tuners.ADMMoptimizerName[4]
#                            0                   1                      2                            3                            4
# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx']
# ----------------------------------------------------------------------------------------------------------------------
option = 0
#            0            1              2              3                 4
OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters', 'calculateDiffCurve']
tuners_tag = OPTION[option]
# ----------------------------------------------------------------------------------------------------------------------
innerIteration = 1
outerIteration = 2000
# ----------------------------------------------------------------------------------------------------------------------
ALPHAS = Tuners.alphas0
# ----------------------------------------------------------------------------------------------------------------------
# calculate difference curves parameters
inners = list(range(innerIteration))
outers = list(range(0, outerIteration+1))
alpha = Tuners.alphas0[0]
MODEL = 'max'
TOGETHER = False
# ----------------------------------------------------------------------------------------------------------------------
vb = 1
threads = 128
REPLICATES = True
replicates = 1
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# ----------------------------------------------------------------------------------------------------------------------
_3NORMS = True  # defaut:True
_2R = True  # defaut:True
_squreNorm = False  # defaut:False
# ----------------------------------------------------------------------------------------------------------------------