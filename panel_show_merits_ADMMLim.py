from show_functions import getDatabasePath
import Tuners


# ----------------------------------------------------------------------------------------------------------------------
SHOW = False  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------------------------------------------------
outputDatabaseNb = '#'  # remember to change _squreNorm
dataFolderPath = '2022-07-27+09-31-46ADMM+initial+i100+o100+t128+a=alphas0+mu1+tau100+rep5+0'
replicates = 1
# ----------------------------------------------------------------------------------------------------------------------
whichADMMoptimizer = Tuners.ADMMoptimizerName[0]
#                            0                   1                      2                            3                            4
# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx']
# ----------------------------------------------------------------------------------------------------------------------
option = 1
#            0            1              2              3                 4
OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters', 'calculateDiffCurve']
tuners_tag = OPTION[option]
# ----------------------------------------------------------------------------------------------------------------------
innerIteration = 100
outerIteration = 100
# ----------------------------------------------------------------------------------------------------------------------
ALPHAS = Tuners.alphas0
# ----------------------------------------------------------------------------------------------------------------------
# calculate difference curves parameters
inners = list(range(innerIteration))
outers = list(range(1, outerIteration+1))
alpha = Tuners.alphas0[0]
MODEL = 'max'
TOGETHER = False
# ----------------------------------------------------------------------------------------------------------------------
vb = 1
threads = 128
REPLICATES = True
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# ----------------------------------------------------------------------------------------------------------------------
_3NORMS = True  # defaut:True
_2R = True  # defaut:True
_squreNorm = False  # defaut:False
# ----------------------------------------------------------------------------------------------------------------------