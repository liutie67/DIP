from show_functions import getDatabasePath
import Tuners


# ----------------------------------------------------------------------------------------------------------------------
SHOW = True  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------------------------------------------------
outputDatabaseNb = '20'  # remember to change _squreNorm
dataFolderPath = '2022-07-11+16-22-01+admm5+ADMMadpAT+i1+o3000+t128+a=1+mu1+tau2'
# ----------------------------------------------------------------------------------------------------------------------
whichADMMoptimizer = Tuners.ADMMoptimizerName[4]
#                            0                   1                      2                            3                            4
# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx']
# ----------------------------------------------------------------------------------------------------------------------
option = 1
#            0            1              2              3                 4
OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters', 'calculateDiffCurve']
tuners_tag = OPTION[option]
# ----------------------------------------------------------------------------------------------------------------------
innerIteration = 1
outerIteration = 3000
# ----------------------------------------------------------------------------------------------------------------------
ALPHAS = [1]
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
replicates = 1
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# ----------------------------------------------------------------------------------------------------------------------
_3NORMS = True  # defaut:True
_2R = True  # defaut:True
_squreNorm = False  # defaut:False
# ----------------------------------------------------------------------------------------------------------------------