from show_functions import getDatabasePath
import Tuners


# ----------------------------------------------------------------------------------------------------------------------
SHOW = False  # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------------------------------------------------
outputDatabaseNb = 'A'  # remember to change _squreNorm
dataFolderPath = '2022-08-01+13-17-24ADMMadpAT+i1+o100*100+t128+a=1+mu1+tau100+rep5+4'
replicates = 3
# ----------------------------------------------------------------------------------------------------------------------
whichADMMoptimizer = Tuners.ADMMoptimizerName[4]
#                            0                   1                      2                            3                            4
# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx']
# ----------------------------------------------------------------------------------------------------------------------
option = 0  # Now, only option 0 and 1 are useful, option 2, 3 and 4 should be ignored
#            0            1              2              3                 4
OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters', 'calculateDiffCurve']
tuners_tag = OPTION[option]
# ----------------------------------------------------------------------------------------------------------------------
innerIteration = 1
outerIteration = 100*100
# ----------------------------------------------------------------------------------------------------------------------
ALPHAS = [1]
# ----------------------------------------------------------------------------------------------------------------------
# parameters for option 4: 'calculateDiffCurve'
inners = list(range(innerIteration))
outers = list(range(1, outerIteration+1))
alpha = Tuners.alphas0[0]
MODEL = 'max'
TOGETHER = False
# ----------------------------------------------------------------------------------------------------------------------
vb = 1  # same verbose in vGeneral.py
threads = 128
REPLICATES = True  # as we use variable 'replicates' above, set it to True
# ----------------------------------------------------------------------------------------------------------------------
_3NORMS = True  # defaut:True
_2R = True  # defaut:True
_squreNorm = False  # defaut:False
# ----------------------------------------------------------------------------------------------------------------------