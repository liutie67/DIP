import time


from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

##1111111111111111111111111111111111111111111111111111##
initialALL()
for rep in range(1, 6):
    specialTask(method=Tuners.ADMMoptimizerName[4],
                replicates=rep,
                alpha=1,
                innerIter=1,
                outerIter=100*100,
                threads=128,)  # mu= ?, tau = ?
moveALL('ADMMadpAT+i1+o100*100+t128+a=1+mu1+tau100+rep5+4', dtnBase='#')
