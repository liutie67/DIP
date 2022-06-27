from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[0],
            inner_special=1,  # real inner
            outer_special=100*100,
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=128)  # mu= ?, tau = ?
moveALL('+2r+new3norms+NonAdp+i1+o100*100+t128+a=alphas0+mu1+tau100')
