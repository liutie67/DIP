from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

lr = 0.04
iter = 400
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'

initialALL()
specialTask(DIP_special=True,
            lr_special=[lr],
            sub_iter_special=[iter],
            skip_special=[skip],
            input_special=[input],
            opti_special=[opti],
            scaling_special=[scaling],)
moveALL('+ADMMi100o100+lr' + str(lr)
        + '+iter' + str(iter)
        + '+skip' + str(skip)
        + '+input' + input
        + '+opti' + opti
        + '+scaling' + scaling)