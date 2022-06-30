import time

from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

lr = 0.04
iter = 1000
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'
thread = 128
optimizer = 'ADMMadpATi1o100*100a1'

initialALL()
timerStart = time.perf_counter()
specialTask(DIP_special=True,
            lr_special=Tuners.lrs1,
            sub_iter_special=[iter],
            skip_special=[skip],
            input_special=[input],
            opti_special=[opti],
            scaling_special=[scaling],
            threads=[thread])
timerEnd = time.perf_counter()
moveALL('+wx+px+' + optimizer
        + '+lr=lrs1'
        + '+iter' + str(iter)
        + '+skip' + str(skip)
        + '+input' + input
        + '+opti' + opti
        + '+scaling' + scaling
        + '+t' + str(thread)
        + '+' + str(int(timerEnd - timerStart)) + 's',
        dtnBase=15
        )
