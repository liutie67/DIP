import time

from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

iter = 5000
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'
thread = 128
optimizer = 'admm5+adpAT+a1+i1o2000'

initialALL()
timerStart = time.perf_counter()
specialTask(method_special='nested',
            DIP_special=True,
            lr_special=[0.006],
            sub_iter_special=[iter],
            skip_special=[skip],
            input_special=[input],
            opti_special=[opti],
            scaling_special=[scaling],
            threads=[thread])
timerEnd = time.perf_counter()
moveALL('+dip5+' + optimizer
        + '+lr=0.006'
        + '+iter' + str(iter)
        + '+skip' + str(skip)
        + '+input' + input
        + '+opti' + opti
        + '+scaling' + scaling
        + '+t' + str(thread)
        + '+' + str(int(timerEnd - timerStart)) + 's',
        dtnBase=20
        )


