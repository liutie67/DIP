import time

from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #
admmInner = 1
admmOuter = 800
admmAlpha = 1

lr = 0.006
iter = 200
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'
thread = 128
optimizer = 'nested'

globalIter = 5
rho = 1e-3

timerStart = time.perf_counter()
'''
initialALL()
specialTask(method_special=optimizer,
            max_iter=[globalIter],
            replicates_special=1,
            lr_special=[lr],
            sub_iter_special=[iter],
            opti_special=[opti],
            skip_special=[skip],
            scaling_special=[scaling],
            input_special=[input],
            inner_special=[admmInner],
            outer_special=[admmOuter],
            alpha_special=[admmAlpha],
            rho_special=[rho],
            nb_subsets=[28],
            threads=[thread])
'''
timerEnd = time.perf_counter()

moveALL('+' + optimizer
        + 'globalIter' + str(globalIter)
        + '+lr=' + str(lr)
        + '+subIter' + str(iter)
        + '+skip' + str(skip)
        + '+input' + input
        + '+opti' + opti
        + '+scaling' + scaling
        + '+t' + str(thread)
        + '+' + str(int(timerEnd - timerStart)) + 's',
        dtnBase=19
        )
