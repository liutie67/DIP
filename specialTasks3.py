import time

from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #
admmInner = 1
admmOuter = 1000
admmAlpha = 1

lr = 0.012
iter = 500
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'
thread = 128
optimizer = 'nested'

globalIter = 4
rho = 1e-3

timerStart = time.perf_counter()

initialALL()

specialTask(method='nested',
            random_seed=True,
            threads=thread,
            max_iter=globalIter,
            nb_subsets=28,
            all_images_DIP="False",
            replicates=1,
            rho=rho,
            lr=lr,
            sub_iter_DIP=iter,
            opti_DIP='Adam',
            skip_connections=0,
            scaling='standardization',
            input='CT',
            innerIter=admmInner,
            outerIter=admmOuter,
            alpha=admmAlpha,
            post_recon=False)

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
        dtnBase='#'
        )
