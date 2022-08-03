import time

from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

iter = 1000
skip = 0
input = 'CT'
opti = 'Adam'
scaling = 'standardization'
thread = 128
rep = 1
optimizer = 'ADMMadpATi1o100*100rep' + str(rep)

initialALL()
timerStart = time.perf_counter()

for lr in [0.012]:
    specialTask(method='nested',
                random_seed=True,      # !
                threads=thread,
                all_images_DIP="True",
                lr=lr,
                sub_iter_DIP=iter,
                opti_DIP=opti,
                skip_connections=skip,
                scaling=scaling,
                input=input,
                post_recon=True)

timerEnd = time.perf_counter()
moveALL('+w50p200+' + optimizer
        + '+lr=0.012'
        + '+iter' + str(iter)
        + '+skip' + str(skip)
        + '+input' + input
        + '+opti' + opti
        + '+scaling' + scaling
        + '+t' + str(thread)
        + '+' + str(int(timerEnd - timerStart)) + 's'
        , dtnBase='D'
        )


