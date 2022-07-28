import time


from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #

##1111111111111111111111111111111111111111111111111111##
#initialALL()
'''
specialTask(method_special=Tuners.ADMMoptimizerName[0],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=5,
            threads=[128])  # mu= ?, tau = ?

moveALL('ADMM+initial+i100+o100+t128+a=alphas0+mu1+tau100+rep5+0', dtnBase='#')
'''

initialALL()
for replicate in range(1, 6):
    for alpha in Tuners.alphas0:
        specialTask(method=Tuners.ADMMoptimizerName[1],
                    replicates=replicate,
                    alpha=alpha,
                    innerIter=100,
                    outerIter=100,
                    threads=128,)  # mu= ?, tau = ?
moveALL('ADMMadpA+i100+o100+t128+a=alphas0+mu10+tau2+rep5+1', dtnBase='#')

time.sleep(10)
initialALL()
for replicate in range(1, 6):
    for alpha in Tuners.alphas0:
        specialTask(method=Tuners.ADMMoptimizerName[2],
                    replicates=replicate,
                    alpha=alpha,
                    innerIter=100,
                    outerIter=100,
                    threads=128,)  # mu= ?, tau = ?
moveALL('ADMMadpAT+i100+o100+t128+a=alphas0+mu2+tau100+rep5+2', dtnBase='#')
'''
##222222222222222222222222222222222222222222222##
time.sleep(60)
initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[1],
            inner_special=[1],  # real inner
            outer_special=[100*100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?

moveALL('ADMMadpA+initial+i1+o100*100+t128+a=alphas0+mu10+tau2+rep1+1', dtnBase='#')

time.sleep(60)
initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[2],
            inner_special=[1],  # real inner
            outer_special=[100*100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpAT+initial+i1+o100*100+t128+a=alphas0+mu2+tau100+rep1+2', dtnBase='#')

##33333333333333333333333333333333333333333333333333##
time.sleep(60)
initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[3],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpA+initial+i100+o100+t128+a=alphas0+mu1+tau100+rep1+3', dtnBase='#')

initialALL()
time.sleep(60)
specialTask(method_special=Tuners.ADMMoptimizerName[4],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpAT+initial+i100+o100+t128+a=alphas0+mu1+tau100+rep1+4', dtnBase='#')

##4444444444444444444444444444444444444444444444444444##
initialALL()
time.sleep(60)
specialTask(method_special=Tuners.ADMMoptimizerName[3],
            inner_special=[1],  # real inner
            outer_special=[100*100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpA+initial+i1+o100*100+t128+a=alphas0+mu1+tau100+rep1+3', dtnBase='#')

initialALL()
time.sleep(60)
specialTask(method_special=Tuners.ADMMoptimizerName[4],
            inner_special=[1],  # real inner
            outer_special=[100*100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpAT+initial+i1+o100*100+t128+a=alphas0+mu1+tau100+rep1+4', dtnBase='#')
'''