import time


from main import specialTask
import Tuners

from show_functions import moveRuns, moveData, initialALL, moveALL


# ADMMoptimizerName = ['ADMMLim_new', 'ADMMLim_adaptiveRho', 'ADMMLim_adaptiveRhoTau', 'ADMMLim_adaptiveRhoTau-m10', 'ADMMLim_adaptiveRhoTau-mx'] #
#                            0                  1                       2                           3                             4               #
'''
initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[4],
            inner_special=[1],  # real inner
            outer_special=[100 * 100],
            alpha_special=Tuners.alphas0,
            replicates_special=1,
            threads=[128])  # mu= ?, tau = ?
moveALL('+2r+new3norms+AdpAT+i1+o100*100+t128+a=alpha0+mu50+tau2')
'''

initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[0],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=3,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMM+i100+o100+t128+a=alphas0+mu1+tau100+rep3', dtnBase='F')

initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[1],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=3,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpA+i100+o100+t128+a=alphas0+mu10+tau2+rep3', dtnBase='F')

initialALL()
specialTask(method_special=Tuners.ADMMoptimizerName[2],
            inner_special=[100],  # real inner
            outer_special=[100],
            alpha_special=Tuners.alphas0,
            replicates_special=3,
            threads=[128])  # mu= ?, tau = ?
moveALL('ADMMadpAT+i100+o100+t128+a=alphas0+mu2+tau100+rep3', dtnBase='F')