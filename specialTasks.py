from main import specialTask
import tuners

from show_functions import moveRuns, moveData, initialALL, moveALL



# MLEM
'''
specialTask(method_special='MLEM', max_iter=1000, nb_subsets=1, mlem_sequence=False)
'''

# DIP(Gong)
'''
for i in range(74, 100):
    specialTask(method_special='Gong',
                sub_iter_special=400,
                lr_special=0.05,
                skip_special=[0, 1, 2, 3],
                input_special=['CT', 'random'],
                opti_special='Adam',
                scaling_special='normalization',
                DIP_special=True)

    moveRuns(i)
    moveData(i)
'''
# DIP
'''
inputs_sp = ['CT']
skips_sp = [0, 3]

for input_sp in inputs_sp:
    for skip_sp in skips_sp:
        for lr_sp in lrs_sp:
            specialTask(method_special='nested',
                        sub_iter_special=1000,
                        lr_special=lr_sp,
                        skip_special=skip_sp,
                        input_special=input_sp,
            #opti_special='Adam',
            #scaling_special='standardization',
                        DIP_special=True)
'''


# ADMMLim
initialALL()
specialTask(method_special='ADMMLim',
            inner_special=50,
            outer_special=70,
            alpha_special=[1])  # mu= 2, tau = 100
moveALL()
'''
specialTask(method_special='ADMMLim',
            inner_special=50,
            outer_special=70,
            alpha_special=tuners.adaptiveAlphas8)  # mu= 10, tau = 2
moveALL()

specialTask(method_special='ADMMLim',
            inner_special=50,
            outer_special=70,
            alpha_special=tuners.adaptiveAlphas9)  # mu= 10, tau = 2
moveALL()

specialTask(method_special='ADMMLim',
            inner_special=50,
            outer_special=70,
            alpha_special=tuners.adaptiveAlphas10)  # mu= 10, tau = 2
moveALL()
'''

# nested
'''
rhos = [0.0001, 0.001, 0.01, 1]
# rhos = [0.1]
for rho in rhos:
    specialTask(method_special='nested',
                inner_special=65,
                outer_special=45,
                alpha_special=0.005,
                rho_special=rho,
                lr_special=0.07,
                sub_iter_special=200)

'''

