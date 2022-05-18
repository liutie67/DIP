from main import specialTask



# MLEM
'''
specialTask(method_special='MLEM', max_iter=1000, nb_subsets=1, mlem_sequence=False)

'''
# DIP(Gong)
lrs_sp = [0.5, 1, 2, 3, 4]
for lr_sp in lrs_sp:
    specialTask(method_special='Gong',
                sub_iter_special=1000,
                lr_special=lr_sp,
                skip_special=0,
                input_special='CT',
                opti_special='LBFGS',
                scaling_special='normalization',
                DIP_special=True)
