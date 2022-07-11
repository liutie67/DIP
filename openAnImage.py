import os
import numpy as np
import matplotlib.pyplot as plt
from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list, getShape

path_img = '/home/liutie/stageSTING/DIP/data/Algo/Data/initialization/out_DIP_post_reco_epoch=311config_rho=0_lr=0.006_sub_i=1000_opti_=Adam_skip_=0_scali=standardization_input=CT_d_DD=4_k_DD=32_sub_i=50_alpha=0.084_mlem_=False.img'
name = ''
for letter in path_img[::-1]:
    if letter != '/':
        name += letter
    elif letter == '/':
        name = name[::-1]
        break

x_out = fijii_np(path_img, shape=getShape(), type='<f')

plt.figure(1)
plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
plt.title(name)
plt.colorbar()

plt.savefig('/home/liutie/Pictures/' + name + '.png')
plt.clf()
