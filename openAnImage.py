import os
import numpy as np
import matplotlib.pyplot as plt
from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list, getShape

'''
open and save an .img image file to /home/liutie/Pictures.
'''

# absolute path
path_img = '/home/liutie/Documents/outputDatabaseA/2022-07-27+09-31-46ADMM+initial+i100+o100+t128+a=alphas0+mu1+tau100+rep5+0/replicate_1/ADMMLim_new/Comparison/ADMMLim_new/config_rho=0_sub_i=100_alpha=0.005_mlem_=False/ADMM_128/0_100_it100.img'
name = ''
FULLCONTRAST = False
for letter in path_img[::-1]:
    if letter != '/':
        name += letter
    elif letter == '/':
        name = name[::-1]
        break

x_out = fijii_np(path_img, shape=getShape(), type='<d')

plt.figure(1)
if FULLCONTRAST:
    plt.imshow(x_out, cmap='gray_r')
else:
    plt.imshow(x_out, cmap='gray_r', vmin=-5000, vmax=5000)
plt.title(name)
plt.colorbar()

plt.savefig('/home/liutie/Pictures/' + name + '.png')
plt.clf()
