import os
import numpy as np
import matplotlib.pyplot as plt
from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list, getShape

'''
open and save an .img image file to /home/liutie/Pictures.
'''

# absolute path
path_img = '/home/liutie/stageSTING/DIP/data/Algo/image0/replicate_1/MLEM/config_mlem_=False_post_=0/MLEM_it1000.img'
name = ''
for letter in path_img[::-1]:
    if letter != '/':
        name += letter
    elif letter == '/':
        name = name[::-1]
        break

x_out = fijii_np(path_img, shape=getShape(), type='<d')

plt.figure(1)
plt.imshow(x_out, cmap='gray_r', vmin=0, vmax=500)
plt.title(name)
plt.colorbar()

plt.savefig('/home/liutie/Pictures/' + name + '.png')
plt.clf()
