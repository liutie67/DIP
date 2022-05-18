import os
import numpy as np
import matplotlib.pyplot as plt
from show_functions import fijii_np, find_nan, read_input_dim, input_dim_str_to_list

databasePATH_root = '/home/liutie/Documents/outputDatabase0'
dataFolderPath = '1000+admm+CT+skip=0+lr=...*18(Adam)'

subroot = os.getcwd() + '/data/Algo'
image = 'image0'
# load ground truth image
PETimage_shape_str = read_input_dim(subroot + '/Data/database_v2/' + image + '/' + image + '.hdr')
PETimage_shape = input_dim_str_to_list(PETimage_shape_str)

alpha = 0.084
inner_iter = 50
iteration = 1000

def show_one(epoch=0):

    replicates = 1
    skip = 0
    INPUT = 'CT'
    lr = 1
    # epoch = 999 # 788 and 797


    opti = 'Adam'  # 'LBFGS' 'Adam'
    scaling = 'standardization'  # normalization' 'standardization'
    filename = 'out_DIP_post_reco_epoch=' + str(epoch) + 'config_rho=0_lr=' + str(lr) + '_sub_i=' \
                           + str(iteration) + '_opti_='+opti+'_skip_=' + str(skip) + '_scali='+scaling+'_input=' + INPUT \
                           + '_d_DD=4_k_DD=32_sub_i=' + str(inner_iter) + '_alpha=' + str(alpha) + '_mlem_=False.img'
    path_img = databasePATH_root + '/' + dataFolderPath + '/24' + str(replicates) + '/' + filename

    # path_img = '/home/liutie/Documents/outputDatabase0/ADMMLim+alpha=...+i50+o70/config_rho=0_sub_i=50_alpha=0.06_mlem_=False/ADMM_64/0_12_it50.img'

    f = fijii_np(path_img, shape=PETimage_shape, type='<f')
    '''
    for i in range(112):
        for j in range(112):
            if f[i, j] < 0:
                f[i, j] = 0
    
    f = f/np.amax(f)
    
    f = f*500
    '''
    f = f
    plt.figure()
    plt.imshow(f, cmap='gray')
    plt.colorbar()

'''
# vmin=0, vmax=500
# path_img = '/home/liutie/Documents/outputDatabase0/ADMMLim+alpha=...+i50+o70/config_rho=0_sub_i=50_alpha=0.06_mlem_=False/ADMM_64/0_13_it50.img'

f = fijii_np(path_img, shape=PETimage_shape, type='<d')

plt.figure()
plt.imshow(-f, cmap='gray')
plt.colorbar()

# path_img = '/home/liutie/Documents/outputDatabase0/ADMMLim+alpha=...+i50+o70/config_rho=0_sub_i=50_alpha=0.06_mlem_=False/ADMM_64/0_14_it50.img'

f = fijii_np(path_img, shape=PETimage_shape, type='<d')

plt.figure()
plt.imshow(-f, cmap='gray')
plt.colorbar()
'''

show_one(607)

'''
show_one(200-1)
show_one(400-1)
show_one(600-1)
show_one(800-1)
show_one(1000-1)
'''
plt.show()
