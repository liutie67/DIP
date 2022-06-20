import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type='<d'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

parser = argparse.ArgumentParser(description='Display absolute difference between 2 images')
parser.add_argument('--img1', type=str, dest='img1', help='first image .img')
parser.add_argument('--img2', type=str, dest='img2', help='second image .img')

args = parser.parse_args()

root = 'data/Algo/'
PETImage_shape = (112,112,1)

img1_np = fijii_np(args.img1, shape=(PETImage_shape))
print("min 1 = " ,np.min(img1_np))
print("mean 1 = " ,np.mean(img1_np))
print("max 1 = " ,np.max(img1_np))
img2_np = fijii_np(args.img2, shape=(PETImage_shape))
print("")
print("min 2 = " ,np.min(img2_np))
print("mean 2 = " ,np.mean(img2_np))
print("max 2 = " ,np.max(img2_np))

plt.figure()
#plt.imshow(np.abs(img1_np - img2_np), cmap='gray_r')
plt.imshow(img1_np - img2_np, cmap='bwr')#,vmin=-5000, vmax=5000)
plt.title('absolute difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'diff_img.png')

plt.figure()
plt.imshow(img1_np / img2_np, cmap='bwr')#,vmin=-70000, vmax=70000)
print('maximum of relative difference: ', np.abs(np.max(img1_np / img2_np)))
print('minimum of relative difference: ', np.abs(np.min(img1_np / img2_np)))

'''
for i in range(img1_np.shape[0]):
    for j in range(img1_np.shape[1]):
        if (np.abs(img1_np[i,j] / img2_np[i,j] > 100)):
            print("img1_np = ",img1_np[i,j])
            print("img2_np = ",img2_np[i,j])
'''
plt.title('relative difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'relative_diff_img.png')

MSE_normed = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])
print("MSE : ",MSE_normed)
print("Numerical error below threshold : ",MSE_normed < 1e-5)

plt.figure()
#plt.imshow(np.abs(img1_np), cmap='gray_r')
plt.imshow(img1_np, cmap='gray_r')
plt.title('img1')
plt.colorbar()
plt.savefig(root+'img1.png')

plt.figure()
#plt.imshow(np.abs(img2_np), cmap='gray_r')
plt.imshow(img2_np, cmap='gray_r')
plt.title('img2')
plt.colorbar()
plt.savefig(root+'img2.png')