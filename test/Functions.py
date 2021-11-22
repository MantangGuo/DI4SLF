from __future__ import print_function, division
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import math
import warnings
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()

          


def CropLF(lf, r_patch_size, range_disp): 
    b,u,v,x,y=lf.shape
    lfStack=[]
    num_x = 0
    for i in range(0, x-r_patch_size,r_patch_size):
        num_x = num_x + 1
        num_y = 0
        for j in range(range_disp//2, y-(r_patch_size+range_disp//2), r_patch_size):
            num_y = num_y + 1
            lf_patch = lf[:,:,:,i:i+r_patch_size, j - range_disp//2 : j + (r_patch_size+range_disp//2)]
            lfStack.append(lf_patch)
    lfStack = torch.stack(lfStack)
    lfStack = lfStack.permute(1,2,3,4,5,0)
    return  lfStack,[num_x,num_y] 


def MergeLF(lfStack,coordinate):
    b,u,v,patch_size,patch_width,n = lfStack.shape
    lfStack = lfStack.reshape(b,u,v,patch_size,patch_width, coordinate[0], coordinate[1]) 
    lfStack = lfStack.permute(0,1,2,5,3,6,4)
    lfMerged  = lfStack.reshape(b,u,v,coordinate[0]*patch_size,coordinate[1]*patch_width)
    return lfMerged 
    
def ComptPSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def Warping(disp, ind_source, ind_target, img_source, an):
    an2 = an*an
    N,h,w = img_source.shape
    ind_source = torch.tensor(ind_source, dtype = torch.float32)
    ind_target = torch.tensor(ind_target, dtype = torch.float32)
    ind_h_source = torch.floor(ind_source / an )
    ind_w_source = ind_source % an
    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = Variable(torch.arange(0,w).view(1,1,w).expand(N,h,w)).type_as(img_source) 
    YY = Variable(torch.arange(0,h).view(1,h,1).expand(N,h,w)).type_as(img_source)
    grid_w = XX + disp * torch.abs(ind_w_target - ind_w_source)
    grid_h = YY + disp * torch.abs(ind_h_target - ind_h_source)
    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) 

    # inverse warp
    img_source = torch.unsqueeze(img_source,1)
    img_target = torch.nn.functional.grid_sample(img_source,grid) 
    img_target = img_target.reshape(N,h,w) 
    return img_target