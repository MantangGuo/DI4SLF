from __future__ import print_function, division
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import math
import warnings
import random
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()




def SetupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def ExtractPatch(lf, flow, warped_flow, disp, ind_input_views, ind_novel_views, range_disp, r_patch_size):

    u,v,x,y =lf.shape 
    patch_size = range_disp + r_patch_size-1

    ind_u = random.randrange(0,u)  
    ind_x = random.randrange(0,x-patch_size,8) 
    ind_y = random.randrange(0,y-patch_size,8)

    lf_sparse = lf[ind_u, ind_input_views, ind_x:ind_x+r_patch_size, ind_y:ind_y+patch_size] 
    sparse_flow = flow[ind_u, :, ind_x:ind_x+r_patch_size, ind_y:ind_y+patch_size] 
    warped_flow = warped_flow[ind_u, :, ind_x:ind_x+r_patch_size, ind_y:ind_y+patch_size,:] 
    disp = disp[ind_u, :, ind_x:ind_x+r_patch_size, ind_y:ind_y+patch_size] 

    novel_x = patch_size//2

    patch_left = np.zeros((len(ind_novel_views),r_patch_size,r_patch_size)) 
    patch_right = np.zeros((len(ind_novel_views),r_patch_size,r_patch_size)) 

    ind = 0
    for ind_v in ind_novel_views:

        if ind_v <= v//2:
            novel_flow = warped_flow[ind_v, :,range_disp//2:range_disp//2+r_patch_size,0] 
            disp_mean = np.mean(novel_flow) 

            left_x = int(round(novel_x - disp_mean * abs(ind_input_views[0] - ind_v))) # -/+ is depended on the flow/disp 
            patch_left[ind] = lf_sparse[0,:,left_x-r_patch_size//2:left_x+r_patch_size//2] 

            right_x = int(round(novel_x + disp_mean * abs(ind_input_views[1] - ind_v))) # -/+ is depended on the flow/disp 
            patch_right[ind] = lf_sparse[1,:,right_x-r_patch_size//2:right_x+r_patch_size//2] 
            
        else:
            novel_flow = warped_flow[ind_v, :,range_disp//2:range_disp//2+r_patch_size,1] 
            disp_mean = np.mean(novel_flow) 

            left_x = int(round(novel_x + disp_mean * abs(ind_input_views[0] - ind_v))) # -/+ is depended on the flow/disp 
            patch_left[ind] = lf_sparse[0,:,left_x-r_patch_size//2:left_x+r_patch_size//2] 

            right_x = int(round(novel_x - disp_mean * abs(ind_input_views[1] - ind_v))) # -/+ is depended on the flow/disp 
            patch_right[ind] = lf_sparse[1,:,right_x-r_patch_size//2:right_x+r_patch_size//2] 

        ind = ind + 1
    
    label =  lf[ind_u, ind_novel_views, ind_x:ind_x+r_patch_size, ind_y:ind_y+patch_size] 

    return lf_sparse, sparse_flow, patch_left, patch_right, label, disp

def Warping(disp, ind_source, ind_target, img_source, an):
    N,h,w = img_source.shape
    ind_source = torch.tensor(ind_source, dtype = torch.float32)
    ind_target = torch.tensor(ind_target, dtype = torch.float32)

    # coordinate for source and target
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

def depth_grad_loss(pred, label, disp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def depth_grad(lf_img, lf_disp):
        N, v, h, w = lf_img.shape
        w1 = torch.abs(lf_disp - torch.floor(lf_disp))
        w2 = torch.abs(lf_disp - torch.ceil(lf_disp))
        lf_disp_ceil = torch.ceil(lf_disp).type(torch.long)
        lf_disp_floor = torch.floor(lf_disp).type(torch.long)

        # lf[:,v+1,:,w+d]
        lf_dir3_0 = torch.zeros_like(lf_img)
        lf_dir3_0[:, :v-1] = lf_img[:, 1:].clone()
        w_grid = torch.arange(0, w).view(1, 1, 1, w).expand(lf_img.shape).to(device).type(torch.long)
        w_grid_1 = w_grid - lf_disp_ceil
        w_grid_1 = torch.clamp(w_grid_1, min=0, max=w-1)
        lf_dir3_1 = torch.gather(lf_dir3_0, dim=3, index=w_grid_1)
        w_grid_2 = w_grid - lf_disp_floor
        w_grid_2 = torch.clamp(w_grid_2, min=0, max=w-1)
        lf_dir3_2 = torch.gather(lf_dir3_0, dim=3, index=w_grid_2)
        lf_dir3 = w1 * lf_dir3_1 + w2 * lf_dir3_2

        # lf[:,v-1,:,w-d]
        lf_dir4_0 = torch.zeros_like(lf_img)
        lf_dir4_0[:, 1:] = lf_img[:, :v-1].clone()
        w_grid_1 = w_grid + lf_disp_ceil
        w_grid_1 = torch.clamp(w_grid_1, min=0, max=w-1)
        lf_dir4_1 = torch.gather(lf_dir4_0, dim=3, index=w_grid_1)
        w_grid_2 = w_grid + lf_disp_floor
        w_grid_2 = torch.clamp(w_grid_2, min=0, max=w-1)
        lf_dir4_2 = torch.gather(lf_dir4_0, dim=3, index=w_grid_2)
        lf_dir4 = w1 * lf_dir4_1 + w2 * lf_dir4_2

        lf_grad = lf_img - (lf_dir3 + lf_dir4)/2.
        lf_grad = lf_grad[:, 1:-1, 4:-4, 4:-4]

        return lf_grad
    L1Loss = torch.nn.L1Loss()
    grad_pred = depth_grad(pred, disp)
    grad_gt = depth_grad(label, disp)

    return L1Loss(grad_pred, grad_gt)


