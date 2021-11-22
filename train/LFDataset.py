import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np
import random
from Functions import ExtractPatch, Warping

# Loading data
class LFDataset(Dataset):
    """Light Field data_set."""

    def __init__(self, opt):

        super(LFDataset, self).__init__()     
        data_set = h5py.File(opt.data_path) 
        self.lf_set = data_set.get('lf')[:].transpose(4,3,2,1,0)  
        self.flow_set = data_set.get('flow')[:].transpose(4,3,2,1,0)  
        self.warped_flow_set  = data_set.get('flow_warped')[:].transpose(5,4,3,2,1,0)  
        self.disp_set  = data_set.get('disparity')[:].transpose(4,3,2,1,0)  

        self.batch_size = opt.batch_size
        self.ind_input_view = opt.ind_input_view
        self.patch_size = opt.refined_patch_size + opt.range_disp-1
        self.r_patch_size = opt.refined_patch_size
        self.ang_res_sparse = opt.ang_res_sparse
        self.ang_res_dense = opt.ang_res_dense
        self.range_disp = opt.range_disp
        


    def __getitem__(self, idx):
        
        idx = idx % self.lf_set.shape[0]
        lf = self.lf_set[idx,:,:,:,:] 
        flow = self.flow_set[idx] 
        warped_flow = self.warped_flow_set[idx] 
        disp = self.disp_set[idx,:,:,:,:] 

        ind_input_view = np.array(self.ind_input_view)-1
        ind_novel_view = np.delete(np.arange(self.ang_res_dense),ind_input_view)

        lf_sparse, flow_sparse, patch_left, patch_right, label, disp = ExtractPatch(lf, flow, warped_flow, disp, ind_input_view, ind_novel_view, self.range_disp, self.r_patch_size) 
        
        
        lf_sparse = torch.from_numpy(lf_sparse.astype(np.float32)/255) 
        flow_sparse = torch.from_numpy(flow_sparse.astype(np.float32)) 
        patch_left = torch.from_numpy(patch_left.astype(np.float32)/255) 
        patch_right = torch.from_numpy(patch_right.astype(np.float32)/255) 
        label = torch.from_numpy(label[:,:,self.range_disp//2:self.range_disp//2+self.r_patch_size].astype(np.float32)/255) 
        disp = torch.from_numpy(disp[:,:,self.range_disp//2:self.range_disp//2+self.r_patch_size].astype(np.float32)) 
        
        warped_lf_sparse = torch.zeros(self.ang_res_sparse, self.r_patch_size, self.patch_size).type_as(lf_sparse) 
        for k_s in range(0,self.ang_res_sparse): 
            ind_s = ind_input_view[::-1][k_s]
            ind_t = ind_input_view[k_s] 
            dispMap = flow_sparse[k_s] / (self.ang_res_dense-1)
            warped_lf_sparse[k_s] = torch.squeeze(Warping(dispMap.unsqueeze(0), ind_s, ind_t, lf_sparse[self.ang_res_sparse-1-k_s].unsqueeze(0), self.ang_res_dense))

        sample = {'lf_sparse':lf_sparse, 'flow_sparse':flow_sparse, 'warped_lf_sparse':warped_lf_sparse, 'patch_left':patch_left,'patch_right':patch_right, 'label':label, 'disp':disp}
        return sample
        
    def __len__(self):
        return max(self.lf_set.shape[0], self.batch_size)



