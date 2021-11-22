from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import scipy.io as scio
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()

# Loading data
class LFDataset(Dataset):
    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        dataSet = scio.loadmat(opt.dataPath) 
        self.LFSet = dataSet['lf']  
        self.flowSet = dataSet['flow']  
        self.warped_flow_set = dataSet['flow_warped']  
        self.lfNameSet = dataSet['LF_name'] 
    
    def __getitem__(self, idx):
        LF=self.LFSet[idx] 
        flow=self.flowSet[idx] 
        warped_flow=self.warped_flow_set[idx]
        lfName=''.join([chr(self.lfNameSet[idx][0][0][i]) for i in range(self.lfNameSet[idx][0][0].shape[0])]) 
        LF= torch.from_numpy(LF.astype(np.float32)/255)
        flow= torch.from_numpy(flow.astype(np.float32))
        warped_flow= torch.from_numpy(warped_flow.astype(np.float32))
        sample = {'LF':LF, 'flow':flow, 'warped_flow':warped_flow, 'lfName':lfName}
        return sample
        
    def __len__(self):
        return self.LFSet.shape[0]



