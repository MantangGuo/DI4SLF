import torch
import matplotlib.pyplot as plt
import warnings
import numpy as np
from RefNet import FlowRefNet, ViewRefNet
from MlpNet import MlpNet

warnings.filterwarnings("ignore")
plt.ion()
       
        
# Main Network construction
class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()
        self.mlpNet = MlpNet(opt)
        self.flowRefNet = FlowRefNet(opt)
        self.viewRefNet = ViewRefNet(opt)
        self.range_disp = opt.range_disp
        self.r_patch_size = opt.refined_patch_size
        self.ind_input_view = opt.ind_input_view

    def forward(self, lf_sparse, flow_sparse, warped_lf_sparse, patch_left, patch_right):
        
        b,an_sparse,r_patch_size,patch_size = lf_sparse.shape
        _,an_novel,_,_ = patch_left.shape
        N = r_patch_size * r_patch_size

        ind_input_view = np.array(self.ind_input_view)-1
        ind_novel_view = np.delete(np.arange(an_sparse+an_novel),ind_input_view)

        lf_sparse = lf_sparse.reshape(-1,1,r_patch_size,patch_size) 
        flow_sparse = flow_sparse.reshape(-1,1,r_patch_size,patch_size) 
        warped_lf_sparse = warped_lf_sparse.reshape(-1,1,r_patch_size,patch_size) 

        # content embeddings
        feat_flow_sparse = self.flowRefNet(flow_sparse,lf_sparse,warped_lf_sparse) 

        # concate spatial and angular code to the content embeddings
        spatialCode = (torch.arange(self.range_disp)-self.range_disp//2).type_as(flow_sparse).reshape(1,1,self.range_disp,1).expand(b*an_novel*an_sparse,-1,-1,N) 
        ang_code = torch.tensor([ind_input_view - ind_novel_view[i] for i in range(len(ind_novel_view))]).type_as(flow_sparse) 
        ang_code = ang_code.reshape(1,an_novel,an_sparse,1,1,1).expand(b,-1,-1,-1,self.range_disp,N).reshape(b*an_novel*an_sparse,1,self.range_disp,N) 
        flow_sparse = torch.nn.functional.unfold(flow_sparse,kernel_size = (1,self.range_disp)).reshape(b,1,an_sparse,1,self.range_disp,N).expand(-1,an_novel,-1,-1,-1,-1).reshape(b*an_novel*an_sparse,1,self.range_disp,N) 
        feat_flow_sparse = torch.nn.functional.unfold(feat_flow_sparse,kernel_size = (1,self.range_disp)).reshape(b,1,an_sparse,64,self.range_disp,N).expand(-1,an_novel,-1,-1,-1,-1).reshape(b*an_novel*an_sparse,64,self.range_disp,N) 
        features = torch.cat([flow_sparse,feat_flow_sparse,spatialCode,ang_code],1) 

        # predict dynamic weights and confidences
        weight, confs = self.mlpNet(features.permute(0,3,2,1).reshape(b*an_novel*an_sparse*N,self.range_disp,67)) 
        weight = weight.reshape(b*an_novel,an_sparse,N,self.range_disp) 
        weight_left = torch.nn.functional.softmax(weight[:,0], dim = 2) 
        weight_right = torch.nn.functional.softmax(weight[:,1], dim = 2) 
        weight = torch.cat([weight_left.unsqueeze(1),weight_right.unsqueeze(1)],1)

        # reconstruct novel views
        sparseEPIs = torch.nn.functional.unfold(lf_sparse,kernel_size = (1,self.range_disp)) 
        sparseEPIs = sparseEPIs.reshape(b,1,an_sparse,self.range_disp,N).expand(b,an_novel,an_sparse,self.range_disp,N).reshape(b*an_novel,an_sparse,self.range_disp,N) 
        leftNovelView = torch.bmm( sparseEPIs.permute(0,3,1,2).reshape(b*an_novel*N,an_sparse,self.range_disp)[:,0:1,:], weight_left.reshape(b*an_novel*N,self.range_disp,1)) 
        rightNovelView = torch.bmm( sparseEPIs.permute(0,3,1,2).reshape(b*an_novel*N,an_sparse,self.range_disp)[:,1:2,:], weight_right.reshape(b*an_novel*N,self.range_disp,1)) 
        novelView = leftNovelView.reshape(b*an_novel,N) * confs[:,0,:] + rightNovelView.reshape(b*an_novel,N) * confs[:,1,:] 

        # geometry-based refinement
        patch_novel = novelView.reshape(b*an_novel,1,r_patch_size,r_patch_size) 
        patch_left = patch_left.reshape(b*an_novel,1,r_patch_size,r_patch_size)
        patch_right = patch_right.reshape(b*an_novel,1,r_patch_size,r_patch_size)
        ref_patch_novel = self.viewRefNet(patch_novel,patch_left,patch_right) 

        return ref_patch_novel.reshape(b,an_novel,r_patch_size,r_patch_size)
        
