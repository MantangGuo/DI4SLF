from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
from LFDataset import LFDataset
from DeviceParameters import to_device
from MainNet import MainNet
from Functions import CropLF, MergeLF,ComptPSNR,Warping
from skimage.measure import compare_ssim 
import numpy as np
import scipy.io as scio 
import os, time
import logging,argparse

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Testing.log')
log.addHandler(fh)

# Testing settings
parser = argparse.ArgumentParser(description="Learning Dynamic Interpolation for Extremely Sparse Light Fields with Wide Baselines")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--refined_patch_size", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--range_disp", type=int, default=161, help="The disparity range")
parser.add_argument("--ind_input_view", nargs='+', type=int, default=[1, 5], help="The index of input views (start index is '1'.)")
parser.add_argument("--ang_res_sparse", type=int, default=2, help="The angular resolution of the sparse LF")
parser.add_argument("--ang_res_dense", type=int, default=5, help="The angular resolution of the original LF")
parser.add_argument("--modelPath", type=str, default='./model/model_0.0001.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='../data/demo_test_inria.mat', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='./testResults/', help="Path for saving results ")

opt = parser.parse_args()
logging.info(opt)

if __name__ == '__main__':

    # load data
    lf_dataset = LFDataset(opt)
    dataloader = DataLoader(lf_dataset, batch_size=opt.batch_size,shuffle=False)
    device = torch.device("cuda:0")

    # load model
    model=MainNet(opt)
    model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    to_device(model,device)


    with torch.no_grad():
        num = 0
        avg_psnr_y = 0
        avg_ssim_y = 0

        # index of input and novel views
        ind_inputViews = np.array(opt.ind_input_view)-1
        ind_novelViews = np.delete(np.arange(opt.ang_res_dense),ind_inputViews)

        # inference
        for _,sample in enumerate(dataloader):
            num=num+1
            sum = 0

            # initialization
            LF=sample['LF'].cuda() 
            flow=sample['flow'].cuda() 
            warped_flow=sample['warped_flow']
            lfName=sample['lfName']

            b,u,v,x,y = LF.shape
            r_patch_size = opt.refined_patch_size
            patch_size = opt.refined_patch_size + opt.range_disp-1
            range_disp = opt.range_disp


            
################################################################################################################################################################################################################
            start = time.time()
            # crop test lf into smaller patches
            LFStack, coordinate = CropLF(LF,r_patch_size, range_disp) 
            flowStack, _ = CropLF(flow,r_patch_size, range_disp) 
            warped_flow_stack, _ = CropLF(warped_flow.permute(0,5,1,2,3,4).reshape(b*opt.ang_res_sparse,u,v,x,y), r_patch_size, opt.range_disp) 
            warped_flow_stack = warped_flow_stack.reshape(b,opt.ang_res_sparse,u,v,r_patch_size,patch_size,-1) 
            n=LFStack.shape[5]          

            estiLFStack = torch.zeros(b,u,v,r_patch_size,r_patch_size,n,device="cuda").type_as(LF)
            patch_left = torch.zeros(b,len(ind_novelViews),r_patch_size,r_patch_size,device="cuda").type_as(LF)
            patch_right = torch.zeros(b,len(ind_novelViews),r_patch_size,r_patch_size,device="cuda").type_as(LF)

            # reconstruct novle views in each lf patch 
            for ind_n in range(n):
                for i in range(u):
                    sparseLF = LFStack[:,i,ind_inputViews,:,:,ind_n] 
                    sparseFlow = flowStack[:,i,:,:,:,ind_n] 

                    # warp input views
                    warpedSparseLF = torch.zeros(b,opt.ang_res_sparse, r_patch_size, patch_size).type_as(sparseLF) 
                    for k_s in range(0,opt.ang_res_sparse): 
                        ind_s = ind_inputViews[::-1][k_s]
                        ind_t = ind_inputViews[k_s] 
                        disp = sparseFlow[:,k_s] / (opt.ang_res_dense-1)
                        warpedSparseLF[:,k_s] = Warping(disp, ind_s, ind_t, sparseLF[:,opt.ang_res_sparse-1-k_s], opt.ang_res_dense)

                    # patch matching from input views
                    ind = 0
                    for j in list(ind_novelViews):                        
                        novel_x = patch_size//2
                        if j <= v//2:
                            novel_flow = warped_flow_stack[:,0,i,j,:,patch_size//2-r_patch_size//2:patch_size//2+r_patch_size//2,ind_n] 
                            # disp_mean =  torch.clamp(torch.squeeze(torch.mean(novel_flow,dim=(1,2))),min=-20, max=20).numpy() #[1]
                            disp_mean =  torch.squeeze(torch.mean(novel_flow,dim=(1,2))).numpy() #[1]

                            left_x = int(round(novel_x - disp_mean * abs(ind_inputViews[0] - j)))
                            patch_left[:,ind] = sparseLF[:,0,:,left_x-r_patch_size//2:left_x+r_patch_size//2] 

                            right_x = int(round(novel_x + disp_mean * abs(ind_inputViews[1] - j)))
                            patch_right[:,ind] = sparseLF[:,1,:,right_x-r_patch_size//2:right_x+r_patch_size//2] 
                        else:
                            novel_flow = warped_flow_stack[:,1,i,j,:,patch_size//2-r_patch_size//2:patch_size//2+r_patch_size//2,ind_n] 
                            # disp_mean =  torch.clamp(torch.squeeze(torch.mean(novel_flow,dim=(1,2))),min=-20, max=20).numpy() #[1]
                            disp_mean =  torch.squeeze(torch.mean(novel_flow,dim=(1,2))).numpy() #[1]

                            left_x = int(round(novel_x + disp_mean * abs(ind_inputViews[0] - j)))
                            patch_left[:,ind] = sparseLF[:,0,:,left_x-r_patch_size//2:left_x+r_patch_size//2] 

                            right_x = int(round(novel_x - disp_mean * abs(ind_inputViews[1] - j)))
                            patch_right[:,ind] = sparseLF[:,1,:,right_x-r_patch_size//2:right_x+r_patch_size//2] 
                        ind = ind + 1
                    
                    # synthesizeing novel views
                    estiLFStack[:,i,ind_novelViews,:,:,ind_n] = model(sparseLF, sparseFlow, warpedSparseLF, patch_left, patch_right)

            # merge the lf patches into whole lf
            estiLF = MergeLF(estiLFStack,coordinate) 
            _,_,_,x_croped,y_croped = estiLF.shape
            end = time.time()
            print('Inference: ' ,(end-start)/(u*(v-2)))
################################################################################################################################################################################################################
            estiLF = estiLF.cpu()
            LF = LF.cpu()
            LF=LF[:,:,:,0:x_croped,opt.range_disp//2:opt.range_disp//2+y_croped]

            # evaluation
            lf_psnr_y = 0
            lf_ssim_y = 0
            for i in range(u):
                for j in list(ind_novelViews):

                    lf_psnr_y += ComptPSNR(np.squeeze(estiLF[0,i,j].numpy()),
                                        np.squeeze(LF[0,i,j].numpy()))  / (u*len(ind_novelViews))
                                        
                    lf_ssim_y += compare_ssim(np.squeeze((estiLF[0,i,j].numpy()*255.0).astype(np.uint8)),
                                            np.squeeze((LF[0,i,j].numpy()*255.0).astype(np.uint8)),gaussian_weights=True,sigma=1.5,use_sample_covariance=False,multichannel=False) / (u*len(ind_novelViews))

            avg_psnr_y += lf_psnr_y / len(dataloader)           
            avg_ssim_y += lf_ssim_y / len(dataloader) 
            log.info('Index: %d  Scene: %s  PSNR: %.2f  SSIM: %.3f'%(num,lfName[0],lf_psnr_y,lf_ssim_y))
            
            # save results
            scio.savemat(os.path.join(opt.savePath,lfName[0]+'_view.mat'),{'lf_recons':torch.squeeze(estiLF).numpy()}) 
            

        log.info('Average PSNR: %.2f  SSIM: %.3f '%(avg_psnr_y,avg_ssim_y))            