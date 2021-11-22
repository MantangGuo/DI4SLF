import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from LFDataset import LFDataset
from Functions import SetupSeed,depth_grad_loss
from DeviceParameters import to_device
from MainNet import MainNet
import itertools,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import torchvision



# Training settings
parser = argparse.ArgumentParser(description="Learning Dynamic Interpolation for Extremely Sparse Light Fields with Wide Baselines")
parser.add_argument("--learningRate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--refined_patch_size", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--range_disp", type=int, default=161, help="The disparity range")
parser.add_argument("--ind_input_view", nargs='+', type=int, default=[1, 5], help="The index of input views (start index is '1'.)")
parser.add_argument("--ang_res_sparse", type=int, default=2, help="The angular resolution of the sparse LF")
parser.add_argument("--ang_res_dense", type=int, default=5, help="The angular resolution of the original LF")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
parser.add_argument("--summaryPath", type=str, default='./log/', help="Path for saving training log ")
parser.add_argument("--data_path", type=str, default='/path/to/training/data', help="Path for loading training data ")

opt = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Training.log')
log.addHandler(fh)
logging.info(opt)


if __name__ == '__main__':

    SetupSeed(1)
    savePath = './model/model_{}.pth'.format(opt.learningRate)
    lfDataset = LFDataset(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batch_size, shuffle=True, num_workers = 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model=MainNet(opt)
    to_device(model,device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %d; Training parameters: %d" %(total_params,total_trainable_params))

    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate) #optimizer
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochNum*0.8, gamma=0.1, last_epoch=-1)
    writer = SummaryWriter(opt.summaryPath)
    


    lossLogger = defaultdict(list)
    batch = 0
    min_loss = 100
    torch.autograd.set_detect_anomaly(True)
    

    for epoch in range(opt.epochNum):
        lossSum = 0
        for _,sample in enumerate(dataloader):
            batch = batch +1
            lf_sparse = sample['lf_sparse']
            flow_sparse = sample['flow_sparse']
            warped_lf_sparse = sample['warped_lf_sparse']
            patch_left = sample['patch_left']
            patch_right = sample['patch_right']
            label = sample['label']
            disp = sample['disp']

            lf_sparse = to_device(lf_sparse,device) 
            flow_sparse = to_device(flow_sparse,device) 
            warped_lf_sparse = to_device(warped_lf_sparse,device) 
            patch_left = to_device(patch_left,device) 
            patch_right = to_device(patch_right,device) 
            label = to_device(label,device) 
            disp = to_device(disp,device) 

            
            ref_patch_novel, patch_novel, left_novel, right_novel  = model( lf_sparse, flow_sparse, warped_lf_sparse, patch_left, patch_right)

            # construct predicted lf and label lf
            b,an_sparse,r_patch_size,_ = lf_sparse.shape
            _,an_novel,_,_ = label.shape
            ind_input_view = np.array(opt.ind_input_view)-1
            ind_novel_view = np.delete(np.arange(opt.ang_res_dense),ind_input_view)
            ref_lf = torch.zeros(b,an_sparse+an_novel,r_patch_size,r_patch_size).type_as(lf_sparse) 
            ref_lf[:,ind_input_view] = lf_sparse[:,:,:,opt.range_disp//2:opt.range_disp//2+r_patch_size]
            ref_lf[:,ind_novel_view] = ref_patch_novel
            label_lf = torch.zeros(b,an_sparse+an_novel,r_patch_size,r_patch_size).type_as(lf_sparse) 
            label_lf[:,ind_input_view] = lf_sparse[:,:,:,opt.range_disp//2:opt.range_disp//2+r_patch_size]
            label_lf[:,ind_novel_view] = label

            # loss
            loss = criterion(ref_patch_novel,label) + criterion(patch_novel,label) + criterion(left_novel,label) + criterion(right_novel,label) + 50*depth_grad_loss(ref_lf,label_lf,disp)
            lossSum += loss.item()
            
            writer.add_scalar('Loss', loss, batch)

            print("Epoch: %d Batch: %d Loss: %.6f" %(epoch,batch,loss.item()))
            
            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()

            
        torch.save(model.state_dict(),savePath)
        if lossSum/len(dataloader) <= min_loss:
            torch.save(model.state_dict(),'./model/model_{}_optimal.pth'.format(opt.learningRate))
            log.info("Epoch: %d Loss: %.6f is the optimal!" %(epoch,lossSum/len(dataloader)))
            min_loss = lossSum/len(dataloader)

        log.info("Epoch: %d Loss: %.6f" %(epoch,lossSum/len(dataloader)))
        scheduler.step()

        #Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(lossSum/len(dataloader))
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
        plt.savefig('Training_{}.jpg'.format(opt.learningRate))
        plt.close()
    

