import torch.nn as nn

class MlpNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(MlpNet, self).__init__()
        self.an_sparse = opt.ang_res_sparse
        self.N = opt.refined_patch_size * opt.refined_patch_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

 
        self.weightAgg = nn.Sequential(
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 1)
        )


        self.confFeatAgg = nn.Sequential(
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67),
            nn.ReLU(inplace=True),
            nn.Linear(67, 67)
        )

        self.confAgg = nn.Sequential(
            nn.Linear(opt.range_disp*67, 1)
        )

    def forward(self, concatedFeatures):
        _,range_disp,_ = concatedFeatures.shape
        concatedFeatures = concatedFeatures.reshape(-1,67)    

        weight = self.weightAgg(concatedFeatures) 

        confs = self.confFeatAgg(concatedFeatures) 
        confs = self.confAgg(confs.reshape(-1,range_disp*67)) 
        confs = confs.reshape(-1,self.an_sparse,self.N)
        confs = self.softmax(confs)
        return weight, confs























































