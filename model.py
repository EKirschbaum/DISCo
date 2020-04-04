import torch
import torch.nn as nn
from inferno.extensions.models.unet import UNet

class DISCoNet(nn.Module):
    def __init__(self, device):
        super(DISCoNet, self).__init__()
        self.device = device
        self.dtype = torch.float
        
        
        corr_channels = 15 # correlations to C=15 neighboring pixels
        combi_channels = corr_channels + 1 # combination of correlations and summary image
        out_channels = 6 # affinities to A=5 neighboring pixels + foreground-prediction
        
        # components of first CNN    
        self.conv1 = nn.Conv3d(corr_channels,2*corr_channels,(4,3,3),padding=(0,1,1))
        self.conv2 = nn.Conv3d(2*corr_channels,4*corr_channels,(4,3,3),padding=(0,1,1))
        self.conv3 = nn.Conv3d(4*corr_channels,corr_channels,(4,3,3),padding=(0,1,1))
        self.relu = nn.ReLU()
        
        # second CNN        
        self.unet = UNet(in_channels=combi_channels,out_channels=out_channels,
                         dim=2,depth=5,initial_features=64,gain=2,
                         final_activation=nn.Sigmoid(),p_dropout=None).to(self.device)
                             
       
    def forward(self,corrs,summ):
        
        # pass segment-wise correlations through first CNN
        corrs = self.relu(self.conv1(torch.transpose(corrs,1,2)))
        corrs = self.relu(self.conv2(corrs))
        corrs = self.relu(self.conv3(corrs))[:,:,0]
            
        # combine output from first CNN and summary image
        combi_input = torch.cat([summ,corrs],dim=1)
            
        # pass combined input through second CNN
        out = self.unet(combi_input)

        return(out)
