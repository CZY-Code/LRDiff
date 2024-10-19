import torch 
import torch.nn as nn
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import parafac #CP Decomposition
from tensorly.decomposition import tucker, non_negative_tucker #Tucker Decomposition
from tensorly.tucker_tensor import tucker_to_tensor
from torch.nn.parallel import DataParallel, DistributedDataParallel
import math
import matplotlib.pyplot as plt

# random_state = 12345
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=2): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class TuckerNetwork(nn.Module):
    def __init__(self, tucker_rank, mid_channel):
        super(TuckerNetwork, self).__init__()
        r_1, r_2, r_3 = tucker_rank
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))
        self.C_net = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(8),
                                   nn.Tanh(),
                                   nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                #    nn.BatchNorm2d(32),
                                #    nn.Tanh(),
                                #    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                   )
        # self.centre = nn.parameter(torch.randn(r_1, r_2, r_3))
        # stdv = 1 / math.sqrt(self.centre.size(0))
        # self.centre.data.uniform_(-stdv, stdv)

    def forward(self, img, U_input, V_input, W_input):
        U = self.U_net(U_input)
        V = self.V_net(V_input)
        W = self.W_net(W_input)
        centre = self.C_net(img.unsqueeze(0)).squeeze(0)
        output = torch.einsum('ijk, hj, wk, ci -> chw', centre, U, V, W)
        # torch.testing.assert_close(Foutput, FO, check_stride=False)
        return output


class TuckerModel(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.tucker_rank = opt['TDNet']['ranks'] #[128, 128, 16]
        self.algo = opt['TDNet']['algo']
        self.mid_channel = opt['TDNet']['mid_channel']
        if self.algo == 'NN':
            batch_size = opt['batch_size']
            self.HW, self.C = opt['GT_size'], 3
            self.net = nn.ModuleList([TuckerNetwork(self.tucker_rank, mid_channel=self.mid_channel) 
                                      for _ in range(batch_size)]).to(self.device) #8个模型
    
    def forward(self, LQImages, GTImages, mask): #输出的范围为[0,1]
        if self.algo == 'HOSVD': #产生的值范围较大，且不同初始化条件下的输出不同
            cores, Ufactors, Vfactors, Wfactors, GTcores = [], [], [], [], []
            for LQimage, GTimage in zip(LQImages, GTImages):
                # core, tucker_factors = non_negative_tucker(image, rank=self.tucker_rank, init='svd', tol=10e-5, random_state=None)
                core, tucker_factors = tucker(LQimage, rank=self.tucker_rank, init='svd', tol=10e-5, random_state=None)
                U, V, W = tucker_factors #[In, Rn] U^T @ U = I
                #factor matrix必须要有正交的性质
                GTcore = tucker_to_tensor(tucker_tensor=(GTimage, (U.transpose(0,1), V.transpose(0,1), W.transpose(0,1))))
                # torch.testing.assert_close(GTcore, core, check_stride=False)
                cores.append(core)
                Ufactors.append(U)
                Vfactors.append(V)
                Wfactors.append(W)
                GTcores.append(GTcore)

            cores = torch.stack(cores)
            Ufactors = torch.stack(Ufactors)
            Vfactors = torch.stack(Vfactors)
            Wfactors = torch.stack(Wfactors)
            GTcores = torch.stack(GTcores)
            # plt.hist((GTcores).numpy().flatten(), bins=100, density=True, alpha=0.6, color='b')
            # plt.title('Probability Distribution of Tensor Elements')
            # plt.xlabel('Value')
            # plt.ylabel('Probability')
            # plt.show()
            # exit(0)
            return cores, Ufactors, Vfactors, Wfactors, GTcores
        
        elif self.algo == 'NN':
            U_input = torch.arange(1, self.HW+1, dtype = torch.float).view(self.HW, 1).to(self.device)
            V_input = torch.arange(1, self.HW+1, dtype = torch.float).view(self.HW, 1).to(self.device)
            W_input = torch.arange(1, self.C+1, dtype = torch.float).view(self.C, 1).to(self.device)
            LQImages = LQImages.to(self.device)
            mask = mask.to(self.device)
            outputs = []
            for i, LQimage in enumerate(LQImages):
                outputs.append(self.net[i](LQimage, U_input, V_input, W_input))
            
            outputs = torch.stack(outputs, dim=0)
            normLoss = torch.norm(outputs*mask - LQImages, 2)
            return outputs, normLoss
            

        