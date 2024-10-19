'''
还没写完，有必要研究Tucker分解的傅里叶形式，
因为图像和掩码X的hadamard积的傅里叶变换应该转换为什么？？
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.fftpack as fp
import scipy
import math
from skimage.metrics import peak_signal_noise_ratio
import cv2

################### 
# Here are the hyperparameters. 
w_decay = 3
lr_real = 0.0001
max_iter =  10001
down = [2,2,1]
omega = 2
###################
# 固定随机种子
torch.manual_seed(1047)
np.random.seed(1047)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=omega): 
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

class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3, mid_channel):
        super(Network, self).__init__()
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))
        
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))
        self.C_net = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.Tanh(),
                                   nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                #    nn.BatchNorm2d(32),
                                #    nn.Tanh(),
                                #    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                   )

    def forward(self, img, U_input, V_input, W_input):
        U = self.U_net(U_input)
        V = self.V_net(V_input)
        W = self.W_net(W_input)
        centre = self.C_net(img.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
        
        # FU = torch.fft.fftn(U)
        # FV = torch.fft.fftn(V)
        # FW = torch.fft.fftn(W)
        # Foutput = torch.einsum('abc,ib,jb,kc -> ijk', centre, FU, FV, FW)

        # Foutput_s = torch.fft.fftshift(Foutput, dim=(-3,-2))
        # output_mag = 20 * torch.log10(0.01+torch.abs(Foutput_s))
        # output_phase = torch.angle(Foutput_s)
        # output_fi = self.freq2img(output_mag, output_phase)
        
        output = torch.einsum('abc,ib,jb,kc -> ijk', centre, U, V, W)
        # torch.testing.assert_close(Foutput, FO, check_stride=False)

        return output
    
    def freq2img(self, img_mag, img_phase):
        freq2 = (10 ** (img_mag / 20) - 0.01) * torch.exp(1j * img_phase)
        freq = torch.fft.ifftshift(freq2, dim=(-3,-2))
        imiFFT = torch.fft.ifft2(freq, dim=(-3,-2)).real
        return imiFFT

def calc_snr(img, axis=0, ddof=0): # 计算信噪比
    a = np.asanyarray(img)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)

def DrawFFT():
    #傅里叶变换
    file_name = 'data/LennaTest.png'
    img = cv2.imread(file_name, 0)
    # mask = np.random.choice([True, False], size=img.shape, p=[0.5, 0.5])
    # img[mask] = 0

    freq = fp.fftn(img)
    im1 = fp.ifftn(freq).real  # 取实部重建图像
    freq2 = fp.fftshift(freq)  # 移位变换系数，使得直流分量在中间   
    img_mag = 20 * np.log10(0.1 + np.abs(freq2))
    img_phase = np.angle(freq2)
    snr = calc_snr(im1, axis=None)
    print('SNR for the image obtained after fft reconstruction is ' + str(snr))
    # Make sure the forward and backward transforms work!
    assert(np.allclose(img, im1))

    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original')
    plt.axis('off')
    plt.subplot(132), plt.imshow(im1, 'gray'), plt.title('ifft2')
    plt.axis('off')
    plt.subplot(133), plt.imshow(img_mag, 'gray'), plt.title('spectrum')
    plt.axis('off')
    plt.show()

def FFT4GRB(img):
    #灰度化
    # im = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    freq = fp.fft2(img, axes=(-3,-2)) #3D图像直接进行傅里叶变换
    freq2 = fp.fftshift(freq, axes=(-3,-2))  # 移位变换系数，使得直流分量在中间   
    img_mag = 20 * np.log10(0.01 + np.abs(freq2))
    img_phase = np.angle(freq2)
    
    # plt.subplot(211)
    # plt.imshow(img_mag.astype(np.uint8)), plt.title('input mag')
    # plt.subplot(212)
    # plt.imshow(img_phase.astype(np.uint8)), plt.title('input phase')
    # plt.show()

    # imiFFT = fp.ifft2(freq).real  # 取实部重建图像
    # assert(np.allclose(img, imiFFT))
    return img_mag, img_phase

def freq2RGB(img_mag, img_phase):
    freq2 = (10 ** (img_mag / 20) - 0.01) * np.exp(1j * img_phase)
    freq = np.fft.ifftshift(freq2)
    imiFFT = np.fft.ifft2(freq).real
    return imiFFT.astype(np.uint8)


def TVLoss(img, beta=2): #HWC
    H, W, C = img.size()
    count_h = (H-1) * W * C
    count_w = H * (W-1) * C
    h_tv = torch.pow((img[1:, :, :] - img[:-1, :, :]), 2).sum()
    w_tv = torch.pow((img[:, 1:, :] - img[:, -1:, :]), 2).sum()
    tvloss =  h_tv + w_tv
    return tvloss

class Gradient_Net(nn.Module):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x): #[H, W, C]
        grad_x = []
        grad_y = []
        for i in range(x.shape[2]):
            grad_x.append(F.conv2d(x[...,i].unsqueeze(0), self.weight_x)) #[1,510,510]
            grad_y.append(F.conv2d(x[...,i].unsqueeze(0), self.weight_y)) #[1,510,510]
        grad_x = torch.concat(grad_x, dim=0).mean(dim=0)
        grad_y = torch.concat(grad_y, dim=0).mean(dim=0)
        norm_x = self.nuclear_norm(grad_x)
        norm_y = self.nuclear_norm(grad_y)
        return norm_x + norm_y
  
    def nuclear_norm(self, matrix):
        _, singular_values, _ = torch.svd(matrix)
        norm = torch.sum(singular_values)
        return norm

def TesnorComplete():#
    n = 8
    sr = 0.5
    G_net = Gradient_Net()

    X_np = cv2.imread('data/Lenna.png')
    X_np = cv2.resize(X_np, (512, 512))
    maskSize = tuple(s//n for s in X_np.shape[:2])#n*n的mask
    mask4noise = np.random.choice([True, False], size=maskSize, p=[1-sr, sr])
    mask4noise = np.repeat(np.repeat(mask4noise, n, axis=0), n, axis=1)
    mask4noise = np.stack([mask4noise]*3, axis=2)
    X_np[mask4noise] = 0
    #FFT
    X_mag, X_phase = FFT4GRB(X_np)

    X_np = cv2.cvtColor(X_np, cv2.COLOR_RGB2BGR)
    X_np = np.array(X_np) / 255.0
    X = torch.from_numpy(X_np).type(dtype).cuda()

    Input_mag = torch.from_numpy(X_mag).type(dtype).cuda()
    Input_phase = torch.from_numpy(X_phase).type(dtype).cuda()

    mask = torch.ones(X.shape).type(dtype)
    mask[X == 0] = 0
    X[mask == 0] = 0

    [n_1,n_2,n_3] = X.shape #[512, 512, 3]
    #下面是秩的定义
    r_1 = int(n_1/down[0]) 
    r_2 = int(n_2/down[1])
    # r_3 = int(n_3/down[2])
    r_3 = 32
    mid_channel = int(n_2)

    gt_np = cv2.imread('data/Lenna.png')
    gt_np = cv2.resize(gt_np, (512, 512))
    gt_mag, gt_phase = FFT4GRB(gt_np)
    gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
    gt_np = np.array(gt_np) / 255.0
    gt = torch.from_numpy(gt_np).type(dtype).cuda()

    # img1 = freq2RGB(X_mag, X_phase) / 255.0
    # print('ori PSNR:', peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(),0,1), img1))
    
    # centre = torch.Tensor(r_1,r_2,r_3).type(dtype) #core tensor C
    # stdv = 1 / math.sqrt(centre.size(0))
    # centre.data.uniform_(-stdv, stdv)
    U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
    V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
    W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)

    model = Network(r_1,r_2,r_3, mid_channel).type(dtype)
    params = []
    params += [x for x in model.parameters()]
    print("Total number of parameters: {}".format(sum(p.numel() for p in params)))
    print("Total number of parameters: {:.2f} M".format(sum(p.numel() for p in params)/ 1e6))

    # centre.requires_grad=True
    # params += [centre]
    optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay) 
    
    psMax = 0
    for iter in range(max_iter):
        # X_Out = model(centre, U_input, V_input, W_input)
        X_Out = model(X, U_input, V_input, W_input)
        nuNorm = G_net(X_Out)
        normLoss = 1.00 * torch.norm(X_Out * mask - X * mask, 2)
            #  + 0.00 * torch.norm(output_mag - Input_mag, 2) \
            #  + 0.00 * torch.norm(output_phase - Input_phase, 2) \
            #  + 1.00 * torch.norm(X_Out * mask - output_fi * mask, 2)
        loss = normLoss + nuNorm
        optimizier.zero_grad()
        loss.backward(retain_graph=True)
        optimizier.step()
        if iter % 100 == 0:
        # if iter % (max_iter-1) == 0 and iter != 0:
            ps = peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(),0,1), X_Out.cpu().detach().numpy())
            print('iteration:',iter,'PSNR',ps)
            psMax = max(ps, psMax)
            print('PSNR MAX: ', psMax)
            continue
            if iter == (max_iter - 1):
                plt.figure(figsize=(15,45))
                
                show = [0,1,2] 
                plt.subplot(131)
                plt.imshow(np.clip(np.stack((gt[:,:,show[0]].cpu().detach().numpy(),
                                        gt[:,:,show[1]].cpu().detach().numpy(),
                                        gt[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('gt')

                plt.subplot(132)
                plt.imshow(np.clip(np.stack((X[:,:,show[0]].cpu().detach().numpy(),
                                        X[:,:,show[1]].cpu().detach().numpy(),
                                        X[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('input')

                plt.subplot(133)
                plt.imshow(np.clip(np.stack((X_Out[:,:,show[0]].cpu().detach().numpy(),
                                        X_Out[:,:,show[1]].cpu().detach().numpy(),
                                        X_Out[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('out')
                # plt.subplot(234), plt.imshow(gt_mag.astype(np.uint8)), plt.title('gt spectrum')
                # plt.subplot(235), plt.imshow(X_mag.astype(np.uint8)), plt.title('input spectrum')
                # plt.subplot(236), plt.imshow(output_mag.cpu().detach().numpy().astype(np.uint8)), plt.title('output spectrum')
                plt.show()

if __name__ == '__main__':
    TesnorComplete()
    # DrawFFT()