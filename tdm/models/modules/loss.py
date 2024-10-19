import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import tucker #Tucker Decomposition
from tensorly.tucker_tensor import tucker_to_tensor
from torch_dct import dct, idct

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, mask, weights=None):
       
        lossm = self.loss_fn(predict * (1 - mask), target * (1 - mask), reduction='none')
        lossm = einops.reduce(lossm, 'b ... -> b (...)', 'mean')
        
        lossu = self.loss_fn(predict * mask, target * mask, reduction='none')
        lossu = einops.reduce(lossu, 'b ... -> b (...)', 'mean')

        loss = lossu + 10 * lossm
        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class LowrankLoss(nn.Module):
    def __init__(self, weight, ranklist, retLoss=True):
        super().__init__()
        self.weight = weight
        self.ranklist = ranklist
        self.retLoss = retLoss

    def soft_threshold(self, x, lambd):
        return torch.sign(x) * torch.max(torch.abs(x) - lambd, torch.tensor(0.0, device=x.device))
    
    def forward(self, predicts, targets, theta_ts, masks):
        lossList = []
        outputs = []
        for predict, target, theta_t, mask in zip(predicts, targets, theta_ts, masks):
            core_targ, tucker_factors = tucker(target, rank=self.ranklist, init='svd', tol=10e-5, random_state=None)
            U, V, W = tucker_factors #[In, Rn] U^T @ U = I
            # core_pred = tucker_to_tensor(tucker_tensor=(predict, (U.T, V.T, W.T))) #[3,64,64]
            core_targ = self.soft_threshold(core_targ, theta_t/2 * self.weight) #XXX 随手给的
            lowRankTar = tucker_to_tensor(tucker_tensor=(core_targ, (U, V, W)))
            if self.retLoss:
                # lossList.append(F.l1_loss(core_pred, core_targ))
                lossList.append(F.l1_loss(predict * (1 - mask), lowRankTar * (1 - mask)))
            else:
                outputs.append(tucker_to_tensor(tucker_tensor=(core_targ, (U, V, W))))
        if self.retLoss:
            return torch.mean(torch.stack(lossList)) * self.weight
        else:
            return torch.stack(outputs, dim=0)
        
        