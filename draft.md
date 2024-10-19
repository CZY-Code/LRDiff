# code for HOSVD Reconstruction
```python
    visuals['Output'] = tucker_to_tensor(tucker_tensor=(visuals['Output'], (Ufactors[0], Vfactors[0], Wfactors[0])))
    visuals['GT'] = tucker_to_tensor(tucker_tensor=(visuals['GT'], (Ufactors[0], Vfactors[0], Wfactors[0])))
    visuals['Input'] = tucker_to_tensor(tucker_tensor=(visuals['Input'], (Ufactors[0], Vfactors[0], Wfactors[0])))
```

# test vanilla Tucker Net
```python
    LQ_ = util.tensor2img((Y_GT*mask).squeeze())
    # cores, Ufactors, Vfactors, Wfactors, GTcores = td_model(Y_GT * mask, Y_GT)
    output = testTDmodel(opt, Y_GT, mask).detach() #[1, 3, 256, 256]
    output = util.tensor2img(output.squeeze())
    mask4pred = mask.squeeze().unsqueeze(-1).cpu().detach().numpy()
    output = output * (1-mask4pred) + LQ_ * mask4pred
    GT_ = util.tensor2img(Y_GT.squeeze())       
    test_PSNR.append(util.calculate_psnr(GT_, output))
    test_SSIM.append(util.calculate_ssim(GT_, output))
```

# low rank loss function
```python
    class LowrankLoss(nn.Module):
    def __init__(self, weight, ranklist, retLoss=True):
        super().__init__()
        self.weight = weight
        self.ranklist = ranklist
        self.retLoss = retLoss

    def forward(self, predicts, targets, theta_ts):
        lossList = []
        outputs = []
        for predict, target, theta_t in zip(predicts, targets, theta_ts):
            core_targ, tucker_factors = tucker(target, rank=self.ranklist, init='svd', tol=10e-5, random_state=None)
            U, V, W = tucker_factors #[In, Rn] U^T @ U = I
            core_pred = tucker_to_tensor(tucker_tensor=(predict, (U.T, V.T, W.T))) #[3,64,64]
            fft_core_targ = dct(core_targ.permute(1,2,0)) #[64,64,3]
            for i in range(fft_core_targ.shape[-1]):
                #这里的超参数threshold给多少？\tau的值越大 解越偏向于低秩，反之越偏向于近似
                fft_core_targ[...,i] = tl.tenalg.proximal.svd_thresholding(fft_core_targ[...,i], threshold=1.0)
            core_targ = idct(fft_core_targ).permute(2,0,1)
            if self.retLoss:
                lossList.append(F.mse_loss(core_pred, core_targ))
            else:
                outputs.append(tucker_to_tensor(tucker_tensor=(core_targ, (U, V, W))))
        if self.retLoss:
            return torch.mean(torch.stack(lossList)) * self.weight
        else:
            return torch.stack(outputs, dim=0)
```

# Resize image in GT_dataset.py
```python
    else:
        img_GT = cv2.resize(np.copy(img_GT), (self.GT_size, self.GT_size), interpolation=cv2.INTER_LINEAR)
        img_LQ = cv2.resize(np.copy(img_LQ), (self.GT_size, self.GT_size), interpolation=cv2.INTER_LINEAR)
        img_Mask = cv2.resize(np.copy(img_Mask), (self.GT_size, self.GT_size), interpolation=cv2.INTER_NEAREST)
```