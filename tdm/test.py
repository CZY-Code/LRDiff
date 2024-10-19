import argparse
import logging
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
import cv2
import options as option
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorly.tucker_tensor import tucker_to_tensor
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import str_utils as str_util
import utils as util
from data.util import bgr2ycbcr
from models import create_model, create_td_model
from data import create_dataloader, create_dataset


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if (mp.get_start_method(allow_none=True) != "spawn"): #Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True) #'spawn' is the default on Windows
    rank = int(os.environ["RANK"]) #system env process ranks
    num_gpus = torch.cuda.device_count() #Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs) #Initializes the default distributed process group


def fft_compute_color(img_col, center=False):
        assert img_col.shape[0]!=1, "Should be color image"
        c, h, w = img_col.shape
        idx_list_ = []
        x_mag = np.zeros((c, h, w))
        x_phase = np.zeros((c, h, w))
        x_fft = []
        for i in range(c):
            img = img_col[i]
            dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
            if center:
                dft = np.fft.fftshift(dft)
            mag = cv2.magnitude(dft[:,:,0],dft[:,:,1])#np.abs(dft[:,:,0])#
            idx = (mag==0)
            mag[idx] = 1.
            magnitude_spectrum = np.log(mag)
            phase_spectrum = cv2.phase(dft[:,:,0],dft[:,:,1])
            x_mag[i] = magnitude_spectrum
            x_phase[i] = phase_spectrum
            idx_list_.append(idx)

        return x_fft, x_mag, x_phase, idx_list_

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="./tdm/options/test/ir-sde-td.yml", type=str, help="Path to option YMAL file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    util.mkdirs((
        path for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key)
    )
    # os.system("rm ./results")
    # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./results")
    
    # Config loggers. Before it, the log will not work
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    
    # mask_root = opt['degradation']['mask_root']
    # train_set_mask = Datasetset_mask(mask_root)
    
    #### create train and val dataloader
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        # train_loader_mask = DataLoader(train_set_mask, batch_size=1, shuffle=True)
        logger.info("Number of test images in [{:s}]: {:d}".format(dataset_opt["name"], len(test_set)))
  

    #### create model
    model = create_model(opt)
    # td_model = create_td_model(opt) #no need to train 
    device = model.device

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
       
    # S_sde = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    # S_sde.set_model(model.models)
    
    test_times = []
    test_PSNR = []
    test_SSIM = []
    # mask_iterator = iter(train_loader_mask)
    for train_data in tqdm(test_loader):
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        logger.info("\nTesting [{:s}]...".format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
        util.mkdir(dataset_dir)

        # need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = train_data["GT_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        Y_GT, Y_LQ, mask = train_data["GT"], train_data["LQ"], train_data["Mask"]

        # GT_path, X_GT, X_LQ = train_data["GT_path"], train_data["GT_gray"], train_data["GT_edge"] #completed grayscale and edge images
        
        dataset_dir = os.path.join(dataset_dir)
        util.mkdir(dataset_dir)
        
        # load mask information 在测试的时候单独加载mask可能导致每次测试的输入不一致
        # try:
        #     mask = next(mask_iterator)
        # except StopIteration:
        #     mask_iterator = iter(train_loader_mask)
        #     mask = next(mask_iterator)

        # LQ_ = util.tensor2img((Y_GT*mask).squeeze())
        # # cores, Ufactors, Vfactors, Wfactors, GTcores = td_model(Y_GT * mask, Y_GT)
        # output = testTDmodel(opt, Y_GT, mask).detach() #[1, 3, 256, 256]
        # output = util.tensor2img(output.squeeze())
        # mask4pred = mask.squeeze().unsqueeze(-1).cpu().detach().numpy()
        # output = output * (1-mask4pred) + LQ_ * mask4pred
        # GT_ = util.tensor2img(Y_GT.squeeze())       
        # test_PSNR.append(util.calculate_psnr(GT_, output))
        # test_SSIM.append(util.calculate_ssim(GT_, output))

        noisy_state = sde.noise_state(Y_LQ)
        model.feed_data(noisy_state, Y_LQ, Y_GT) #state, LQ, GT
        tic = time.time()
        model.test(sde, save_states=False, save_dir=dataset_dir)
        toc = time.time()
        test_times.append(toc - tic)
        visuals = model.get_current_visuals()

        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        # mask4pred = torch.where(mask>=1.0, 1.0, 0.0)
        # output = visuals["Output"] * (1-mask4pred) + visuals["Input"] * mask4pred
        output = visuals["Output"]
        output = util.tensor2img(output.squeeze())  # uint8
        test_PSNR.append(util.calculate_psnr(GT_, output))
        test_SSIM.append(util.calculate_ssim(GT_, output))

        if opt["suffix"]:
            save_img_path = os.path.join(dataset_dir, img_name + opt["suffix"] + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + "_f.png")
        util.save_img(output, save_img_path)
        LQ_img_path = os.path.join(dataset_dir, img_name + "_l.png")
        util.save_img(LQ_, LQ_img_path)
        GT_img_path = os.path.join(dataset_dir, img_name + "_r.png")
        util.save_img(GT_, GT_img_path)
    
    print(f"PSNR: {np.mean(test_PSNR)}, SSIM: {np.mean(test_SSIM)}")


def testTDmodel(opt, Y_GT, mask):
    td_model = create_td_model(opt) #8个模型
    params = []
    params += [x for x in td_model.parameters()]
    w_decay = 3
    lr_real = 0.0001
    optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay)
    Y_Outs = None
    for iter in range(opt['TDNet']['max_iter']):
        Y_Outs, normLoss = td_model(Y_GT*mask, Y_GT, mask) #Y_GT没有用到
        optimizier.zero_grad()
        normLoss.backward(retain_graph=True)
        optimizier.step()
        if iter % 100 == 0:
            ps = peak_signal_noise_ratio(np.clip(Y_GT.cpu().detach().numpy(),0,1), Y_Outs.cpu().detach().numpy())
            print('iteration:',iter,'PSNR',ps)
    return Y_Outs


from PIL import Image
class Datasetset_mask(Dataset):
    """The class to load the dataset"""
    def __init__(self, THE_PATH):
        data = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data.append(osp.join(root, name))
                
        self.data = data   
        print("mask dataset length: {}".format(len(self.data)))
        self.image_size = 256

        self.transform = transforms.Compose([
        	transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
        	transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        print(path)
        mask = self.transform(Image.open(path).convert('1'))
        return 1 - mask   #  0 is masked, 1 is unmasked


if __name__ == "__main__":
    import os
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
