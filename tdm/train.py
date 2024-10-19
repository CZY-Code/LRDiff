import argparse
import logging
import math
import os
import sys
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os.path as osp
import os
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from torch import optim
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

import options as option
import utils as util
from data.data_sampler import DistIterSampler
import str_utils as str_util
from models import create_model, create_td_model
from data import create_dataloader, create_dataset


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (mp.get_start_method(allow_none=True) != "spawn"): #Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  #'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs) # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="./tdm/options/train/ir-sde-td.yml", type=str, help="Path to option YMAL file.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]
    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        # Returns the number of processes in the current process group
        world_size = (torch.distributed.get_world_size())
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location = lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(opt["path"]["experiments_root"]) #rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
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
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                raise NotImplementedError('update your pytorch version!')
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=False)
        logger = logging.getLogger("base") 
    
    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None

            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

    assert train_loader is not None

    #### create model
    model = create_model(opt)
    device = model.device
    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model) #设置去噪网络 ConditionalUNet

    #### training
    logger.info("Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step))
    error = mp.Value('b', False)
    for epoch in tqdm(range(start_epoch, total_epochs)):
        
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        with tqdm(total=len(train_loader)) as pbar:
            for train_data in train_loader:
                current_step += 1

                if current_step > total_iters:
                    break

                Y_GT, Y_LQ, mask = train_data["GT"], train_data["LQ"], train_data["Mask"]
                # GT_path, X_GT, X_LQ = train_data["GT_path"], train_data["GT_gray"], train_data["GT_edge"] #completed grayscale and edge images
                
                # Y_Outs = trainTDmodel(opt, Y_GT, mask).detach()
                # Y_Outs = Y_GT * mask

                # timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_GT*mask) #timestep>2
                timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_LQ) #timestep>2
                model.feed_data(states, Y_LQ, Y_GT, mask=mask) # xt, mu, x0
                model.optimize_parameters(timesteps, sde)
                model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])

                if current_step % opt["logger"]["print_freq"] == 0:
                    logs = model.get_current_log()
                    message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                        epoch, current_step, model.get_current_learning_rate())
                    for k, v in logs.items():
                        message += "{:s}: {:.4e} ".format(k, v)
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

                if error.value:
                    sys.exit(0)
            
                pbar.set_postfix({'loss': f"{model.get_current_log()['loss']:.4f}",
                                  'lowR': f"{model.get_current_log()['lowR']:.4f}"})
                pbar.update()

        #### save models and training states
        logger.info("Saving models and training states.")
        model.save("new", epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


def trainTDmodel(opt, Y_GT, mask):
    #-------初始化---------
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
            transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
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
