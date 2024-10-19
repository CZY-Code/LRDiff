import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss, LowrankLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():
    return transforms.ToPILImage()


def image_to_tensor():
    return transforms.ToTensor()


def gray_to_edge(image, sigma):
    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    return edge


class AdversarialLoss(nn.Module):
  r"""
  Adversarial loss
  https://arxiv.org/abs/1711.10337
  """

  def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
    r"""
    type = nsgan | lsgan | hinge
    """
    super(AdversarialLoss, self).__init__()
    self.type = type
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))

    if type == 'nsgan':
      self.criterion = nn.BCELoss()
    elif type == 'lsgan':
      self.criterion = nn.MSELoss()
    elif type == 'hinge':
      self.criterion = nn.ReLU()

  def patchgan(self, outputs, is_real=None, is_disc=None):
    if self.type == 'hinge':
      if is_disc:
        if is_real:
          outputs = -outputs
        return self.criterion(1 + outputs).mean()
      else:
        return (-outputs).mean()
    else:
      labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
      loss = self.criterion(outputs, labels)
      return loss

  def __call__(self, outputs, is_real=None, is_disc=None):
    return self.patchgan(outputs, is_real, is_disc)


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        #if opt["dist"]:
            #self.rank = torch.distributed.get_rank()
        #else:
            #self.rank = -1  # non dist training
        train_opt = opt["train"]
        
        # define network and load pretrained models
        self.model = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.model = DataParallel(self.model)
        self.load()

        
        
        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.lowRankExp = LowrankLoss(weight=0.1, ranklist=[3, 64, 64], retLoss=True)
            # self.loss_tri = nn.TripletMarginLoss().to(self.device)
            # self.adversarial_loss = AdversarialLoss(type = 'hinge').to(self.device)
            # self.weight = opt['train']['weight']
            
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (k,v) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            # self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

        else:
            self.lowRankExp = LowrankLoss(weight=0.1, ranklist=[3, 64, 64], retLoss=False)

    def feed_data(self, state, LQ, GT, mask=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        #if GT is not None: 
        self.state_0 = GT.to(self.device)  # GT
        if mask == None:
            self.mask = torch.ones_like(state).to(self.device)
        else:
            self.mask = mask.to(self.device) # mask
        # self.S_sde = S_sde
        # self.S_GT = S_GT.to(self.device)
        # self.S_LQ = S_LQ.to(self.device)


    def optimize_parameters(self, timesteps, sde=None):
        sde.set_mu(self.condition)

        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps) #按照常规解出最优解
        timesteps = timesteps.to(self.device)
        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps) #概率对数函数对x的梯度-得分函数
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps) #xt-\delta*x_i即x_{t-1}的期望
        # yt_1_expection = self.lowRankExp(yt_1_expection, yt_1_expection)
        self.optimizer.zero_grad()
        rankLoss = self.lowRankExp(yt_1_expection, yt_1_optimum, sde.thetas[timesteps], self.mask) #计算exp和optim的低秩解的mse损失
        loss = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask) + rankLoss
        # print(self.lowRankExp(yt_1_expection, yt_1_optimum).item())
        loss.backward()
        self.optimizer.step()
        # set log
        self.log_dict["loss"] = loss.item()
        self.log_dict["lowR"] = rankLoss.item()


    def test(self, sde=None, save_states=False, save_dir='save_dir'):
        sde.set_mu(self.condition)
        self.model.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states, save_dir=save_dir, lowRanker = None)
        # self.model.train() #这行有啥意义？？？

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, DistributedDataParallel):
            net_struc_str = "{} - {}".format(self.model.__class__.__name__, self.model.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info("Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt["path"]["pretrain_model"]
        if load_path is not None:
            print('load-------------------------------')
            logger.info("Loading model for G [{:s}] ...".format(load_path))
            self.load_network(load_path, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label, epoch=None, current_step=None):
        self.save_network(self.model, "TDMNet", iter_label)
        if epoch is not None and current_step is not None:
            self.save_training_state(epoch, current_step)
        
