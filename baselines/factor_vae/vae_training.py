import json
from itertools import combinations
from numbers import Number
from random import shuffle
from collections import Counter
import torch
import time
import logging
import argparse
import os
import math
import random
import numpy as np
import pickle
from pandas import DataFrame
from torch import nn

# index is the fold number, revise it to train on index = 0, 1, 2, 3, 4
index = 0 

from dataloader import *
from transform import *

#data preprocessing
with open('../../inputs_new.pickle','rb') as handle:
    inputs = pickle.load(handle)

#change column name orlogid_encoded to orlogid
inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)

train_data = pickle.loads(open('./preops/X_sampled_non_cardiac.pickle', 'rb').read())
train_outcome = pickle.loads(open('./preops/outcome_data_non_cardiac.pickle', 'rb').read())
types = pickle.loads(open('./preops/surgery_type_non_cardiac.pickle', 'rb').read())
surgery_types = types.to_numpy()


outcomes_name = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']

outcomes = train_outcome[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()

#convert to binary, if >0, then 1, else 0
outcomes = np.where(outcomes > 0, 1, 0)

folder = ('./preops_cv/')
outcome = 'arrest'
foldername = folder+outcome+'/'
train_data2 = pickle.load(open(foldername+'X_train_'+str(index)+'.pickle', 'rb'))
train_data = np.concatenate((train_data, train_data2), axis=0)
train_data = np.delete(train_data, 163, axis=1)
train_ids2 = pickle.load(open(foldername+'outcome_data_train_ids_'+str(index)+'.pickle', 'rb'))
train_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(train_ids2))].copy()
outcomes2 = train_outcome2[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
#convert to binary, if >0, then 1, else 0
outcomes2 = np.where(outcomes2 > 0, 1, 0)
outcomes = np.concatenate((outcomes, outcomes2), axis=0)

#types2 is of length of train_data2 and all 2.0
types2 = np.full((train_data2.shape[0],), 2.0)

surgery_types = np.concatenate((surgery_types, types2), axis=0)

import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from vae_model import VAE_attention, Discriminator
from dataset import return_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)




def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.epoch)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.recons_weight = 0.1
        self.kld_weight = 1
        self.tc_weight = 1
        self.mi_weight = 1
        self.contrastive_weight = 10
        self.matching_weight = 1000
        self.prediction_weight = 100


        # Data
        self.batch_size = args.batch_size
        self.data_loader = return_data(self.batch_size, train_data, outcomes, surgery_types)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        self.anneal_steps = 2000
        self.alpha = float(1.)
        self.beta = float(6.)
        self.num_iter = 0
        self.dataset_size = len(self.data_loader.dataset)
        print("dataset_size", self.dataset_size)
        self.pbar = tqdm(total=self.max_iter*(np.ceil(self.dataset_size/self.batch_size)))


        self.VAE = VAE_attention(input_dim=train_data.shape[1], z_dim=self.z_dim).to(self.device)
 

        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))
        self.nets = [self.VAE, self.D]
        


    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        vae_recon_losses = []
        vae_kld_losses = []
        vae_tc_losses = []

        
        epochs = self.max_iter

        for epoch in range(epochs):
            print("Epoch:", epoch)

            for x_true1, x_true2 in self.data_loader:

                    self.global_iter += 1
                    self.num_iter += 1

                    self.pbar.update(1)

                    x_true1 = x_true1.to(self.device)
                    x_recon, mu, logvar, z = self.VAE(x_true1)
                    vae_recon_loss = F.mse_loss(x_recon, x_true1, reduction='sum')/ (x_true1.size(0))
                    vae_kld = kl_divergence(mu, logvar)
                    self.D = self.D.eval()
                    D_z = self.D(z)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                    vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

                    self.optim_VAE.zero_grad()
                    vae_loss.backward()
                    self.optim_VAE.step()
                    self.D = self.D.train()
                    x_true2 = x_true2.to(self.device)
                    z_prime = self.VAE(x_true2, no_dec=True)
                    z_pperm = permute_dims(z_prime).detach()

                    z = z.detach()
                    D_z = self.D(z)
                             
                    D_z_pperm = self.D(z_pperm)
                    D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                    

                    self.optim_D.zero_grad()
                    D_tc_loss.backward()
                    self.optim_D.step()

                    vae_recon_losses.append(vae_recon_loss.item())
                    vae_kld_losses.append(vae_kld.item())
                    vae_tc_losses.append(vae_tc_loss.item())





                    if self.global_iter%self.print_iter == 0:
                        self.pbar.write('[{}] vae_loss:{:.3f} recon_loss:{:.3f} kld:{:.3f} tc:{:.3f}'.format(
                            self.global_iter, vae_loss.item(), vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item()))
                            
                        




        self.pbar.write("[Training Finished]")
        #save model parameters
        model_folder = './factor_vae_model_att_fold' + str(index) + '/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        pickle.dump(self.VAE, open(model_folder + 'model.pickle', 'wb'))

        self.pbar.close()
        return vae_recon_losses, vae_kld_losses, vae_tc_losses
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--epoch', default=10, type=int, help='maximum training iteration')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    parser.add_argument('--z_dim', default=64, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=6.4, type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-3, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-3, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')


    parser.add_argument('--print_iter', default=100, type=int, help='print losses iter')

    args = parser.parse_args()

    net = Solver(args)
    vae_recon_losses, vae_kld_losses, vae_tc_losses = net.train()
    
