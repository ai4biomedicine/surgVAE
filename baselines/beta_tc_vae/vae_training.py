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

from vae_model import VAE_attention
from dataset import return_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        self.nc = 1

        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.nets = [self.VAE]
        self.cn_loss = nn.BCEWithLogitsLoss(reduction='mean')



    def log_density_gaussian(self, x, mu, logvar):

        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M).to(self.device)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def matrix_log_density_gaussian(self, x, mu, logvar):

        batch_size, dim = x.shape
        x = x.view(batch_size, 1, dim)
        mu = mu.view(1, batch_size, dim)
        logvar = logvar.view(1, batch_size, dim)
        return self.log_density_gaussian(x, mu, logvar)

    def _get_log_pz_qz_prodzi_qzCx(self, latent_sample, latent_dist, n_data, is_mss=False):
        batch_size, hidden_dim = latent_sample.shape

        # calculate log q(z|x)
        log_q_zCx = self.log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

        # calculate log p(z)
        # mean and log var is 0
        zeros = torch.zeros_like(latent_sample)
        log_pz = self.log_density_gaussian(latent_sample, zeros, zeros).sum(1)

        if not is_mss:
            log_qz, log_prod_qzi = self._minibatch_weighted_sampling(latent_dist,
                                                                latent_sample,
                                                                n_data)

        else:
            log_qz, log_prod_qzi = self._minibatch_stratified_sampling(latent_dist,
                                                                  latent_sample,
                                                                  n_data)

        return log_pz, log_qz, log_prod_qzi, log_q_zCx

    def _minibatch_weighted_sampling(self, latent_dist, latent_sample, data_size):

        batch_size = latent_sample.size(0)

        mat_log_qz = self.matrix_log_density_gaussian(latent_sample, *latent_dist)

        log_prod_qzi = (logsumexp(mat_log_qz, dim=1, keepdim=False) -
                        math.log(batch_size * data_size)).sum(dim=1)
        log_qz = logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False
                                 ) - math.log(batch_size * data_size)

        return log_qz, log_prod_qzi

    def _minibatch_stratified_sampling(self, latent_dist, latent_sample, data_size):

        batch_size = latent_sample.size(0)

        mat_log_qz = self.matrix_log_density_gaussian(latent_sample, *latent_dist)

        log_iw_mat = self.log_importance_weight_matrix(batch_size, data_size).to(latent_sample.device)
        log_qz = logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = logsumexp(log_iw_mat.view(batch_size, batch_size, 1) +
                                       mat_log_qz, dim=1, keepdim=False).sum(1)

        return log_qz, log_prod_qzi

    def loss_function(self, recons, input, mu, log_var, z, dataset_size, training, is_mss=True):


        batch_size = input.size(0)
        dim = input.size(1)
        recons_loss = F.mse_loss(recons, input, reduction='sum')/ (batch_size)
        #recons_loss = F.binary_cross_entropy(recons, input, reduction="sum")
        latent_dist = [mu, log_var]

        log_pz, log_qz, log_prod_qzi, log_q_zCx = self._get_log_pz_qz_prodzi_qzCx(z,
                                                                             latent_dist,
                                                                                dataset_size,
                                                                                is_mss=is_mss)

        mi_loss = (log_q_zCx - log_qz).mean()

        tc_loss = (log_qz - log_prod_qzi).mean()

        kld_loss = (log_prod_qzi - log_pz).mean()


        if training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss =  recons_loss + self.alpha * mi_loss + self.beta * tc_loss + anneal_rate * self.gamma * kld_loss

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss,
                'TC_Loss': tc_loss,
                'MI_Loss': mi_loss}

    def train(self):
        self.net_mode(train=True)

        vae_recon_losses = []
        vae_kld_losses = []
        vae_tc_losses = []
        vae_mi_losses = []
        
        epochs = self.max_iter

        for epoch in range(epochs):
            print("Epoch:", epoch)

            for x_true, outcomes,types in self.data_loader:

                    self.global_iter += 1
                    self.num_iter += 1

                    self.pbar.update(1)

                    x_true = x_true.to(self.device)

                    x_recon, mu, logvar, z = self.VAE(x_true)



                    vae_losses = self.loss_function(x_recon, x_true, mu, logvar, z, self.dataset_size, True, True)
                    vae_loss = vae_losses['loss']
                    vae_recon_loss = vae_losses['Reconstruction_Loss']
                    vae_kld = vae_losses['KLD']
                    vae_tc_loss = vae_losses['TC_Loss']
                    vae_mi_loss = vae_losses['MI_Loss']


                    self.optim_VAE.zero_grad()
                    vae_loss.backward()
                    self.optim_VAE.step()




                    if self.global_iter%self.print_iter == 0:
                        self.pbar.write('[{}] vae_loss:{:.3f} recon_loss:{:.3f} kld:{:.3f} tc:{:.3f} mi:{:.3f}'.format(
                            self.global_iter, vae_loss.item(), vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), vae_mi_loss.item()))

                        vae_recon_losses.append(vae_recon_loss.item())
                        vae_kld_losses.append(vae_kld.item())
                        vae_tc_losses.append(vae_tc_loss.item())
                        vae_mi_losses.append(vae_mi_loss.item())




        self.pbar.write("[Training Finished]")
        #save model parameters
        model_folder = './beta_tc_vae_model_att_fold' + str(index) + '/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        pickle.dump(self.VAE, open(model_folder + 'model.pickle', 'wb'))

        self.pbar.close()
        return vae_recon_losses, vae_kld_losses, vae_tc_losses, vae_mi_losses
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

    parser = argparse.ArgumentParser(description='VAE')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--epoch', default=10, type=int, help='maximum training iteration')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    parser.add_argument('--z_dim', default=64, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=1., type=float, help='gamma hyperparameter')
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
    vae_recon_losses, vae_kld_losses, vae_tc_losses, vae_mi_losses = net.train()
    
