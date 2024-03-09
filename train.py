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

from dataloader import *
from transform import *
import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import VAE_attention
from dataset import return_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

    
class PairSelector:
    def get_pairs(self, embeddings, targets):
        positive_pairs = []
        negative_pairs = []
        unique_targets = torch.unique(targets)

        for target in unique_targets:
            target_indices = (targets == target).nonzero(as_tuple=True)[0]
            non_target_indices = (targets != target).nonzero(as_tuple=True)[0]

            # Positive pairs (all combinations within the same surgery type)
            positive_pairs.extend(list(combinations(target_indices, 2)))

            # Negative pairs (all combinations between different surgery types)
            for non_target_index in non_target_indices:
                negative_pairs.extend([(i, non_target_index) for i in target_indices])

        return torch.tensor(positive_pairs), torch.tensor(negative_pairs)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin, pair_selector):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

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
def recon_loss(x, x_recon):
    loss = F.mse_loss(x_recon, x, reduction='sum')/x.size(0)

    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld
def gaussian_kernel(x1, x2):
    x1_size = x1.size(0)
    x2_size = x2.size(0)
    dim = x1.size(1)

    x1 = x1.unsqueeze(1)  # Shape: [x1_size, 1, dim]
    x2 = x2.unsqueeze(0)  # Shape: [1, x2_size, dim]

    tile_x1 = x1.expand(x1_size, x2_size, dim)
    tile_x2 = x2.expand(x1_size, x2_size, dim)

    kernel = torch.exp(-torch.mean((tile_x1 - tile_x2) ** 2, dim=2) / dim)
    return kernel

def mmd_loss(x1, x2):
    x1_kernel = gaussian_kernel(x1, x1)
    x2_kernel = gaussian_kernel(x2, x2)
    cross_kernel = gaussian_kernel(x1, x2)

    mmd = x1_kernel.mean() + x2_kernel.mean() - 2 * cross_kernel.mean()
    return mmd

def compute_matching_loss(latents, surgery_types):
    unique_surgery_types = torch.unique(surgery_types)
    total_loss = 0.0

    for surgery_type1, surgery_type2 in combinations(unique_surgery_types, 2):
        indices1 = (surgery_types == surgery_type1).nonzero(as_tuple=True)[0]
        indices2 = (surgery_types == surgery_type2).nonzero(as_tuple=True)[0]

        latents1 = latents[indices1]
        latents2 = latents[indices2]



        total_loss += mmd_loss(latents1, latents2)
    total_loss /= len(list(combinations(unique_surgery_types, 2)))

    return total_loss




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
        #self.prediction_weight = 20


        # Data
        self.batch_size = args.batch_size
        self.data_loader = return_data(self.batch_size, train_data, outcomes, surgery_types)
        self.val_data_loader = return_data(self.batch_size, train_data2, outcomes2, types2)
        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

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
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.fold_idx = args.fold_idx


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
            log_qz, log_prod_qz1, log_prod_qz2 = self._minibatch_weighted_sampling(latent_dist,
                                                                latent_sample,
                                                                n_data)

        else:
            log_qz, log_prod_qz1, log_prod_qz2 = self._minibatch_stratified_sampling(latent_dist,
                                                                  latent_sample,
                                                                  n_data)

        return log_pz, log_qz, log_prod_qz1, log_prod_qz2, log_q_zCx

    def _minibatch_weighted_sampling(self, latent_dist, latent_sample, data_size):

        batch_size = latent_sample.size(0)

        mat_log_qz = self.matrix_log_density_gaussian(latent_sample, *latent_dist)

        #get first half of z to calculate log_prod_qz1
        log_prod_qz1 = (logsumexp(mat_log_qz[:, :, :self.z_dim//2].sum(2), dim=1, keepdim=False
                                    ) - math.log(batch_size * data_size))
        #get second half of z to calculate log_prod_qz2
        log_prod_qz2 = (logsumexp(mat_log_qz[:, :,  self.z_dim//2:].sum(2), dim=1, keepdim=False
                                    ) - math.log(batch_size * data_size))

        log_qz = logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False
                                 ) - math.log(batch_size * data_size)

        return log_qz, log_prod_qz1, log_prod_qz2

    def _minibatch_stratified_sampling(self, latent_dist, latent_sample, data_size):

        batch_size = latent_sample.size(0)

        mat_log_qz = self.matrix_log_density_gaussian(latent_sample, *latent_dist)

        log_iw_mat = self.log_importance_weight_matrix(batch_size, data_size).to(latent_sample.device)
        log_qz = logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)
        # get first half of z to calculate log_prod_qz1
        log_prod_qz1 = logsumexp(log_iw_mat + mat_log_qz[:, :, :self.z_dim//2].sum(2), dim=1, keepdim=False)
        # get second half of z to calculate log_prod_qz2
        log_prod_qz2 = logsumexp(log_iw_mat + mat_log_qz[:, :, self.z_dim//2:].sum(2), dim=1, keepdim=False)
        
        return log_qz, log_prod_qz1, log_prod_qz2

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution
        with diagonal covariance and a unit normal distribution.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim) where
            D is dimension of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        storer : dict
            Dictionary in which to store important variables for vizualisation.
        """
        latent_dim = mean.size(1)
        # batch mean of kl for each latent dimension
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()

        return total_kl
    def pred_loss(self, classifications, outcomes):

        outcomes = outcomes.to(self.device)
        #loss = self.cn_loss(classifications, outcomes)
        loss_list = []
        
        for i in range( len(classifications[0])):
            loss = self.ce_loss(classifications[:,i, :], outcomes[:,i])
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        return loss
    def loss_function(self, recons, input, mu, log_var, z, dataset_size, training, classifications,types, outcomes,is_mss=True):

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        latent_dist = [mu, log_var]

        log_pz, log_qz, log_prod_qz1, log_prod_qz2, log_q_zCx = self._get_log_pz_qz_prodzi_qzCx(z,
                                                                             latent_dist,
                                                                                dataset_size,
                                                                                is_mss=is_mss)

        #mi_loss = (log_q_zCx - log_qz).mean()

        #tc_loss = (log_qz - log_prod_qzi).mean()
        tc_loss = (log_qz - log_prod_qz1 - log_prod_qz2).mean()

        #kld_loss = (log_prod_qzi - log_pz).mean()
        original_KL = self._kl_normal_loss(mu, log_var)
        contrastive = ContrastiveLoss(1, PairSelector())

        contrastive_loss = contrastive(z[:, self.z_dim//2:], types)
        matching_loss = compute_matching_loss(z[:, :self.z_dim//2], types)

        prediction_loss = self.pred_loss(classifications, outcomes)



        if training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = (self.recons_weight*recons_loss/self.batch_size + self.beta * tc_loss
                + self.gamma * original_KL
                + self.contrastive_weight*contrastive_loss + self.matching_weight*matching_loss + self.prediction_weight * prediction_loss)

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss* self.recons_weight/self.batch_size,
                'KLD': original_KL* self.gamma,
                'TC_Loss': tc_loss* self.beta,
                'Contrastive_Loss': contrastive_loss* self.contrastive_weight,
                'Matching_Loss': matching_loss* self.matching_weight,
                'Prediction_Loss': prediction_loss* self.prediction_weight }

    def train(self):
        self.net_mode(train=True)

        vae_recon_losses = []
        vae_kld_losses = []
        vae_tc_losses = []
        vae_mi_losses = []
        vae_contrastive_losses = []
        vae_matching_losses = []
        vae_prediction_losses = []
        epochs = self.max_iter
        previous_prediction_loss = np.inf
        for epoch in range(epochs):
            print("Epoch:", epoch)

            for x_true, outcomes,types in self.data_loader:

                    self.global_iter += 1
                    self.num_iter += 1

                    self.pbar.update(1)

                    x_true = x_true.to(self.device)

                    x_recon, mu, logvar, z, classifications = self.VAE(x_true)



                    vae_losses = self.loss_function(x_recon, x_true, mu, logvar, z, self.dataset_size, True, classifications, types, outcomes,True)
                    vae_recon_loss = vae_losses['Reconstruction_Loss']
                    vae_kld = vae_losses['KLD']
                    vae_tc_loss = vae_losses['TC_Loss']
                    vae_loss = vae_losses['loss']
                    vae_contrastive_loss = vae_losses['Contrastive_Loss']
                    vae_matching_loss = vae_losses['Matching_Loss']
                    vae_prediction_loss = vae_losses['Prediction_Loss']


                    self.optim_VAE.zero_grad()
                    vae_loss.backward()
                    self.optim_VAE.step()




                    if self.global_iter%self.print_iter == 0:
                        self.pbar.write('[{}] vae_loss:{:.3f} recon_loss:{:.3f} kld:{:.3f} tc:{:.3f} contrastive:{:.3f} matching:{:.3f} prediction:{:.3f}'.format(
                            self.global_iter, vae_loss.item(), vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), vae_contrastive_loss.item(), vae_matching_loss.item(), vae_prediction_loss.item()
                        ))

                        vae_recon_losses.append(vae_recon_loss.item())
                        vae_kld_losses.append(vae_kld.item())
                        vae_tc_losses.append(vae_tc_loss.item())
                        vae_contrastive_losses.append(vae_contrastive_loss.item())
                        vae_matching_losses.append(vae_matching_loss.item())
                        vae_prediction_losses.append(vae_prediction_loss.item())
   
            self.VAE.eval()
            var_pred_loss = []
            with torch.no_grad():
                for var_x_true, var_outcomes, var_types in self.val_data_loader:
                    var_x_true = var_x_true.to(self.device)
                    
                    var_x_recon, var_mu, var_logvar, var_z, var_classifications = self.VAE(var_x_true)
                    var_pred_loss.append(self.pred_loss(var_classifications, var_outcomes).item())
            self.VAE.train()
            print("Validation Prediction Loss:", np.mean(var_pred_loss))
            if np.mean(var_pred_loss) < previous_prediction_loss:
                #save model parameters
                model_folder = './distangle_vae_model_fold'
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                pickle.dump(self.VAE, open(model_folder + '/model_fold'+str(self.fold_idx)+'.pickle', 'wb'))
                previous_prediction_loss = np.mean(var_pred_loss)
                print("Model Saved")

            model_folder = './distangle_vae_model_epoch'+str(epoch)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            pickle.dump(self.VAE, open(model_folder + '/model_fold'+str(self.fold_idx)+'.pickle', 'wb'))
            previous_prediction_loss = np.mean(var_pred_loss)
            print("Model Saved")



        self.pbar.write("[Training Finished]")
           

        self.pbar.close()
        return vae_recon_losses, vae_kld_losses, vae_tc_losses, vae_contrastive_losses, vae_matching_losses, vae_prediction_losses
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
    parser.add_argument('--epoch', default=8, type=int, help='maximum training iteration')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    parser.add_argument('--z_dim', default=64, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=1., type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')

    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')



    parser.add_argument('--fold_idx', default=0, type=int, help='fold index')
    #usage: python distangle_vae.py --fold_idx=0


    parser.add_argument('--print_iter', default=100, type=int, help='print losses iter')

    args = parser.parse_args()
    with open('../../../inputs_new.pickle','rb') as handle:
        inputs = pickle.load(handle)
    print(inputs)
    #change column name orlogid_encoded to orlogid
    inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
    inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
    print(inputs.outcomes.columns)

    train_data = pickle.loads(open('../preops/X_sampled_non_cardiac.pickle', 'rb').read())
    train_outcome = pickle.loads(open('../preops/outcome_data_non_cardiac.pickle', 'rb').read())
    types = pickle.loads(open('../preops/surgery_type_non_cardiac.pickle', 'rb').read())
    surgery_types = types.to_numpy()


    outcomes_name = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']
    for outcome in outcomes_name:
        #check if outcome is in the outcome_data and not nan
        if outcome in train_outcome.columns:
            print(outcome)
            print(train_outcome[outcome].isnull().sum())
            print(train_outcome[outcome].value_counts())

    outcomes = train_outcome[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
    print(outcomes.shape)
    #convert to binary, if >0, then 1, else 0
    outcomes = np.where(outcomes > 0, 1, 0)
    print(outcomes[0])

    folder = ('../preops_cv/')
    outcome = 'arrest'
    foldername = folder+outcome+'/'
    train_data2 = pickle.load(open(foldername+'X_train_'+str(args.fold_idx)+'.pickle', 'rb'))
    train_data = np.delete(train_data, 163, axis=1)
    train_data2 = np.delete(train_data2, 163, axis=1)
    train_data = np.concatenate((train_data, train_data2), axis=0)
    train_ids2 = pickle.load(open(foldername+'outcome_data_train_ids_'+str(args.fold_idx)+'.pickle', 'rb'))
    train_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(train_ids2))].copy()
    outcomes2 = train_outcome2[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
    print(outcomes2.shape)
    #convert to binary, if >0, then 1, else 0
    outcomes2 = np.where(outcomes2 > 0, 1, 0)
    print(outcomes2[0])
    outcomes = np.concatenate((outcomes, outcomes2), axis=0)

    #types2 is of length of train_data2 and all 2.0
    types2 = np.full((train_data2.shape[0],), 2.0)

    surgery_types = np.concatenate((surgery_types, types2), axis=0)

    print(train_data.shape)
    print(train_data2.shape)
    print(Counter(surgery_types))
    net = Solver(args)


   
    vae_recon_losses, vae_kld_losses, vae_tc_losses, vae_contrastive_losses, vae_matching_losses, vae_prediction_losses = net.train()
 
