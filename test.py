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
import pickle as pkl
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

from dataloader import *
import argparse
from transform import *
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--epoch', type=int, default=7, help='number of epochs')
    args = parser.parse_args()
    with open('../../../inputs_new.pickle','rb') as handle:
        inputs = pkl.load(handle)
    print(inputs)
    #change column name orlogid_encoded to orlogid
    inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
    inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
    print(inputs.outcomes.columns)
    folder = ('../preops_cv/')

    outcomes = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']
    for idx, outcome in enumerate(outcomes):
        foldername = folder+'arrest'+'/'
        aurocs = []
        auprcs = []
        for i in range(5):
            X_train = pkl.load(open(foldername + 'X_train_' + str(i) + '.pickle', 'rb'))
            X_test = pkl.load(open(foldername + 'X_test_' + str(i) + '.pickle', 'rb'))

            ids_train = pkl.load(open('../preops_cv/arrest/' + 'train_ids_' + str(i) + '.pickle', 'rb'))
            ids_test = pkl.load(open('../preops_cv/arrest/' + 'test_ids_' + str(i) + '.pickle', 'rb'))
            train_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_train))].copy()
            train_outcome2 = train_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
            train_outcome2 = np.where(train_outcome2> 0, 1, 0)
            test_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_test))].copy()
            test_outcome2 = test_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
            test_outcome2 = np.where(test_outcome2> 0, 1, 0)
            y_train = train_outcome2[:, idx]
            y_test = test_outcome2[:, idx]
            print(outcome + ' ' + "fold " + str(i))

            vae_model = pkl.load(open('./distangle_vae_model_epoch' + str(args.epoch) + '/model_fold' + str(i) + '.pickle', 'rb'))

            # transform to float32, delete column 163
            X_train = np.delete(X_train, 163, axis=1)
            X_test = np.delete(X_test, 163, axis=1)

            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            # log transform
            X_train = np.log(X_train + 1)
            X_test = np.log(X_test + 1)
            vae_model = vae_model.to("cuda")
            vae_model.eval()
            x_test = torch.from_numpy(X_test).to("cuda")
            # as float32
            x_test = x_test.float()
            # dataloader
            test_dataset = torch.utils.data.TensorDataset(x_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
            y_pred = []
            for batch_idx, (x_test) in enumerate(test_loader):
                x_test = x_test[0]
                x_recon, mu, logvar, z, classifications = vae_model(x_test)
                torch.cuda.empty_cache()
                classification = classifications[:, idx, :]
                #softmax, take first 2 logits
                classification = nn.Softmax(dim=1)(classification[:, :2])
                # predict
                y_pred_ = classification[:, 1].detach().cpu().numpy()
                y_pred.append(y_pred_)
            y_pred = np.concatenate(y_pred, axis=0)

            # calculate AUC
            auroc = roc_auc_score(y_test, y_pred)
            #print("AUC: ", auroc)
            aurocs.append(auroc)
            # calculate AUPRC
            auprc = average_precision_score(y_test, y_pred)
            #print("AUPRC: ", auprc)
            auprcs.append(auprc)
        #average AUC
        print("average AUC: ", np.mean(aurocs))
        #std AUC
        print("std AUC: ", np.std(aurocs))
        #average AUPRC
        print("average AUPRC: ", np.mean(auprcs))
        #std AUPRC
        print("std AUPRC: ", np.std(auprcs) )
