import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

#data preprocessing
folder = ('./preops_cv/')
outcomes = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']
with open('../../inputs_new.pickle','rb') as handle:
    inputs = pkl.load(handle)
#change column name orlogid_encoded to orlogid
inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

outcomes = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']
for idx, outcome in enumerate(outcomes):
    foldername = folder+'arrest'+'/'
    aurocs = []
    auprcs = []
    for i in range(5):
        
        #set model, 5 ML models
        model =  LogisticRegression(random_state=0) #change from RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, MLPClassifier, xgb.XGBClassifier
        
        X_train = pkl.load(open(foldername + 'X_train_' + str(i) + '.pickle', 'rb'))
        X_test = pkl.load(open(foldername + 'X_test_' + str(i) + '.pickle', 'rb'))
        # load X_train, X_test, y_train, y_test
        train_data = pkl.loads(open('./preops/X_sampled_non_cardiac.pickle', 'rb').read())
        X_train2 = train_data
        train_outcome = pkl.loads(open('./preops/outcome_data_non_cardiac.pickle', 'rb').read())
        y_train2 = train_outcome[outcome].to_numpy()

        y_train2 = np.where(y_train2 > 0, 1, 0)
        ids_train = pkl.load(open('./preops_cv/arrest/' + 'train_ids_' + str(i) + '.pickle', 'rb'))
        ids_test = pkl.load(open('./preops_cv/arrest/' + 'test_ids_' + str(i) + '.pickle', 'rb'))
        train_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_train))].copy()
        train_outcome2 = train_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
        
        train_outcome2 = np.where(train_outcome2> 0, 1, 0)
        test_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_test))].copy()
        test_outcome2 = test_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
        test_outcome2 = np.where(test_outcome2> 0, 1, 0)
        y_train = train_outcome2[:, idx]

        y_test = test_outcome2[:, idx]
        print(outcome + ' ' + "fold " + str(i))
       
        X_train = np.delete(X_train, 163, axis=1)
        X_test = np.delete(X_test, 163, axis=1)

        # transform to float32, true to 1, false to 0
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # train 
        model.fit(X_train, y_train)
        # predict
        y_pred = model.predict_proba(X_test)[:, 1]

        # calculate AUC
        auroc = roc_auc_score(y_test, y_pred)
        aurocs.append(auroc)
        
        # calculate AUPRC
        auprc = average_precision_score(y_test, y_pred)
        auprcs.append(auprc)
        
    #average AUC
    print("average AUC: ", np.mean(aurocs))
    #std AUC
    print("std AUC: ", np.std(aurocs))
    #average AUPRC

    print("average AUPRC: ", np.mean(auprcs))
    #std AUPRC
    print("std AUPRC: ", np.std(auprcs))
