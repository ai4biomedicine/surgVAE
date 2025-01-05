import json
from random import shuffle
from collections import Counter
import torch
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
import time
import logging
import argparse
import os

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from maml import Learner
import random
import numpy as np
import pickle
from pandas import DataFrame

from dataloader import *
from transform import *

#Data loading and preprocessing
with open('../../../inputs_new.pickle','rb') as handle:
    inputs = pickle.load(handle)
    
#change column name orlogid_encoded to orlogid
inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)

train_data = pickle.loads(open('../preops/X_sampled_non_cardiac.pickle', 'rb').read())
train_data = np.delete(train_data, 163, axis=1)

train_data = train_data.astype(np.float32)
train_data = np.log(train_data + 1)

train_outcome = pickle.loads(open('../preops/outcome_data_non_cardiac.pickle', 'rb').read())
types = pickle.loads(open('../preops/surgery_type_non_cardiac.pickle', 'rb').read())
surgery_types = types.to_numpy()


outcomes_name = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']

outcomes = train_outcome[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()

#convert to binary, if >0, then 1, else 0
outcomes = np.where(outcomes > 0, 1, 0)

classes = np.unique(surgery_types)
train_support = []
train_query = []
train_support_labels = []
train_query_labels = []
for c in classes:
    for idx, outcome in enumerate(outcomes_name):
        ids = np.where(surgery_types == c)[0]
        #random split ids into 80% train and 20% test
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)

        train_support.append(train_data[train_ids])
        train_query.append(train_data[test_ids])
        train_support_labels.append(outcomes[train_ids][:,idx])
        train_query_labels.append(outcomes[test_ids][:,idx])

folder = ('../preops_cv/')
outcomes = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']
test_support = []
test_query = []
test_support_labels = []
test_query_labels = []

for idx, outcome in enumerate(outcomes):
    foldername = folder+'arrest'+'/'
    for i in range(5):
        X_train = pickle.load(open(foldername + 'X_train_' + str(i) + '.pickle', 'rb'))

        X_test = pickle.load(open(foldername + 'X_test_' + str(i) + '.pickle', 'rb'))

        X_train = np.delete(X_train, 163, axis=1)

        X_test = np.delete(X_test, 163, axis=1)

        # transform to float32, true to 1, false to 0
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # log transform
        X_train = np.log(X_train + 1)
        X_test = np.log(X_test + 1)

        ids_train = pickle.load(open('../preops_cv/arrest/' + 'train_ids_' + str(i) + '.pickle', 'rb'))
        ids_test = pickle.load(open('../preops_cv/arrest/' + 'test_ids_' + str(i) + '.pickle', 'rb'))
        train_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_train))].copy()
        train_outcome2 = train_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
        train_outcome2 = np.where(train_outcome2> 0, 1, 0)
        test_outcome2 = inputs.outcomes[(inputs.outcomes.orlogid.isin(ids_test))].copy()
        test_outcome2 = test_outcome2[['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()
        test_outcome2 = np.where(test_outcome2> 0, 1, 0)
        y_train = train_outcome2[:, idx]
        y_test = test_outcome2[:, idx]
        test_support.append(X_train)
        test_query.append(X_test)
        test_support_labels.append(y_train)
        test_query_labels.append(y_test)

#surgery_types = np.concatenate((surgery_types, types2), axis=0)
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from numpy import inf

def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def create_batch_of_tasks(X_support, y_support, X_query, y_query, batch_size=4, is_shuffle=True):
    # Create a list of tasks, each task is a tuple of support and query set
    task_list = []
    for i in range(len(X_support)):
        task_list.append(([X_support[i], y_support[i]], [X_query[i], y_query[i]]))

    if is_shuffle:
        random.shuffle(task_list)

    # Batch up the tasks, return a list of batches containing batch_size tasks

    batches = [task_list[i:i + batch_size] for i in range(0, len(task_list), batch_size)]

    return batches

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=10, type=int,
                        help="Number of outer interation")


    parser.add_argument("--outer_batch_size", default=6, type=int,
                        help="Batch of task size")

    parser.add_argument("--inner_batch_size", default=256, type=int,
                        help="Training batch size in inner iteration")

    parser.add_argument("--outer_update_lr", default=1e-3, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=1e-3, type=float,
                        help="Inner update learning rate")

    parser.add_argument("--inner_update_step", default=10, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--inner_update_step_eval", default=40, type=int,
                        help="Number of interation in the inner loop during test time")
    # New arguments for RNN
    parser.add_argument("--input_size", default=663, type=int, help="Input size for RNN")
    parser.add_argument("--hidden_size", default=128, type=int, help="Hidden size for RNN")
    parser.add_argument("--num_layers", default=2, type=int, help="Number of RNN layers")

    args = parser.parse_args()

    learner = Learner(args)

    global_step = 0

    for epoch in range(args.epoch):

        train_batches = create_batch_of_tasks(train_support, train_support_labels,train_query, train_query_labels,
                                                batch_size=args.outer_batch_size, is_shuffle=True)
        test_batches = create_batch_of_tasks(test_support, test_support_labels, test_query, test_query_labels,
                                             batch_size=args.outer_batch_size, is_shuffle=False)

        for step, task_batch in enumerate(train_batches):


            acc = learner(task_batch)


            print('Step:', step, '\ttraining Acc:', acc)

            if epoch == args.epoch - 1:
                random_seed(42)
                print("\n-----------------Testing Mode-----------------\n")
                # Use create_batch_of_tasks function for testing
                acc_all_test = []
                dfs = []
                for test_batch in test_batches:
                    # Assuming the Learner's __call__ method takes X, y as arguments
                    acc, df = learner(test_batch, training=False)

                    acc_all_test.append(acc)
                    dfs.append(df)
                df = pd.concat(dfs)
                df.to_csv('results_DNN.csv', index=False)

                print('Step:', step, 'Test ACC:', np.mean(acc_all_test))

                random_seed(int(time.time() % 10))

            global_step += 1

if __name__ == "__main__":
    main()