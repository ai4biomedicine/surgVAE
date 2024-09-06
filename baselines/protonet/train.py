import argparse
from collections import Counter
import os.path as osp
import pickle
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import Averager, euclidean_metric
from dataloader import *
from transform import *
from protonet import ProtoNet

from imblearn.over_sampling import RandomOverSampler

from dataset import *
import pickle as pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()
    
    #data preprocessing
    with open('../../../inputs_new.pickle','rb') as handle:
        inputs = pickle.load(handle)

    #change column name orlogid_encoded to orlogid
    inputs.outcomes.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)
    inputs.texts.rename(columns={'orlogid_encoded':'orlogid'}, inplace=True)

    train_data = pickle.loads(open('../preops/X_sampled_non_cardiac.pickle', 'rb').read())
    train_outcome = pickle.loads(open('../preops/outcome_data_non_cardiac.pickle', 'rb').read())
    types = pickle.loads(open('../preops/surgery_type_non_cardiac.pickle', 'rb').read())
    surgery_types = types.to_numpy()


    outcomes_name = ['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']

    outcomes = train_outcome[['cardiac', 'AF','arrest', 'DVT_PE', 'post_aki_status', 'total_blood']].to_numpy()

    #convert to binary, if >0, then 1, else 0
    outcomes = np.where(outcomes > 0, 1, 0)

    #folder = ('../preops_cv/')
    #outcome = 'arrest'
    #foldername = folder+outcome+'/'
    #train_data2 = pickle.load(open(foldername+'X_train_'+str(args.fold_idx)+'.pickle', 'rb'))
    train_data = np.delete(train_data, 163, axis=1)
    #train_data duplicate 6
    train_data_list_pos = []
    train_y_list_pos = []
    train_data_list_neg = []
    train_y_list_neg = []

    for idx, outcome_name in enumerate(outcomes_name):
        train_data_ = train_data.copy()
        train_y_ = outcomes[:,idx].copy()
        #use oversampling to balance the data
        ros = RandomOverSampler(random_state=0)
        train_data_, train_y_ = ros.fit_resample(train_data_, train_y_)
        #split the data into positive and negative
        train_data_pos = train_data_[train_y_==1]
        train_data_neg = train_data_[train_y_==0]
        #check if the data is balanced
        print(train_data_pos.shape)
        print(train_data_neg.shape)
        #shape the data to drop the last batch
        train_data_pos = train_data_pos[:-(train_data_pos.shape[0]%128)]
        train_data_neg = train_data_neg[:-(train_data_neg.shape[0]%128)]
        train_data_list_pos.append(train_data_pos)
        train_data_list_neg.append(train_data_neg)
        train_y_list_pos.append(np.ones(train_data_pos.shape[0]))
        train_y_list_neg.append(np.zeros(train_data_neg.shape[0]))



    print(train_data.shape)


    model = ProtoNet(x_dim=train_data.shape[1]).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    
    train_data_pos = np.concatenate(train_data_list_pos, axis=0)
    train_data_neg = np.concatenate(train_data_list_neg, axis=0)
    train_y_pos = np.concatenate(train_y_list_pos, axis=0)
    train_y_neg = np.concatenate(train_y_list_neg, axis=0)

    train_data_pos_loader = return_data(128, train_data_pos, train_y_pos)
    train_data_neg_loader = return_data(128, train_data_neg, train_y_neg)

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()
        
        for i, (batch_pos, batch_neg) in enumerate(zip(train_data_pos_loader, train_data_neg_loader), 1):
            #p = args.shot * args.train_way
            #data_shot, data_query = data[:p], data[p:]
            
            (train_batch_pos, train_batch_pos_outcome) = batch_pos
            (train_batch_neg, train_batch_neg_outcome) = batch_neg

            batch_size = train_batch_pos.shape[0]
            
            
            
            
            data_shot = torch.concatenate((train_batch_neg[:batch_size//2], train_batch_pos[:batch_size//2])).cuda()
            data_query = torch.concatenate((train_batch_neg[batch_size//2:], train_batch_pos[batch_size//2:])).cuda()

            proto = model(data_shot)
            #proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto_neg = proto[:batch_size//2].mean(dim=0) 
            proto_pos = proto[batch_size//2:].mean(dim=0)
            proto = torch.stack((proto_neg, proto_pos)) #shape (2, 128)
            
            


            #label = torch.arange(args.train_way).repeat(args.query)
            label_pos = train_batch_pos_outcome[batch_size//2:]
            label_neg = train_batch_neg_outcome[batch_size//2:]
            label = torch.concatenate((label_neg, label_pos)).cuda()
            
            label = label.type(torch.cuda.LongTensor).cuda()


            logits = euclidean_metric(model(data_query), proto).cuda()
            loss = F.cross_entropy(logits, label)
            pred = torch.softmax(logits, dim=1)
            acc = torch.mean((torch.argmax(pred, dim=1) == label).type(torch.cuda.FloatTensor)).item()

            ta.add(acc)



            tl.add(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proto = None
            logits = None
            loss = None

        tl = tl.item()
        ta = ta.item()
        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))

   
    model.eval()
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
           
            X_train = np.delete(X_train, 163, axis=1)
            X_test = np.delete(X_test, 163, axis=1)

            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            

            #split the data into positive and negative
            X_train_pos = X_train[y_train==1]
            X_train_neg = X_train[y_train==0]

            X_train = np.concatenate((X_train_neg, X_train_pos))
            y_train = np.concatenate((np.ones(X_train_neg.shape[0]), np.zeros(X_train_pos.shape[0])))
            
            
            proto = model(torch.from_numpy(X_train).to("cuda").float())

            proto_neg = proto[:X_train_neg.shape[0]].mean(dim=0)
            proto_pos = proto[X_train_neg.shape[0]:].mean(dim=0)
            proto = torch.stack((proto_neg, proto_pos))
            
            label = torch.from_numpy(y_train).long().to("cuda")
            logits = euclidean_metric(model(torch.from_numpy(X_test).to("cuda").float()), proto)

            pred = torch.softmax(logits, dim=1)
            y_pred = pred[:,1].cpu().detach().numpy()
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

        #standard deviation AUC 
        print("std AUC: ", np.std(aurocs))
        #average AUPRC
        print("average AUPRC: ", np.mean(auprcs))
        #standard deviation AUPRC
        print("std AUPRC: ", np.std(auprcs))


