from imblearn.over_sampling import SMOTE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import gc
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_labels):
        super(DNN, self).__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        out = self.fc3(x)
        return out

class Learner(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()

        self.num_labels = args.num_labels
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = SimpleRNN(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_labels=self.num_labels)
        self.model = DNN(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_labels=self.num_labels)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.model.to(self.device)
        self.model.train()

    def forward(self, batch_tasks, training=True):

        task_accs = []
        task_preds = []
        task_labels = []
        task_ids = []
        task_probs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval
        df = pd.DataFrame()

        for task_id, task in enumerate(batch_tasks):
            support , query = task
            # Convert the numpy array to float32 type
            support_float = support[0].astype(np.float32)

            # Convert the numpy array to a PyTorch tensor
            support_features = torch.tensor(support_float, dtype=torch.float32)

            # Assuming they are floating-point values
            support_labels = torch.tensor(support[1], dtype=torch.long)  # Assuming these are long integers

            # Now create the TensorDataset
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)

            support_features = torch.tensor(support_features, dtype=torch.float32)


            support = TensorDataset(support_features, support_labels)


            # Convert the numpy array to float32 type
            query_float = query[0].astype(np.float32)

            # Convert the numpy array to a PyTorch tensor
            query_features = torch.tensor(query_float, dtype=torch.float32)

            query_features = torch.tensor(query_features, dtype=torch.float32)

            query_labels = torch.tensor(query[1], dtype=torch.long)  
           
            query = TensorDataset(query_features, query_labels)


            support_dataloader = DataLoader(support, sampler=None, batch_size=self.inner_batch_size)

            inner_optimizer = Adam(fast_model.parameters(), lr=self.inner_update_lr)
            fast_model.train()

            print('----Task', task_id, '----')
            for i in range(0, num_inner_update_step):
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    sequences, label_id = batch
                    outputs = fast_model(sequences)

                    loss = CrossEntropyLoss()(outputs, label_id)
                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()

                    all_loss.append(loss.item())

                if i % 4 == 0:
                    print("Inner Loss: ", np.mean(all_loss))

            query_dataloader = DataLoader(query, sampler=None, batch_size= 1024)
            q_outputs = []
            q_label_id = []
            for inner_step, batch in enumerate(query_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                sequences, label_id = batch
                outputs = fast_model(sequences)
                q_label_id.append(label_id)
                q_outputs.append(outputs)

            q_outputs = torch.cat(q_outputs, dim=0)
            q_label_id = torch.cat(q_label_id, dim=0)


            if training:
                q_loss = CrossEntropyLoss()(q_outputs, q_label_id)
                q_loss.backward()
                fast_model.cuda()
                for i, params in enumerate(fast_model.parameters()):
                    if task_id == 0:
                        sum_gradients.append(deepcopy(params.grad))
                    else:
                        sum_gradients[i] += deepcopy(params.grad)

            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)

            task_preds.extend(pre_label_id)
            task_labels.extend(q_label_id)
            task_ids.extend([task_id] * len(pre_label_id))
            task_probs.extend(q_logits.detach().cpu().numpy().tolist())

            df = pd.DataFrame({'task_id': task_ids, 'label': task_labels, 'pred': task_preds, 'prob': task_probs})

            del fast_model, inner_optimizer
            torch.cuda.empty_cache()

        if training:
            # Average gradient across tasks
            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            # Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                if params.grad is None:
                    params.grad = torch.zeros_like(params, device=params.device, dtype=params.dtype)
                params.grad = sum_gradients[i]
                print(params.dtype)
                print(params.grad)

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            del sum_gradients
            gc.collect()

        #check if df single or list
        if isinstance(df, list):
            df = pd.concat(df)
        return np.mean(task_accs), df
