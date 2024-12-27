import math
import numpy as np
import torch
import torch.distributions as td
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm,trange
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def weights_init(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)

        
class ClinicalVAE(nn.Module):
    def __init__(
        self, input_dimension=64, hidden_layer_width=2000, n_epochs=50, 
        number_of_labels=1, number_of_classes=[4,2,2], weight=1, device=None, class_weight=None,latent_dimension=10
    ): 
        super().__init__()
        #multi-task 
        self.n_epochs=n_epochs
        self.hidden_layer_width = hidden_layer_width
        self.number_of_labels = number_of_labels #supervised dimension
        self.pred_weight = weight
        self.latent_dimension = latent_dimension
        self.input_dimension = input_dimension

        #Below are best hyperparameters finetuned for the model

        self.pred_weight = weight
        self.beta = .1
        self.recon_weight = 1
        self.KL_weight=1
        
        self.starting_index = 0
        for i in range(number_of_labels):
            self.starting_index+=number_of_classes[i]
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_dimension, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.input_dimension),
        ).to(device)
        self.encoder = nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, (2 * self.latent_dimension)),
        ).to(device)
        # self.predictor =nn.Sequential(
        #     nn.Linear(self.latent_dimension*2,self.hidden_layer_width),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_layer_width,number_of_classes[0])
        #                              )
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters())
                                                  , lr=1e-4, weight_decay = 5e-6
        )
        self.number_of_classes = number_of_classes
        self.criterion = nn.CrossEntropyLoss(weight = class_weight,
                                             reduction='elementwise_mean')
        self.mse = nn.MSELoss()
        self.device=device
        self.batch_size = 128
        self.plot=False
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)
    def matrix_log_density_gaussian(self, x, mu, logvar):
        # broadcast to get probability of x given any instance(row) in (mu,logvar)
        # [k,:,:] : probability of kth row in x from all rows in (mu,logvar)
        x = x.view(self.batch_size, 1, self.latent_dimension)
        mu = mu.view(1, self.batch_size, self.latent_dimension)
        logvar = logvar.view(1, self.batch_size, self.latent_dimension)
        return td.Normal(loc=mu, scale=(torch.exp(logvar)) ** 0.5).log_prob(x)

    def log_importance_weight_matrix(self):
        """
        Calculates a log importance weight matrix
        Parameters
        ----------
        batch_size: int
            number of training images in the batch
        dataset_size: int
        number of training images in the dataset
        """
        N = self.n_data
        M = self.batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(self.batch_size, self.batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def get_log_qz_prodzi(self, latent_sample, latent_dist, is_mss=True):
        mat_log_qz = self.matrix_log_density_gaussian(
            latent_sample,
            latent_dist[..., : self.latent_dimension],
            latent_dist[..., self.latent_dimension :],
        )
        if is_mss:
            # use stratification
            log_iw_mat = self.log_importance_weight_matrix().to(
                latent_sample.device
            )
            mat_log_qz = mat_log_qz + log_iw_mat.view(self.batch_size, self.batch_size, 1)
            log_qz = torch.logsumexp(
                log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False
            )
            log_prod_qzi = torch.logsumexp(
                log_iw_mat.view(self.batch_size, self.batch_size, 1) + mat_log_qz,
                dim=1,
                keepdim=False,
            ).sum(1)

        else:
            log_prod_qzi = (
                torch.logsumexp(
                    mat_log_qz, dim=1, keepdim=False
                )  # sum of probabilities in each latent dimension
                - math.log(self.batch_size * self.n_data)
            ).sum(1)
            log_qz = torch.logsumexp(
                mat_log_qz.sum(2),  # sum of probabilities across all latent dimensions
                dim=1,
                keepdim=False,
            ) - math.log(self.batch_size * self.n_data)

        return log_qz, log_prod_qzi

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
    def pred_loss(self,targets,out_encoder):
        loss_prediction = 0
        starting_index = 0
        for label in range(self.number_of_labels): 
            target = targets[:,label]
            # print(torch.unique(target))
            if torch.isnan(target).all():
                continue
            else:
                nans = torch.isnan(target)
                preds =  out_encoder[~nans,starting_index:starting_index+self.number_of_classes[label]]
                # preds = self.predictor(out_encoder[~nans])
                loss_prediction+= self.criterion(
                    preds,
                    target[~nans].long()
                )
            starting_index += self.number_of_classes[label]
        return loss_prediction
    def forward(self, train_data):
        return self.encoder(train_data)
    def compute_loss(self, data,targets):
        out_encoder = self.encoder(data)
        #resample latent variables 
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_dimension],
                scale=torch.exp(out_encoder[..., self.latent_dimension :]) ** 0.5,
            ),
            1,
        )  # each row is a latent vector
        zgivenx_flat = q_zgivenxobs.rsample()
        zgivenx = zgivenx_flat.reshape((-1, self.latent_dimension))

        # calculate reconstruction loss
        out_decoder = self.decoder(zgivenx)
        recon_loss = self.mse(out_decoder, data)

        # Calculate the original KL in VAE
        original_KL = self._kl_normal_loss(
            out_encoder[..., :self.latent_dimension],
            out_encoder[..., self.latent_dimension:],
        )
        # prob of z given observations x
        log_pz = (
            td.Independent(
                td.Normal(
                    loc=torch.zeros_like(zgivenx), scale=torch.ones_like(zgivenx)
                ),
                1,
            )
            .log_prob(zgivenx)
            .mean()
        )
        log_q_zCx = q_zgivenxobs.log_prob(zgivenx).mean()

        log_qz, log_prod_qzi = self.get_log_qz_prodzi(
            zgivenx, out_encoder
        )
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        
        prediction_losses = self.pred_loss(targets,out_encoder)

        neg_bound = (
                self.pred_weight * prediction_losses
                + recon_loss * self.recon_weight
                + original_KL * self.KL_weight
                + tc_loss * self.beta
        )

        return neg_bound

    
    def predict_proba(self,test_data):
        preds = []
        with torch.no_grad():
            starting_index = 0
            out_encoder = self.encoder(torch.Tensor(test_data).to(self.device))
            for label in range(self.number_of_labels): 
                curr_pred =  out_encoder[:,starting_index:starting_index+self.number_of_classes[label]]
                # curr_pred =  self.predictor(out_encoder)
                preds.append(self.softmax(curr_pred).detach().cpu().numpy())
                starting_index += self.number_of_classes[label] 
        return preds
    def fit(self,data,train_label
                ):
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.early_stopper = EarlyStopper(patience=1, min_delta=0.01)
        self.n_data = len(data)
        #train_eval split
        if len(train_label.shape)==1:
            train_label = train_label.reshape(-1,1)
        #make sure stratify works
        indices_train, indices_eval, train_label, eval_label = train_test_split(np.arange(0,len(train_label)), train_label,
                                                    test_size=0.15, stratify=train_label)
        train_loss = []
        val_loss = []        
        train_data = torch.Tensor(data[indices_train])
        eval_data = torch.Tensor(data[indices_eval]).to(self.device)
        
        train_label = torch.Tensor(train_label)
        eval_label = torch.Tensor(eval_label).to(self.device)
        for epoch in trange(self.n_epochs, desc='epochs',leave=False):
            epoch_train_loss=[]
            train_set = torch.utils.data.TensorDataset(train_data, train_label)
            train_loader = DataLoader(train_set, shuffle=True,
                                      num_workers=0, drop_last=True,
                                      batch_size=self.batch_size)
            for i, batch_data in enumerate(train_loader,0):
                data,label = batch_data
                self.optimizer.zero_grad()
                loss = self.compute_loss(data=data.to(self.device),targets = label.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters())
                              ,5)
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
   
            epoch_pred = self.encoder(eval_data)
            with torch.no_grad():
                prediction_losses=self.pred_loss(eval_label ,
                                    epoch_pred
                                   )
            train_loss.append(np.mean(epoch_train_loss,axis=0))
            val_loss.append(prediction_losses.item())
            if self.early_stopper.early_stop(prediction_losses):
                break

