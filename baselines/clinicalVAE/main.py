import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
from VAE import ClinicalVAE  # Import the VAE model class

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data loading and preprocessing
folder = ('../preops_cv/')
outcomes = ['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']

with open('../inputs_new.pickle', 'rb') as handle:
    inputs = pkl.load(handle)

# Rename columns for consistency
inputs.outcomes.rename(columns={'orlogid_encoded': 'orlogid'}, inplace=True)
inputs.texts.rename(columns={'orlogid_encoded': 'orlogid'}, inplace=True)
#set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for idx, outcome in enumerate(outcomes):
    foldername = folder + outcome + '/'
    aurocs = []
    auprcs = []
    for i in range(5):
        # Load training and testing data
        X_train = pkl.load(open(foldername + 'X_train_' + str(i) + '.pickle', 'rb'))
        X_test = pkl.load(open(foldername + 'X_test_' + str(i) + '.pickle', 'rb'))

        # Load train and test IDs
        ids_train = pkl.load(open(foldername + 'train_ids_' + str(i) + '.pickle', 'rb'))
        ids_test = pkl.load(open(foldername + 'test_ids_' + str(i) + '.pickle', 'rb'))

        # Get outcomes for training and testing sets
        train_outcome2 = inputs.outcomes[inputs.outcomes.orlogid.isin(ids_train)].copy()
        y_train = train_outcome2[outcome].to_numpy()
        y_train = np.where(y_train > 0, 1, 0).reshape(-1, 1)  # Reshape for VAE input

        test_outcome2 = inputs.outcomes[inputs.outcomes.orlogid.isin(ids_test)].copy()
        y_test = test_outcome2[outcome].to_numpy()
        y_test = np.where(y_test > 0, 1, 0)

        print(f'{outcome} fold {i}')

        # Remove specific feature
        X_train = np.delete(X_train, 163, axis=1)
        X_test = np.delete(X_test, 163, axis=1)

        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        # log transform
        X_train = np.log(X_train + 1)
        X_test = np.log(X_test + 1)


        # Initialize the VAE model, best hyperparameters finetuned
        model = ClinicalVAE(
            input_dimension=X_train.shape[1],
            hidden_layer_width=1000,
            n_epochs=50,
            number_of_labels=1,
            number_of_classes=[2],  # Binary classification
            weight=1,
            device=device,
            class_weight=None,
            latent_dimension=10
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict probabilities on the test set
        y_pred_proba = model.predict_proba(X_test)
        y_pred = y_pred_proba[0][:, 1]  # Probability of the positive class

        # Calculate evaluation metrics
        auroc = roc_auc_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_pred)

        aurocs.append(auroc)
        auprcs.append(auprc)

    # Print average and standard deviation of metrics
    print(f"{outcome} average AUC: {np.mean(aurocs):.4f}")
    print(f"{outcome} std AUC: {np.std(aurocs):.4f}")
    print(f"{outcome} average AUPRC: {np.mean(auprcs):.4f}")
    print(f"{outcome} std AUPRC: {np.std(auprcs):.4f}")