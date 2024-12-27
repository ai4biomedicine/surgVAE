# surgVAE

The implementation of surgVAE (surgical Variational Autoencoder).

A Novel Generative Multi-Task Representation Learning Approach for Predicting Postoperative Complications in Cardiac Surgery Patients. 

The repo contains training and testing codes for surgVAE and baseline models.

The arxiv preprint of the paper can be accessed via [Link](https://www.arxiv.org/pdf/2412.01950).

# Overview of the framework

surgVAE is a unified model with auxiliary predictors tailored for N = 6 important postoperative complications after cardiac surgery (AF, Cardiac Arrest, DVT/PE, Post AKI Status, Blood Transfusion, Intraop Cardiac Events), enabling simultaneous prediction across different complications. 

![Pipeline](https://github.com/user-attachments/assets/7adf3f35-971b-44c9-8208-170f5dc92098)

Trained model files can be downloaded from here: <https://figshare.com/s/e44f0120502b01583ba2>. Trained under 5-fold cross-validation settings specified in the paper.

# Train & test settings

All methods are trained and tested under stratified 5-fold cross-validation, with slight differences in their settings.

For each one train and test split, 

1. **surgVAE, and Multi-task DNN** is trained using the train set containing both cardiac surgery and non-cardiac surgery data, and tested on the cardiac surgery test set. **surgVAE** is trained/tested **once** to make predictions for all postoperative complications. 

2. **Vanilla VAE, Factor VAE, and Beta TC VAE** are first pre-trained using the train set containing both cardiac surgery and non-cardiac surgery data. Then, DNN Multi-Layer Perceptron classifiers are trained upon the pre-trained encoders of **Vanilla VAE, Factor VAE, and Beta TC VAE**, using the train set containing only the cardiac surgery data. For each model, the encoder is pre-trained **once** and the classifier is trained/tested **6 times** for N = 6 different outcomes.

**Note**: these VAE-based methods (surgVAE, Vanilla VAE, Factor VAE, and Beta TC VAE) share the same encoder and decoder architectures.

3. **MAML, Prototypical network** are first pre-trained using the train set containing only non-cardiac surgery data. Then, they are further adapted/finetuned on the train set containing only the cardiac surgery data. Each model is pre-trained **once** and finetuned/tested **6 times** for N = 6 different outcomes.

4. **Other ML baselines (XGBoost, DNN, LR, RF, GBM), and Clinical VAE** are trained using the train set containing both cardiac surgery and non-cardiac surgery data. Each model is trained/tested **6 times** for N = 6 different outcomes.

There are 5 train and test fold splits in the 5-fold cross-validation setting.
