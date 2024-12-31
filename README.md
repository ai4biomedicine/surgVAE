# surgVAE

This repository contains code for the paper ***A Novel Generative Multi-Task Representation Learning Approach for Predicting Postoperative Complications in Cardiac Surgery Patients***, which is accepted by *Journal of the American Medical Informatics Association (JAMIA)*.

The repo contains the implementation of surgVAE (surgical Variational Autoencoder) and baseline models.

The paper can be accessed via [Link](https://doi.org/10.1093/jamia/ocae316).

# Overview of the framework

surgVAE is a unified model with auxiliary predictors tailored for N = 6 important postoperative complications after cardiac surgery (AF, Cardiac Arrest, DVT/PE, Post AKI Status, Blood Transfusion, Intraop Cardiac Events), enabling simultaneous prediction across different complications. 

![Pipeline](https://github.com/user-attachments/assets/7adf3f35-971b-44c9-8208-170f5dc92098)

# Train & test settings

All methods are trained and tested under stratified 5-fold cross-validation, with slight differences in their settings.

For each one train and test split, 

1. **surgVAE, and Multi-task DNN** is trained using the train set containing both cardiac surgery and non-cardiac surgery data, and tested on the cardiac surgery test set. **surgVAE** is trained/tested **once** to make predictions for all postoperative complications. 

2. **Vanilla VAE, Factor VAE, and Beta TC VAE** are first pre-trained using the train set containing both cardiac surgery and non-cardiac surgery data. Then, DNN Multi-Layer Perceptron classifiers are trained upon the pre-trained encoders of **Vanilla VAE, Factor VAE, and Beta TC VAE**, using the train set containing only the cardiac surgery data. For each model, the encoder is pre-trained **once** and the classifier is trained/tested **6 times** for N = 6 different outcomes.

**Note**: these VAE-based methods (surgVAE, Vanilla VAE, Factor VAE, and Beta TC VAE) share the same encoder and decoder architectures.

3. **MAML, Prototypical network** are first pre-trained using the train set containing only non-cardiac surgery data. Then, they are further adapted/finetuned on the train set containing only the cardiac surgery data. Each model is pre-trained **once** and finetuned/tested **6 times** for N = 6 different outcomes.

4. **Other ML baselines (XGBoost, DNN, LR, RF, GBM), and Clinical VAE** are trained using the train set containing both cardiac surgery and non-cardiac surgery data. Each model is trained/tested **6 times** for N = 6 different outcomes.

There are 5 train and test fold splits in the 5-fold cross-validation setting.
