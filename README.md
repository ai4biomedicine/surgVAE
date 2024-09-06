# surgVAE

The implementation of surgVAE (surgical Variational Autoencoder), 

the innovative and highly accurate VAE-based Perioperative Prediction Framework (Semi-supervised and Multi-tasking model), 

capable of simultaneous prediction across N = 6 different crucial outcomes in high-risk cardiac surgery.

![image](https://github.com/JunboShen/cardiacVAE/assets/93768018/bcd73c3d-f2ea-4a80-97ef-b5d92613f99e)


The repo contains training and testing codes for surgVAE and baseline models.

Trained model files can be downloaded from here: <https://figshare.com/s/e44f0120502b01583ba2>. Trained under 5-fold cross-validation settings specified in the paper.

# Training settings

All methods are trained and tested under stratified 5-fold cross-validation, with slight differences in their settings.

For 1 train and test split, 

1. **surgVAE** is trained using the train set containing both cardiac surgery and non-cardiac surgery data, and tested on the cardiac surgery test set. **surgVAE** is trained/tested **once** to make predictions for all postoperative complications. 

2. **Vanilla VAE, Factor VAE, and Beta TC VAE** are first pre-trained using the train set containing both cardiac surgery and non-cardiac surgery data. Then, DNN Multi-Layer Perceptron classifiers are trained upon the pre-trained encoders of **Vanilla VAE, Factor VAE, and Beta TC VAE**, using the train set containing only the cardiac surgery data. For each model, the encoder is pre-trained **once** and the classifier is trained/tested **6 times** for N = 6 different outcomes. All VAE-based methods (surgVAE, Vanilla VAE, Factor VAE, and Beta TC VAE) share the same encoder and decoder architectures.

3. **MAML, Prototypical network** are first pre-trained using the train set containing only non-cardiac surgery data. Then, they are further adapted/finetuned on the train set containing only the cardiac surgery data. Each model is pre-trained **once** and finetuned/tested **6 times** for N = 6 different outcomes.

4. **Other machine learning baselines** are trained using the train set containing both cardiac surgery and non-cardiac surgery data. Each model is trained/tested **6 times** for N = 6 different outcomes.

There are 5 train and test fold splits in the 5-fold cross-validation setting.
