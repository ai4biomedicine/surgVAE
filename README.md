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

## Citations <a name="citations"></a>

If you find the methods useful in your research, please kindly cite our paper:

```bibtex
@article{10.1093/jamia/ocae316,
    author = {Shen, Junbo and Xue, Bing and Kannampallil, Thomas and Lu, Chenyang and Abraham, Joanna},
    title = {A novel generative multi-task representation learning approach for predicting postoperative complications in cardiac surgery patients},
    journal = {Journal of the American Medical Informatics Association},
    pages = {ocae316},
    year = {2024},
    month = {12},
    abstract = {Early detection of surgical complications allows for timely therapy and proactive risk mitigation. Machine learning (ML) can be leveraged to identify and predict patient risks for postoperative complications. We developed and validated the effectiveness of predicting postoperative complications using a novel surgical Variational Autoencoder (surgVAE) that uncovers intrinsic patterns via cross-task and cross-cohort presentation learning.This retrospective cohort study used data from the electronic health records of adult surgical patients over 4 years (2018-2021). Six key postoperative complications for cardiac surgery were assessed: acute kidney injury, atrial fibrillation, cardiac arrest, deep vein thrombosis or pulmonary embolism, blood transfusion, and other intraoperative cardiac events. We compared surgVAE’s prediction performance against widely-used ML models and advanced representation learning and generative models under 5-fold cross-validation.89 246 surgeries (49\% male, median [IQR] age: 57 [45-69]) were included, with 6502 in the targeted cardiac surgery cohort (61\% male, median [IQR] age: 60 [53-70]). surgVAE demonstrated generally superior performance over existing ML solutions across postoperative complications of cardiac surgery patients, achieving macro-averaged AUPRC of 0.409 and macro-averaged AUROC of 0.831, which were 3.4\% and 3.7\% higher, respectively, than the best alternative method (by AUPRC scores). Model interpretation using Integrated Gradients highlighted key risk factors based on preoperative variable importance.Our advanced representation learning framework surgVAE showed excellent discriminatory performance for predicting postoperative complications and addressing the challenges of data complexity, small cohort sizes, and low-frequency positive events. surgVAE enables data-driven predictions of patient risks and prognosis while enhancing the interpretability of patient risk profiles.},
    issn = {1527-974X},
    doi = {10.1093/jamia/ocae316},
    url = {https://doi.org/10.1093/jamia/ocae316},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocae316/61289374/ocae316.pdf},
}
```
