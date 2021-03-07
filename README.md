# Credit-Card-Fraud
The aim is to produce a model that is able to accurately predict Fradulent vs Non-Fradulent transactions. Given the imbalanced dataset (very limited number of Fradulent transactions vs abundant Non-Fradulent transactions), ML models have very little Fradulent data to train on. To circumvent this, undersampling or oversampling methods are used to allow for better training performance, before the models are exposed to imbalanced datasets for actual testing (reflecting actual propotion of class to predict).

In this project, I wanted to investigate the differences in performance of a few ML models with undersampling and oversampling methods with the use of an imbalanced dataset. 

Data used: Credit Card Fraud Detection Dataset from Kaggle - Anonymized credit card transactions labeled as fraudulent or genuine (https://www.kaggle.com/mlg-ulb/creditcardfraud)

## EDA on data
![image](https://user-images.githubusercontent.com/75196868/110233997-efb3e980-7f62-11eb-928d-b68fc4b58d42.png)

This is a highly imbalanced dataset, with 284315 Non-Fradulent transactions but only 492 Fradulent transactions! The minority class (Fradulent transactions) accounts for only 0.173% of all transactions.

There is also a higher proportion of Fradulent transactions with the Amount involved being $0, at 1.48%. 
While non $0 transactions have 0.1643% Fradulent cases.

# Building Machine Learning Models
The following models were used on the dataset:
1. Logistic Regression
2. Random Forest
3. Support Vector Classifier
(Default parameters are used)

Each of them were fitted with data first without any sampling method. Note that this is a wrong approach as it would result in a high accuracy just by predicting all transactions to be Non-Fradulent. For the purposes of comparison, this shall be called the 'baseline' models. 

Next, the datasets are split into train-test sets. Undersampling and oversampling methods are then applied to the training sets.

Methods of undersampling are used, namely:
A. Random undersampling
B. Cluster Centroids undersampling
C. Near Miss undersampling

Methods of oversampling are also used, namely:
D. Random oversampling
E. SMOTE
F. ADASYN

The undersampled/oversampled training data is then passed into the above mentioned models, and subsequently scored against testing data - which remains imbalanced.
It is to be expected that the unsampled 'baseline' model will give a high accuracy score. A better reflection of actual accuracy scores/confusion matrix/precision-recall would be of those models trained with undersampled or oversampled training datasets. The testing data should remain imbalanced as there should not be any information leakage into the testing data.

# Accuracy Score
![image](https://user-images.githubusercontent.com/75196868/110236284-e67d4980-7f6f-11eb-806d-d71bb02e7f17.png)

# Confusion Matrix
![image](https://user-images.githubusercontent.com/75196868/110236304-0280eb00-7f70-11eb-9682-ba552e37116c.png)
