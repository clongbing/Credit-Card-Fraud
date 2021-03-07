# Credit-Card-Fraud
The aim is to produce a model that is able to accurately predict Fradulent vs Non-Fradulent transactions. Given the imbalanced dataset (very limited number of Fradulent transactions vs abundant Non-Fradulent transactions), ML models have very little Fradulent data to train on. To circumvent this, undersampling or oversampling methods are used to allow for better training performance, before the models are exposed to imbalanced datasets for actual testing (reflecting actual proportion of classes to predict).


In this project, I wanted to investigate the differences in performance of various ML models with undersampling and oversampling methods with the use of an imbalanced dataset. 


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


Each of them were fitted with data first without any sampling method. Note that this is not an ideal approach as it would result in a high accuracy just by predicting all transactions to be Non-Fradulent. For the purposes of comparison, these shall be called the 'baseline' models. 


Next, the datasets are split into train-test sets. Undersampling and oversampling methods are then applied to the training sets.


Methods of undersampling are used, namely:
- Random undersampling
- Cluster Centroids undersampling
- Near Miss undersampling


Methods of oversampling are also used, namely:
- Random oversampling
- SMOTE
- ADASYN


The undersampled/oversampled training data is then passed into the above mentioned models, and subsequently scored against testing data - which remains imbalanced.
It is to be expected that the unsampled 'baseline' model will give a high accuracy score. A better reflection of actual accuracy scores/confusion matrix/precision-recall would be of those models trained with undersampled or oversampled training datasets. The testing data should remain imbalanced as there should not be any information leakage into the testing data.



# Results
### Accuracy Score
Please refer to Accuracy.csv
![image](https://user-images.githubusercontent.com/75196868/110236284-e67d4980-7f6f-11eb-806d-d71bb02e7f17.png)

### Confusion Matrix
Please refer to Confusion_Matrix.png
![image](https://user-images.githubusercontent.com/75196868/110236576-796ab380-7f71-11eb-8dbf-5ff4f96577b6.png)

### Precison-Recall Curve
Please refer to Precision-Recall_Curve.png
![image](https://user-images.githubusercontent.com/75196868/110239140-7b883e80-7f80-11eb-8a87-d53181780c80.png)

# Conclusion
Highest Accuracy: 0.999391407 - Using Random Forest with Random Oversampling

Highest AP: 0.89 - Using Random Forest with SMOTE. This is the only model that performed better than the 'baseline' model


We see that Random Forest performs well overall despite the sampling methods used, with the exception of CC undersampling.
While it is clear that RF with Random Oversampling should be used to maximise Accuracy, it is not clear if we should use RT with SMOTE in practise, as RF is harder to interpret and might require some parameter tuning.


Furthermore, business input is required in choosing the probability threshold along the precision recall curve. For instance, a reputable bank that aims to protect its customer's interests at the expense of higher operational costs would choose a probability threshold that would maximise recall. In this case, the number of False Negative cases (predicted Non-Fradulent but is actually Fradulent) would be minised at the expense of higher operational cost - employees might need more man hours to sieve through more False Positive cases

# Future Improvements
- Exclude transactions where Amount = $0
- Feature selection

# Credits
https://github.com/annsiong for advice - creating src.model_results function to print Accuracy, Confusion Matrix and PR curve.


