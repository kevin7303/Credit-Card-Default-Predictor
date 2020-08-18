# Credit-Card-Default-Predictor
Machine Learning/ Deep Learning model to predict bank accounts that default 


# Project Overview 
* Cleaned and analyzed data found on Kaggle : https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
* Provided detailed visual analysis of the Ames Housing dataset to reach relational insight between features and data structures
* Engineered features such as Total Square footage and total bathroom numbers by combining columns into insightful information 
* Optimized Random Forest, Gradient Boosting Regressor, Ridge Regression, Lasso Regression and Elastic Net using GridsearchCV to reach the best model
 

## Code and Resources Used 
**Python Version:** 3.8 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn , keras 
**Original Kaggle Dataset:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Data Overview
* Data contains 30,000 rows and 25 columns:
* Columns (features) are:
* ID: ID of each client
* LIMIT_BAL: Amount of given credit in NT dollars 
* SEX: Gender
* EDUCATION
* MARRIAGE: Marital status 
* AGE: Age in years
* PAY_0: Repayment status in September, 2005
* PAY_2: Repayment status in August, 2005 
* PAY_3: Repayment status in July, 2005 
* PAY_4: Repayment status in June, 2005
* PAY_5: Repayment status in May, 2005 
* PAY_6: Repayment status in April, 2005 
* BILL_AMT1: Amount of bill statement in September, 2005 
* BILL_AMT2: Amount of bill statement in August, 2005 
* BILL_AMT3: Amount of bill statement in July, 2005 
* BILL_AMT4: Amount of bill statement in June, 2005 
* BILL_AMT5: Amount of bill statement in May, 2005 
* BILL_AMT6: Amount of bill statement in April, 2005 
* PAY_AMT1: Amount of previous payment in September, 2005
* PAY_AMT2: Amount of previous payment in August, 2005 
* PAY_AMT3: Amount of previous payment in July, 2005 
* PAY_AMT4: Amount of previous payment in June, 2005 
* PAY_AMT5: Amount of previous payment in May, 2005
* PAY_AMT6: Amount of previous payment in April, 2005 
* default.payment.next.month: Default payment 


Target Variable: 
default.payment.next.month: Does the account default on Payment

## Data Cleaning
I did extensive data cleaning in order to facilitate the exploratory analysis and the model building process:

*	Coerced the features to their respective correct data types after handling unique complications
* Encoded categorical features with appropriate ordinal values when ordinality is present
* Encoded categorical values with one hot encoding when ordinality is not present
* Used data visualization to safely remove outliers that were most likely erroneous 
*	Dropped irrelevant features such as ID
*	Performed transformation to target variable to remove skewness 
*	Improved model efficiency by using Standard Scaling to better handle outliers

## EDA
The expoloraty data analysis was done to better visualize and understand data set before undergoing the model building process.

Explored the target data and its distribution.
Below are some of the graphs created with seaborn:


![alt text](https://github.com/kevin7303/Credit-Card-Default-Predictor/blob/master/Age.PNG "Age")
![alt text](https://github.com/kevin7303/Credit-Card-Default-Predictor/blob/master/Education.PNG "Education")



## Model Building 
I wanted to create a model that accurately classify real accounts that would default based on their previous payments

*Evaluation Metric*

The specific metric used to evaluate the model was Accuracy with attention to precision, recall and F1 score.


**Steps Taken**

Performed One hot encoding on the categorical variables in order to accomodate Sklearn Decision trees treatment of categorical variables as continuous

Used an assortment of simple and complex classification models to create a robust model to accurately predict Default Class.
Started with base model to evaluate the performance of an unoptimzed and unfitted model on the problem and later used RandomSearchCV to optimize the hyper parameters.
All of these were done using a 5 fold cross validation.

Developed a simple deep learning model using a standard 4 layered neural network approach. One input layer, two hidden layer and an output layer.

Compared the machine learning model results with the deep learning model to find the best accuracy.

Models Used:
* **Logistic Regression** (Base)
* **Decision Tree** (Base)
* **XGBoosting Classifier** (Tuned)
* **Random Forest Classifier** (Tuned)

Deep Methods Used:
* **Neural Network** (Tuned)



## Model performance
Tuning was done on all of the functions above to increase prediction accuracy.


**Model Parameters

* **XGBoosting Classifier** - n_estimators=600, max_depth=20, learning_rate=0.05, eval_metric= 'error', random_state=99
* **Random Forest Classifier** - n_estimators = 400, max_features = 'sqrt', max_depth= 50, criterion = 'gini', min_samples_split = 2, random_state=99

* **Neural Network** -

* Input - input_dim = 66
* Hidden - Dense(16, activation = 'relu')
* Hidden - Dense(16, activation = 'relu')
* Dropout(0.5)
* Output -Dense(1, activation = 'sigmoid')

* Adam Optmizer (lr=1e-3)

* batch = 16
* epochs = 100
* earlystop = EarlyStopping(patience=30, restore_best_weights= True)
* callback = [earlystop]

* model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])

## **Results**
# The best algorithm was Random Forest Classifier with test data accuracy: 0.81681

The other tuned models resulted in:

XGB Classifer:
*Test data accuracy: 0.80761

Neural Networks 
*loss: 0.6938 - accuracy: 0.5571 - val_loss: 0.5809 - val_accuracy: 0.7825
