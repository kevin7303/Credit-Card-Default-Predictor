import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Import CSV file
df = pd.read_csv('UCI_Credit_Card.csv')

#Analyze Dataframe attributes and properties
df.head()
df.columns
df.dtypes

#Check data for complete values
df.count()
df.isna().sum()
df.shape

#Drop ID column
df = df.drop(['ID'], axis = 1)

#Summary Statistics for Columns
(df.iloc[:, 0:6]).describe()
(df.iloc[:, 6:12]).describe()
(df.iloc[:, 12:18]).describe()
(df.iloc[:, 18:24]).describe()

#Check for inconsistent values in Columns with Value_counts and Graphs
df.SEX.value_counts()
df.EDUCATION.value_counts() #Incorrect Labels / Undocumented
df.MARRIAGE.value_counts() #Incorrect Label 0
df.AGE.value_counts()

#Remove Outliers Bill Amount < -50000 
bill = ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
for x in bill:
    df.drop(df[df[x] < -50000].index, inplace = True)

#Graph Personal Information Columns 
cat = ['SEX', 'EDUCATION', 'MARRIAGE']

fig, ax = plt.subplots(1,3)

for i, x in enumerate(cat):
    sns.countplot(df[x], ax = ax[i])
    ax[i].set_title(x + ' Distribution')
plt.show()

#Graph Age Column
sns.distplot(df.AGE)
plt.show()

#Graph Received Payment Columns
pay = ['PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

fig, ax = plt.subplots(1, 6)

for i, x in enumerate(pay):
    sns.countplot(df[x], ax=ax[i])
    ax[i].set_title(x + ' Distribution')
plt.show()


#Graph Payment Amount vs AGE
pay_am = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ]
fig, ax = plt.subplots(1, 6)

for i, x in enumerate(pay_am):
    sns.scatterplot(x=df[x], y = df['AGE'], ax=ax[i])
    ax[i].set_title(x + ' Distribution')
plt.show()

#Data Imputation

#Education Column
df.loc[(df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0), 'EDUCATION'] =4
df.EDUCATION.value_counts()

#Marriage Column
df.MARRIAGE.value_counts()
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

#Rename Columns for clarity
df = df.rename(columns={'default.payment.next.month': 'def_pay',
                        'PAY_0': 'PAY_1'})
df.head()

#Payment Status - All negative number as 0
pay = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for x in pay:
     df.loc[df[x] <= 0, x] = 0

df.columns
#Slice dataframe for normalization
data = df.iloc[:, 1:11]
data.head()

data2 = df.loc[:, ['LIMIT_BAL','BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
data2[['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()

#Scale Age and Payment amounts with StandardScaler
scale = StandardScaler()
scaled = scale.fit_transform(data2.values)

#Convert numpy array back to Dataframe
data2scaled = pd.DataFrame(scaled, index = data2.index, columns = data2.columns)
data2scaled

#Concatenate Categorical and Scaled Numerical Columns
new_df = pd.concat([data, data2scaled], axis = 1)
new_df['def_pay'] = df['def_pay']
new_df.columns

df.to_csv('eda_data_removed_outliers.csv', index = False)
new_df.to_csv('cleaned_data_removed_outliers.csv', index = False)
