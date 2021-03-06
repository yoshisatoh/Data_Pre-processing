
#################### Data Pre-processing: Split data into three sets (train, validation, and test) in an random manner ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/03
# Last Updated: 2021/10/03
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/data_split/train_validation_and_test/time-series/temporal/data_split_train_valid_and_test__ts_temporal.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/data_split/train_validation_and_test/time-series/temporal/data_split_train_valid_and_test__ts_temporal.py
#
#
########## Input Data File(s)
#
#df.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python data_split_train_valid_and_test__ts_temporal.py df.csv y date 0.8 0.1
#
#Generally,
#python data_split_train_valid_and_test__ts_temporal.py (dfname: data file name) (column name of target y to predict) (column in datetime type) (train_size) (valid_size)
#
#You do not need to specify test_size and valid size as it is set as follows.
#(test_size) = 1 - (train_size) - (valid_size)
#
#
########## References
#
#How to split data into three sets (train, validation, and test) And why?
#https://towardsdatascience.com/how-to-split-data-into-three-sets-train-validation-and-test-and-why-e50d22d3e54c
#
#See the section:
#2) Splitting using the temporal component
#ii. Using Fast_ml → ‘train_valid_test_split’
#
#
####################




########## Definition of Training/Validation/Test Datasets

'''
You take a given dataset and divide it into three subsets.


1. Training Dataset (train)

Simply put, it is used for model PARAMETER learning.

Set of data used for learning by the model, that is,
choosing the parameters of the machine learning model so that
the predicated data by the model can be fitted to labeled data given in supervised learning

Parameters include, but not limited to, weights and a bias used between each node in deep learning.
Weights and biases are used to calculate input values and output values of nodes.


2. Validation Dataset (valid)

Simply put, it is used for model HYPERPARAMETER learning.

Set of data used to provide an unbiased evaluation of a model
fitted on the training dataset while tuning model hyperparameters.

Also it plays a role in other forms of model preparation, such as
feature selection, threshold cut-off selection.


3. Test Dataset (test)

Simply put, it is used for evaluation of outputs from the final model after learning, NOT for model learning.

Set of data used to provide an unbiased evaluation of a final model fitted on the training dataset and validation dataset.
'''




########## import Python libraries

import sys

import pandas as pd

#from sklearn.model_selection import train_test_split    # i. Using Sklearn → ‘train_test_split’

from fast_ml.model_development import train_valid_test_split    # ii. Using Fast_ml → ‘train_valid_test_split’




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #data_split_train_valid_and_test.py

dfname     = str(sys.argv[1])    #df.csv

ycol       = str(sys.argv[2])    #y

datecol    = str(sys.argv[3])    #date

train_size = float(sys.argv[4])    #0.8

valid_size = float(sys.argv[5])    #0.1

#test_size  = sys.argv[4]    #0.1
test_size  = 1 - train_size - valid_size   #0.1




########## 2) Splitting using the temporal component


##### ii. Using Fast_ml → ‘train_valid_test_split’

'''
Using that temporal (=recent) variable is a more reliable way of splitting datasets
whenever the dataset includes the date variable, and we want to predict something in the future.

Hence we must use the latest (=recent past) samples for creating the VALIDATION and TEST dataset.

Distant past samples are used as TRAINING dataset.

If the characteristics of data in the distant past remain the same to the recent past,
your model predictions will be good for TEST dataset.
Also, forecasting the recent past (and the near future) is usually very important.
'''

df = pd.read_csv(dfname, parse_dates=[datecol], low_memory=False)

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df,
                                                                            target = ycol, 
                                                                            method = 'sorted',
                                                                            sort_by_col= datecol,
                                                                            train_size=train_size,
                                                                            valid_size=valid_size,
                                                                            test_size=test_size)



'''
print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)
'''

X_train.to_csv('X_train.csv', header=True, index=False)
y_train.to_csv('y_train.csv', header=True, index=False)

X_valid.to_csv('X_valid.csv', header=True, index=False)
y_valid.to_csv('y_valid.csv', header=True, index=False)

X_test.to_csv('X_test.csv', header=True, index=False)
y_test.to_csv('y_test.csv', header=True, index=False)
