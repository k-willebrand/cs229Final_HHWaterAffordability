# Baseline model: naive model that does not use additional information sources to guide prediction.

# load packages
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import pickle
from sklearn.model_selection import train_test_split

# set working directory
os.chdir('/Users/keaniw/Documents/Classes/CS229 Machine Learning/Project/Project Code/cs229_project/src')

#%% utility functions
def picklesave(filename, obj):
    """save python object for easy future loading using pickle (binary)

        Args:
             filename: filename as string to save object to (e.g., sample.txt)
             obj: the object/variable to saved

        """
    # write binary
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickleload(filename):
    """load previously saved python object from pickle (binary)

        Args:
             filename: filename as string where object previously saved to with pickle (e.g., sample.txt)

        Returns:
             obj: the previously saved python object

        """
    # read binary
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

#%% load the merged database of household level data and affordability metrics and specify training and test dataset

# load the merged dataset
df_merged = pickleload('data/df_merged_2019_agg.txt')
df_merged.drop(columns=['account', 'is_delinq'], axis=1, inplace=True)

#%% prepare data for use with sklearn

# one-hot encode categorical variables
df_merged = pd.get_dummies(df_merged)

# separate data into features (x) and response variables (y)
X = df_merged.drop(columns=['pen_freq', 'debt_dur', 'debt_val'])  # candidate features
X_names = list(X.columns)
Y = df_merged[['pen_freq', 'debt_dur', 'debt_val']]  # candidate response variables
Y_names = list(Y.columns)

# convert data to numpy array for use with sklearn algorithm
X = np.array(X)
Y = np.array(Y)

# create a random training and validation set of features (x) and response variables (y)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'Training features shape: {train_x.shape}')
print(f'Training response shape: {train_y.shape}')
print(f'Test features shape: {test_x.shape}')
print(f'Test response shape: {test_y.shape}')

#%% train and test naive baseline model (no additional knowledge)

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

for i, metric in enumerate(aff_metrics):

    #  make predictions on training and test data
    print('')
    print(f'Baseline model for {metric}:')
    pred_y_train = np.ones_like(train_y[:, i]) * np.mean(train_y[:, i])  # predict average metric
    pred_y_test = np.ones_like(test_y[:, i]) * np.mean(train_y[:, i])  # predict average metric
    picklesave(f'baseline/pred_y_train_{metric}_2019.txt', pred_y_train)
    picklesave(f'baseline/pred_y_test_{metric}_2019.txt', pred_y_test)

    # training data MSE
    print('')
    print(f'Baseline model for {metric}:')
    MSE = np.mean((train_y[:, i] - pred_y_train) ** 2)
    print(f'Train MSE: {MSE}')

    # test data MSE
    MSE = np.mean((test_y[:, i] - pred_y_test) ** 2)
    print(f'Test MSE: {MSE}')

    # calculate additional test data metrics
    abs_err = np.sum(abs((test_y[:, i] - pred_y_test)))  # L1
    print(f'Test absolute error: {abs_err}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i])) ** 2)
    RSS = np.sum((test_y[:, i] - pred_y_test) ** 2)  # L2^2
    print(f'Test RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'Test R2: {R2}')
    Adj_R2 = 1 - (1 - R2) * len(test_y[:, i]) / (len(test_y[:, i]) - 0 - 1)
    print(f'Adjusted R2: {Adj_R2}')
    SMAPE = np.sum(abs((test_y[:, i] - pred_y_test))) / np.sum(
        test_y[:, 0] + pred_y_test)  # alternative symmetric mean absolute error
    print(f'SMAPE: {SMAPE}')

    #%% scatter plot true vs predicted values
    plt.figure()
    plt.scatter(test_y[:, i], pred_y_test, color='chocolate', s=10, alpha=0.5, label='Random Forest Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Baseline model')
    plt.legend()
    plt.grid()
    plt.savefig(f'baseline/scatter_{metric}_2019.tif', bbox_inches='tight')
    plt.show()

