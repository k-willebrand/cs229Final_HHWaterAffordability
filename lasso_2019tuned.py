# example implementation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# set working directory
os.chdir('/Users/keaniw/Documents/Classes/CS229 Machine Learning/Project/Project Code/cs229_project/src')

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
# load the merged dataset
df_merged = pickleload('data/df_merged_2019_agg.txt')
df_merged.drop(columns=['is_delinq'], axis=1, inplace=True)

# separate data into features (x) and response variables (y)
X = df_merged.drop(columns=['account', 'pen_freq', 'debt_dur', 'debt_val'])  # candidate features
Y = df_merged[['pen_freq', 'debt_dur', 'debt_val']]  # candidate response variables

# drop X columns with only one value (std=0)
cols_drop = np.where(X.loc[:, X.dtypes != 'category'].std(numeric_only=True, axis=0) == 0)  # index rows with std = 0
X.drop(X.columns[cols_drop], axis=1, inplace=True)

# standardize numeric features (for regression)
X.loc[:, X.dtypes != 'category'] = (X.loc[:, X.dtypes != 'category'] - X.loc[:, X.dtypes != 'category'].mean(numeric_only=True))\
                                   / X.loc[:, X.dtypes != 'category'].std(numeric_only=True, axis=0)
# keep mean and std of response variables for transforming predictions back to un-standardized units
Y_mean = Y.loc[:, Y.dtypes != 'category'].mean(numeric_only=True)
Y_std = Y.loc[:, Y.dtypes != 'category'].std(numeric_only=True, axis=0)

# standardize target variables (for regression)
Y.loc[:, Y.dtypes != 'category'] = (Y.loc[:, Y.dtypes != 'category'] - Y.loc[:, Y.dtypes != 'category'].mean(numeric_only=True))\
                                   / Y.loc[:, Y.dtypes != 'category'].std(numeric_only=True, axis=0)

# one-hot encode categorical variables
X = pd.get_dummies(X)

# get feature and predictor names
X_names = list(X.columns)
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

#%% first tune the alpha hyperparameter for Lasso regression for each metric

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# range of alphas to test with grid search
alphas = np.arange(0.1, 20.1, 0.1)

# use grid search to find the best alpha for each metric
alpha_tuned = np.zeros(len(aff_metrics))
for i, metric in enumerate(aff_metrics):
    pipeline = Pipeline(['model', linear_model.Lasso()])
    search = GridSearchCV(linear_model.Lasso(), {'alpha': alphas}, cv=10, scoring="neg_mean_squared_error", verbose=0)
    search.fit(train_x, train_y[:, i])

    # save and print to console the selected alpha for each metric
    alpha_tuned[i] = search.best_params_['alpha']
    picklesave(f'lasso/randombestparams_{metric}.txt', search)
    print(f'{metric} {search.best_params_}')

#%% Fit the Lasso Regression Model for each affordability metric using selected alpha value

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# switch: set true to run baseline model for comparison within loop below
calc_baseline = False

for i, metric in enumerate(aff_metrics):

    # load the best parameters
    search = pickleload(f'lasso/randombestparams_{metric}.txt')
    alpha_tuned = search.best_params_['alpha']

    # create lasso regression model object
    clf = linear_model.Lasso(alpha=alpha_tuned, fit_intercept=True)

    # fit the lasso regression model for the household level affordability metric
    clf.fit(train_x, train_y[:, i])
    picklesave(f'lasso/params_{metric}_2019.txt', clf)

    # get the parameters of the model
    params = clf.get_params()

    # make a prediction using the test and training data
    pred_y_train = clf.predict(train_x)
    pred_y_test = clf.predict(test_x)
    picklesave(f'lasso/pred_y_train_{metric}_2019_std.txt', pred_y_train)
    picklesave(f'lasso/pred_y_test_{metric}_2019_std.txt', pred_y_train)
    pred_y_train_ = (pred_y_train * Y_std[i]) + Y_mean[i]  # un-standardized
    pred_y_test_ = (pred_y_test * Y_std[i]) + Y_mean[i]  # un-standardized
    picklesave(f'lasso/pred_y_train_{metric}_2019_unstd.txt', pred_y_train_)
    picklesave(f'lasso/pred_y_test_{metric}_2019_unstd.txt', pred_y_train_)

    # training data MSE
    print('')
    print(f'LASSO model for {metric}:')
    train_y_ = (train_y[:, i] * Y_std[i]) + Y_mean[i]
    MSE = np.mean((train_y_ - pred_y_train_) ** 2)
    print(f'Train MSE: {MSE}')  # un-standardized

    # test data MSE
    test_y_ = (test_y[:, i] * Y_std[i]) + Y_mean[i]
    MSE = np.mean((test_y_ - pred_y_test_) ** 2)
    print(f'Test MSE: {MSE}')  # un-standardized

    # calculate additional test data metrics (using un-standardized data)
    abs_err = np.sum(abs((test_y_ - pred_y_test_)))  # L1
    print(f'Test absolute error: {abs_err}')
    TSS = np.sum((test_y_ - np.mean(test_y_)) ** 2)
    RSS = np.sum((test_y_ - pred_y_test_) ** 2)  # L2^2
    print(f'Test RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'Test R2: {R2}')
    p = np.sum(clf.coef_ != 0)
    Adj_R2 = 1 - (1 - R2) * len(test_y_) / (len(test_y_) - p - 1)
    print(f'Adjusted R2: {Adj_R2}')
    SMAPE = np.sum(abs((test_y_ - pred_y_test_))) / np.sum(
        test_y_ + pred_y_test_)  # alternative symmetric mean absolute error
    print(f'SMAPE: {SMAPE}')

    #%% make a bar plot of variable coefficients (weight feature importance)
    plt.figure()
    plt.bar(list(range(len(clf.coef_))), clf.coef_, color='gray', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(clf.coef_))), X_names)
    plt.xticks(fontsize=5, rotation=90)
    plt.ylabel('Coefficient Value')
    plt.xlabel('Feature Name')
    plt.title(f'Coefficient Value vs. Feature for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'lasso/coeff_{metric}_2019V2.tif', bbox_inches='tight')
    plt.show()

    #%% plot 15 most important features for each metric by abs on coeff
    plt.figure()
    idx_top_feat = np.argsort(np.abs(clf.coef_))[len(clf.coef_)-15:len(clf.coef_)+1][::-1]  # up to 15 largest coefficients
    top_feat_import_val = np.abs(itemgetter(*idx_top_feat)(clf.coef_))
    top_feat_import_names = np.array(list(itemgetter(*idx_top_feat)(X_names)))
    top_feat_import_names[top_feat_import_val == 0] = ' '
    plt.bar(list(range(len(top_feat_import_val))), top_feat_import_val, color='gray', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(top_feat_import_val))), top_feat_import_names)
    plt.xticks(fontsize=10, rotation=90)
    plt.ylabel('|Coeffient|', fontsize=13)
    plt.ylim([0, 0.4])
    plt.title(f'{metric}', fontsize=13)
    plt.gca().yaxis.grid(True, which='both')
    plt.gcf().set_size_inches(6.5, 2.9)
    plt.tight_layout()
    plt.savefig(f'lasso/subset_featimport_{metric}_2019V2.tif', bbox_inches='tight')
    plt.show()

    #%% scatter plot true vs predicted values (standardized)
    plt.figure()
    plt.scatter(test_y[:, i], pred_y_test, color='chocolate', s=10, alpha=0.5, label='LASSO Regression Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='Baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Lasso Regression Model')
    plt.legend()
    plt.grid()
    plt.savefig(f'lasso/scatter_{metric}_2019stdV2.tif', bbox_inches='tight')
    plt.show()

    #%% scatter plot true vs predicted values (un-standardized)
    plt.figure()
    test_y_unstd = (test_y[:, i] * Y_std[i]) + Y_mean[i]
    pred_y_test_unstd = (pred_y_test * Y_std[i]) + Y_mean[i]
    picklesave(f'lasso/pred_y_test_{metric}_2019.txt', pred_y_test_unstd)  # save unstandardized form
    plt.scatter(test_y_unstd, pred_y_test_unstd, color='chocolate', s=10, alpha=0.5, label='LASSO Regression Model')
    plt.scatter(test_y_unstd, np.ones_like(test_y_unstd) * np.mean((train_y[:, i] * Y_std[i]) + Y_mean[i]),
                color='steelblue', s=10, alpha=0.5, label='Baseline Model')
    plt.plot(test_y_unstd, test_y_unstd, 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Lasso Regression Model')
    plt.legend()
    plt.grid()
    plt.savefig(f'lasso/scatter_{metric}_2019V2.tif', bbox_inches='tight')
    plt.show()

#%% optional: grid search tuning the alpha hyperparameter for Lasso regression for each metric

# switch whether to run or not
plots = False

if plots is True:
    # the number of different affordability metrics to try to predict
    aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

    # set of alphas to test
    alphas = np.arange(0.1, 10.1, 0.1)

    # preallocate arrays
    train_MSE = np.zeros(shape=(len(aff_metrics), alphas.shape[0]))
    test_MSE = np.zeros(shape=(len(aff_metrics), alphas.shape[0]))

    # train and test fitted lasso regression for each model
    for i, metric in enumerate(aff_metrics):

        for j, alpha in enumerate(alphas):

            # create lasso regression model object
            clf = linear_model.Lasso(alpha=alpha, fit_intercept=True)

            # fit the lasso regression model for the household level affordability metric
            clf.fit(train_x, train_y[:, i])

            # make a prediction using the test data
            pred_y_train = clf.predict(train_x)
            pred_y_test = clf.predict(test_x)

            # save train and test MSE
            train_MSE[i, j] = np.mean((train_y[:, i] - pred_y_train) ** 2)
            test_MSE[i, j] = np.mean((test_y[:, i] - pred_y_test) ** 2)

    # plot train MSE and test MSE vs. alpha for each affordability metric
    for i, metric in enumerate(aff_metrics):
        plt.figure()
        plt.plot(alphas, train_MSE[i, :], color='chocolate', label='training MSE')
        plt.plot(alphas, test_MSE[i, :], color='steelblue', label='test MSE')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid()
        plt.title(f'{metric}')
        plt.savefig(f'lasso/alpha_{metric}_2019.tiff', bbox_inches='tight')
        plt.show()