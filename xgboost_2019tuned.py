# example implementation: https://mljar.com/blog/feature-importance-xgboost/

# Description: get feature importance from parallel boosting trees algorithm (Xgboost model)

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns  # for correlation heatmap
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

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

#%% tune parameters for use with XGBoost
# Parameters to tune:
# max_depth (Optional[int]) – Maximum tree depth for base learners.
# max_leaves – Maximum number of leaves; 0 indicates no limit.
# max_bin – If using histogram-based algorithm, maximum number of bins per feature
# grow_policy – Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.
# learning_rate

random_grid = {'n_estimators': [100, 500, 1000],
          'max_depth': [3, 5, 6, 10, 15, 20],
          'learning_rate': [0.01, 0.1, 0.2, 0.3],
          'subsample': np.arange(0.5, 1.0, 0.1),
          'colsample_bytree': np.arange(0.4, 1.0, 0.1),
          'colsample_bylevel': np.arange(0.4, 1.0, 0.1)}
# print parameters to randomly sample
print(random_grid)

# Use the random grid to search for best hyperparameters for each metric
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
params_tuned = []   # intialize list for saving params
for i, metric in enumerate(aff_metrics):
    # Random search of parameters, using 5-fold cross validation,
    # search across n_iter=35 different combinations, and use all available cores
    xgb = XGBRegressor(random_state=0, tree_method='approx', booster='gbtree', validate_parameters=True,
                       num_parallel_tree=1)  # approx allows for categorical predictors
    search = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, scoring='neg_mean_squared_error',
                                n_iter=35, cv=5, verbose=2, random_state=1, n_jobs=-1)
    search.fit(train_x, train_y[:, i])

    picklesave(f'xgboost/randombestparams_{metric}.txt', search)
    params_tuned.append(search.best_params_)
    print(f'{metric} random best hyperparams: {search.best_params_}')
    print("Lowest RMSE: ", (-search.best_score_)**(1/2.0))


#%% fit the XGBoost Regression model

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# switch: set true to run baseline model for comparison within loop below
calc_baseline = False

# fit XGBoost for each
for i, metric in enumerate(aff_metrics):

    if calc_baseline is True:
        # first fit baseline model
        print('')
        print(f'baseline model for {metric}:')
        pred_y = np.ones_like(test_y[:, i]) * np.mean(train_y[:, i])  # predict average metric
        abs_err = np.sum(abs((pred_y - test_y[:, i])))  # L1
        print(f'absolute error: {abs_err}')
        MSE = np.mean((pred_y - test_y[:, i]) ** 2)
        print(f'MSE: {MSE}')
        TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i])))
        RSS = np.sum((pred_y - test_y[:, i]) ** 2)  # L2^2
        print(f'RSS: {RSS}')
        R2 = (TSS - RSS) / TSS
        print(f'R2: {R2}')

    # create xgboost model object
    #xgb = XGBRegressor(n_estimators=params_tuned[i]['n_estimators'], max_depth=params_tuned[i]['max_depth'],
    #                   learning_rate=params_tuned[i]['learning_rate'], subsample=params_tuned[i]['subsample'],
    #                   colsample_bytree=params_tuned[i]['colsample_bytree'],
    #                   colsample_bylevel=params_tuned[i]['colsample_bylevel'],
    #                   random_state=0, n_jobs=-1)  # use the selected best model inputs

    xgb = XGBRegressor(n_estimators=params_tuned[i]['n_estimators'], max_depth=params_tuned[i]['max_depth'],
                       learning_rate=params_tuned[i]['learning_rate'], subsample=params_tuned[i]['subsample'],
                       colsample_bytree=params_tuned[i]['colsample_bytree'],
                       colsample_bylevel=params_tuned[i]['colsample_bylevel'],
                       random_state=1, booster='gbtree', validate_parameters=True, num_parallel_tree=1)  # set num_parallel_tree = 1 for boosting (-1 bagging -> RF)

    # train and save the fitted model
    xgb.fit(train_x, train_y[:, i])
    picklesave(f'xgboost/params_{metric}_2019tuned.txt', xgb)

    # make predictions on training and test data
    pred_y_train = xgb.predict(train_x)
    pred_y_test = xgb.predict(test_x)
    picklesave(f'xgboost/pred_y_train_{metric}_2019tuned.txt', pred_y_train)
    picklesave(f'xgboost/pred_y_test_{metric}_2019tuned.txt', pred_y_test)

    # training data MSE
    print('')
    print(f'XGBoost model for {metric}:')
    MSE = np.mean((train_y[:, i] - pred_y_train) ** 2)
    print(f'Train MSE: {MSE}')

    # test data MSE
    MSE = np.mean((test_y[:, i] - pred_y_test) ** 2)
    print(f'Test MSE: {MSE}')

    # calculate other test metrics
    print('')
    print(f'XGBoost model for {metric}:')
    abs_err = np.sum(abs((test_y[:, i] - pred_y_test)))  # L1
    print(f'absolute error: {abs_err}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i])) ** 2)
    RSS = np.sum((test_y[:, i] - pred_y_test) ** 2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS - RSS) / TSS
    print(f'R2: {R2}')

    # plot and save the feature importances (bar)
    # bst.get_booster().get_score(importance_type='gain')
    # get numerical feature importances
    importances = list(xgb.feature_importances_)
    feat_import = [(feature, round(importance, 5)) for feature, importance in zip(X_names, importances)]
    feat_import = sorted(feat_import, key=lambda x: x[1], reverse=True)  # sort from highest to lowest importance
    picklesave(f'xgboost/feat_import_{metric}_2019tuned.txt', feat_import)

    plt.bar(list(range(len(importances))), importances, color='steelblue', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(importances))), X_names)
    plt.xticks(fontsize=5, rotation=90)
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature Name')
    plt.title(f'Feature Importance for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'xgboost/featimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    # make line plot of cumulative importances
    sorted_feat = [importance[0] for importance in feat_import]
    sorted_import = [importance[1] for importance in feat_import]
    cum_import = np.cumsum(sorted_import)
    plt.plot(list(range(len(importances))), cum_import, 'chocolate')
    plt.xticks(list(range(len(importances))), sorted_feat)
    plt.xticks(fontsize=5, rotation=90)
    plt.xlabel('Feature Name')
    plt.ylabel('Cumulative Feature Importance')
    plt.title(f'Cumulative Feature Importance for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'xgboost/cumfeatimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    # identify the 15 most important features for each metric
    top_feat_import_names = [y[0] for x, y in enumerate(feat_import[0:15])]
    top_feat_import_val = [y[1] for x, y in enumerate(feat_import[0:15])]

    print('')
    print(f'{metric} feature importance:')
    print(np.vstack(feat_import[0:15]))

    # bar chart plot of top 15 most important features
    plt.figure()
    plt.bar(list(range(len(top_feat_import_val))), top_feat_import_val, color='grey', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(top_feat_import_val))), top_feat_import_names)
    plt.xticks(fontsize=8, rotation=90)
    plt.ylabel('Importance')
    plt.ylim([0, 0.08])
    # plt.xlabel('Feature Name')
    plt.title(f'{metric}')
    plt.gca().yaxis.grid(True, which='both')
    plt.gcf().set_size_inches(6, 3)
    plt.tight_layout()
    plt.savefig(f'xgboost/subset_featimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    # scatter plot true vs predicted values
    plt.figure()
    plt.scatter(test_y[:, i], pred_y_test, color='chocolate', s=10, alpha=0.5, label='Random Forest Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for baseline model')
    plt.legend()
    plt.grid()
    plt.savefig(f'xgboost/scatter_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

