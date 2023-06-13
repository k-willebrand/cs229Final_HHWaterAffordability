# example implementation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import os
import pickle

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

#%% tune the hyperparameters for the random forest model to find best performing model
# Parameters to tune:
# n_estimators = number of trees in the forest
# max_features = max number of features considered for splitting a node
# max_depth = max number of levels in each decision tree
# min_samples_split = min number of data points placed in a node before the node is split
# min_samples_leaf = min number of data points allowed in a leaf node
# bootstrap = method for sampling data points (with or without replacement)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]  # prior: start=200, stop=2000
# Number of features to consider at every split
max_features = [1.0, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# print parameters to randomly test
print(random_grid)

# Use the random grid to search for best hyperparameters for each metric
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
params_tuned = []  # intialize list for saving params
for i, metric in enumerate(aff_metrics):
    # Random search of parameters, using 5-fold cross validation,
    # search across n_iter=35 different combinations, and use all available cores
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), param_distributions=random_grid, n_iter=35, cv=5,
                                verbose=2, random_state=0, n_jobs=-1)
    # Fit the random search model
    search.fit(train_x, train_y[:, i])

    picklesave(f'randomforest/randombestparams_{metric}.txt', search)
    params_tuned.append(search.best_params_)
    print(f'{metric} random best hyperparams: {search.best_params_}')

#%% perform random forest for each of the candidate affordabilty metrics

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# switch: set true to run baseline model for comparison within loop below
calc_baseline = False

# fit random forest for each metric
for i, metric in enumerate(aff_metrics):

    # load the best parameters
    search = pickleload(f'randomforest/randombestparams_{metric}.txt')
    params_tuned = search.best_params_

    # create random forest model object
    rf = RandomForestRegressor(n_estimators=params_tuned['n_estimators'], min_samples_split=params_tuned['min_samples_split'],
                               min_samples_leaf=params_tuned['min_samples_leaf'], max_features=params_tuned['max_features'],
                               max_depth=params_tuned['max_depth'], bootstrap=params_tuned['bootstrap'], n_jobs=-1)

    # train the model and save fitted parameters
    rf.fit(train_x, train_y[:, i])
    params = rf.get_params
    picklesave(f'randomforest/params_{metric}_2019tuned.txt', rf)

    # make predictions on training and test data
    pred_y_train = rf.predict(train_x)
    pred_y_test = rf.predict(test_x)
    picklesave(f'randomforest/pred_y_train_{metric}_2019tuned.txt', pred_y_train)
    picklesave(f'randomforest/pred_y_test_{metric}_2019tuned.txt', pred_y_test)

    # training data MSE
    print('')
    print(f'RF model for {metric}:')
    MSE = np.mean((train_y[:, i] - pred_y_train) ** 2)
    print(f'Train MSE: {MSE}')

    # test data MSE
    MSE = np.mean((test_y[:, i] - pred_y_test) ** 2)
    print(f'Test MSE: {MSE}')

    # calculate additional test data metrics
    abs_err = np.sum(abs(test_y[:, i] - pred_y_test))  # L1
    print(f'absolute error: {abs_err}')
    MAE = abs_err / len(test_y[:, i])
    print(f'MAE: {MAE}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i]))**2)
    RSS = np.sum((test_y[:, i] - pred_y_test)**2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS-RSS) / TSS
    print(f'R2: {R2}')
    Adj_R2 = 1 - (1-R2) * len(test_y[:, i])/(len(test_y[:, i]) - rf.n_features_in_ - 1)
    print(f'Adjusted R2: {Adj_R2}')
    SMAPE = np.sum(abs((test_y[:, i] - pred_y_test))) / np.sum(test_y[:, 0] + pred_y_test)  # alternative symmetric mean absolute error
    print(f'SMAPE: {SMAPE}')
    #accuracy = 100 - mape
    #print(f'Accuracy: {accuracy} %')

    # get numerical feature importances
    importances = list(rf.feature_importances_)
    feat_import = [(feature, round(importance, 5)) for feature, importance in zip(X_names, importances)]
    feat_import = sorted(feat_import, key=lambda x: x[1], reverse=True)  # sort from highest to lowest importance
    picklesave(f'randomforest/feat_import_{metric}_2019tuned.txt', feat_import)

    #%% make bar plot chart of feature importances
    #plt.figsize = (4.8, 100)  # default(6.4, 4.8)
    plt.bar(list(range(len(importances))), importances, color='steelblue', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(importances))), X_names)
    plt.xticks(fontsize=5, rotation=90)
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature Name')
    plt.title(f'Feature Importance for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'randomforest/featimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    #%% make line plot of cumulative importances
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
    plt.savefig(f'randomforest/cumfeatimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    #%% scatter plot true vs predicted values
    plt.figure()
    plt.scatter(test_y[:, i], pred_y_test, color='chocolate', s=10, alpha=0.5, label='Random Forest Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Random Forest model')
    plt.legend()
    plt.grid()
    plt.savefig(f'randomforest/scatter_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

#%% print and graph 15 most important features for each metric
for i, metric in enumerate(aff_metrics):
    feat_import = pickleload(f'randomforest/feat_import_{metric}_2019tuned.txt')
    top_feat_import_names = [y[0] for x, y in enumerate(feat_import[0:15])]
    top_feat_import_val = [y[1] for x, y in enumerate(feat_import[0:15])]

    print('')
    print(f'{metric} feature importance:')
    print(np.vstack(feat_import[0:15]))

    # bar chart top 15 most important features
    plt.figure()
    plt.bar(list(range(len(top_feat_import_val))), top_feat_import_val, color='grey', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(top_feat_import_val))), top_feat_import_names)
    plt.xticks(fontsize=8, rotation=90)
    plt.ylabel('Importance')
    plt.ylim([0, 0.4])
    #plt.xlabel('Feature Name')
    plt.title(f'{metric}')
    plt.gca().yaxis.grid(True, which='both')
    plt.gcf().set_size_inches(5, 3)
    plt.tight_layout()
    plt.savefig(f'randomforest/subset_featimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()


#%% optional: refit model with top 15 predictors for each metric

# the number of different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']

# fit random forest for each
for i, metric in enumerate(aff_metrics):

    # create random forest model object
    rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None)

    # pick the top 15 (?) most important predictors and refit tree model
    feat_import = pickleload(f'randomforest/feat_import_{metric}_2019.txt')
    feat_import = np.array(feat_import)
    feat_subset = list(feat_import[0:15, 0])  # first 30 most important features

    # update subset of features for training RF model
    both = set(X_names).intersection(feat_subset)
    idx = [X_names.index(x) for x in both]

    # train the model and save fitted parameters
    rf.fit(train_x[:, idx], train_y[:, i])
    params = rf.get_params
    picklesave(f'randomforest/subset_params_{metric}_2019.txt', params)

    # make predictions on test data
    pred_y = rf.predict(test_x[:, idx])

    # calculate mean absolute percentage error (MAPE), to then compute accuracy
    print('')
    print(f'RF model for {metric}:')
    abs_err = np.sum(abs((pred_y - test_y[:, i])))  # L1
    print(f'absolute error: {abs_err}')
    MSE = np.mean((pred_y - test_y[:, i]) ** 2)
    print(f'MSE: {MSE}')
    TSS = np.sum((test_y[:, i] - np.mean(test_y[:, i]))**2)
    RSS = np.sum((pred_y - test_y[:, i])**2)  # L2^2
    print(f'RSS: {RSS}')
    R2 = (TSS-RSS) / TSS
    print(f'R2: {R2}')
    #mape = np.mean(100 * (abs((pred_y - test_y[:, i]) / np.maximum(1e-10, test_y[:, 0]))))  # mean abs percentage error
    #accuracy = 100 - mape
    #print(f'Accuracy: {accuracy} %')

    # get numerical feature importances
    importances = list(rf.feature_importances_)
    feat_import = [(feature, round(importance, 5)) for feature, importance in zip(X_names, importances)]
    feat_import = sorted(feat_import, key=lambda x: x[1], reverse=True)  # sort from highest to lowest importance
    picklesave(f'randomforest/subset_feat_import_{metric}_2019.txt', feat_import)

    # make bar plot chart of feature importances
    #plt.figsize = (4.8, 100)  # default(6.4, 4.8)
    plt.bar(list(range(len(importances))), importances, color='steelblue', edgecolor='white', linewidth=1)
    plt.xticks(list(range(len(importances))), feat_import)
    plt.xticks(fontsize=5, rotation=90)
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature Name')
    plt.title(f'Feature Importance for Metric: {metric}')
    plt.gca().yaxis.grid(True)
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()
    plt.savefig(f'randomforest/subset_featimport_{metric}_2019tuned.tif', bbox_inches='tight')
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
    plt.savefig(f'randomforest/subset_cumfeatimport_{metric}_2019tuned.tif', bbox_inches='tight')
    plt.show()

    # scatter plot true vs predicted values
    plt.figure()
    plt.scatter(test_y[:, i], pred_y, color='chocolate', s=10, alpha=0.5, label='Random Forest Model')
    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='steelblue', s=10,
                alpha=0.5, label='baseline Model')
    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Theoretical Perfect Predictive Model (x=y)')
    plt.xlabel(f'True {metric}')
    plt.ylabel(f'Predicted {metric}')
    plt.title(f'True vs. Predicted {metric} for Refined Random Forest Model')
    plt.legend()
    plt.grid()
    plt.savefig(f'randomforest/subset_scatter_{metric}_2019.png', bbox_inches='tight')
    plt.show()


