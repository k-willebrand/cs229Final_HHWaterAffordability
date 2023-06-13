# run all analyses and create combined figures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from operator import itemgetter
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

#%% load and preprocess training and test data

# load the merged dataset
df_merged = pickleload('data/df_merged_2019_agg.txt')
df_merged.drop(columns=['account', 'is_delinq'], axis=1, inplace=True)

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


#%% Plot a histogram of the response variables to describe the data

# the different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
aff_metrics = ['Penalty Frequency (-)', 'Debt Duration (months)', 'Debt Severity ($)']

# initialize figure
plt.subplots(1, 3, constrained_layout=True)

for i, metric in enumerate(aff_metrics):

    # update subplot index
    plt.subplot(int(f'13{i+1}'))

    # plot histogram
    plt.hist(Y[:, i], 15, color='darkred')
    #plt.xlabel('Value')
    plt.title('')
    plt.xlim([0, max(Y[:, i])])
    plt.ylabel('Frequency', fontsize=18)
    plt.xlabel(f'{metric}', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid('minor')

plt.gcf().set_size_inches(10, 3)
# plt.suptitle(f'True vs. Predicted Metrics')
plt.savefig(f'hist_all_2019.tif', bbox_inches='tight')
plt.savefig(f'hist_all_2019.png', bbox_inches='tight')
plt.show()


#%% plot true vs predicted for each metric: random forest and lasso

# the different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
aff_metrics_title = ['Penalty Frequency', 'Debt Duration', 'Debt Severity']

# initialize figure
plt.subplots(1, 3, constrained_layout=True)

for i, metric in enumerate(aff_metrics):

    # update subplot index
    plt.subplot(int(f'13{i+1}'))

    # -- THEORETICAL PERFECT PREDICTION --

    plt.plot(test_y[:, i], test_y[:, i], 'k-', label='Perfect Prediction')

    # -- RANDOM FOREST MODEL --

    # load predictions from fitted random forest models
    pred_y_test = pickleload(f'randomforest/pred_y_test_{metric}_2019tuned.txt')

    # scatter plot true vs predicted values
    plt.scatter(test_y[:, i], pred_y_test, color='chocolate', s=12, alpha=0.5, label='Random Forest')

    # -- XGBOOST MODEL --

    # load the predictions from fitted XGBoost Model
    pickleload(f'xgboost/pred_y_test_{metric}_2019tuned.txt')

    # scatter plot true vs predicted values
    plt.scatter(test_y[:, i], pred_y_test, color='steelblue', s=8, alpha=0.5, marker="*", label='XGBoost')

    # -- LASSO REGRESSION MODEL --

    # load un-standardized predictions from fitted LASSO regression models
    pred_y_test = pickleload(f'lasso/pred_y_test_{metric}_2019.txt')

    # scatter plot true vs predicted values
    plt.scatter(test_y[:, i], pred_y_test, color='darkred', s=8, alpha=0.5, marker="d", label='LASSO Regression')

    # -- BASELINE MODEL --

    plt.scatter(test_y[:, i], np.ones_like(test_y[:, i]) * np.mean(train_y[:, i]), color='dimgrey', s=10,
                alpha=0.6, label='baseline Model')

    # -- FINALIZE FIGURE --
    plt.xlabel(f'True')
    plt.ylabel(f'Predicted')
    plt.xlim([0, 1.05*np.max(test_y[:, i])])
    plt.ylim([0, 1.05*np.max(test_y[:, i])])
    plt.title(f'{aff_metrics_title[i]}')
    plt.grid()

plt.gcf().set_size_inches(10, 4)
# plt.suptitle(f'True vs. Predicted Metrics')
plt.legend(loc="upper right")
#plt.savefig(f'scatter_all_2019.tif', bbox_inches='tight')
#plt.savefig(f'scatter_all_2019.png', bbox_inches='tight')
plt.show()

#%% plot feature importances (9 x 9 horizontal bar)

# number of features to show
n = 10

# the different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
models = ['randomforest', 'xgboost', 'lasso']

# initialize figure
fig, axes = plt.subplots(3, 3, constrained_layout=True)

for j, model in enumerate(models):
    for i, metric in enumerate(aff_metrics):

        # update current axis
        ax = axes[j, i]

        # load features
        if j == 0 or j == 1:  # random forest or xgboost
            # feat_import = pickleload(f'{model}/feat_import_{metric}_2019tuned.txt')
            feat_import = pickleload(f'{model}/feat_import_{metric}_2019tuned.txt')
            top_feat_import_names = [y[0] for x, y in enumerate(feat_import[0:n])]
            top_feat_import_val = [y[1] for x, y in enumerate(feat_import[0:n])]
        else:  # lasso
            clf = pickleload(f'{model}/params_{metric}_2019.txt')
            idx_top_feat = np.argsort(np.abs(clf.coef_))[len(clf.coef_) - n:len(clf.coef_) + 1][::-1]  # up to 10 largest coefficients
            top_feat_import_val = np.abs(itemgetter(*idx_top_feat)(clf.coef_))
            top_feat_import_names = np.array(list(itemgetter(*idx_top_feat)(X_names)))
            top_feat_import_names[top_feat_import_val == 0] = ' '


        # make labeled horizontal bar plot of top n features
        hbars = ax.barh(list(range(len(top_feat_import_val))), top_feat_import_val, color='gray', edgecolor='white', linewidth=1)
        ax.invert_yaxis()
        #plt.xticks(list(range(len(top_feat_import_val))), top_feat_import_names)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('|Coeffient|', fontsize=13)
        if j == 0:
            ax.set_title(f'{metric}', fontsize=13)
        if j == 0 or j == 1:
            ax.set_xlim((0, 1.15 * ax.get_xlim()[1]))  # accomodate variable name label
        #plt.gca().yaxis.grid(True, which='both')
        ax.grid(axis='y', which='both')
        ax.bar_label(hbars, label_type='edge', labels=top_feat_import_names)

plt.gcf().set_size_inches(20, 10)
plt.tight_layout()
#plt.savefig('topfeatimport_all_2019.png', bbox_inches='tight')
#plt.savefig('topfeatimport_all_2019.tif', bbox_inches='tight')
plt.show()

#%% plot feature importances (1 x 3 joined horizontal bar)

# number of features to show
n = 7

# the different affordability metrics to try to predict
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
models = ['randomforest', 'xgboost', 'lasso']
models_title = ['Random Forest', 'XGBoost', 'LASSO']
colors = ['chocolate', 'steelblue', 'darkred']

# initialize figure
fig, axes = plt.subplots(1, 3)

for j, model in enumerate(models):

    # update current axis
    ax = axes[j]

    df = pd.DataFrame({'rank': np.arange(n) + 1})

    #  load importances for each metric for given model and store in a dataframe df
    for i, metric in enumerate(aff_metrics):

        if j == 0 or j == 1:  # random forest or xgboost
            feat_import = pickleload(f'{model}/feat_import_{metric}_2019tuned.txt')
            top_feat_import_names = [y[0] for x, y in enumerate(feat_import[0:n])]
            top_feat_import_val = [y[1] for x, y in enumerate(feat_import[0:n])]
            #ax.set_xlabel('Importance', fontsize=13)
        else:  # lasso
            clf = pickleload(f'{model}/params_{metric}_2019.txt')
            idx_top_feat = np.argsort(np.abs(clf.coef_))[len(clf.coef_) - n:len(clf.coef_) + 1][::-1]  # up to 10 largest coefficients
            top_feat_import_val = np.abs(itemgetter(*idx_top_feat)(clf.coef_))
            top_feat_import_names = np.array(list(itemgetter(*idx_top_feat)(X_names)))
            top_feat_import_names[top_feat_import_val == 0] = ' '
            #ax.set_xlabel('|Coeffient|', fontsize=13)
        df[f'{metric}_names'] = top_feat_import_names
        df[f'{metric}_val'] = top_feat_import_val

    # make labeled horizontal group bar plot of top n features for each metric

    width = 0.28

    for i, metric in enumerate(aff_metrics):
        hbars = ax.barh(list(np.arange(n) + i * width), df[f'{metric}_val'], width, color=colors[i],
                        edgecolor='white', linewidth=1)
        ax.bar_label(hbars, label_type='edge', labels=df[f'{metric}_names'], fontsize=8)

    if j == 1 or j == 2:
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel('Rank', fontsize=13)
    ax.set(yticks=np.arange(n) + width, yticklabels=list((np.arange(n) + 1).astype(str)))
    if j == 0 or j == 2:
        ax.set_xlim((0, 1.15 * ax.get_xlim()[1]))
    else:
        ax.set_xlim((0, 1.7 * ax.get_xlim()[1]))
    ax.invert_yaxis()
    ax.set_title(f'{models_title[j]}', fontsize=13)
    if j == 1:
        ax.set_xlabel('Relative Importance', fontsize=13)
    if j == 2:
        plt.legend(['Penalty Frequency', 'Debt Duration', 'Debt Severity'], loc='lower center', fontsize=8)
    #plt.gca().xaxis.grid(True, which='both')
    #ax.grid(axis='y', which='both')


plt.gcf().set_size_inches(8.5, 5)
plt.gcf().set_size_inches(10, 4)
plt.tight_layout()
#plt.savefig('topfeatimport_all_compact_2019.png', bbox_inches='tight')
#plt.savefig('topfeatimport_all_compact_2019.tif', bbox_inches='tight')
plt.show()
