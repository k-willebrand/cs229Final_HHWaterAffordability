# Description: merges household level and affordability metrics data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sqlite3
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

#%% load household level affordability metric data for 2019 and 2020

# load file household level affordability metric data
# note: file contains affordability metrics data for 24452 unique accounts
csvpath = 'data/acc_metric_data_2019.csv'
afford_data_2019 = pd.read_csv(csvpath, header=0)  # fields: account, is_deliq, acc_length

# save selected household level affordability metrics by account in pandas dataframe
df_afford = afford_data_2019[['account', 'is_delinq', 'acc_length', 'pen_freq', 'debt_dur', 'debt_val', 'avg_wuse']]
df_afford['is_delinq'] = df_afford['is_delinq'].astype('category')

# calculate number of unique accounts in df_afford (out: 61611)
#num_accounts = df_afford.drop_duplicates(subset='account', inplace=False).shape[0]

#%% specify feature names for household level sociodemographic and housing data for SQL query

# load file containing candidate feature names form the current unmerged database
csvpath = 'data/Database_Column_Description_select_2019.csv'
feat_descript = pd.read_csv(csvpath, header=0)
feat_names = np.array(feat_descript['Column Header'])  # array of feature names
feat_query = ", ".join(feat_names)  # common seperated string for SQL query

#%% load household level sociodemographic and housing data from SQL database

# specify SQL database file path
dbpath = 'data/wudb.db'

# create a SQL connection to the SQLite database
conn = sqlite3.connect(dbpath)

# create cursor
c = conn.cursor()

# run SQL query on database and extract feature data
# select single family households with billing data in 2019 or 2020 only
c.execute(" ".join(["SELECT", feat_query, "FROM wudata WHERE restype = 'SF' and eyr = 2019 and bdate != '' and edate != ''"]))
wudata = c.fetchall()

# close SQL connection to SQLite database
conn.close()

# save variable as list of tuples using pickle
picklesave('data/wudata_2019.txt', wudata)

#%% create and save selected sociodemographic and housing data in pandas dataframe

# note: this dataframe is wudb data prior to merge and preprocessing
df_wudata = pd.DataFrame(wudata, columns=feat_names)
picklesave('data/df_wudata_2019.txt', df_wudata)

#%% perform additional preproccessing on df_wudata to investigate available features

# load the saved dataframe
df_wudata = pickleload('data/df_wudata_2019.txt')

# remove commas in strings for data type compatability
df_wudata = df_wudata.replace(',', '', regex=True)

# omit billing data features with no data
df_wudata.drop(columns=['rts34', 'rts1', 'rts15', 'rts2', 'rts3', 'rts4', 'rts6', 'rts8'], inplace=True)
df_wudata.drop(columns=['sbmtr', 'sbmtu', 'sbmtc'], inplace=True)
df_wudata.drop(columns=['dcrf58', 'dcrf34', 'dcrf1', 'dcrf15', 'dcrf2', 'dcrf3', 'dcrf4', 'dcrf6'], inplace=True)

# omit field for whether the account is continuous 2009 - 2021
df_wudata.drop(columns=['bmon', 'byr', 'acc_con'], inplace=True)
#df_wudata.drop(columns=['bill_length'], inplace=True)

# omit census tract and block group numbers: we are interested in the properties of census groups
#df_wudata.drop(columns=['Tract_y', 'BkGp_y'], inplace=True)

# check datatypes in dataframe and specify desired column dtypes
feat_dtypes = df_wudata.dtypes

intcols = ['or_tot', 'yr_bt_own', 'yr_bt_rnt', 'X_of_Units', 'Bedrooms', 'Effective_Year', 'Fireplaces',
           'Room_Count', 'Year_Built']
catcols = ['Tract_y', 'BkGp_y', 'emon', 'eyr', 'account', 'Bathrooms_F_H', 'Condition', 'General_Plan', 'Heat',
           'Pool', 'Roof', 'Sanitation', 'Spa', 'Topography', 'View', 'Water', 'Zoning']
floatcols = ['t5r', 't5c', 't1aor', 't1aoc', 'irfaor', 'irfaou', 'irfaoc', 'totwuse', 'normtotwuse', 'hc_own_20K_20',
             'hc_own_20K_29', 'hc_own_20K_30_p', 'hc_rnt_20K_20', 'hc_rnt_20K_29', 'hc_rnt_20K_30_p',
             'hc_rnt_35K_20', 'hc_rnt_35K_29', 'hc_rnt_35K_30_p', 'hc_rnt_50K_20', 'hc_rnt_50K_29', 'hc_rnt_50K_30_p',
             'aggrm_own', 'aggrm_rnt', 'Garage', 'Main_Area', 'Parcel_Size_acres', 'tax_value']

# replace occurrence of 'NA' and 'N/A' strings appropriate nulltype and update datatypes

# integers
df_wudata[intcols] = df_wudata[intcols].fillna(-9999)
df_wudata[intcols] = df_wudata[intcols].replace('NA', -9999)
df_wudata[intcols] = df_wudata[intcols].replace('N/A', -9999)
df_wudata[intcols] = df_wudata[intcols].astype('int64')

# categoricals
df_wudata[catcols] = np.where(df_wudata[catcols] == 'NA', np.nan, df_wudata[catcols])
df_wudata[catcols] = np.where(df_wudata[catcols] == 'N/A', np.nan, df_wudata[catcols])
df_wudata[catcols] = np.where(df_wudata[catcols] == 'nan', np.nan, df_wudata[catcols])
df_wudata[catcols] = np.where(df_wudata[catcols] == 'None', np.nan, df_wudata[catcols])
df_wudata[catcols] = df_wudata[catcols].astype('category')

# floats
df_wudata[floatcols] = np.where(df_wudata[floatcols] == 'NA', np.nan, df_wudata[floatcols])
df_wudata[floatcols] = np.where(df_wudata[floatcols] == 'N/A', np.nan, df_wudata[floatcols])
df_wudata[floatcols] = np.where(df_wudata[floatcols] == 'nan', np.nan, df_wudata[floatcols])
df_wudata[floatcols] = np.where(df_wudata[floatcols] == 'None', np.nan, df_wudata[floatcols])
df_wudata[floatcols] = df_wudata[floatcols].astype('float64')
df_wudata[floatcols] = df_wudata[floatcols].round(decimals=5)  # account for rounding errors

# check updated datatypes in dataframe to confirm correct
feat_dtypes = df_wudata.dtypes

# update water billing nan fields with 0's rather than nans as appropriate
billcols = ['t1aor', 't1aou', 't1aoc', 'irfaor', 'irfaou', 'irfaoc', 'elscr', 'elscu', 'elscc', 'rsfr', 'rsfu', 'rsfc']
for i in range(5):
    billcols.append(f't{i+1}r')
    billcols.append(f't{i + 1}u')
    billcols.append(f't{i + 1}c')
for i in range(4):
    billcols.append(f'irft{i + 1}r')
    billcols.append(f'irft{i + 1}u')
    billcols.append(f'irft{i + 1}c')
df_wudata[billcols] = df_wudata[billcols].fillna(0)

# save current non-aggregated version (monthly account info)
picklesave('data/df_wudata_2019_mon.txt', df_wudata)

#%% create annual aggregation of 2019 data

# take annual average value of numeric datatypes for each unique account
df_wudata_num = df_wudata.groupby('account', as_index=False).mean()  # numeric data

# keep most complete housing data per account
# df_wudata.groupby('account')[catcols].agg(pd.Series.mode)
df_wudata['num_nans'] = df_wudata[catcols].isnull().sum(1)
df_wudata = df_wudata.sort_values(by=['account', 'num_nans'], ascending=[True, True])
df_wudata_cat = df_wudata.drop_duplicates(subset='account', keep='first')[catcols]

# merge categorical and numerical data by account number
df_wudata_agg = pd.merge(df_wudata_num, df_wudata_cat, how="inner", on='account')

#%% merge with affordability metrics data and save merged dataset

# merge datasets by account number
df_merged = pd.merge(df_wudata_agg, df_afford, how="inner", on='account')

# drop features with excessive missing data: num_nans (>=3,000)
num_nans = df_merged.isnull().sum(0)  # missing data in each column
cols_drop = np.where(num_nans >= 3000)
df_merged.drop(df_merged.columns[cols_drop], axis=1, inplace=True)

# drop rows/accounts with debt_val > 900 (see Rachunok et al.)
rows_drop = np.where([df_merged['debt_val'] >= 900])[1]
df_merged.drop(index=df_merged.index[rows_drop], axis=0, inplace=True)

# finally, drop rows/accounts with missing data fields: num_nans (> 0)
num_nans = df_merged.isnull().sum(1)  # missing data in each row
rows_drop = np.where(num_nans > 0)[0]
df_merged.drop(index=df_merged.index[rows_drop], axis=0, inplace=True)

# drop monthly fields not applicable at annual scale
df_merged.drop(columns=['emon', 'eyr', 'acc_length'])

# calculate number of unique accounts in df_merged (out: 10,101)
num_accounts = df_merged.shape[0]
# note: df_afford contains 61611 unique account (no duplicates)

# save the merged aggregate dataset for easy future loading
picklesave('data/df_merged_2019_agg.txt', df_merged)

#%% plot histogram of affordability metrics in the merged data
aff_metrics = ['pen_freq', 'debt_dur', 'debt_val']
fig, axes = plt.subplots(nrows=len(aff_metrics), ncols=1)
fig.tight_layout()
for i, metric in enumerate(aff_metrics):

    plt.subplot(int(str(len(aff_metrics)) +str(1) + str(i+1)))
    plt.hist(df_merged[metric], color='steelblue', bins=50)
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {metric}')

plt.savefig('data/hist_metrics.png', bbox_inches='tight')
