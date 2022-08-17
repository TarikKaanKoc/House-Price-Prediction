################################################
##                                            ##
## 02-Data_Preprocessing_Feature_Engineering  ##
##                                            ##
################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.width', 600)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df = pd.concat((train, test)).reset_index(drop=True)

# Drop columns - that have many NAN values
df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','LotFrontage'], axis = 1, inplace = True)

# I'll fly the id variable because it doesn't make any sense
df.drop('Id', axis = 1, inplace=True)

# Above, we transformed year variables from numerical to categorical variables. In order to make calsulations, we need take back the tramsformation.
df['YearBuilt'] = df['YearBuilt'].astype(int)
df['YearRemodAdd'] = df['YearRemodAdd'].astype(int)
df['YrSold'] = df['YrSold'].astype(int)


# Columns to be transformed
transform_num_to_cat = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath',
                        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                        'Fireplaces', 'GarageCars', 'MoSold']

# Conversion Process
for col in transform_num_to_cat:
    df[col] = df[col].astype(str)

df.info()

# DETECTION OF CATEGORİCAL VARİABLES
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
len(cat_cols)
# DETECTION OF NUM VARİABLES
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Id", 'SalePrice']]
len(num_cols)



#                                                       * Feature Engineering and Data Prepocessing *
# -------------------------------------------------------------------------------------------------------------------


# age of the building
df["Building_Age"] = (df["YearBuilt"].max() - df["YearBuilt"])
num_cols.append('Building_Age')
# Shows the age of the building when it was sold
df['Building_Sold_Age'] = df['YrSold'] - df['YearBuilt']
num_cols.append('Building_Sold_Age')

# Shows how new the building was when sold
df['So_New_At_Sold'] = df['YrSold'] - df['YearRemodAdd']
num_cols.append('So_New_At_Sold')



'''
# We've seen in EDA Operations that there are some classes of nan, 
which means these buildings don't have garages.
 (From GarageYrBlt feature)
'''

df['GarageYrBlt'].isnull().sum() # Total of : 159 Number of buildings without garage
df["GarageYrBlt"].fillna(0, inplace=True) # fillna @ 0

# In the next step we are going to create a new column called 'HasGarage' that shows if the building has a garage or not.
df['GarageYrBlt'] = df['GarageYrBlt'].astype(float)
# Shows the age of the garage when it was sold
df['Garage_Age']= df['YrSold'] - df['GarageYrBlt']
df.drop('GarageYrBlt', axis = 1, inplace=True)
num_cols.remove('GarageYrBlt')
df['Garage_Age'].isnull().sum() # out: 0
# Customize
num_cols.append('Garage_Age')

# Shows if the house has a garage or not
df['Has_Garage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has_Garage'] = df['Has_Garage'].astype(str)

# Drop the coluns, that we have used and we do no need anymore
df.drop(['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold'], axis = 1, inplace = True)
num_cols.remove('YearBuilt')
num_cols.remove('YearRemodAdd')
num_cols.remove('LowQualFinSF')
num_cols.remove('YrSold')
cat_cols.remove('MoSold')

# Relationship between 'PoolArea' and 'SalePrice'
sns.jointplot(x="PoolArea", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()
'''
We see that they are very rare, 
but there is a meaningful difference between the ones with and without a pool
So, let's create a feature that shows if the house has a pool or not , and then drop 'PoolArea'
'''
df['Has_Pool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has_Pool'] = df['Has_Pool'].astype(str)
df.drop('PoolArea', axis = 1, inplace = True)

# Customize to lists
cat_cols.append('Has_Pool')
num_cols.remove('PoolArea')

''''
# New Features 
1- Shows if the house has the second floor or not
2- Shows if the house has a basement or not
3- Shows if the house has a fireplace or not
4- shows if the building has masonry area or not
5- Shows the total number of bathrooms
6- Shows if the the surface of the building
7- Shows the total surface of porches
'''
# CRE Features
df['Has_2nd_Floor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['Has_Bsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['MasVnrArea' + 'HasOrNot'] = np.where(df['MasVnrArea'] > 0, 1, 0)
# transform for calcu
df['Fireplaces'] = df['Fireplaces'].astype(int)
# apply
df['Has_Fireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Transform
df['Has_2nd_Floor'] = df['Has_2nd_Floor'].astype(str)
df['Has_Bsmt'] = df['Has_Bsmt'].astype(str)
df['MasVnrAreaHasOrNot'] = df['MasVnrAreaHasOrNot'].astype(str)
df['Fireplaces'] = df['Fireplaces'].astype(str)
df['Has_Fireplace'] = df['Has_Fireplace'].astype(str)
# Customize to lists
cat_cols.append('Has_2nd_Floor')
cat_cols.append('Has_Bsmt')
cat_cols.append('Has_Fireplace')
cat_cols.append('MasVnrAreaHasOrNot')

# We need to transform the variables to float data type, to be able to make calculations. (for: total number of bathrooms)
for col in ['HalfBath', 'BsmtHalfBath', 'BsmtFullBath', 'FullBath']:
    df[col] = df[col].astype(float)

# Shows the total number of bathrooms
df["Total_Bathroom"] = ((0.5 * df["HalfBath"]) + (0.5 * df["BsmtHalfBath"]) + (df["BsmtFullBath"]) + (df["FullBath"]))

df["Total_Bathroom"].describe()

'''
count   2917.000
mean       2.218
std        0.808
min        1.000
25%        1.500
50%        2.000
75%        2.500
max        7.000
'''

# Relationship between 'Total_Bathroom' and 'SalePrice'
sns.jointplot(x="Total_Bathroom", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()

num_cols.append('Total_Bathroom')

drop_list_bath = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
cat_cols = [col for col in cat_cols if col not in drop_list_bath]
len(cat_cols)
# Drop other bathroom columns
for col in drop_list_bath:
    df.drop(col, axis=1, inplace=True)

# Shows if the the surface of the building
df['Total_SF'] = (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])
# Show the correlations
df[['Total_SF', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF']].corr()
''''
             Total_SF  TotalBsmtSF  1stFlrSF  2ndFlrSF
Total_SF        1.000        0.829     0.794     0.298
TotalBsmtSF     0.829        1.000     0.802    -0.206
1stFlrSF        0.794        0.802     1.000    -0.250
2ndFlrSF        0.298       -0.206    -0.250     1.000
'''
# Relationship between 'Total_SF' and 'SalePrice'
sns.jointplot(x="Total_SF", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show() # very nice result ! :)

num_cols.append('Total_SF')

# Shows the total surface of porches
df["Total_Porch_SF"] = (df["ScreenPorch"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["WoodDeckSF"])
# Show the correlations
df[['Total_Porch_SF', 'ScreenPorch', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'WoodDeckSF', 'SalePrice']].corr()

'''
                Total_Porch_SF  ScreenPorch  OpenPorchSF  EnclosedPorch  3SsnPorch  WoodDeckSF  SalePrice
Total_Porch_SF           1.000        0.300        0.444          0.254      0.127       0.740      0.391
ScreenPorch              0.300        1.000        0.048         -0.064     -0.030      -0.052      0.111
OpenPorchSF              0.444        0.048        1.000         -0.060     -0.009       0.038      0.316
EnclosedPorch            0.254       -0.064       -0.060          1.000     -0.033      -0.119     -0.129
3SsnPorch                0.127       -0.030       -0.009         -0.033      1.000      -0.004      0.045
WoodDeckSF               0.740       -0.052        0.038         -0.119     -0.004       1.000      0.324
SalePrice                0.391        0.111        0.316         -0.129      0.045       0.324      1.000

'''
# Relationship between 'TotPorchSF' and 'SalePrice'
sns.jointplot(x="Total_Porch_SF", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show() # very nice result ! :)

num_cols.append('Total_Porch_SF')

# Fill missing values for the buildings with these variables with 0, because they do not have any garage.
for col in ('GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)


# Function for to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


has_outliers(df, num_cols)
''''
LotArea : 24
MasVnrArea : 6
BsmtFinSF1 : 2
BsmtFinSF2 : 6
TotalBsmtSF : 2
1stFlrSF : 3
LowQualFinSF : 40
GrLivArea : 2
WoodDeckSF : 3
OpenPorchSF : 6
EnclosedPorch : 3
3SsnPorch : 37
ScreenPorch : 5
MiscVal : 103
Features: 
['LotArea',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'TotalBsmtSF',
 '1stFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'MiscVal']
'''
# Function to reassign up/low limits to the ones above/below up/low limits
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Assign outliers thresholds values for all the numerical variables
for col in num_cols:
    replace_with_thresholds(df, col)

# Check for outliers, again
has_outliers(df, num_cols) # Out : [] {(zero outlier)}




# Function to catch missing variables, count them and find ratio (in descending order)
def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=False)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df).sort_values(by="Ratio", ascending=False)
    return missing_df

df[num_cols].isnull().sum().sum() # OUT: 30
df[cat_cols].isnull().sum().sum() # OUT: 1075
missing_values_analysis(df)
# OUT:
'''
             Total Missing Values  Ratio
SalePrice                     1459 49.980
GarageCond                     159  5.450
GarageQual                     159  5.450
GarageFinish                   159  5.450
GarageType                     157  5.380
BsmtCond                        82  2.810
BsmtExposure                    82  2.810
BsmtQual                        81  2.770
BsmtFinType2                    80  2.740
BsmtFinType1                    79  2.710
MasVnrType                      24  0.820
MasVnrArea                      23  0.790
MSZoning                         4  0.140
Functional                       2  0.070
Utilities                        2  0.070
BsmtUnfSF                        1  0.030
TotalBsmtSF                      1  0.030
Electrical                       1  0.030
KitchenQual                      1  0.030
BsmtFinSF1                       1  0.030
GarageArea                       1  0.030
Exterior2nd                      1  0.030
Exterior1st                      1  0.030
SaleType                         1  0.030
BsmtFinSF2                       1  0.030

'''


'''
İMPORTANT-1: Some missing values are intentionally left blank, 
for example: In the Alley feature there are blank values meaning that there are no alley's in that specific house.

İMPORTANT-2: Impute median values for missing values for numeral variables
'''
fill_columns =["GarageFinish","GarageType",
                  'BsmtFinType1', "GarageCond",
                  'GarageQual', 'BsmtQual',
                   'BsmtExposure','BsmtCond',
                  'BsmtFinType2', 'MasVnrType']

for i in fill_columns:
    df[i] = df[i].fillna('None')

df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)


# The most logical and common classes. (fill with)
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

fillna_cat_columns = ['Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'SaleType', 'Electrical']

for col in fillna_cat_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df[cat_cols].isnull().sum().sum() # 0

# ONE HOT ENCODİNG
def one_hot_encoder(df, cat_cols, nan_as_category=True):
    original_columns = list(df.columns)
    dataframe = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_columns = one_hot_encoder(df, cat_cols)
len(new_columns) # 326

# Check if there are any dulicated columns!
duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:, ~df.columns.duplicated()]


# Robust Scaler

transfrm = RobustScaler()
df[num_cols] = transfrm.fit_transform(df[num_cols])


# Descriptive Statistics
def descriptive_statistics(df):
    describe_ = df.describe().T
    describe_df = pd.DataFrame(index=df.columns,
                               columns=describe_.columns,
                               data=describe_)

    f, ax = plt.subplots(figsize=(22,7))
    sns.heatmap(describe_df,
                annot=True,
                cmap= ['#581845', '#C70039'],
                fmt='.3f',
                ax=ax,
                linecolor='#C6D3E5',
                linewidths=3,
                cbar=False,
                annot_kws={"size": 15})
    plt.xticks(size=20)
    plt.yticks(size=20,
               rotation=0)
    plt.title("\nDescriptive Statistics\n", size=25)
    plt.show()


num_desc = df[num_cols]
descriptive_statistics(num_desc)

tarin_dataframe = df[df['SalePrice'].notnull()]
tarin_dataframe.shape # Out: (1460, 351)
test_dataframe = df[df['SalePrice'].isnull()]
test_dataframe.shape # Out: (1459, 351)

tarin_dataframe.to_pickle("train_final_v.pkl")
test_dataframe.to_pickle("test_final_v.pkl")