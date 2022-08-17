################################################
##                                            ##
##        01-Exploratory_Data_Analysis        ##
##                                            ##
################################################

import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.style as style
import missingno as msno
from termcolor import colored
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
cat_feat_constraints = ['#7FB3D5','#76D7C4','#F7DC6F','#85929E','#283747']
constraints = ['#581845', '#C70039']


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.shape, test.shape # Out: ((1459, 80), (1459, 80))

df = pd.concat((train, test)).reset_index(drop=True)

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=False)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df).sort_values(by="Ratio", ascending=False)
    return missing_df

def check_classes(df):
    dict = {}
    for i in list(df.columns):
        dict[i] = df[i].value_counts().shape[0]

    unq = pd.DataFrame(dict,index=["Unique Count"]).transpose().sort_values(by="Unique Count", ascending=False)
    return unq

def check_df(df, head=5, tail=5):
    print(f"\nShape of dataset: {colored(df.shape, 'red')}\n")
    print(' Duplicate Values Analysis '.center(60, '~'))
    print("\n",df.duplicated().sum(),"\n")

check_df(df)

"""
Shape of dataset: (2918, 80)
Duplicate Values: None / (0)
"""

# Now, we can see the relationship between missing values
msno.matrix(df)
plt.show()

missing_values_analysis(df)

""""
TOP 10 : 

             Total Missing Values  Ratio
PoolQC                        2912 99.790
MiscFeature                   2816 96.500
Alley                         2704 92.670
Fence                         2338 80.120
FireplaceQu                   1460 50.030
LotFrontage                    454 15.560
GarageCond                     156  5.350
GarageYrBlt                    156  5.350
GarageQual                     156  5.350
GarageFinish                   156  5.350
"""
# Drop columns - that have many NAN values
df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','LotFrontage'], axis = 1, inplace = True)

# I'll fly the id variable because it doesn't make any sense
df.drop('Id', axis = 1, inplace=True)

df.shape # Out: (2918, 73)

# Check the types again
df.info()

#  Above, we transformed year variables from numerical to categorical variables. In order to make calsulations, we need take back the tramsformation.
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


def detectionof_features(df, cat_th=15, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
            df: Dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                threshold value for numeric but categorical variables
        car_th: int, optinal
                threshold value for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical but cardinal variable list

    Examples
    ------
        You just need to call the function and send the dataframe.)

        --> grab_col_names(df)

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 returned lists equals the total number of variables:
        cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                   df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"\n\nCATEGORCİAL FEATURES: \n\n", cat_cols, "\n\n")
    print("".center(167, '~'))
    print(f"\n\nNUM FEATURES: \n\n", num_cols, "\n\n")
    print("".center(167, '~'))
    print(f"\nTotal Categorical Features: ", len(cat_cols))
    print(f"Total Numerical Features: ", len(num_cols), "\n\n")
    print("".center(167, '~'))

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = detectionof_features(df)


''''
CATEGORCİAL FEATURES: 

 ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 
 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 
 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
  'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'MoSold', 'YrSold', 'SaleType', 
  'SaleCondition', '3SsnPorch', 'PoolArea'] 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NUM FEATURES: 

 ['LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
 '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
 'EnclosedPorch', 'ScreenPorch', 'MiscVal'] 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Categorical Cols:  53
Total Numerical Cols:  21 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Now, we have 52 categorical and 20 numerical columns excluding target variable (dependent variable)

'''

check_classes(df)

'''
Ex out: 

               Unique Count
LotArea                1106
GrLivArea               879
BsmtUnfSF               793
1stFlrSF                789
TotalBsmtSF             736
BsmtFinSF1              669
GarageArea              459
etc...
...
...
'''


#                                                       * Categorical Features Analysis *
# -------------------------------------------------------------------------------------------------------------------


# cat_summary function. extra feature --> it describes target, too.
def cat_summary(df, cat_cols, label, classes=20, plot=False):
    count = 0  # How many categorical variables will be reported
    exceptions = []  # Variables with more than a certain number of classes will be stored.
    for i in cat_cols:
        # Label in box plot visualization and categorical variable breakdown
        if len(df[i].value_counts()) <= classes:  # choose by number of classes
            if plot:
                sns.boxplot(x=i, y=label, data=df)
                plt.show()
            print(pd.DataFrame({i: df[i].value_counts(),
                                "Ratio": 100 * df[i].value_counts() / len(df),
                                "TARGET_MEAN": df.groupby(i)[label].mean(),
                                "TARGET_MEDIAN": df.groupby(i)[label].median()}), end="\n\n\n")
            count += 1 # We increment the counter by one to count the categorical variable
        else:
            exceptions.append(df[i].name)
    # Report
    print('Total as %d categorical feature have been described!' % count, end="\n\n")
    print('There are', len(exceptions), "variables have more than", classes, "classes", end="\n\n")
    print('Features with more than %d classes:' % classes, exceptions, end="\n\n")

cat_summary(df, cat_cols, "SalePrice", plot=True)


#                                                       * Numerical Features Analysis *
# -------------------------------------------------------------------------------------------------------------------

def num_features_hist(df, column_name, i, hue):
    rcParams['figure.figsize'] = 30, 50
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(5, 2, i)
    sns.histplot(data=df, x=column_name, hue=hue, kde=True,color=constraints)
    plt.show()


i = 0
for column_name in num_cols:
    i +=  1
    num_features_hist(df, column_name, i, hue='SalePrice')

# Function for analysis of numerical features
def num_plot(data,num_cols):

    for i in num_cols:
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        data.hist(str(i), bins=10, ax=axes[0])
        data.boxplot(str(i), ax=axes[1], vert=False);
        try:
            sns.kdeplot(np.array(data[str(i)]))
        except:
            ValueError

        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        axes[0].set_title(i + " | Histogram")
        axes[1].set_title(i + " | Boxplot")
        axes[2].set_title(i + " | Density")
        plt.show()

num_plot(df, num_cols)


#                                                       * Label Analysis *
# -------------------------------------------------------------------------------------------------------------------


df.SalePrice.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99,1])

"""
# The distribution of the target value with respect to quantiles

count     1460.000
mean    180921.196
std      79442.503
min      34900.000
5%       88000.000
10%     106475.000
25%     129975.000
50%     163000.000
75%     214000.000
80%     230000.000
90%     278000.000
95%     326100.000
99%     442567.010
max     755000.000
"""
# Visualize: distribution of the target feature
df.SalePrice.hist(color=constraints[1])
plt.show()

np.log1p(df.SalePrice).hist(color=constraints[1])
plt.show()

#                                                       * Correlation Analysis *
# -------------------------------------------------------------------------------------------------------------------

style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(30,20))
sns.heatmap(df.corr(),
            annot=True,
            center = 0)
plt.title("Heatmap of all the Features", fontsize = 30)
plt.show()

def corr_map(df, width=23, height=7):
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df.corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap=constraints,
                mask=mtx,
                linewidth=0.4,
                linecolor="black",
                annot_kws={"size": 15})
    plt.yticks(rotation=0, size=15)
    plt.xticks(rotation=75, size=15)
    plt.title('\nCorrelation Map\n', size=40)
    plt.show()


corr_map(df)


'''
# Correlations of target with numerical independent variables by using a correlation limit 0.50
'''
def find_correlation(df, corr_limit=0.50):
    high_corr = []
    low_corr = []
    for i in num_cols:
        if i == "SalePrice":
            pass

        else:
            corr = df[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(i, corr)
            if abs(corr) > corr_limit:
                high_corr.append(i + ": " + str(corr))
            else:
                low_corr.append(i + ": " + str(corr))
    return low_corr, high_corr

low_corrs, high_corrs = find_correlation(df)

# pairplot
corr_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.set(font_scale=1.2,
        style="whitegrid",
        palette= constraints,
        font="sans-serif")
sns.pairplot(train[corr_cols],
             hue='SalePrice',
             corner = True,
             kind = 'reg');