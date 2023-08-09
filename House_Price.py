#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#loading the dataset
data=pd.read_csv("data.csv")


# In[4]:


# looking the top 5 rows of dataset
data.head()


# In[5]:


#data types of variable
data.dtypes


# In[7]:


#finding the variable having int64 datatype
data.dtypes[data.dtypes=='int64']


# In[8]:


#converitng the below variable into category variable
data['MSSubClass']=data['MSSubClass'].astype('category')
data['OverallQual']=data['OverallQual'].astype('category')
data['OverallCond']=data['OverallCond'].astype('category')
data['YrSold']=data['YrSold'].astype('category')
data['MoSold']=data['MoSold'].astype('category')


# In[9]:


#converting all the object datatype to category
for i in data.columns[1:]:
  if data[i].dtypes=='object':
    data[i]=data[i].astype('category')


# In[10]:


#collecting all the category variable and continuous variable
category_vars=[]
continuous_vars=[]
for i in data.columns[1:]:
  if data[i].dtypes=='int64' or data[i].dtypes=='float64':
    continuous_vars.append(i)
  elif data[i].dtypes=='category':
    category_vars.append(i)

category_vars


# In[11]:


#missing values in category and continuous 'LotFrontage', 'MasVnrArea', 'GarageYrBlt
data.isna().sum().sum()
data[continuous_vars].isna().sum(),data[category_vars].isna().sum()


# In[12]:


# finding the mode and median of all the category and continuous variable having missing values
Alley_m=data['Alley'].mode()
MasVnrType_m=data['MasVnrType'].mode()
BsmtQual_m=data['BsmtQual'].mode()
BsmtCond_m    =  data['BsmtCond'].mode()
BsmtExposure_m   =data['BsmtExposure'].mode()
BsmtFinType1_m=data['BsmtFinType1'].mode()
BsmtFinType2_m=data['BsmtFinType2'].mode()
Electrical_m=data['Electrical'].mode()
FireplaceQu_m=data['FireplaceQu'].mode()
GarageType=data['GarageType'].mode()
GarageFinish=data['GarageFinish'].mode()
GarageQual=data['GarageQual'].mode()
GarageCond =data['GarageCond'].mode()
PavedDrive =data['PavedDrive'].mode()
PoolQC =data['PoolQC'].mode()
Fence   =data['Fence'].mode()
MiscFeature  =data['MiscFeature'].mode()
LotFrontage =data['LotFrontage'].median()
MasVnrArea  =data['MasVnrArea'].median()
GarageYrBlt =data['GarageYrBlt'].median()


# In[13]:


#Filling the null values with fillna funciton
data['Alley']=data['Alley'].fillna(Alley_m.values[0])
data['MasVnrType']=data['MasVnrType'].fillna(MasVnrType_m.values[0])
data['BsmtQual']=data['BsmtQual'].fillna(BsmtQual_m.values[0])
data['BsmtCond']=data['BsmtCond'].fillna(BsmtCond_m.values[0])
data['BsmtExposure']=data['BsmtExposure'].fillna(BsmtExposure_m.values[0])
data['BsmtFinType1']=data['BsmtFinType1'].fillna(BsmtFinType1_m.values[0])
data['BsmtFinType2']=data['BsmtFinType2'].fillna(BsmtFinType2_m.values[0])
data['Electrical']=data['Electrical'].fillna(Electrical_m.values[0])
data['FireplaceQu']=data['FireplaceQu'].fillna(FireplaceQu_m.values[0])
data['GarageType']=data['GarageType'].fillna(GarageType.values[0])
data['GarageFinish']=data['GarageFinish'].fillna(GarageFinish.values[0])
data['GarageQual']=data['GarageQual'].fillna(GarageQual.values[0])
data['GarageCond']=data['GarageCond'].fillna(GarageCond.values[0])
data['PavedDrive']=data['PavedDrive'].fillna(PavedDrive.values[0])
data['PoolQC']=data['PoolQC'].fillna(PoolQC.values[0])
data['Fence']=data['Fence'].fillna(Fence.values[0])
data['MiscFeature']=data['MiscFeature'].fillna(MiscFeature.values[0])
data['LotFrontage']=data['LotFrontage'].ffill()
data['MasVnrArea']=data['MasVnrArea'].fillna(MasVnrArea)
data['GarageYrBlt']=data['GarageYrBlt'].fillna(GarageYrBlt)


# In[14]:


#checking for null values
data.isna().sum().sum()


#  **Univariate Analysis**
# 

# In[25]:


# 1. custom function for easy and efficient analysis of numerical univariate

def UVA_numeric(data, var_group):
  '''
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
  '''

  size = len(var_group)


  #looping for each variable
  for j,i in enumerate(var_group):

    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information

    sns.kdeplot(data[i], fill=True)
    plt.xlabel('{}'.format(i), fontsize = 10)
    plt.ylabel('density')
    plt.title('std={}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format(round(st_dev),(round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))
    plt.show()


# In[26]:


#Calling UVA_numeric funciton
UVA_numeric(data,continuous_vars)


# **Summary**:
# 1**.LotArea**-
#        The variable exhibits a relatively high standard deviation, indicating a wide spread of values around the mean. The positive kurtosis value suggests a heavy-tailed distribution with a relatively large number of outliers or extreme values. The positive skewness indicates that the distribution is skewed to the right, with a longer tail on the right side.
# Overall, the variable shows significant variability and deviation from a normal distribution, with a pronounced right skewness and high kurtosis, indicating the presence of outliers or extreme values in the upper range of the distribution
# 
# 2.**YearBuilt**-The variable has a relatively low standard deviation of 30, indicating a narrow spread of values around the mean. The negative kurtosis value suggests a slightly flatter distribution compared to a normal distribution. The negative skewness indicates that the distribution is slightly skewed to the left, with a longer tail on the left side
# 
# 3.**BsmtUnfSF**:The variable has a high standard deviation of 442, indicating a wide spread of values around the mean. The positive kurtosis value of 0.47 suggests a slightly heavier tail compared to a normal distribution. The positive skewness of 0.92 indicates that the distribution is skewed to the right, with a longer tail on the right side.
# 4.1stFlrSF:the variable exhibits a wide range of values with a relatively high standard deviation and positive skewness, indicating a skewed distribution with a longer tail on the right side. The high positive kurtosis suggests a significant presence of outliers or extreme values in the upper range of the distribution
# 
# 5.**Overall** summary :In the above graph we can see the some variables are normally distributed and some have low outliers while some have high outliers. we can also see there is left skewness and right skewness.
# positive kurtosis indicates a more peaked and heavy-tailed distribution, while negative kurtosis indicates a flatter and light-tailed distribution relative to a normal distribution
# 

# In[30]:


# 2. function for visualisation of Categorical Variables and it bar graph distribution
def UVA_category(data, var_group):

  '''
  Univariate_Analysis_categorical
  takes a group of variables (category) and plot/print all the value_counts and barplot.
  '''
  # setting figure_size
  size = len(var_group)


  # for every variable
  for j,i in enumerate(var_group):
    norm_count = data[i].value_counts(normalize = True)
    n_uni = data[i].nunique()

  # Define a color palette with different colors for each category
    colors = sns.color_palette('Set3', n_colors=n_uni)

  #Plotting the variable with every information
    plt.bar(norm_count.index,norm_count,color=colors,width=1)
    plt.xlabel('SaleCondition', fontsize = 20)
    plt.ylabel('Count', fontsize = 20)
    plt.title('top 3 value counts\n{};'.format(norm_count[0:3]))
    plt.show()


# In[31]:


#calling UVA_category function
UVA_category(data,category_vars)


# Summary:
# **MSSubClass**: The majority of the dwellings involved in the sale belong to two categories, 20 and 60,
# which together account for more than 58% of the total MSSubClass values.
# 
# **MSZoning:** The RL type of zoning is the most prevalent, constituting approximately 79% of the observations.
# 
# **Street**: The vast majority, around 99.5%, of the properties have a paved street, while a very small percentage have a gravel street.
# 
# **Alley**: The variable "Alley" represents the type of alley access to the property.
# It is observed that the number of properties with a gravel alley is slightly higher compared to other types of alley access.
# 
# **LotShape**: The general shape of the properties is primarily regular, accounting for approximately 63% of the observations.
# 
# **LandContour:** The majority of the properties exhibit a flat level, with approximately 90% of the data falling under this category.
# 
# **Utilities:** The "Utilities" variable indicates the type of utilities available. The vast majority, around 99%, have access to all public utilities, including electricity, gas, water, and sewage.
# 
# **HouseStyle:** The analysis reveals that the most common house style is "1Story," which comprises approximately 50% of the observations.
# 
# **RoofStyle:** The "RoofStyle" variable indicates the style of the roof. The analysis shows that "Gable" roofs are the most prevalent, accounting for around 78% of the data, followed by approximately 20% of roofs with a "Hip" style.
# In summary:
# 
# the analysis provides insights into various features of the dataset. It reveals the predominant categories within each variable,
# such as the most common dwelling type, zoning type, street type, and property shape.
# These findings can help in understanding the distribution and characteristics of the data,
# which may be useful for further analysis or decision-making processes related to the dataset.
# 
# 

# In[32]:


#removing the saleprice from list as saleprice is target variable
continuous_vars.remove('SalePrice')


# In[33]:


#Removing the outliers do we can clearnly visualize the grap and see the trend
# Dealing with outliers in the numerical variables
for i ,j in enumerate(continuous_vars):
  quant25=data[j].quantile(0.25)
  quant75=data[j].quantile(0.75)
  IQR=quant75-quant25
  median=data[j].median()
  # whisker
  whis_low=median-(1.5*IQR)
  whis_high=median+(1.5*IQR)
  # replacing outliers with max/min whisker

  data[j][data[j]>whis_high] = whis_high+1
  data[j][data[j]<whis_low] = whis_low-1


# **Bivariate Analysis**
# 
# 

# Visualize the relationship between continuous variables and the target variable

# In[ ]:


# 2.1 Plot  between continuous variable and target variable
# Visualize the relationship between continuous variables and the target variable
for var in continuous_vars[:-1]:  # Excluding 'SalePrice'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=var, y='SalePrice', color='darkblue', alpha=0.7)
    plt.xlabel(var)
    plt.ylabel('SalePrice')
    plt.title(f"{var} vs SalePrice")
    plt.show()



# **Visulaize the relationship between category variables and target variable(SalePrice)**

# In[ ]:


# 2.2 Plot relationship between category variables and target variable(SalePrice)

# Iterate over each categorical variable and plot the relationship with 'SalePrice'
for var in category_vars:
    plt.figure(figsize=(12, 6))
    #sns.barplot(x=var, y='SalePrice', data=data)
    sns.boxplot(x=var, y='SalePrice', data=data)
    plt.xlabel(var)
    plt.ylabel('SalePrice')
    plt.title('Bar Plot: {} vs SalePrice'.format(var))
    plt.xticks(rotation=90)
    plt.show()


# Correlation

# Continuous-Continuous Variables

# In[ ]:


# 3.1 Calculate correlation coefficient between the pairs of continuous-continuous variables
#adding SalePrice in continuous_vars list
continuous_vars.append('SalePrice')
# calculating Pearson correlation coefficient
data[continuous_vars].corr()


# The variable with the highest positive correlation to "SalePrice" is "GrLivArea" with a correlation coefficient of 0.709.
# Other variables with relatively high positive correlations include "GarageCars" (0.640), "GarageArea" (0.623), "TotalBsmtSF" (0.614), and "1stFlrSF" (0.606).
# Variables like "FullBath", "TotRmsAbvGrd", "YearBuilt", and "YearRemodAdd" also show moderate positive correlations.
# 
# 

# In[ ]:


# 3.1 Calculate correlation coefficient between the pairs of categorical-continuous variables

#The correlation ratio (eta) can be used to see if one or more categories have more influence among all categories

def correlation_ratio(categories, measurements):
  '''
  1.First, we calculate the mean value of each category and the mean of all values.
  2.we calculate the ratio of the weighted sum of the squares of the differences between each category’s average and
    overall average to the sum of squares between each value and overall average
        Parameters:
        categories (array-like): Categorical variable.
        measurements (array-like): Measurement variable

        retrun :
        float: Correlation ratio value.
  '''
  fcat, _ = pd.factorize(categories)
  cat_num = np.max(fcat)+1
  y_avg_array = np.zeros(cat_num)
  n_array = np.zeros(cat_num)
  for i in range(0,cat_num):
      cat_measures = measurements[np.argwhere(fcat == i).flatten()]
      n_array[i] = len(cat_measures)
      y_avg_array[i] = np.average(cat_measures)
  y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
  numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
  denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
  if numerator == 0:
      eta = 0.0
  else:
      eta = np.sqrt(numerator/denominator)
  return eta


# In[ ]:


import itertools
#function can calculate eta for various columns in a data frame.
def cat_cont_eta(df, categorical_features, continuous_features):
    '''
    Calculates the correlation ratio (eta) between categorical and continuous variables.

    Parameters:
        df (DataFrame): Input DataFrame containing the data.
        categorical_features (list): List of categorical variable column names.
        continuous_features (list): List of continuous variable column names.

    Returns:
        DataFrame: Correlation ratio matrix between categorical and continuous variables.
    '''
    eta_corr = []
    for pair in itertools.product(categorical_features, continuous_features):
        try:
            eta_corr.append(correlation_ratio(df[pair[0]], df[pair[1]]))
        except ValueError:
            eta_corr.append(0)
    eta_corr = np.array(eta_corr).reshape(len(categorical_features), len(continuous_features))
    eta_corr = pd.DataFrame(eta_corr, index=categorical_features, columns=continuous_features)
    return eta_corr


# In[ ]:


#Calling cat_cont funciton for calculting eta values
eta_df=cat_cont_eta(data,category_vars,continuous_vars)
# dataframe have correlation coefficient between the pairs of categorical-continuous variables
eta_df


# The range of eta is between 0 and 1. A value closer to 0 indicates all
# categories have similar values, and any single category doesn’t have more
# influence on variable y. A value closer to 1 indicates one or more
# categories have different values than other categories and have more influence on variable y.

# In[ ]:


# Calculate correlation coefficient between the pairs of categorical-categorical variables

import scipy
def cramers_v(x, y):
  #this function calculate correlation between two categorical variables based on the chi2 test statistic known as  Cramer’s V.
    confusion_matrix = pd.crosstab(x,y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[ ]:


#function that is use to find Cramer’s V in a data frame with categorical variable
def cat_cat_cramers_v(df, cat_features):
    cramers_v_corr = []
    for pair in itertools.product(cat_features, repeat=2):
        try:
            cramers_v_corr.append(cramers_v(df[pair[0]], df[pair[1]]))
        except ValueError:
            cramers_v_corr.append(0)
    cramers_v_corr = np.array(cramers_v_corr).reshape(len(cat_features),len(cat_features))
    cramers_v_corr = pd.DataFrame(cramers_v_corr, index=cat_features, columns=cat_features)
    return cramers_v_corr


# In[ ]:


#Calling the cat_cat_cramers_v function to see the correlation between two categorical variables
cramers=cat_cat_cramers_v(data,category_vars)
cramers


# In[ ]:


#3.4 descriptive statistics of a variable
variable_stats = data.describe()
variable_stats


# The summary provides information about the  mean, standard deviation (measure of variability),
#  minimum, quartiles (25th, 50th, and 75th percentiles), and maximum values for each variable in the dataset

# In[ ]:


# 9. Preprocessing the data

#checking for missing values
data.isna().sum().sum()


# In[ ]:


#Importing the library for preprocessing
from sklearn.preprocessing import LabelEncoder,StandardScaler

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical variable
for i,j in enumerate(category_vars):
  data[category_vars[i]] = label_encoder.fit_transform(data[category_vars[i]])


# In[ ]:


#depndent and independent variable
X=data.drop('SalePrice',axis=1)
y=data['SalePrice']


# In[ ]:


# Perform data preprocessing
#Scalling the values using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X=pd.DataFrame(X,columns=['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'])
X.head()


# In[ ]:


#train_test_split function from scikit-learn to split your data into training and testing sets
from sklearn.model_selection import train_test_split,KFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#Models

#Importing the necessary algorithm as told in question
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingRegressor


# In[ ]:


# 4. 1 building the linear regression
Lr=LinearRegression()
Lr.fit(X_train,y_train)
predict_test=Lr.predict(X_test)
predict_train=Lr.predict(X_train)
error_test=np.sqrt(msle(np.abs(y_test),np.abs(predict_test)))
error_train=np.sqrt(msle(np.abs(y_train),np.abs(predict_train)))
print('RMSLE for train: ',error_train)
print('RMSLE for test: ' ,error_test)


# In[ ]:



# Importing ridge from sklearn's linear_model module
from sklearn.linear_model import Lasso

#Define the alpha values to test [0, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
alpha_lasso = [250,400,600,700,800,900,1100,1500,2000,3000]


# In[ ]:


# 4.2 defining a function which will fit lasso regression model and return the coefficients
def lasso_regression(train_x, train_y, test_x, test_y, alpha):
  lassoreg = Lasso(alpha=alpha)
  lassoreg.fit(train_x,train_y)
  train_y_pred = lassoreg.predict(train_x)
  test_y_pred = lassoreg.predict(test_x)
  #root means squared log error
  rmsle_train=np.sqrt(msle(train_y,train_y_pred))
  rmsle_test=np.sqrt(msle(test_y,test_y_pred))
  ret=[rmsle_train]
  ret.extend([rmsle_test])

  ret.extend([lassoreg.intercept_])
  ret.extend(lassoreg.coef_)


  return ret


# In[ ]:


#Initialize the dataframe to store coefficients
col = ['rmsle_train','rmsle_test','intercept'] + ['coef_Var_%d'%i for i in range(1,81)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)


# In[ ]:


#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(X_train, y_train, X_test, y_test, alpha_lasso[i])


# In[ ]:


#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_lasso


# In[ ]:


coef_matrix_lasso[['rmsle_train',	'rmsle_test']].plot()
plt.xlabel('Alpha Values')
plt.ylabel('MRSS')
plt.legend(['train', 'test'])


# In[ ]:


# 4.3 Now using the optimized alpha value for the Lasso model so we can decrease the error
alpha=1500
lassoreg = Lasso(alpha=1500)
lassoreg.fit(X_train,y_train)
train_y_pred = lassoreg.predict(X_train)
test_y_pred = lassoreg.predict(X_test)
#root means squared log error
rmsle_train=np.sqrt(msle(y_train,train_y_pred))
rmsle_test=np.sqrt(msle(y_test,test_y_pred))
print("RMSLE for train: ",rmsle_train)
print('RMSLE for test: ',rmsle_test)


# **Great we can be able to decrease the error by using Regularization from approx 0.18 to 0.14 **

# In[ ]:


#4.4  Bagging model (Random Forest)
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
predict_train=random_forest.predict(X_train)
predict_test= random_forest.predict(X_test)

rmslet=np.sqrt(msle(y_train,predict_train))
rmslete=np.sqrt(msle(y_test,predict_test))
print('RMSLE for train(RandomFores): ',rmslet)
print('RMSLE for test(RandomFores): ',rmslete)



# 4.5 Boosting model (AdaBoost)
adaboost = AdaBoostRegressor()
adaboost.fit(X_train, y_train)
y_pred_train=adaboost.predict(X_train)
y_pred_test = adaboost.predict(X_test)
error_train=np.sqrt(msle(y_train,y_pred_train))
error_test=np.sqrt(msle(y_test,y_pred_test))

print('RMSLE for train (AdaBoost):', error_train)
print('RMSLE for train (AdaBoost):', error_test)


# In[ ]:


# 5 We have define a function which takes input linear model instance,independent variable,dependent variable
# It return the RMSLE for the given model

def model_error(ml_model,x,y,rstate = 11):
    i = 1
    rmsle1= []
    x=x
    y=y
    #Creating instance of KFold() and puting n_splits=5
    kf = KFold(n_splits=5,random_state=rstate,shuffle=True)
    for train_index,test_index in kf.split(x,y):
        print('\n{} of kfold {}'.format(i,kf.n_splits))
        x_train,x_test = x.loc[train_index],x.loc[test_index]
        y_train1,y_test1 = y[train_index],y[test_index]

        model = ml_model
        model.fit(x_train, y_train1)
        pred_test = model.predict(x_test)

        rmsle =np.sqrt(msle(np.abs(pred_test),np.abs(y_test1)))
        sufix = ""
        msg = ""

        msg += "Valid RMSLE: {:.5f}".format(rmsle)
        print("{}".format(msg))
        # Save scores
        rmsle1.append(rmsle)
        i+=1
    return rmsle1


# In[ ]:


#Importing the decision tree regressor
from sklearn.tree import DecisionTreeRegressor

#Creating instance of decisiontreeregressor  and putting hyperparameter
dtr=DecisionTreeRegressor(random_state=12,max_depth=15,min_samples_leaf=25, min_samples_split=25)


# In[ ]:


##Calling the function model_score()
dtr1=model_error(dtr,X,y)


# In[ ]:


#Call model error with linear regression instance
lr=model_error(Lr,X,y)


# In[ ]:


#calling model error funciton with random forest instance
rfr=model_error(random_forest,X,y)


# In[ ]:


#shape of train and test
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# **Linear regression and randomforest is working well**

# In[ ]:


# 6: Hyperparameter Tuning
# Grid search for Random Forest hyperparameters
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]    #If our model not perform well on this hyperparameter then we can all add more hyperparameter

}


grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=3)
grid_search_rf.fit(X_train, y_train)

#  the best Random Forest model
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
rmsle_b = np.sqrt(msle(y_test, y_pred_best_rf))
print('best rmsle (Random Forest):', rmsle_b)


# Grid search for AdaBoost hyperparameters
param_grid_adaboost = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

grid_search_adaboost = GridSearchCV(estimator=adaboost, param_grid=param_grid_adaboost, cv=3)
grid_search_adaboost.fit(X_train, y_train)

# Get the best AdaBoost model
best_adaboost = grid_search_adaboost.best_estimator_
y_pred_best_adaboost = best_adaboost.predict(X_test)
rmsle_add = np.sqrt(msle(y_test, y_pred_best_adaboost))
print('Best rmsle (AdaBoost):', rmsle_add)


# **Well** we can see the lowest error is in  random forest approx 0.14 which is better than addaboost

# In[ ]:


# 7 Feature selection techniques
# importing the SequentialFeatureSelector from mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# initallization SequentialFeatureSelector with randomforest instance and other parameter of sfs (forward feature selection)
sfs1 = sfs(best_rf, k_features=4, forward=True, verbose=2, scoring='neg_mean_squared_log_error',n_jobs=-1)

#sfs1 = sfs(best_rf, k_features=30, forward=False, verbose=2, scoring='neg_mean_squared_log_error') (backward elimination)


# In[ ]:


#fittig SequentialFeatureSelector model
sfs1 = sfs1.fit(X, y)


# In[ ]:


feat_names = list(sfs1.k_feature_names_)
print(feat_names)


# In[ ]:


# creating a new dataframe using the above variables and adding the target variable
new_data = data[feat_names]
new_data['SalePrice'] = data['SalePrice']


# In[ ]:


# first five rows of the new data
new_data.head()


# In[ ]:


X_new=new_data.drop('SalePrice',axis=1)
y_new=new_data['SalePrice']


# In[ ]:


# 8 : Stacking of multiple models to improve the model performance


# Creating the list of estimators
estimators = [Lasso(alpha=alpha),AdaBoostRegressor()]
sr = StackingRegressor(estimators,RandomForestRegressor(n_estimators=10,random_state=42))
sr.fit(X_train, y_train)
# Predict using the stacking regressor
predicttrain=sr.predict(X_train)
predicttest=sr.predict(X_test)

errortrain=np.sqrt(msle(y_train,predicttrain))
errortest=np.sqrt(msle(y_test,predicttest))
print('RMSLE for train (Stacking):',errortrain)
print('RMSLE for test (Stacking):',errortest)


# With the help of stacking we can be able to decrease the error value

# In[ ]:


# 9 checking the residual for homoscedasticity and normal distribution
# Arranging and calculating the Residuals
residuals = pd.DataFrame({
    'fitted values' : y_test,
    'predicted values' :y_pred_best_rf,
})
residuals['residuals'] = residuals['fitted values'] - residuals['predicted values']
residuals.head()


#  Plotting residual curve (if there constant Variance OR **Homoscedastic**?)

# In[ ]:


plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')
l=len(y_test)
f = range(0,l)
k = [0 for i in range(0,l)]
plt.scatter( f, residuals.residuals[:], label = 'residuals')
plt.plot( f, k , color = 'red', label = 'regression line' )
plt.xlabel('fitted points ')
plt.ylabel('residuals')
plt.title('Residual plot')
plt.ylim(-5000, 5000)
plt.legend()


# # From the above scatter plot we can assume that there is not any pattern in the residuals

# Checking the **normal** distribution of the residual or not

# In[ ]:


#Histogram for distribution
plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')
plt.hist(residuals.residuals, bins = 100)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Error Terms')
plt.show()


# Residual values is normally distributed but having some outliers

# In[ ]:


#QQplot for distribution
from statsmodels.graphics.gofplots import qqplot
fig,ax=plt.subplots(figsize=(12,4),dpi=120)
#plotting qqplot
qqplot(residuals.residuals,line='s',ax=ax)
plt.ylabel('Residual Quantiles')
plt.xlabel('Ideal Scaled Quantiles')
plt.title('Checking distribution of Residual Errors')
plt.show()


# # The Above QQ-plot clearly tell that the residual are normally distributed with few outliers

# In[ ]:


# 10 Analysis of Feature Importance
# Random Forest Feature Importance
importances = best_rf.feature_importances_
#creating dataframe
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
#soritng the features in decending order
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
print('Random Forest Feature Importance:')
print(feature_importance_rf)


# # RandomForest feature_importances_ tells that BedroomAbFGr ,OverallQual,Utilites ,HeatingQC and Street are most important feature

# In[ ]:


# 11 Analysis of Feature Importance
# Lasso Feature Importance
importances = lassoreg.coef_
#creating dataframe
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
#soritng the features in decending order
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
print('Lasso regression Feature Importance:')
print(feature_importance_rf)


# The top features according to lasso regression are OverallQual,BedroomAbvGr and Utilites.

# In[ ]:


# 12 Analysis of Feature Importance
# AdaBoost Feature Importance
importances = best_adaboost.feature_importances_
feature_importance_adaboost = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_adaboost = feature_importance_adaboost.sort_values(by='Importance', ascending=False)
print('AdaBoost Feature Importance:')
print(feature_importance_adaboost)


# # **According** to **importance** of feature we can train our model while using the **top** **features**

# # Collecting the top features according to feature selection and feature importance from different models

# In[ ]:


#features collection
features_of_models=['BedroomAbvGr','OverallQual','Utilities','HeatingQC',
                    'Street','TotRmsAbvGrd','Condition2','GrLivArea']


# In[ ]:


#dataset of new features collection
newest_data=data[features_of_models]


# In[ ]:


#Independent variables
X=newest_data
#dependent variable
y=data['SalePrice']


# In[ ]:


# Perform data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X=pd.DataFrame(X,columns=['BedroomAbvGr','OverallQual','Utilities','HeatingQC',
                    'Street','TotRmsAbvGrd','Condition2','GrLivArea'])
X.head()


# In[ ]:


#Importing module
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Spiliting into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#(Random Forest)
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
predict_train=random_forest.predict(X_train)
predict_test= random_forest.predict(X_test)

rmslet=np.sqrt(msle(y_train,predict_train))
rmslete=np.sqrt(msle(y_test,predict_test))
r2_score1=r2_score(y_test,predict_test)
print("r2_score: ", r2_score1)
print('RMSLE for test(RandomFores): ',rmslete)


# In[ ]:


# Hyperparameter Tuning
# Grid search for Random Forest hyperparameters
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]    #If our model not perform well on this hyperparameter then we can all add more hyperparameter

}


grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=3)
grid_search_rf.fit(X_train, y_train)

#  the best Random Forest model
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
rmsle_b = np.sqrt(msle(y_test, y_pred_best_rf))
print('best rmsle (Random Forest):', rmsle_b)


# Grid search for AdaBoost hyperparameters
param_grid_adaboost = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

grid_search_adaboost = GridSearchCV(estimator=adaboost, param_grid=param_grid_adaboost, cv=3)
grid_search_adaboost.fit(X_train, y_train)

# Get the best AdaBoost model
best_adaboost = grid_search_adaboost.best_estimator_
y_pred_best_adaboost = best_adaboost.predict(X_test)
rmsle_add = np.sqrt(msle(y_test, y_pred_best_adaboost))
print('Best rmsle (AdaBoost):', rmsle_add)


# In[ ]:


# Stacking of multiple models to improve the model performance

# Creating the list of estimators
estimators = [Lasso(alpha=alpha),AdaBoostRegressor()]
sr = StackingRegressor(estimators,RandomForestRegressor(n_estimators=10,random_state=42))
sr.fit(X_train, y_train)
# Predict using the stacking regressor
predicttrain=sr.predict(X_train)
predicttest=sr.predict(X_test)

errortrain=np.sqrt(msle(y_train,predicttrain))
errortest=np.sqrt(msle(y_test,predicttest))
r2_score2=r2_score(y_test,predicttest)
print('RMSLE for test (Stacking):',errortest)
print('R2_score : ',r2_score2)


# In[ ]:


#Shape of new formed dataset
newest_data.shape


# # Wonderful! with the help of feature selection ,feature importance ,stacking,hyperparameter tuning in ensemble model we are successfully able to decrease the number of features from 80 to 8.
# 
# The best thing is we are getting same RMSLE as we are getting with the help of 80 features
