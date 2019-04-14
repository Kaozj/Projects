
# coding: utf-8

# In[51]:

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import copy

pd.options.display.max_rows=99


# In[3]:

train=pd.read_csv('train.csv',index_col=False)
test=pd.read_csv('test.csv',index_col=False)
df=pd.concat([train,test],ignore_index=True)


# In[4]:

print len(df)
df=df.drop(df[(df['GrLivArea']>4000)&(df['SalePrice']<300000)].index).reset_index(drop=True)
print len(df)


# In[5]:

#designate field types

cont_fields=['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
             'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
             'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
             'GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
             'ScreenPorch','PoolArea','MiscVal','YrSold']

ord_fields=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC',
            'KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC']
qual_map={np.nan:0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
ord_mappings=[{'Reg':1,'IR1':2,'IR2':3,'IR3':4},{'Gtl':1,'Mod':2,'Sev':3},qual_map,qual_map,qual_map,qual_map,
              {np.nan:0,'No':1,'Mn':2,'Av':3,'Gd':4},qual_map,qual_map,
              {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8},qual_map,
              {np.nan:0,'Unf':1,'RFn':2,'Fin':3},qual_map,qual_map,qual_map]
print len(ord_fields)==len(ord_mappings)

nom_fields=['MSSubClass','MSZoning','Street','Alley','LandContour','Utilities','LotConfig','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
            'MasVnrType','Foundation','BsmtFinType1','BsmtFinType2','Heating','CentralAir','Electrical',
            'GarageType','PavedDrive','Fence','MiscFeature','MoSold','SaleType','SaleCondition']

len(cont_fields+ord_fields+nom_fields)==len(df.columns)-2


# In[6]:

#map ordinal field values to numerical ranks

for col,mapping in zip(ord_fields,ord_mappings):
    df[col]=df[col].replace(mapping)


# In[7]:

#impute missing values

df['MiscFeature']=df['MiscFeature'].fillna('None')
df['Alley']=df['Alley'].fillna('None')
df['Fence']=df['Fence'].fillna('None')
df['LotFrontage']=df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col]=df[col].fillna(0)
df['GarageType']=df['GarageType'].fillna('None')
for col in ('BsmtFinType1', 'BsmtFinType2'):
    df[col]=df[col].fillna('None')
df['MasVnrType']=df['MasVnrType'].fillna('None')
df['MasVnrArea']=df['MasVnrArea'].fillna(0)
df['MSZoning']=df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df=df.drop('Utilities',1)
nom_fields.remove('Utilities')
df['Functional']=df['Functional'].fillna(df['Functional'].mode()[0])
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col]=df[col].fillna(0)
df['SaleType']=df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Exterior1st']=df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd']=df['Exterior2nd'].replace({'Brk Cmn':'BrkComm','CmentBd':'CemntBd','Wd Shng':'WdShing'})
df['Exterior2nd']=df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])

df_na = (df.isnull().sum()/len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
print missing_data

# fields_with_nulls.remove('GarageYrBlt')
# imputed_values=[0,'None','None',0,'None','None','None','None','None','None']
# for col,imp_val in zip(fields_with_nulls,imputed_values):
#     df[col]=df[col].fillna(imp_val)

# df['GarageYrBlt']=[i if pd.isnull(i)==False else j+1 for i,j in zip(df['GarageYrBlt'],df['YrSold'])]


# In[ ]:

#compute pearson correlation and plot scatterplots for each continuous and ordinal field

for col in [i for i in df.columns if i not in nom_fields+['Id','SalePrice']]:
    print col
    print pearsonr(list(df[col]),list(df['SalePrice']))
    plt.scatter(df[col],df['SalePrice'])
    plt.show()
    plt.close()


# In[8]:

#create new features

def get_count(sf1,sf2):
    return sum(i>0 for i in [sf1,sf2])

def get_avg(total,count):
    if total!=0:
        return float(total)/count
    else:
        return 0

df['BsmtFinSFMax']=[max(sf1,sf2) for sf1,sf2 in zip(df['BsmtFinSF1'],df['BsmtFinSF2'])]
df['BsmtFinSFTotal']=df['BsmtFinSF1']+df['BsmtFinSF2']
df['BsmtFinCount']=[get_count(sf1,sf2) for sf1,sf2 in zip(df['BsmtFinSF1'],df['BsmtFinSF2'])]
df['BsmtFinSFAvg']=[get_avg(total,count) for total,count in zip(df['BsmtFinSFTotal'],df['BsmtFinCount'])]

df['FlrSFMax']=[max(sf1,sf2) for sf1,sf2 in zip(df['1stFlrSF'],df['2ndFlrSF'])]
df['FlrSFTotal']=df['1stFlrSF']+df['2ndFlrSF']
df['FlrCount']=[get_count(sf1,sf2) for sf1,sf2 in zip(df['1stFlrSF'],df['2ndFlrSF'])]
df['FlrSFAvg']=[get_avg(total,count) for total,count in zip(df['FlrSFTotal'],df['FlrCount'])]

df['TotalSF']=df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']

df['AggBsmtBath']=df['BsmtFullBath']+0.5*df['BsmtHalfBath']
df['AggBath']=df['FullBath']+0.5*df['HalfBath']
df['IndoorsArea']=df['GrLivArea']+df['TotalBsmtSF']+df['GarageArea']
df['AgeSinceBuilt']=df['YrSold']-df['YearBuilt']
df['AgeSinceRemod']=df['YrSold']-df['YearRemodAdd']
df['AvgQual']=[np.mean([a,b,c,d,e,f,g]) for a,b,c,d,e,f,g in zip(df['ExterQual'],df['BsmtQual'],df['HeatingQC'],
                                                                 df['KitchenQual'],df['FireplaceQu'],df['GarageQual'],
                                                                 df['PoolQC'])]
df['AvgCond']=[np.mean([a,b,c]) for a,b,c in zip(df['ExterCond'],df['BsmtCond'],df['GarageCond'])]


# In[ ]:

#checking number of observations for each level of a categorical field

for col in nom_fields:
    if col not in ['Condition1','Condition2','Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2']:
        num_cat=len(set(df[col]))
        print col,'Num cats:',num_cat
        ranked_cats=df[col].value_counts(normalize=True)
        print pd.DataFrame({'Cum_Count':(1-ranked_cats.cumsum()+ranked_cats).round(6)})
        print '-'*30


# In[9]:

#collapse levels that are too sparse, delete fields that don't have at least 2 non-sparse levels, bucketize month field

def collapse_levels(col,num):
    levels_to_collapse=df[col].value_counts().tail(num).index
    df[col]=df[col].replace(levels_to_collapse,'Others')

collapse_levels('MSSubClass',6)
collapse_levels('MSZoning',3)
collapse_levels('Alley',2)
collapse_levels('LandContour',2)
collapse_levels('LotConfig',2)
collapse_levels('Neighborhood',5)
collapse_levels('BldgType',2)
collapse_levels('HouseStyle',4)
collapse_levels('RoofStyle',5)
collapse_levels('MasVnrType',2)
collapse_levels('Foundation',4)
collapse_levels('Electrical',4)
collapse_levels('GarageType',4)
collapse_levels('PavedDrive',2)
collapse_levels('Fence',2)
collapse_levels('MiscFeature',4)
collapse_levels('SaleType',7)
collapse_levels('SaleCondition',4)

to_drop=['Street','RoofMatl','Heating']
df=df.drop(to_drop,1)
nom_fields=[i for i in nom_fields if i not in to_drop]

# for i,j in zip([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],['Q1','Q2','Q3','Q4']):
#     df['MoSold']=df['MoSold'].replace(i,j)


# In[10]:

#collapse levels for fields that take up multiple columns

def get_cum_count(col1,col2):
    ranked_cats_1=df[col1].value_counts(normalize=True)
    ranked_cats_2=df[col2].value_counts(normalize=True)
    index_diff=set(ranked_cats_1.index)-set(ranked_cats_2.index)
    ranked_cats_2=ranked_cats_2.append(pd.Series([0]*len(index_diff),index=index_diff))
    index_diff=set(ranked_cats_2.index)-set(ranked_cats_1.index)
    ranked_cats_1=ranked_cats_1.append(pd.Series([0]*len(index_diff),index=index_diff))
    merged_ranked_cats=(ranked_cats_1+ranked_cats_2).sort_values(ascending=False)
    print (2-merged_ranked_cats.cumsum()+merged_ranked_cats).round(6)
    return merged_ranked_cats
    
def collapse_levels_multicol(col1,col2,num,series):
    to_replace=series.tail(num).index
    for col in [col1,col2]:
        df[col]=df[col].replace(to_replace,'Others')

series=get_cum_count('Condition1','Condition2')
collapse_levels_multicol('Condition1','Condition2',5,series)

series=get_cum_count('Exterior1st','Exterior2nd')
collapse_levels_multicol('Exterior1st','Exterior2nd',7,series)

series=get_cum_count('BsmtFinType1','BsmtFinType2')        


# In[11]:

#convert to dummy variables

for col in nom_fields:
    if col not in ['Condition1','Condition2','Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2']:
        df_dummies=pd.get_dummies(df[col])
        new_colnames=['{}#{}'.format(col,i) for i in df_dummies.columns]
        df_dummies=df_dummies.rename(columns=dict(zip(df_dummies.columns,new_colnames)))
        df=pd.concat([df,df_dummies],1)
        df=df.drop(col,1)
    elif col not in ['Condition2','Exterior2nd','BsmtFinType2']:
        col2=col.replace('1st','2nd').replace('1','2')
        new_col=col.replace('1st','').replace('1','')
        vals=set(df[col]).union(set(df[col2]))
        for val in vals:
            df['{}#{}'.format(new_col,val)]=[sum(i==val) for i in np.array(df[[col,col2]])]
        df=df.drop([col,col2],1)


# In[12]:

#define functions for later, prepare X and Y

def rmsle(y, y_pred):
    y=np.array(y)
    y_pred=np.array(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def get_fitted_clf(X_train,Y_train,features):
    clf=XGBRegressor(seed=0)
    clf.fit(X_train[features],Y_train)
    return clf

def get_score(clf,X_test,Y_test,features):
    Y_pred=clf.predict(X_test[features])
    return rmsle(Y_test,Y_pred)

X=df.drop(['Id','SalePrice'],1)[pd.isnull(df['SalePrice'])==False].reset_index(drop=True)
Y=df[['SalePrice']][pd.isnull(df['SalePrice'])==False].reset_index(drop=True)


# In[14]:

#tuning xgboost hyperparameters

print 'SELECTING FEATURES...'
kf=KFold(n_splits=5,random_state=0)
features_to_keep_folds=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    features_to_keep=X.columns
    clf=get_fitted_clf(X_train,Y_train,features_to_keep)
    cur_score=get_score(clf,X_test,Y_test,features_to_keep)
    prev_score=999
    while cur_score<prev_score:
        prev_score=cur_score
        save_if_good=features_to_keep
        X_train=X_train[features_to_keep]
        features_to_keep=X_train.columns[clf.feature_importances_>
                                               [i for i in set(sorted(clf.feature_importances_)) if i!=0][0]]
        print 'Features to keep:',len(features_to_keep)
        clf=get_fitted_clf(X_train,Y_train,features_to_keep)
        cur_score=get_score(clf,X_test,Y_test,features_to_keep)
        print 'Cur score:',cur_score
    features_to_keep_folds.append(save_if_good)
    print '-'*30
selected_features=set.intersection(*[set(i) for i in features_to_keep_folds])
print len(selected_features)

print 'TUNING HYPERPARAMS...'
rmsle_scorer=make_scorer(rmsle,greater_is_better=False)
params={'max_depth':[3,4,5],
        'n_estimators':[100,300,500],
        'min_child_weight':[1,3,5],
        'gamma':[0,0.5,1]}
grid=GridSearchCV(XGBRegressor(seed=0),params,cv=5,scoring=rmsle_scorer,verbose=5)
grid.fit(X[list(selected_features)],Y)
means = grid.cv_results_['mean_test_score']

stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.6f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print 'Best Params:',grid.best_params_


# In[80]:

#tuning ridge regression hyperparameters

def rmse_cv(model,X,Y):
    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

log_Y=copy.deepcopy(Y)
log_Y['SalePrice']=np.log1p(log_Y['SalePrice'])

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 25, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha),X,log_Y).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

alphas = [20, 21, 22, 23, 24, 25]
cv_ridge = [rmse_cv(Ridge(alpha = alpha),X,log_Y).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

print cv_ridge


# In[83]:

#stacking predictions

test=df[pd.isnull(df['SalePrice'])]

grid.best_params_['seed']=0
xgb=XGBRegressor(**grid.best_params_)
xgb.fit(X[list(selected_features)],Y)

rdg=Ridge(alpha=22)
rdg.fit(X,Y)

test['xgb_pred']=xgb.predict(test[list(selected_features)])
test['rdg_pred']=rdg.predict(test[X.columns])

test['SalePrice']=[np.mean([i,j]) for i,j in zip(test['xgb_pred'],test['rdg_pred'])]

test[['Id','SalePrice']].to_csv('submission_8.csv',index=False)
