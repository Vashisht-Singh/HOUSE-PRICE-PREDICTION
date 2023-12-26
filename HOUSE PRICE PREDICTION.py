#!/usr/bin/env python
# coding: utf-8

# In[19]:



import pandas as pd
housing=pd.read_csv(".\Housing.csv")


# In[13]:


housing.head()


# In[14]:


housing.info()


# In[15]:


housing['CHAS'].value_counts()


# In[ ]:


housing.describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# housing.hist(bins=50,figsize=(20,15))


# In[ ]:


# import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[ ]:


# train_set,test_set=split_train_test(housing,0.2)
# print(f"the training number is {len(train_set)}")
# print(f"the testing number is {len(test_set)}")


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
# train_set,test_set=split_train_test(housing,0.2)
print(f"the training number is {len(train_set)}")
print(f"the testing number is {len(test_set)}")


# In[ ]:


housing['CHAS'].value_counts()


# In[ ]:



import numpy as np
np.any(np.isnan(housing))


# In[ ]:


np.all(np.isfinite(housing))


# In[ ]:


housing.isnull().sum()
housing.dropna(axis=0,inplace=True)


# In[ ]:


np.all(np.isfinite(housing))


# In[ ]:


# housing=strat_train_set.copy()


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[ ]:


print(strat_train_set['CHAS'].value_counts())


# In[ ]:


strat_test_set['CHAS'].value_counts()


# In[ ]:


housing=strat_train_set.copy()


# In[ ]:


# plt.plot(housing['CHAS'])

# fig = plt.figure(figsize =(31,31))
# plt.plot(housing['CHAS'])


# In[ ]:


# housing['ZN'].describe()


# In[ ]:


corr_matrix=housing.corr()


# In[ ]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[ ]:


plt.scatter(housing['ZN'],housing['CHAS'])


# In[ ]:


from pandas.plotting import scatter_matrix


# In[ ]:


attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(10,15))


# In[ ]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[ ]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[ ]:


corr_matrix=housing.corr()


# In[ ]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[ ]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV")


# In[ ]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()
# print(housing_features)
# print(housing_labels)


# In[ ]:


# median=housing['RM'].median()
# housing['RM'].fillna(median)


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(housing)


# In[ ]:


# imputer.statistics_


# In[ ]:


X=imputer.transform(housing)
X


# In[ ]:


housing_tr=pd.DataFrame(X,columns=housing.columns)
housing_tr


# In[ ]:


housing_tr.info()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


# In[ ]:


housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr


# In[ ]:


housing_num_tr.shape


# In[ ]:


# housing1=pd.DataFrame(housing_num_tr,labels=[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV])

# housing1
# # housing1.drop("14",axis=1)

# housing1.drop("MEDV",axis=1)
# housing2=housing1.drop("MEDV",axis=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
# print(some_data)
some_labels=housing_labels.iloc[:5]
# print(some_labels)
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[ ]:


some_labels


# In[ ]:


housing.head()


# In[ ]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[ ]:


lin_mse


# In[ ]:


# from sklearn.tree import DecisionTreeRegressor
# model=DecisionTreeRegressor()
# model.fit(housing_num_tr,housing_labels)
# some_data=housing.iloc[:5]
# # print(some_data)
# some_labels=housing_labels.iloc[:5]
# # print(some_labels)
# prepared_data=my_pipeline.transform(some_data)
# model.predict(prepared_data)


# In[ ]:


some_labels


# In[ ]:


# from sklearn.metrics import mean_squared_error
# housing_predictions=model.predict(housing_num_tr)
# mse=mean_squared_error(housing_labels,housing_predictions)
# rmse=np.sqrt(lin_mse)


# In[ ]:


# rmse


# In[ ]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[ ]:


rmse_scores


# In[ ]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# In[ ]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)


# In[ ]:


final_rmse


# In[ ]:


prepared_data[0]


# PRECTING THE HOUSE PRICE
# 

# In[ ]:


from joblib import dump,load
model=load('Dragon.joblib')
model.predict([[-0.44499072,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])


# In[ ]:




