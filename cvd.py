#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np


# In[65]:


data=pd.read_excel('cep_dataset.xlsx')


# In[66]:


data.head()


# In[67]:


data.shape


# In[68]:


data.info()


# In[69]:


data.isnull().sum(axis=0)


# In[70]:


data.isnull().sum(axis=1)


# In[73]:


data['target'].value_counts()


# In[74]:


data.describe()


# In[77]:


data.dtypes


# In[78]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[83]:


sns.countplot(x ='sex', data = data,hue='sex')
plt.show()


# In[82]:


sns.countplot(x='target',data=data,hue='target')
plt.show()


# In[86]:


categorical_val = []
continous_val = []
for column in data.columns:
    print("--------------------")
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[87]:


import hvplot.pandas


# In[90]:


data.target.value_counts().hvplot.bar(
    title="Heart Disease Count", xlabel='Heart Disease', ylabel='Count', 
    width=250, height=400
)


# In[91]:


sns.countplot(x ='sex', data = data,hue='target')
plt.show()


# In[102]:


have_disease = data.loc[data['target']==1, 'sex'].value_counts().hvplot.bar(alpha=0.4) 
no_disease = data.loc[data['target']==0, 'sex'].value_counts().hvplot.bar(alpha=0.4) 

(no_disease * have_disease).opts(
    title="Heart Disease by Sex", xlabel='Sex', ylabel='Count',
    width=500, height=450, legend_cols=2, legend_position='top_right'
)


# In[103]:


have_disease = data.loc[data['target']==1, 'sex'].value_counts()


# In[104]:


have_disease 


# In[105]:


no_disease = data.loc[data['target']==0, 'sex'].value_counts()
no_disease


# In[108]:


have_disease_1 = data.loc[data['target']==1, 'age'].value_counts()
have_disease_1


# In[109]:


have_disease_2 = data.loc[data['target']==1, 'trestbps'].value_counts()
have_disease_2


# In[110]:


have_disease_3 = data.loc[data['target']==1, 'chol'].value_counts()
have_disease_3


# In[111]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True,fmt='.0%')


# In[112]:


plt.figure(figsize=(15,10))
sns.pairplot(data.select_dtypes(exclude='object'))
plt.show()


# In[209]:


x=data.drop(columns=['target','age','sex','chol','restecg','exang','oldpeak','trestbps'],axis=1)
print(x)


# In[210]:


Y=data['target']
print(Y)


# In[211]:


x.corrwith(Y).plot.bar(
    figsize=(16,4),title='Correlation with CVD',fontsize=15,
    rot=90,grid=True)


# In[236]:


from scipy import stats
from sklearn.model_selection import train_test_split


# In[240]:


X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.3, random_state=42)


# In[241]:


x_train.shape


# In[242]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[243]:


from sklearn.linear_model import LogisticRegression


# In[244]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
  


# In[245]:


from imblearn.under_sampling import NearMiss
sm = NearMiss() 


# In[246]:


X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[252]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))


# In[253]:


lr1 = LogisticRegression()
lr1.fit(X_train_res, y_train_res)
predictions = lr1.predict(X_test)
  
# print classification report
print(classification_report(y_test, predictions))


# In[254]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train_res,y_train_res)


# In[255]:


predictions1 = rfc.predict(X_test)


# In[256]:


print(classification_report(y_test,predictions1))


# In[257]:


rfc.score(X_test, y_test)


# In[258]:


print(classification_report(y_train, rfc.predict(X_train)))


# In[259]:


import statsmodels.api as sm
logit_model=sm.Logit(Y,x)
result=logit_model.fit()
print(result.summary())


# In[ ]:





# In[ ]:




