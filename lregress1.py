#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import seaborn as sns

data = pd.read_csv('diabetes.csv')
data.head()


# In[40]:


# remove null values
data['Glucose'] = data['Glucose'].replace(0,np.nan)
data['BloodPressure'] = data['BloodPressure'].replace(0,np.nan)
data['SkinThickness'] = data['SkinThickness'].replace(0,np.nan)
data['Insulin'] = data['Insulin'].replace(0,np.nan)
data['BMI'] = data['BMI'].replace(0,np.nan)
data.isnull().sum()


# In[41]:


# mean and stardard deviation for attibute for 2-6
print('Mean for Glucose:',data.loc[:,'Glucose'].mean())
print('Mean for Blood Pressure:',data.loc[:,'BloodPressure'].mean())
print('Mean for Skin Thickness:',data.loc[:,'SkinThickness'].mean())
print('Mean for Insulin:',data.loc[:,'Insulin'].mean())
print('Mean for BMI:',data.loc[:,'BMI'].mean())
print('')
print('Standard Deviation for Glucose:',data.loc[:,'Glucose'].std())
print('Standard Deviation for Blood Pressure:',data.loc[:,'BloodPressure'].std())
print('Standard Deviation for Skin Thickness:',data.loc[:,'SkinThickness'].std())
print('Standard Deviation for Insulin:',data.loc[:,'Insulin'].std())
print('Standard Deviation for BMI:',data.loc[:,'BMI'].std())


# In[42]:


# Finding the covarience
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].cov()


# In[43]:


# Finding the correlation for 10 pairs and 5 attributes
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].corr()


# In[44]:


# Scatter plot of attributes 3 and 6
plot1 = data.plot.scatter(x='BloodPressure',
                         y='BMI')


# In[45]:


# Scatter plot for attributes 2 and 7
plot2 = data.plot.scatter(x='Glucose',
                         y='DiabetesPedigreeFunction')


# In[46]:


# Histgrams for attributes 2,3 and 6
hist1 = data.hist(column=['Glucose'])
plt.xlabel("Plasma Level")
plt.ylabel("Frequency")
hist2 = data.hist(column=['BloodPressure'])
plt.xlabel("mmHg")
plt.ylabel("Frequency")
hist3 = data.hist(column=['BMI'])
plt.xlabel("kg/m^2")
plt.ylabel("Frequency")


# In[47]:


# Create histograms for 3 attributes of class 1 and class 0
hist4 = data[data['Outcome']==0].hist(column=['Glucose'])
plt.title('Class 0')
plt.xlabel("Plasma Level")
plt.ylabel("Frequency")
hist4 = data[data['Outcome']==1].hist(column=['Glucose'])
plt.title('Class 1')
plt.xlabel("Plasma Level")
plt.ylabel("Frequency")
print('')
hist5 = data[data['Outcome']==0].hist(column=['BloodPressure'])
plt.title('Class 0')
plt.xlabel("mmHg")
plt.ylabel("Frequency")
hist5 = data[data['Outcome']==1].hist(column=['BloodPressure'])
plt.title('Class 1')
plt.xlabel("mmHg")
plt.ylabel("Frequency")
print('')
hist6 = data[data['Outcome']==0].hist(column=['BMI'])
plt.title('Class 0')
plt.xlabel("kg/m^2")
plt.ylabel("Frequency")
hist6 = data[data['Outcome']==1].hist(column=['BMI'])
plt.title('Class 1')
plt.xlabel("kg/m^2")
plt.ylabel("Frequency")


# In[48]:


bar1 = data.boxplot(column=['Glucose'])
plt.title('Glucose Bar Plot')
plt.ylabel('Plasma Level')


# In[49]:


bar2 = data[data['Outcome']==0].boxplot(column=['Glucose'])
plt.title('Glucose Bar Plot Class 0')
plt.ylabel('Plasma Level')


# In[50]:


bar3 = data[data['Outcome']==1].boxplot(column=['Glucose'])
plt.title('Glucose Bar Plot Class 1')
plt.ylabel('Plasma Level')


# In[51]:


bar4 = data.boxplot(column=['DiabetesPedigreeFunction'])
plt.title('Diabetes Pedigree Function')


# In[52]:


bar5 = data[data['Outcome']==0].boxplot(column=['DiabetesPedigreeFunction'])
plt.title('Diabetes Pedigree Function Class 0')


# In[53]:


bar6 = data[data['Outcome']==1].boxplot(column=['DiabetesPedigreeFunction'])
plt.title('Diabetes Pedigree Function Class 1')


# In[54]:


bar7 = data.boxplot(column=['Age'])
plt.title('Age')
plt.ylabel('Years')


# In[55]:


bar8 = data[data['Outcome']==0].boxplot(column=['Age'])
plt.title('Age Class 0')
plt.ylabel('Years')


# In[33]:


bar9 = data[data['Outcome']==1].boxplot(column=['Age'])
plt.title('Age Class 1')
plt.ylabel('Years')


# In[66]:


sns.pairplot(data,vars=data.columns[1:6])


# In[37]:


# create two 3D scatter plots for 2,3,6 and 2,4,6
fig = plt.figure()
scatter3d = Axes3D(fig)
scatter3d.scatter(data['Glucose'],data['BloodPressure'],data['BMI'])
scatter3d.set_xlabel('Plasma Glucose axis')
scatter3d.set_ylabel('mmHg axis')
scatter3d.set_zlabel('kg/m^2 axis')


# In[38]:


fig = plt.figure()
scatter3d = Axes3D(fig)
scatter3d.scatter(data['Glucose'],data['SkinThickness'],data['BMI'])
scatter3d.set_xlabel('Plasma Glucose axis')
scatter3d.set_ylabel('mm axis')
scatter3d.set_zlabel('kg/m^2 axis')


# In[34]:


# Fit a linear regression which fits the class attribute and reports the R^2 values
data1 = pd.read_csv('diabetes.csv')
mlr_x = data1[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
mlr_y = data1['Outcome']
model = sm.OLS(mlr_y,mlr_x).fit()
model.summary()


# In[35]:


# Drop two values that have the closest coeffcients
mlr_x = data1[["Pregnancies","Glucose","BloodPressure","BMI","DiabetesPedigreeFunction","Age"]]
mlr_y = data1['Outcome']
model = sm.OLS(mlr_y,mlr_x).fit()
model.summary()


# In[ ]:




