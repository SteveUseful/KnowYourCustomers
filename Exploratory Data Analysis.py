#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[36]:


df = pd.read_csv("C:/Data/Telco-Customer-Churn.csv")
df.head()


# In[37]:


df.shape


# In[38]:


df.isna().sum()


# ## Analysis

# We do not need customer ID in our analysis as it does not help us predict whether the cutomer will churn or not also, it increases the dimensionality.

# In[39]:


df.drop(["customerID"], inplace = True, axis = 1)


# In[40]:


def stacked_plot(df, group, target):
    """
    Function to generate a stacked plots between two variables
    """
    fig, ax = plt.subplots(figsize = (6,4))
    temp_df = (df.groupby([group, target]).size()/df.groupby(group)[target].count()).reset_index().pivot(columns=target, index=group, values=0)
    temp_df.plot(kind='bar', stacked=True, ax = ax, color = ["green", "darkred"])
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlabel(group)
    ax.set_ylabel('Churn Percentage')


# ### Gender, SeniorCitizen, Partner, Dependents

# In[8]:


stacked_plot(df, "gender", "Churn")
stacked_plot(df, "SeniorCitizen", "Churn")
stacked_plot(df, "Partner", "Churn")
stacked_plot(df, "Dependents", "Churn")


# From above plots, we can say following:
# - Gender alone does not help us predict the customer churn.
# - If a person is young and has a family, he or she is less likely to stop the service as we can see below. The reason might be the busy life, more money or another factors.

# In[9]:


df[(df.SeniorCitizen == 0) & (df.Partner == 'Yes') & (df.Dependents == 'Yes')].Churn.value_counts()


# In[10]:


df[(df.SeniorCitizen == 0) & (df.Partner == 'Yes') & (df.Dependents == 'No')].Churn.value_counts()


# In[11]:


df[(df.SeniorCitizen == 0) & (df.Partner == 'No') & (df.Dependents == 'Yes')].Churn.value_counts()


# In[12]:


df[(df.SeniorCitizen == 0) & (df.Partner == 'No') & (df.Dependents == 'No')].Churn.value_counts()


# ### Tenure

# In[10]:


df['tenure'].describe()


# In[11]:


df['tenure'].value_counts().head(10)


# In[12]:


plt.figure(figsize=(16,8))
sns.countplot(x="tenure", hue="Churn", data=df)
plt.show()


# As we can see the higher the tenure, the lesser the churn rate. This tells us that the customer becomes loyal with the tenure.

# Converting into 5 groups to reduce model complexity.

# In[41]:


def tenure(t):
    if t<=12:
        return 1
    elif t>12 and t<=24:
        return 2
    elif t>24 and t<=36:
        return 3
    elif t>36 and t<=48:
        return 4
    elif t>48 and t<=60:
        return 5
    else:
        return 6

df["tenure_group"]=df["tenure"].apply(lambda x: tenure(x))


# In[42]:


df["tenure_group"].value_counts()


# In[43]:


sns.countplot(x="tenure_group", hue="Churn", data=df)


# ### Phone Service and MultipleLines 

# In[19]:


stacked_plot(df, "PhoneService", "Churn")


# In[20]:


stacked_plot(df, "MultipleLines", "Churn")


# As we can see multiplelines and phoneservice do not add value in the model having similar churn rate.

# ### OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

# In[18]:


stacked_plot(df, "OnlineSecurity", "Churn")
stacked_plot(df, "OnlineBackup", "Churn")
stacked_plot(df, "DeviceProtection", "Churn")
stacked_plot(df, "TechSupport", "Churn")
stacked_plot(df, "StreamingTV", "Churn")
stacked_plot(df, "StreamingMovies", "Churn")


# In all above categories we see consistent results. If a person does not opt for internet service, the customer churning is less. The reason might be the less cost of the service. Also, if they have internet service and does not opt for specific service their probability of churning is high.

# In[33]:


sns.distplot(df.tenure[df.OnlineSecurity == "No"], hist_kws=dict(alpha=0.3), label="No")
sns.distplot(df.tenure[df.OnlineSecurity == "Yes"], hist_kws=dict(alpha=0.3), label="Yes")
sns.distplot(df.tenure[df.OnlineSecurity == "No internet service"], hist_kws=dict(alpha=0.3), label="No Internet Service")
plt.title("Tenure Distribution by Online Security Service Subscription")
plt.legend()
plt.show()


# In[32]:


sns.distplot(df.tenure[df.StreamingTV == "No"], hist_kws=dict(alpha=0.3), label="No")
sns.distplot(df.tenure[df.StreamingTV == "Yes"], hist_kws=dict(alpha=0.3), label="Yes")
sns.distplot(df.tenure[df.StreamingTV == "No internet service"], hist_kws=dict(alpha=0.3), label="No Internet Service")
plt.title("Tenure Distribution by Streaming TV Service Subscription")
plt.legend()
plt.show()


# In[31]:


sns.distplot(df.tenure[df.StreamingMovies == "No"], hist_kws=dict(alpha=0.3), label="No")
sns.distplot(df.tenure[df.StreamingMovies == "Yes"], hist_kws=dict(alpha=0.3), label="Yes")
sns.distplot(df.tenure[df.StreamingMovies == "No internet service"], hist_kws=dict(alpha=0.3), label="No Internet Service")
plt.title("Tenure Distribution by Streaming Movies Service Subscription")
plt.legend()
plt.show()


# As we can see, when the customers are new they do not opt for various services and their churning rate is very high.

# ### InternetService

# In[34]:


stacked_plot(df, "InternetService", "Churn")


# When the internet service is Fiber Optic, the churn rate is very high. Fiber Optics provides highr speed compared to DSL. The reason might be the higher cost of fiber optics.

# In[37]:


sns.countplot(df.InternetService, hue = df.Dependents)


# In[43]:


stacked_plot(df[df.InternetService == "Fiber optic"], "Dependents", "Churn")


# Mostly people without dependents go for fiber optic option as Internnet Service and their churning percentage is high.

# In[38]:


sns.countplot(df.InternetService, hue = df.Partner)


# In[39]:


sns.countplot(df.InternetService, hue = df.SeniorCitizen)


# In[42]:


stacked_plot(df[df.InternetService == "Fiber optic"], "SeniorCitizen", "Churn")


# As we can see, Partner and Senior Citizen do not tell us anything about why fiber optics have higher churning rate.

# In[40]:


sns.distplot(df.tenure[df.InternetService == "No"], hist_kws=dict(alpha=0.3), label="No")
sns.distplot(df.tenure[df.InternetService == "DSL"], hist_kws=dict(alpha=0.3), label="DSL")
sns.distplot(df.tenure[df.InternetService == "Fiber optic"], hist_kws=dict(alpha=0.3), label="Fiber optic")
plt.title("Tenure Distribution by Internet Service type")
plt.legend()
plt.show()


# Also, the tenure distribution of customers with different internet service is similar.

# In[44]:


df[df.InternetService == 'No'].head()


# In[45]:


df[df.InternetService == 'No'].OnlineSecurity.value_counts()


# In[46]:


df[df.InternetService == 'No'].OnlineBackup.value_counts()


# In[48]:


df[df.InternetService == 'No'].DeviceProtection.value_counts()


# In[49]:


df[df.InternetService == 'No'].TechSupport.value_counts()


# In[50]:


df[df.InternetService == 'No'].StreamingMovies.value_counts()


# In[52]:


df[df.InternetService == 'No'].StreamingTV.value_counts()


# We need to encode these variables to remove dependancy in the model.

# ### Contract

# In[21]:


stacked_plot(df, "Contract", "Churn")


# In the case of Month-to-month contract Churn rate is very high. There is also a posibility of having customers in the dataframe who are still in their two-year or one-year contract plan.

# In[41]:


sns.countplot(df.InternetService, hue = df.Contract)


# Many of the people of who opt for month-to-month Contract choose Fiber optic as Internet service and this is the reason for higher churn rate for fiber optic Internet service type.

# ### PaymentMethod

# In[44]:


group = "PaymentMethod"
target = "Churn"
fig, ax = plt.subplots(figsize = (12,5))
temp_df = (df.groupby([group, target]).size()/df.groupby(group)[target].count()).reset_index().pivot(columns=target, index=group, values=0)
temp_df.plot(kind='bar', stacked=True, ax = ax, color = ["green", "darkred"])
ax.xaxis.set_tick_params(rotation=0)
ax.set_xlabel(group)
ax.set_ylabel('Churn Percentage');


# In the case of Electronic check, churn is very high. 

# In[47]:


fig, ax = plt.subplots(figsize = (12,5))
sns.countplot(df.PaymentMethod, hue = df.Contract, ax = ax)


# People having month-to-month contract prefer paying by Electronic Check mostly or mailed check. The reason might be short subscription cancellation process compared to automatic payment. 

# ### PaperlessBilling

# In[48]:


stacked_plot(df, "PaperlessBilling", "Churn")


# ### TotalCharges

# In[22]:


df.TotalCharges.describe()


# In[53]:


df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)
df['TotalCharges'].isna().sum() 


# In[54]:


df[df["TotalCharges"].isnull()]


# All the customers having tenure = 0 have null total charges which means that these customers recently joined and we can fill those missing values as 0.

# In[55]:


df.loc[df["TotalCharges"].isnull(), 'TotalCharges'] = 0
df.isnull().any().any()


# In[56]:


df['TotalCharges'] = df["TotalCharges"].astype(float)

Churn = df[df.Churn=="Yes"]
Not_Churn = df[df.Churn=="No"]


# In[57]:


fig, ax = plt.subplots()
sns.kdeplot(Churn["TotalCharges"],label = "Churn", ax= ax)
sns.kdeplot(Not_Churn["TotalCharges"], label = "Not Churn", ax=ax)
ax.set_xlabel("Total Charges");


# The density of total charges for churning customers are high around 0. As many customers cancel the subsription in 1-2 months.

# ### Monthly Charges

# In[18]:


df.MonthlyCharges.describe()


# In[19]:


df.MonthlyCharges.isna().sum()


# In[20]:


sns.kdeplot(Churn["MonthlyCharges"], label = "Churn")
sns.kdeplot(Not_Churn["MonthlyCharges"], label = "Not Churn")


# The customers paying high monthly fees churn more.

# Let's see the correlation of total charges and (monthly charges x tenure) to check if we have redundant information.

# In[28]:


np.corrcoef(df.TotalCharges, df.MonthlyCharges*df.tenure)


# Let's keep total charges as it shows the interaction between tenure and monthly charges

# ## Fucntion to prepare data for model building based on EDA

# In[62]:


def datapreparation(filepath):
    
    df = pd.read_csv(filepath)
    df.drop(["customerID"], inplace = True, axis = 1)
    
    df.TotalCharges = df.TotalCharges.replace(" ",np.nan)
    df.TotalCharges.fillna(0, inplace = True)
    df.TotalCharges = df.TotalCharges.astype(float)
    
    cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
    for col in cols1:
        df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)
   
    df.gender = df.gender.apply(lambda x: 0 if x == "Male" else 1)
    df.MultipleLines = df.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})
    
    cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols2:
        df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})
    
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
    
    return df

