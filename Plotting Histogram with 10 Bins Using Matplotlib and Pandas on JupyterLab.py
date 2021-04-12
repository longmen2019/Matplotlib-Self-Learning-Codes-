#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('subset_census-income_data.csv')


# In[4]:


print(df)


# In[5]:


print(df.to_string())


# In[6]:


print(df.info())


# In[7]:


#Calculate the mean, median, range, and variance of this attribute.
x = df["age"].mean()
print(x)


# In[ ]:





# In[8]:


y = df["age"].median()


# In[9]:


print(y)


# In[10]:


z = df["age"].mode()


# In[11]:


print(z)


# In[12]:


#Plot a histogram of the age attribute using 10 bins.
df.plot()


# In[13]:


df.hist()


# In[19]:


df.hist(bins=10)


# In[17]:


df.plot.hist(bins=10)


# In[18]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist = (df, bins=10 , edgecolor='red')


# In[11]:


import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('subset_census-income_data.csv')
ages = data['age']
bins = 10
plt.hist(ages, bins=bins, edgecolor='black' , log= True)
#plt.legend()
plt.title('Histogram of Age Attribute')
plt.xlabel('Ages')
plt.ylabel('Number of bins')
plt.tight_layout()
plt.show


# In[ ]:




