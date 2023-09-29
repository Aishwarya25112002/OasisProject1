#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pylab
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("D:/Oasis internship/CarPrice_Assignment.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


# Checking NaN values in our Dataset.

df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})


# In[10]:


# Checking Duplicate Values.

print("Duplicate Values =",df.duplicated().sum())


# In[11]:


# Showing Only Categorical Features.

df.select_dtypes(include="object").head()


# In[12]:


#  Showing only the Numerical Features.

df.select_dtypes(include=["int","float"]).head()


# In[14]:


#DATA CLEANING
# Cleaning the CarName Feature

df.head()


# In[15]:


Company_Name = df["CarName"].apply(lambda x: x.split(" ")[0])
df.insert(2,"CompanyName",Company_Name)

# Now we can drop the CarName Feature.
df.drop(columns=["CarName"],inplace=True)


# In[16]:


df.head()


# In[17]:


# Checking the Unique Car Company Names.

df["CompanyName"].unique()


# In[18]:


# Creating a Function to Replace the Values.

def replace(a,b):
    df["CompanyName"].replace(a,b,inplace=True)

replace('maxda','mazda')
replace('porcshce','porsche')
replace('toyouta','toyota')
replace('vokswagen','volkswagen')
replace('vw','volkswagen')


# In[19]:


df["CompanyName"].unique()


# # Exploratory data analysis

# In[24]:


# Visualizing our Target Feature.

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.distplot(df["price"],color="blue",kde=True)
plt.title("Car Price Distribution",fontweight="black",pad=15,fontsize=15)

plt.subplot(1,2,2)
sns.boxplot(y=df["price"],palette="Set1")
plt.title("Car Price Spread",fontweight="black",pad=15,fontsize=15)
plt.tight_layout()
plt.show()


# In[25]:


df["price"].agg(["min","mean","median","max","std","skew"]).to_frame().T


# In[26]:


# Visualizing Total No. of cars sold by different company

plt.figure(figsize=(14,6))
counts = df["CompanyName"].value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Car Company")
plt.ylabel("Total No. of cars sold")
plt.title("Total Cars produced by Companies", pad=15, fontweight="black", fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[27]:


df[df["CompanyName"]=="mercury"]


# In[28]:


df[df["CompanyName"]=="Nissan"]


# In[29]:


df[df["CompanyName"]=="renault"]


# In[30]:


#  Visualizing Car Company w.r.t Price

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.boxplot(x="CompanyName",y="price",data=df)
plt.xticks(rotation=90)
plt.title("Car Company vs Price", pad=10, fontweight="black", fontsize=20)

plt.subplot(1,2,2)
x = pd.DataFrame(df.groupby("CompanyName")["price"].mean().sort_values(ascending=False))
sns.barplot(x=x.index,y="price",data=x)
plt.xticks(rotation=90)
plt.title("Car Company vs Average Price", pad=10, fontweight="black", fontsize=20)
plt.tight_layout()
plt.show()


# In[31]:


df[df["CompanyName"]=="mercury"]


# In[33]:


# Visualizing Car Fuel Type Feature

def categorical_visualization(cols):
    plt.figure(figsize=(20,8))
    plt.subplot(1,3,1)
    sns.countplot(x=cols,data=df,palette="Set3",order=df[cols].value_counts().index)
    plt.title(f"{cols} Distribution",pad=10,fontweight="black",fontsize=20)
    plt.xticks(rotation=90)

    plt.subplot(1,3,2)
    sns.boxplot(x=cols,y="price",data=df,palette="Set3")
    plt.title(f"{cols} vs Price",pad=20,fontweight="black",fontsize=20)
    plt.xticks(rotation=90)

    plt.subplot(1,3,3)
    x=pd.DataFrame(df.groupby(cols)["price"].mean().sort_values(ascending=False))
    sns.barplot(x=x.index,y="price",data=x,palette="Set2")
    plt.title(f"{cols} vs Average Price",pad=20,fontweight="black",fontsize=20)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

categorical_visualization("fueltype")


# In[34]:


# Visualizing Aspiration Feature

categorical_visualization("aspiration")


# In[35]:


# Visualizing Door Nubmer Feature

categorical_visualization("doornumber")


# In[36]:


# Visualizing Car Body Type Feature

categorical_visualization("carbody")


# In[37]:


#  Visualizing Engine Location Feature

categorical_visualization("enginelocation")


# In[38]:


df[df["enginelocation"]=="rear"]


# In[39]:


# Visualizing Engine Type Feature

categorical_visualization("enginetype")


# In[40]:


df[df["enginetype"]=="rotor"]


# In[41]:


# Visualizing Cyclinder Number Feature

categorical_visualization("cylindernumber")


# In[42]:


df[df["cylindernumber"]=="three"]


# In[43]:


# Visualizing Fuel System Feature

categorical_visualization("fuelsystem")


# In[44]:


# Visualizing "CarLength", "CarWidth","Carheight" Features w.r.t "Price"

def scatter_plot(cols):
    x=1
    plt.figure(figsize=(15,6))
    for col in cols:
        plt.subplot(1,3,x)
        sns.scatterplot(x=col,y="price",data=df,color="blue")
        plt.title(f"{col} vs Price",fontweight="black",fontsize=20,pad=10)
        plt.tight_layout()
        x+=1


# In[45]:


scatter_plot(["carlength","carwidth","carheight"])


# In[46]:


scatter_plot(["carlength","carwidth","carheight"])


# In[47]:


scatter_plot(["carlength","carwidth","carheight"])


# In[49]:


# Creating new DataFrame with all the useful Features.

new_df = df[['fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'
             ,'wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg',
             'price']]


# In[51]:


new_df.head()


# In[52]:


# Selecting Features & Labels for Model Training & Testing

x = new_df.drop(columns=["price"])
y = new_df["price"]


# In[53]:


x.shape


# In[54]:


y.shape


# In[ ]:




