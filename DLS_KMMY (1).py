#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#import dataset
automobile = pd.read_csv(r"C:\Users\USER\Desktop\M_S_D_S\automobileEDA.csv")
automobile.head()


# ### DATA CLEANING

# In[4]:


automobile.info()


# In[5]:


automobile.isna().sum()


# In[6]:


#assessing for duplicates if any
automobile_clean=automobile.drop_duplicates(keep="first")


# In[7]:


automobile_clean.isna().sum()


# In[8]:


#imputing the missing values in the stroke column   with the average per column per make
automobile_clean["stroke"]=automobile_clean["stroke"].fillna(automobile_clean.groupby("make")["stroke"].transform("mean"))


# In[9]:


#deleted the missing value in the horsepower-binned column
automobile_clean= automobile_clean.dropna(subset=['horsepower-binned'])


# In[10]:


automobile_clean.isna().sum()


# In[16]:


#HANDLING OUTLIERS
continuous_variables= automobile_clean[['symboling','normalized-losses','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke',
    'compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price','city-L/100km','diesel','gas']]
for column in continuous_variables:
    Q1 = continuous_variables[column].quantile(0.25)
    Q3 = continuous_variables[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = automobile_clean[(automobile_clean[column] < lower_bound) |(automobile_clean[column] > upper_bound)]

    # Plot boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=continuous_variables[column])

    plt.title(f'Outlier Detection: {column}')
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()


# In[17]:


automobile_clean.to_csv("melody.csv")


# In[18]:


automobile_final= pd.read_csv("melody.csv")


# ### DESCRIPTIVE STATISTICS

# In[13]:


continuous_variables= automobile_final[['symboling','normalized-losses','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke',
    'compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price','city-L/100km','diesel','gas']]


# In[14]:


categorical_variables = automobile_final.columns.drop(continuous_variables)


# In[15]:


print("\n mean\n")
print(continuous_variables. mean())
print("\n median \n")
print(continuous_variables. median())
print("\n mode \n")
print(continuous_variables. mode())


# In[16]:


#frequencies for categorical variables
for col in categorical_variables:
    print(automobile_final[col].value_counts())
    print("\n")


# In[17]:


#heatmap showing the correlation coefficients of the continuous variables
plt.figure(figsize=(15,6))
corr_matrix= continuous_variables.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',vmin=-1,vmax=1,linewidths=0.5)
plt.title('Heat map with continuous variables (focus on the price)')
plt.tight_layout()
plt.show()


# In[18]:


#average price per categorical variable
categorical_variables = automobile_final[['drive-wheels','body-style','make','aspiration','num-of-doors','engine-location','engine-type',
    'num-of-cylinders','fuel-system','horsepower-binned']]
price= automobile_final['price']
cats_meanprice={}
for category in categorical_variables:
    cats_meanprice[category] = automobile_final.groupby(category)['price'].mean().to_dict()

print(cats_meanprice)


# In[19]:


for category, subcategories in cats_meanprice.items():
    plt.figure(figsize=(10, 5))
    plt.bar(subcategories.keys(),subcategories.values())
    plt.title(f'Mean Automobile Price by {category}')
    plt.xlabel(category)
    plt.ylabel('Mean Price')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[20]:


#TWO CATEGORICAL VARIABLES

plt.figure(figsize=(10, 6))
sns.countplot(data=automobile_final, x='make', hue='engine-type')
plt.title('Distribution of make by engine_type')
plt.xlabel('make')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='engine-type')
plt.tight_layout()
plt.show()


# #### FEATURE SELECTION WITH RECURSIVE FEATURE ELIMINATION

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

X = automobile_final.drop(columns=['price'])
y = automobile_final['price']

X_encoded = pd.get_dummies(X, columns= ['drive-wheels','body-style','make','aspiration','num-of-doors','engine-location','engine-type','num-of-cylinders',
                                       'fuel-system','horsepower-binned'], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

rfe_selector = RFE(estimator=model, n_features_to_select=4, step=1)

rfe_selector.fit(X_train, y_train)

ranking_df = pd.DataFrame({'Specification': X_encoded.columns,'Is_Top_Driver': rfe_selector.support_,  
    'RFE_Rank': rfe_selector.ranking_}).sort_values(by='RFE_Rank')

print(ranking_df.to_string(index=False))



#  ## LINK TO GITHUB

# In[ ]:


https://github.com/kayenymiriammelody/automobile

