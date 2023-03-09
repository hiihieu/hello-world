#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Load necessary packages 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, ensemble
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import pprint
import sklearn.impute as sk


# In[ ]:





# In[72]:


# Load data
data_train = pd.read_csv("train.csv", sep =",")
data_test = pd.read_csv("test.csv", sep = ",")


# In[ ]:





# In[73]:


# Extract Id column
test_ids = data_test['Id']

# Drop the 'Id' column we wont need for the prediction process.
data_train.drop("Id", axis = 1, inplace = True)
data_test.drop("Id", axis = 1, inplace = True)


# In[ ]:





# In[74]:


# Drop column 'SalePrice" to use for prediction. 
y = data_train['SalePrice']
X = data_train.drop('SalePrice', axis=1)


# In[ ]:





# In[75]:


# Split data_train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





# In[76]:


# Define imputers for categorical and numerical features
#Train Set
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numerical_cols = X_train.select_dtypes(exclude='object').columns.tolist()
#Test Set
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='mean')


# In[ ]:





# In[77]:


# Categorical cleaning 
# Replace missing values in categorical features with the most frequent category
X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
X_val[categorical_cols] = categorical_imputer.transform(X_val[categorical_cols])
data_test[categorical_cols] = categorical_imputer.transform(data_test[categorical_cols])


# In[ ]:





# In[78]:


# Numerical Cleaning
# Replace missing values in numerical features with the mean value
X_train[numerical_cols] = numerical_imputer.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = numerical_imputer.transform(X_val[numerical_cols])
data_test[numerical_cols] = numerical_imputer.transform(data_test[numerical_cols])


# In[ ]:





# In[79]:


# Perform one-hot encoding on categorical columns
X_train_encoded = pd.get_dummies(X_train)
X_val_encoded = pd.get_dummies(X_val)
data_test_encoded = pd.get_dummies(data_test)


# In[ ]:





# In[80]:


# Reindex the validation and test sets to have the same column order as the training set
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
data_test_encoded = data_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# In[ ]:





# In[81]:


# Define and train the model
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, loss='ls')
model.fit(X_train_encoded, y_train)


# In[ ]:





# In[82]:


# Evaluate the model on the validation set
y_val_pred = model.predict(X_val_encoded)
mse = mean_squared_error(y_val, y_val_pred)
print("MSE on validation set:", mse)


# In[ ]:





# In[83]:


#Find how best model fits the data quantitatively. sklearn
print(model.score(X_train_encoded, y_train))

#max_depth =3 , score of 96% 
#max_depth = 5 , score of 99%


# In[68]:


# Make predictions on the test set and save them to a CSV file
test_pred = model.predict(data_test_encoded)


# In[ ]:





# In[69]:


# Combine the predicted SalePrice with the test IDs
output = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})

# Save to csv file 
output.to_csv('submission_March8_ver2.csv', index=False)

# Save to csv file 
output.to_csv('submission_March8_ver2.csv', index=False)


# In[ ]:





# In[70]:


# Make predictions on the validation set
y_val_pred = model.predict(X_val_encoded)

# Create a scatter plot to show actual vs. predicted SalePrice
plt.scatter(y_val, y_val_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs. Predicted SalePrice')
plt.show()


# In[51]:


import seaborn as sns


# Select the variables to include in the pairplot
vars = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

# Create the pairplot
sns.pairplot(data_train[vars], diag_kind='kde')


# In[ ]:





# In[52]:


import seaborn as sns

# Compute the correlation matrix
corr = data_train.corr()

# Create a heatmap plot of the correlation matrix
sns.heatmap(corr, cmap='coolwarm')


# In[ ]:





# In[53]:


# Compute the correlation coefficients
correlations = data_train.corr()

# Extract the correlation coefficients for SalePrice
corr_saleprice = correlations['SalePrice']

# Sort the correlation coefficients in descending order
corr_saleprice = corr_saleprice.sort_values(ascending=False)

# Print the top 10 correlations
print(corr_saleprice.head(10))


# In[ ]:





# In[ ]:


#Submitted for Score 0.14068 - Rank 1611

