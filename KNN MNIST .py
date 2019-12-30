#!/usr/bin/env python
# coding: utf-8

# # Recognising Handwritten Digits on MNIST Dataset using KNN 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Step 1. Data Preparation

# In[2]:


df = pd.read[InternetShortcut]
URL=http://localhost:8888/notebooks/KNN%20MNIST%20.ipynb#
_csv('train.csv')
#print(df.shape)


# In[3]:


#print(df.columns)


# In[5]:


#df.head(n=5)


# In[5]:


data = df.values
#print(data.shape)
#print(type(data))


# In[6]:


X = data[:,1:]
Y = data[:,0]

print(X.shape,Y.shape)


# In[4]:


split = int(0.8*X.shape[0])
print(split)

X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)


# In[8]:


#Visualise Some Samples

def drawImg(sample):
    img = sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()
    
drawImg(X_train[0])
print(Y_train[0])
    
    


# ### Step 2. K-NN

# In[32]:


# Can we apply KNN to this data ?


# In[9]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred
    
    
    


# ### Step 3 : Make Predictions 

# In[10]:


pred = knn(X_train,Y_train,X_test[1])

print(int(pred))


# In[11]:



drawImg(X_test[1])
print(Y_test[1])


# In[ ]:


# Write one method which computes accuracy of KNN over the test set !

