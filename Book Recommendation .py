#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup
import pandas as pd 


# In[3]:


import pandas as pd 
import numpy as np 


# In[4]:


book = pd.read_csv("Books.csv")
rating = pd.read_csv("Ratings.csv")
user = pd.read_csv("Users.csv")


# In[5]:


book


# In[6]:


rating


# In[7]:


user


# In[8]:


rating['User-ID'].value_counts()


# In[9]:


rating['User-ID'].unique().shape


# In[10]:


x = rating['User-ID'].value_counts() > 100


# In[11]:


x[x].shape


# In[12]:


y = x[x].index
y


# In[13]:


rating = rating[rating['User-ID'].isin(y)]


# In[14]:


rating


# In[15]:


books_ratings = rating.merge(book,on='ISBN')


# In[16]:


books_ratings


# In[17]:


num_rating = books_ratings.groupby('Book-Title')['Book-Rating'].count().reset_index()


# In[18]:


num_rating


# In[19]:


num_rating.rename(columns={'Book-Rating':'No_of_ratings'},inplace=True)


# In[20]:


num_rating


# In[21]:


final_rating = books_ratings.merge(num_rating,on = 'Book-Title')


# In[22]:


final_rating


# In[23]:


final_rating.drop_duplicates(['User-ID','Book-Title'],inplace=True)


# In[24]:


final_rating


# In[25]:


book_pivot = final_rating.pivot_table(columns='User-ID',index='Book-Title',values='Book-Rating')


# In[26]:


book_pivot


# In[27]:


book_pivot.shape


# In[28]:


book_pivot.fillna(0,inplace=True)


# In[29]:


book_pivot


# In[30]:


from scipy.sparse import csc_matrix


# In[31]:


book_sparse = csc_matrix(book_pivot)


# In[32]:


book_sparse


# In[33]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')


# In[34]:


model.fit(book_sparse)


# In[35]:


distance , suggestion = model.kneighbors(book_pivot.iloc[3,:].values.reshape(1,-1),n_neighbors=6)


# In[36]:


distance


# In[37]:


suggestion


# In[38]:


for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])


# In[39]:


book_pivot.index[3]


# In[40]:


book_name = book_pivot.index


# In[45]:


import os
import pickle

# Create the 'Book_Recommender' directory if it doesn't exist
if not os.path.exists('Book_Recommender'):
    os.makedirs('Book_Recommender')

# Save the files in the 'Book_Recommender' directory
pickle.dump(model, open('Book_Recommender/model.pkl', 'wb'))
pickle.dump(book_name, open('Book_Recommender/book_name.pkl', 'wb'))
pickle.dump(final_rating, open('Book_Recommender/final_rating.pkl', 'wb'))
pickle.dump(book_pivot, open('Book_Recommender/book_pivot.pkl', 'wb'))


# In[46]:


def book_recommend(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance , suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    
    for i in range(len(suggestion)):
        books =  book_pivot.index[suggestion[i]]
        for j in books:
            print(j)


# In[47]:


book_name = 'Something Blue'
book_recommend(book_name)


# In[ ]:




