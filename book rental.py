#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


books=pd.read_csv('Books.csv')


# In[3]:


books.head()


# In[4]:


books.shape


# In[5]:


books=books.drop(['Image-URL-S','Image-URL-M','Image-URL-L'],axis=1)


# In[31]:


books=books.rename({'ISBN':'isbn','Book-Title':'title','Book-Author':'author','Year-Of-Publicatio':'year','Publisher':'publish'},axis='columns')


# In[32]:


books.isnull().sum(axis=0)


# In[34]:


books.info()


# In[35]:


books.isna().sum()


# In[18]:


ratings=pd.read_csv('Ratings.csv')


# In[19]:


ratings.head()


# In[29]:


ratings=ratings.rename({'User-ID':'id','ISBN':'isbn','Book-Rating':'rate'},axis='columns')


# In[30]:


ratings['id'].value_counts()


# In[36]:


# y is storing the value of x which is having the value greater than 200,specially kept

x = ratings['id'].value_counts() > 200
y = x[x].index  #user_ids
print(y.shape)
ratings = ratings[ratings['id'].isin(y)]


# In[39]:


y


# In[41]:


ratings # it is holding the value that is made by the combination of 899 id have all by meeting them given all the result 
# of that.


# In[42]:


rating_with_books = ratings.merge(books, on='isbn')
rating_with_books.head()


# #### Extract books that have received more than 50 ratings.Now dataframe size has decreased and we have 4.8 lakh because when we merge the dataframe, all the book id-data we were not having. Now we will count the rating of each book so we will group data based on title and aggregate based on rating.
# #### we have to drop duplicate values because if the same user has rated the same book multiple times so it will create a problem

# In[43]:


number_rating = rating_with_books.groupby('title')['rate'].count().reset_index()
number_rating.rename(columns= {'rate':'number_of_ratings'}, inplace=True)
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating.shape
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.drop_duplicates(['id','title'], inplace=True)


# #### we will create a pivot table where columns will be user ids, the index will be book title and the value is ratings. And the user id who has not rated any book will have value as NAN so impute it with zero.

# In[45]:


book_pivot = final_rating.pivot_table(columns='id', index='title', values="rate")
book_pivot.fillna(0, inplace=True)


# In[51]:


book_pivot.head(240)


# #### we have lots of zero values and on clustering, this computing power will increase to calculate the distance of zero values so we will convert the pivot table to the sparse matrix and then feed it to the model.

# In[46]:


from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)


# #### Now we will train the nearest neighbors algorithm. here we need to specify an algorithm which is brute means find the distance of every point to every other point.

# In[47]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)


# #### Let’s make a prediction and see whether it is suggesting books or not. we will find the nearest neighbors to the input book id and after that, we will print the top 5 books which are closer to those books. It will provide us distance and book id at that distance. let us pass harry potter which is at 237 indexes.

# In[48]:


distances, suggestions = model.kneighbors(book_pivot.iloc[237, :].values.reshape(1, -1))


# In[49]:


for i in range(len(suggestions)):
  print(book_pivot.index[suggestions[i]])


# In[ ]:




