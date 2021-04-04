#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# get data files
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'
users_filename = "BX-Users.csv"

df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['bookId', 'title', 'author'],
    usecols=['bookId', 'title', 'author'],
    dtype={'bookId': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['userId', 'bookId', 'rating'],
    usecols=['userId', 'bookId', 'rating'],
    dtype={'userId': 'int32', 'bookId': 'str', 'rating': 'float32'})

# get list of users to remove
user_ratingCount = (df_ratings.
     groupby(by = ['userId'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['userId', 'totalRatingCount']]
)
users_to_remove = user_ratingCount.query('totalRatingCount >= 200').userId.tolist()

# merge rating and catalog by bookId
df = pd.merge(df_ratings,df_books,on='bookId')

# create totalRatingCount
book_ratingCount = (df.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )

rating_with_totalRatingCount = df.merge(book_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# remove books with less than 100 ratings
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= 100')

# remove from the dataset users with less than 200 ratings 
rating_popular_movie = rating_popular_movie[rating_popular_movie['userId'].isin(users_to_remove)]

# drop duplicates
rating_popular_movie.drop_duplicates(subset=['title'])

# pivot table and create matrix
book_features_df = rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
book_features_df_matrix = csr_matrix(book_features_df.values)

# function to return recommended books - this will be tested
def get_recommends(book = ""):
    model_knn = NearestNeighbors(metric = 'cosine', n_neighbors=5, algorithm='auto')
    model_knn.fit(book_features_df_matrix)

    # found book TODO: user a better search
    for query_index in range(len(book_features_df)):
        if book_features_df.index[query_index] == book:
            break

    # creating return structure
    ret = [book_features_df.index[query_index], []]
    distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index,:].values.reshape(1, -1))
    # now we located the book. lets show the recomendations
    for i in range(1, len(distances.flatten())):
        ret[1].append([book_features_df.index[indices.flatten()[i]], distances.flatten()[i]])
    return ret

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False

  print(recommends)
  print()

  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]

  print(recommended_books)
  print(recommended_books_dist)

  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You havn't passed yet. Keep trying!")

test_book_recommendation()
