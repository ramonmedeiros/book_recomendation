import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# count ratings
rating_count = pd.DataFrame(df_ratings.groupby('isbn')['rating'].count())
rating_count.sort_values('rating', ascending=False).head()

# make average of it
average_rating = pd.DataFrame(df_ratings.groupby('isbn')['rating'].mean())
average_rating['ratingCount'] = pd.DataFrame(df_ratings.groupby('isbn')['rating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()

# merge
combine_book_rating = pd.merge(average_rating, df_books, on='isbn')

# remove duplicated
combine_book_rating = combine_book_rating.drop_duplicates(['title'])

# pivot and generate matrix
combine_rating_pivot = combine_book_rating.pivot(index = 'title', columns='ratingCount', values = 'rating').fillna(0)
combine_rating_matrix = csr_matrix(combine_rating_pivot.values)

# create model
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(combine_rating_matrix)

# function to return recommended books - this will be tested
def get_recommends(book = ""):

    for i in combine_book_rating:
        import ipdb;ipdb.set_trace()

    distances, indices = model.kneighbors(combine_book_rating.iloc[book, :].reshape(1, -1), n_neighbors = 6)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(combine_book_rating.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, combine_book_rating.index[indices.flatten()[i]], distances.flatten()[i]))
        return distances[5]

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
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
