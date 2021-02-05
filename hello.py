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


#plt.rc("font", size=15)
#df_ratings.rating.value_counts(sort=False).plot(kind='bar')
#plt.title('Rating Distribution\n')
#plt.xlabel('Rating')
#plt.ylabel('Count')
#plt.savefig('system1.png', bbox_inches='tight')
#plt.show()

# count ratings
rating_count = pd.DataFrame(df_ratings.groupby('isbn')['rating'].count())
rating_count.sort_values('rating', ascending=False).head()

# make average of it
average_rating = pd.DataFrame(df_ratings.groupby('isbn')['rating'].mean())
average_rating['ratingCount'] = pd.DataFrame(df_ratings.groupby('isbn')['rating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()

# remove small ratings?
counts1 = df_ratings['user'].value_counts()
ratings = df_ratings[df_ratings['user'].isin(counts1[counts1 >= 200].index)]
counts = df_ratings['rating'].value_counts()
ratings = df_ratings[df_ratings['rating'].isin(counts[counts >= 100].index)]

# merge
import pdb;pdb.set_trace()
ratings_pivot = ratings.pivot(index='user', columns='isbn').rating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.shape)

ratings_pivot.head()

# function to return recommended books - this will be tested
def get_recommends(book = ""):
    X = np.array(df_books['title'])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
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
