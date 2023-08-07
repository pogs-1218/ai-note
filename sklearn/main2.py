'''
machine learning basics with sklearn
key concenpts
- what is meachine learning?
- category of machine learning
  : supservised learning
  : unsupervised learning
- how does machine learning work? 
- features are the most important

why scikit-learn?
categories of machine learning
: regression, classification

k-neaarest neighbor(KNN) classification
'''
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
print(iris.feature_names)
print(f'{type(iris.target)} | {iris.target}')
print(f'tareget shape: {iris.target.shape}')
print(f'data shape: {iris.data.shape}')
print(iris['target_names'])
# print(f'{type(iris)}\n {iris.data}')

# iris data has 4 features
# target has 3 categories
X, y = iris.data, iris.target

def no_split():
  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(X, y)
  y_pred = knn.predict(X)
  print(metrics.accuracy_score(y, y_pred))

  knn5 = KNeighborsClassifier(n_neighbors=5)
  knn5.fit(X, y)
  y_pred = knn5.predict(X)
  print(metrics.accuracy_score(y, y_pred))

  logreg = LogisticRegression()
  logreg.fit(X, y)
  y_pred = logreg.predict(X)
  print(metrics.accuracy_score(y, y_pred))

def split():
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  print(f'knn with k=1 : {metrics.accuracy_score(y_test, y_pred)}')

  knn5 = KNeighborsClassifier(n_neighbors=5)
  knn5.fit(X_train, y_train)
  y_pred = knn5.predict(X_test)
  print(f'knn with k=5 : {metrics.accuracy_score(y_test, y_pred)}')

  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  print(f'logistic reg : {metrics.accuracy_score(y_test, y_pred)}')

# TODO 
def find_best():
  pass

# no_split()
split()

# bias and variance tradeoff
# overfitting
# evaluating model 