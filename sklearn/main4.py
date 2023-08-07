'''
k-fold cross validation
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pred))

# kf = KFold(5, shuffle=False)
# print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
# for i, data in enumerate(kf.split(X)):
#     print('{:^9} {} {:^25}'.format(i, data[0], data[1]))

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())

# TODO: find optimal!

# TODO: KNN vs logistic regression

linreg = LinearRegression()
df = pd.read_csv('dataset/Advertising.csv', index_col=0)

features_cols = ['TV', 'Radio', 'Newspaper']
X = df[features_cols]
y = df['Sales']
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')
print(np.sqrt(-scores).mean())

features_cols = ['TV', 'Radio']
X = df[features_cols]
y = df['Sales']
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')
print(np.sqrt(-scores).mean())

