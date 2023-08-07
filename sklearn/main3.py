import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('dataset/Advertising.csv', index_col=0)
# sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales',size=5, aspect=0.7, kind='reg')
# plt.show()

# features_cols = ['TV', 'Radio', 'Newspaper']
features_cols = ['TV', 'Radio']
X = df[features_cols]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(f'{X_train.shape}/{X_train.shape}')

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(f'{linreg.intercept_} / {linreg.coef_}')
# feature_with_coef = zip(features_cols, linreg.coef_)
y_pred = linreg.predict(X_test)

# mean absolute error(MAE)
# mean squared error(MSE)
# root mean squared error(RMSE)
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))