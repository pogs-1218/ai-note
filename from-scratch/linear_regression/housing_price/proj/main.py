import numpy as np
from utils import *

data_file_path = './data/data.csv'
#features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
#            'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
#
#(train_data, train_label, test_data, test_label) = load_data(data_file_path, features)
#plot_features(train_data, train_label, features)

# As a result, I choose bathrooms, sqft_living, sqft_above.
features = ['bathrooms', 'sqft_living', 'sqft_above']
(x_train, y_train, x_test, y_test) = load_data(data_file_path, features)

#data_file_path = './data/houses.csv'
#(x_train, y_train) = load_data2(data_file_path)

#print(f'y_train shape: {y_train.shape}')
#print(f'x_train shape: {x_train.shape}')
#print(f'{y_train[:3]}')
#print(f'{x_train[:3]}')
#x_train = x_train[:, 0:4:3]

w_init = np.zeros((x_train.shape[1], 1))
b_init = 0.0
steps = 100
learning_rate = 1e-7
w_final, b_final, j_history = gradient_descent(w_init, b_init, x_train, y_train, learning_rate, steps)

# todo: cost graph!
