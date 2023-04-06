from utils import *

data_file_path = './data/data.csv'
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
            'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

(train_data, train_label, test_data, test_label) = load_data(data_file_path, features)
plot_features(train_data, train_label, features)

# As a result, I choose bathrooms, sqft_living, sqft_above.

