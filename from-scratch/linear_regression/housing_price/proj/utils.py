import pandas as pd
import matplotlib.pyplot as plt
import math

def load_data(path: str):
  '''
    Return:
      train_data
      train_label
      test_data
      test_label
  '''
  df = pd.read_csv(path) 
  print(f"length: {len(df)}")

  # Split train data and test data, 8:2
  offset = math.ceil(len(df) * 0.8)
  train_df = df[:offset]
  test_df = df[offset:]
  print(f"{len(train_df)}, {len(test_df)}")

  # Choose features, 
  features = ['bedrooms', 'bathrooms']
  target = ['price']

  train_data = train_df.loc[:,features]
  train_label = train_df.loc[:,target]
  print("train ================")
  print(f"{type(train_data)}\n {train_data[:3]} ")
  print(f"{train_label[:3]}")

  print("convert pd.DataFrame ot numpy.ndarray")
  train_data = train_data.to_numpy()
  train_label = train_label.to_numpy()
  print("train as numpy -----------")
  print(f"{type(train_data)} with {train_data.shape}\n {train_data[:3]}")
  print(f"{type(train_label)} with {train_label.shape}\n {train_label[:3]}")

  test_data = test_df.loc[:,features]
  test_label = test_df.loc[:,target]
  test_data = test_data.to_numpy()
  test_label = test_label.to_numpy()

  return (train_data, train_label, test_data, test_label)
