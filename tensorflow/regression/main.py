import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

def tf_linear_reg():
  x_train = np.array([[1.], [2.]], dtype=np.float32)
  y_train = np.array([[300.], [500.]], dtype=np.float32)

#fig, ax = plt.subplots(1, 1)
#ax.scatter(x_train, y_train, marker='x', c='r', label='Data Points')
#plt.show()

  linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
#w, b = linear_layer.get_weights()
#print(f'{w}, {b}')

  a1 = linear_layer(x_train[0].reshape(1, 1))
  print(a1)

  w, b = linear_layer.get_weights()
  print(f'{w}, {b}')

def tf_log_reg():
  # should reshape??
  x_train = np.array([0., 1., 2., 3., 4., 5.], dtype=np.float32).reshape(-1, 1)
  y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

  # x_train.shape will be (6,) -> (6, 1)

  print(f'{x_train.shape}, {y_train.shape}')

  pos = y_train == 1
  neg = y_train == 0

  model = Sequential(
      [
        tf.keras.layers.Dense(units=1, input_dim=1, activation='sigmoid', name='L1')
      ])

#model.summary()
  logistic_layer = model.get_layer('L1')
  w, b = logistic_layer.get_weights()
  print(w, b)

  set_w = np.array([[2]])
  set_b = np.array([-4.5])
  logistic_layer.set_weights([set_w, set_b])

  # inference
  a1 = model.predict(x_train[0].reshape(1,1))
  print(a1)
  print(a1.numpy())

def test_tf_layer1():
  x = np.array([[200., 17.],
                [120., 5.],
                [425., 20.],
                [212., 18.]])
  y = np.array([1, 0, 0, 1])
  l1 = tf.keras.layers.Dense(units=3, activation='sigmoid') 
  l2 = tf.keras.layers.Dense(units=1, activation='sigmoid') 
  model = tf.keras.Sequential([l1, l2])

tf_log_reg()

# tf.Tensor is a data type
#tf.Tensor([[0.2, 0.7, 0.3]], shape=(1, 3), dtype=float32)
