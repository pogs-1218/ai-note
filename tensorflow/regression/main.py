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
  a1 = model.predict(x_train[0].reshape(1,1))
  print(a1)

tf_log_reg()
