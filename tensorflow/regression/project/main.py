import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from coursera lecture.
def load_data():
  rng = np.random.default_rng(2)
  X = rng.random(400).reshape(-1,2)
  X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
  X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
  Y = np.zeros(len(X))
  i=0
  for t,d in X:
    y = -3/(260-175)*t + 21
    if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
      Y[i] = 1
    else:
      Y[i] = 0
    i += 1
  return (X, Y.reshape(-1,1))

x_train, y_train = load_data()
print('Original training data =================')
print(x_train.shape, y_train.shape)
print(f'{x_train[0]}')
# 200 training examples with 2 features.

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
norm_layer = tf.keras.layers.Normalization(axis=1)
norm_layer.adapt(x_train)
norm_x_train = norm_layer(x_train)
print('Normalized training data =================')
print(f'{norm_x_train[0]}')

xt = np.tile(norm_x_train, (1000, 1))
yt = np.tile(y_train, (1000, 1))
print('Tiled training data shape =====================')
print(f'{xt.shape}, {yt.shape}')

# only for testing
tf.random.set_seed(1234)

# https://www.tensorflow.org/guide/keras/sequential_model?hl=ko
# create a sequential layer
model = Sequential([
          tf.keras.Input(shape=(2,)),
          Dense(3, activation='sigmoid', name='layer1'),
          Dense(1, activation='sigmoid', name='layer2')
        ])
# The sequential model has no weights if no input layer.
# Think! why?
# So, summary() couldn't be called in that case.
#print(model.weights) 
#model.summary()
# how about now?
#model(tf.ones((1, 4)))

model.summary()

w1, b1 = model.get_layer('layer1').get_weights()
w2, b2 = model.get_layer('layer2').get_weights()
print("layer1's parameters =============================")
print(f"{w1.shape}, {w1}\n {b1.shape}, {b1}\n")
print("layer2's parameters =============================")
print(f"{w2.shape}, {w2}\n {b2.shape}, {b2}")

# https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
# compile is configuring
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy())

# Train the model with epochs
model.fit(xt, yt, epochs=10)

# ok, then? 
w1, b1 = model.get_layer('layer1').get_weights()
w2, b2 = model.get_layer('layer2').get_weights()

model.get_layer('layer1').set_weights([w1, b1])
model.get_layer('layer2').set_weights([w2, b2])

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
x_test = np.array([[200, 13.9], [200, 17]])
norm_x_test = norm_layer(x_test)
predictions = model.predict(norm_x_test)
print('Predictions = \n', predictions)


