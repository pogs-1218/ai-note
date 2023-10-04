from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

'''
Q: SGDClassifier is in the sklearn.linear_model. why it is there? what's the meaning of linear model?
Q: What is binary classification problem?

'''

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    # plt.axis('off')

# 784 = 28*28
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
print(X.shape)
print(y.shape)

idx = 0
some_digit = X[idx]
print(y[idx])
# plot_digit(some_digit)
# plt.show()

# Prepare train set and test set
# NOTE: cross-validation
# NOTE: fetch_openml()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

print(y_train[:10])
print(y_train_5[:10])

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
result = sgd_clf.predict([some_digit])
print(result)