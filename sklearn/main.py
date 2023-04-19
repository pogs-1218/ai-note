import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#sklearn.show_versions()
RANDOM_STATE=55

df = pd.read_csv('heart.csv')
#print(df.head())

cat_variables = ['Sex', 
                 'ChestPainType',
                 'RestingECG',
                 'ExerciseAngina',
                 'ST_Slope']
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
#print(df.head())

features = [x for x in df.columns if x not in 'HeartDisease']
print(len(features))

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#help(train_test_split)

X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE)
print(f'train samples: {len(X_train)}, validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train)}')

# Building the Models
# Decision Tree
min_samples_split_list=[2, 10, 30, 50, 100, 200, 300, 700]
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
  model = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                 random_state=RANDOM_STATE)
  model.fit(X_train, y_train)
  pred_train = model.predict(X_train)
  pred_val = model.predict(X_val)

  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
  accuracy_train = accuracy_score(pred_train, y_train)
  accuracy_val = accuracy_score(pred_val, y_val)
  accuracy_list_train.append(accuracy_train)
  accuracy_list_val.append(accuracy_val)

#plt.xlabel('min_samples_split')
#plt.ylabel('accuracy')
#plt.plot(accuracy_list_train)
#plt.plot(accuracy_list_val)
#plt.legend(['Train', 'Validation'])
#plt.show()


