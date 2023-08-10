'''
columns:
  PassengerId
    : sequentially increment number
  Survived
    : 0 or 1
  Pclass
    : 1 or 2 or 3
  Name
    : string
    : includes one of Mr, Mrs, Miss, Master
  Sex
    : male or female
  Age
    : numerical value including NaN
  SibSp
    : numerical value(0~5)
  Parch
    : nemerical value(0~5)
  Ticket
    : ?? not unique
  Fare
    : nemerical value
  Cabin
    : string including NaN
  Embarked
    : S, C, Q (including NaN)
* 데이터로 부터 생존자를 예측해야하는 문제. 지도학습이므로 target output이 있어야함 Survived로 사용(0 혹은 1)
우선 supervised learning이면서 categorical문제이다. 생존 혹은 죽음.
categorical 문제를 해결하기 위해 적용가능한 모델은 KNN, Logstic Regresssion을 배웠었다.
supervised learning의 데이터에서 출력에 영향을 주는 feature를 찾아야 할 듯?
이름이 생존에 영향이 있을까? 없을 것 이다.
--> https://www.kaggle.com/c/titanic/data
data dictionary가 있음

* feature를 분류할때 데이터가 아주 많으면 어떻게 하나? 지금은 하나씩 둘러봤는데 데이터가 클때도 가능할까?
생각해보니 이것도 프로그램화(?)하는 게 가능하다. categorical 유형의 경우 해당 컬럼의 값들을 읽어서 중복된 정도를 카운팅하면 된다.
아마 머신러닝 라이브러리들에서도 지원할 듯?
* 빈값도 마찬가지. 모두 눈으로 찾아야만 할까?

먼저 프로그램적으로 데이터 분포, 연관관계 등을 볼 수 있도록 해보자. 
1. 전체 데이터 셋에서 남,녀의 분포는 어떻게 될까?

==> the area of feature engineering
It is important!!


'''
import pandas as pd
import matplotlib.pyplot as plt

origin_data = pd.read_csv('dataset/titanic/train.csv')

def feature_eng():
  ''' 
  '''
  e = origin_data['Embarked']
  vc = e.value_counts()
  print(type(vc))
  print(vc.values)

feature_eng()
