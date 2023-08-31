## Overview
데이터를 살펴보고 정리하는것.
모델을 학습하기 전에 어떤 아웃풋이 나와야 하는지를 먼저 파악해야함.
위/경도에 따른 위치와 해당 위치의 집의 다양한 속성들이 있음.
이때 이러한 데이터가 주어졌을 때 집값의 중간값을 찾는 것이 목표.

그럼 기존 데이터를 분석해서 어떤 요인들이 집값에 얼마나 영향을 미치는지 알아야함.
이런 상관관계를 파악한 이후에
없거나 더 유용한 데이터는 더하고,
필요없는 데이터는 제거.

이렇게 준비된 데이터를 학습모델의 입력으로 사용할 것.

데이터는 어떻게 수집할 것인가?
데이터를 수집햇으면, 테스트셋을 준비해야 하낟.

테스트셋은 왜 필요할까?
테스트셋은 모델학습 및 튜닝이 완료된 최종모델의 테스트를 위해 사용된다.

테스트셋 추출의 특징
학습셋과 유사한 모양(?)을 가져야 한다.
학습셋의 크기에 따라 비율을 조정한다.

Data Snooping
overfitting이랑 같은 개념으로 볼수 있을까?
-> 관점이 다르다. 
모델의 성능을 높이기 위해 훈련데이터를 반복적으로 조작함으로 발생함.
모델 개발 과정에서 데이터를 다루는 방식에 관한 문제이고,
overfitting은 데이터자체와 상관없이 모델자체가 훈련데이터에 과적합된 것.

훈련, 검증, 테스트 데이터가 필요하다.
훈련데이터로 모델을 학습하고
검증데이터로 학습된 모델을 테스트, 튜닝, 모델선택 등에 활용
테스트 데이터는 최종모델의 성능평가에만 사용.(일반화된 테스트가 가능함)
https://scikit-learn.org/stable/_images/grid_search_cross_validation.png

머신러닝의 중요한 특징중 하나가 일반화 인것 같다.
전통적인 알고리즘은 일반화를 사람이 고안하지만
머신러닝에서는 데이터 학습을 통해서 이루어지는것이 다를뿐.

https://scikit-learn.org/stable/_images/grid_search_workflow.png
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

* stratified sampling
strata

sklearn의 대표 인터페이스
estimators:
fit() API
dataset을 받아서 파라미터를 설정함

transformers:
transform() API
데이터를 변환

predictors:
predict() API
predict수행

이제 데이터에서 비어있는 값을 채워보자.
sklearn문서에는 preprocessing다음에 imputer가 소개된다.
순서가 영향이 있는걸까?

sklearn의 imputer는 missing value를 채워주기위한 class
여러 종류의 imputer가 있고 일단은 

* categorical data를 encoding할 때, missing value에 대한 처리를 할 수 있다.
* 그전에 missing value를 처리하고 수행하는 거랑 큰차이가 있을까?

## Preprocessing
### Concepts
---
#### Missing Values

#### Standardization

#### Categorical Features

### Data Overview
총 20640개의 데이터.
ocean_proximity는 문자(object). 나머지는 숫자(float64) -> 문자열 처리는 어떻게 할 것인가?
데이터 크기는 1.6MB -> 데이터 크기를 더 줄일수 있을까?
total_bedrooms는 20433개로, 빈값이 존재. -> 빈값에 대한 처리를 어떻게 할것인가?

total_rooms, total_bedrooms, median_income, population
: 왼쪽으로 치우진 분포
median_house_value, housing_media_age
: 전반적으로 고른 분포이지만 가장 오른쪽에 튀어나온 값(outlier)있음



## Appendix
### mean, median, and mode

### Standard Deviation

### Normal(Gaussian) Distribution

### Standard Normal Distribution


## Analyze
### Take a look

## sklearn
파이프라인을 만들어서 붙이는게 확장성과 유지보수가 좋아보임
결국 입력과 출력의 연속이기 때문.


### corr() function
상관계수
correration coefficient
https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
Pearson's R

## drop()
drop specific labels from rows or columns

## imputer
missing value를 보정하기 위핸 클래스.
numeric데이터에만 적용됨.
missing value를 채우는 옵션을 선택할 수 있음 mean, most_frequent등

데이터를 먼저 분석하는게 중요하다.
어떤 상관관계가 있고 
학습데이터와 테스트데이터를 분리하는 방법


nemeric데이터와 그렇지 않은 데이터(문자열 같은?)의 처리 방법을 구분해서 알고있어야 함.
생각해보면 머신러닝 자체가 대부분 구조화된 데이터에 적용되는 것 같은 느낌이다.
supervised learning만 그런걸까?

## feature scaling
왜 필요한가?

### normalization

### standardization
- standard normal distribution을 일단 알아야함
z-distribution

- normal(gaussian) distribution


- mean
referred as 'average'
전부더하고 개수로 나눠서 구함.
outliers의 영향이 있음. 

- mean, median, mode
이 세개를 구분해서 알고있어야함
각 상황에 따라 어떤값을 central tendency로 사용할지 정해짐.

- standard deviation
중심값(mean?)에서 얼마만큼의 다른 데이터가 있냐를 나타내는 정도!!
sd가 높다면 mean값에서 먼데이터가 많다는 것이고 그래프에서는 넓은 종모양이됨
sd가 작다면 mean값에서 가까운 데이터가 많다는 것이고 그래프에서는 작은 종모양이됨. 중앙밀집형


- variance
np.std**2 임.
표현방식의 차이


### outlier


## rbf 


### evaluation of model


## sklearn's preprocessing data
link: https://scikit-learn.org/stable/modules/preprocessing.html

