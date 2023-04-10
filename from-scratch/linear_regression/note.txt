housing price example
basic of machine learning

linesar regression

todo: 
download house price dataset

regression vs classfication

notation
x: input feature, 
y: output/target variable
m: nmber of training examples
this is supervised learning so that it has data and label
x is data, y is label
(x, y) is single training example which is data with a corresponding label.
x_i, y_i : ith training example.
x -> f -> y_hat
feature -> model -> prediction
How to represent f?
f_wb = WX+B

why linear function?

linear + regression!
with one variable(single feature)

f_wb = wx_b -> linear!
f_wb: taret(output) 

* the goal is to find w, b(parameters) that is close to y(target)
  (x, y) : train data is fixed!(supervised learning)

* cost function
- squared error cost function
if 0 <= i <= m
j_wb = np.sum((y_hat[i]-y[i])**2) / 2m


* why gradient descent?
start with some w, b
keep changing w, b to reduce J(w, b)
until minium
w = w - (alpha * dj_dw)

* about learning rate
if it is too large?
if it is too small?


linear regression with multiple features
- what's different with single feature linear regression
x_vec = [x1, x2, x3, ... xn]
w_vec = [w1, w2, w3, ... wn]
* b is scalar value.
f_wb = w1x1 + w2x2 + w3x3 + .. + b

-> so that how to compute those more efficiently?
f_wb = np.dot(x_vec, w_vec) + b
 * parelle! how? through numpy? how?


* an alternative to gradient descent
1. normal equation
2. 


* feature scaling
why?
much faster.
why?
too large (or too small) predict value(y)
if a feature(x1) is large, the parameter(w1) will be small
if the fature(x2) is small, the parameter(w2) will be more large!

so that features should be rescaled.
(think about it with graph)

ideal range of the feature is 0 to 1
how?
if 30 < x < 100
x_scaled = x/100
= 0.3 < x_scaled < 1

Scaling methods
1. mean normalization
get average.
if 300 < x < 2000, mu(average) is 600
x_scaled = x-mu(avg) / 2000(max)-300(min)

2. z-score normalization

As a result, if the feature is too large or too small, the feature should be rescaled.

! How to make sure gradient descent is working correctly?
todo: make a graph that has x-axis is the number of iterations and y-axis is the cost value(j)
think about the goal of the cost function. The output of cost function should be decreased!(after every iteration)

* Automatic convergence test
Let e(epsilon) be 1e-3
If the cost decreases by <e in one iteration, declare convergence.

! How to choose the learning rate?
too small? : so slow!!
too large? : not be convergence well!
TODO: Also make a graph with different learning rate. (x-axis: parameters, y-axis: cost)


* Classification
vs linear regression

logistic regression
either 0 or 1
linear regression is not sufficient for classfication problem.
let's look at the simple graph.

the probability!!!
if f(x) = 0.7
70% change that y, f(x) is 1
P(y=0) + P(y=1) = 1

if z >= 0, y = 1
else y = 0

* decision boundary
g_z = g(w1x1 + w2x2 + b)
z = w_vec * x_vec + b = 0
if w1=1, w2=1, b=3
z = x1+x2+3 = 0

* cost function of the logistic regression
- Compare with the loss function.
c = sum(0->m) of loss / m
loss(L) is for single example
if y[i] = 1, -log(f(x[i]))
if y[i] = 0, -log(1-f(x[i]))
** Think within a graph.

so the loss function L is
Loss(f(x[i], y[i]) = (-y[i] * log(f(x[i]))) - (1-y[i]) * log(1-f(x[i]))


- why machine learning?

- linear regression(one variable)
  - ex. housing price prediction
  
  - Prediction
  - make a model
  - Evaluate
    - cost function
  - Train(Learning)
    - gradient descent
    - pyplot graph

- linear regression(multiple variable)
  - vectorization
  - polynomial regression
  - feature scaling
  - scikit-learn

- classfication with logisitic regression
  - logistic regression
  - decision boundary
  - cost function
  - gradient descent
  - overfitting
  - regularization
