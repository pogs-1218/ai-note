Activation function
---
By now, I used the sigmoid function for binary classfication.  
So, what's the alternatives to that?  
How to choose a proper activation function to solve a specifc problem?  

1. linear activation function
g(z) = z

2. Sigmoid
0 < g(z) < 1

3. ReLU 
Rectified Linear Unit?
g(z) = 0 if z < 0
g(z) = z if z >= 0

Coosing activation functions
Depends on what the target label.


Output layer
---
Binary classification
Sigmoid, y = 0/1

Regression
target y is negative or positive
Linear activation function, y = +/-

Regression
target y is only positive
ReLU, y = 0 or +


Hidden layer
---
ReLU is the most common choice.
ReLU vs Sigmoid
ReLU is more faster.
ReLU is flat or not flat but, sigmoid is flat (0, 1)
 -> slower in gradient descent. : how to prove it?
 -> when g(z) is flat, dj_dw ~= 0 


Why do we need activation functions?

