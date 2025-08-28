# Implementation-of-Linear-Regression-Using-Gradient-Descent
# AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

# Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
# Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

# Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SAI PRASATH.P
RegisterNumber: 212224230238
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using the in a linear regression model
  """
  m=len(y) # length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

from matplotlib.container import ErrorbarContainer
from IPython.core.interactiveshell import error
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take the numpy array X,y,theta and update theta by taking the num_tiers gradient with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)
    J_history=[]

    for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions -y))
      descent=alpha *1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction"

def predict(x,theta):
  """
  Tkes in numpy array of x and theta and return the predicted value of y base
  """

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
# Output:
# Profit Prediction Graph :
<img width="697" height="449" alt="image" src="https://github.com/user-attachments/assets/46d8dc65-64fb-4914-a198-6a663a22ebe5" />
<img width="340" height="65" alt="image" src="https://github.com/user-attachments/assets/22a77e46-2d89-4124-b57e-47b0f5d002ce" />

# Compute Cost Value :
<img width="280" height="56" alt="image" src="https://github.com/user-attachments/assets/594fcf37-a100-4e82-9159-c48e0bf18ceb" />

# h(x) Value :
<img width="623" height="454" alt="image" src="https://github.com/user-attachments/assets/03a5839f-f767-475f-8118-8fce2e75ae26" />

# Cost function using Gradient Descent Graph :
<img width="703" height="45" alt="image" src="https://github.com/user-attachments/assets/3f0011f8-a2c0-4d56-b3d7-9eeb4c7e4e61" />

# Profit for the Population 35,000 :
<img width="621" height="70" alt="image" src="https://github.com/user-attachments/assets/7d442c4b-c544-495b-b3db-9681d8ad7754" />


# Profit for the Population 70,000 :
<img width="703" height="45" alt="image" src="https://github.com/user-attachments/assets/6f735003-d5a8-4e64-92c2-0af01aeb3661" />

# Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
