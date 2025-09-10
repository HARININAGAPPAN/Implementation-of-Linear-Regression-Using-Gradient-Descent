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
Developed by: Harini N
RegisterNumber:  212223040057
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
# output :
<img width="780" height="804" alt="image" src="https://github.com/user-attachments/assets/1e646f76-66e5-4252-ada0-ce45f325bc18" />

<img width="723" height="796" alt="image" src="https://github.com/user-attachments/assets/9ffce248-81db-4f98-b762-e2788a9a970c" />

<img width="841" height="624" alt="image" src="https://github.com/user-attachments/assets/9f73f53c-d455-4968-a05d-b88292a0f2a4" />


# Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
