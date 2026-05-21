# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Store the independent variable (hours studied) and the dependent variable (marks obtained) in arrays and reshape the input data for model training.
3. Create and train the Linear Regression model using least squares.
4. Predict the marks using the input taken from the user.
5. Finally, display the predicted marks and a graph showing the data points and the regression line for visualization.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MAHAGANAPATHI R
RegisterNumber:  212225040221
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

C = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
D = np.array([35, 50, 65, 70, 85])

model = LinearRegression()
model.fit(C, D)

m = model.coef_[0]
b = model.intercept_
print("Slope (m):", m)
print("Intercept (b):", b)

c_input = float(input("Enter hours studied: "))
predicted_marks = model.predict([[c_input]])
print("Predicted Marks:", predicted_marks[0])

D_pred = model.predict(C)

plt.scatter(C, D, label="Actual Data")
plt.plot(C, D_pred, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression (Using sklearn)")
plt.legend()
plt.show()

```

## Output:

<img width="1919" height="908" alt="image" src="https://github.com/user-attachments/assets/64ae2d11-a9c5-42e7-818f-2073a772137e" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
