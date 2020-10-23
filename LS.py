# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:06:33 2020
@author: Raj Shinde
@author: Shubham Sonawane
@author: Prasheel Renkuntla

@file:   LS.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_1.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
x_mean = np.mean(x)

#least square
X = np.array([x**2 ,x ,np.ones(250)]).T
Y = np.array([y]).T

B = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))

a = B[0]
b = B[1]
c = B[2]

Y_eqn = a * x * x + b * x + c
plt.scatter(x,y) 
plt.plot(x,Y_eqn,color="red")
plt.title("LS")
print("The equation of the curve is y = "+str(np.round(float(a),3))+"x**2 + "+str(np.round(float(b),3))
+"x + "+str(np.round(float(c),3)))