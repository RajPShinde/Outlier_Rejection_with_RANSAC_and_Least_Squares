# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 10:12:54 2020
@author: Raj Shinde
@author: Shubham Sonawane
@author: Prasheel Renkuntla

@file:   ransac+LS.py
"""

import numpy as np

import matplotlib.pyplot as plt
import random
from random import randint

# Original data Plot
data = pd.read_csv('data_1.csv')
# plt.scatter(list(data["x"]),list(data["y"]))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data')
# plt.show()

################# Definations #################

# Define Model
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    poly=[]
    poly.clear()  
    denom = (x1-x2)*(x1-x3)*(x2-x3)
    poly.append((x3 * (y2-y1) +
                 x2 * (y1-y3) +
                 x1 * (y3-y2)) / denom)
    poly.append((x3*x3 * (y1-y2) +
                 x2*x2 * (y3-y1) +
                 x1*x1 * (y2-y3)) / denom)
    poly.append((x2 * x3 * (x2-x3) *
                 y1+x3 * x1 * (x3-x1) *
                 y2+x1 * x2 *(x1-x2) *
                 y3) / denom)
    return poly

# Define Error
def error(j, data, model):
  df = pd.DataFrame(data)
  x=df.iloc[j,0]
  y=df.iloc[j,1]
  poly=calc_parabola_vertex(
      model[0],model[1],model[2],
      model[3],model[4],model[5])
  error = abs(y - (poly[0]*x**2+poly[1]*x+poly[2]))
  return error

# Define RANSAC
def my_ransac(data, max_iterations,
              min_inliers, threshold):
  prev_inliers=0
  model=[]
  ran= []
  for i in range(max_iterations):
    inliers=0
    model.clear()
    ran.clear()
    d = pd.DataFrame(data)
    for k in range(3):
      r=randint(0, 249)
      x=d.iloc[r,0]
      y=d.iloc[r,1]
      model.append(x)
      model.append(y)
      ran.append(r)
    for j in range(len(data)):
      if error(j, data, model) < threshold:
        inliers=inliers+1
    if inliers>min_inliers and inliers>prev_inliers:
      prev_inliers=inliers
      final_model=model.copy()
      print(prev_inliers,'Inliers in')
      print('Model No',i,final_model)
  print('Final Model',final_model,
        'has',prev_inliers,'inliers')
  return final_model


# Define Least Square
def leastsquare(mod,data,thershold):
  x=[]
  y=[]
  k = pd.DataFrame(data)
  for j in range(len(data)):
    if error(j, data, mod) < threshold:
      x.append(k.iloc[j,0])
      y.append(k.iloc[j,1])
  x = np.array(x)
  y = np.array(y)
  x_mean = np.mean(x)
  X = np.array([x**2 ,x ,np.ones(len(x))]).T
  Y = np.array([y]).T
  B = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
  print(B)
  return B[0],B[1],B[2]

#############################################

# Parameters
max_iterations= 7000
min_inliers= 200
threshold= 25

# Run
A,B,C=leastsquare(my_ransac(data, max_iterations,
              min_inliers, threshold),data,threshold)
xval = np.arange(0, 500, 1)
yval=A*xval**2+B*xval+C
plt.plot(xval, yval,'red')
plt.scatter(list(data["x"]),list(data["y"]))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
plt.show()
