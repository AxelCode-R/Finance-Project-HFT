import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
import time




start = time.time()


eta=0.03
n_iterations=40000 

X = np.array([
  [1.0,1.0,1.0],
  [1.0,0.0,1.0],
  [1.0,1.0,0.0],
  [1.0,0.0,0.0]]) 

Y = np.array([
  [1.0,0.0], 
  [1.0,1.0], 
  [1.0,1.0], 
  [1.0,0.0]]) 

W_01 = np.array([
  [-0.251, 0.901, 0.464], 
  [0.197, -0.688, -0.688], 
  [-0.884, 0.732, 0.202]])

W_12 = np.array([
  [0.416, -0.959, 0.940], 
  [0.665, -0.575, -0.636]])


def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s))



for i in range(n_iterations):
  for k in range(len(X)):
    # Forward
    IN_01 = X[k]
    OUT_01 = sigmoid(np.dot(W_01, IN_01))   
    IN_12 = OUT_01
    OUT_12 = sigmoid(np.dot(W_12, IN_12))
    error =  Y[k] - OUT_12 
    print(np.sum(0.5 * (error) ** 2))
    # Backward
    grad_12 = OUT_12 * (1-OUT_12) * error
    grad_01 = OUT_01 * (1-OUT_01) * np.dot(grad_12.T, W_12)
    
    W_01 = W_01 + eta * np.outer(grad_01.T, IN_01)
    W_12 = W_12 + eta * np.outer(grad_12.T, IN_12)
  

  
end = time.time()
print(end - start)

np.sum(0.5 * (Y.T -sigmoid((W_12 @ sigmoid((W_01 @ X.T))))) ** 2)

# 0.00327170262382105
# 12.906689405441284









start = time.time()


eta=0.03
n_iterations=40000 

X = np.array([
  [1.0,1.0,1.0],
  [1.0,0.0,1.0],
  [1.0,1.0,0.0],
  [1.0,0.0,0.0]]) 

Y = np.array([
  [1.0,0.0], 
  [1.0,1.0], 
  [1.0,1.0], 
  [1.0,0.0]]) 

W_01 = np.array([
  [-0.251, 0.901, 0.464], 
  [0.197, -0.688, -0.688], 
  [-0.884, 0.732, 0.202]])

W_12 = np.array([
  [0.416, -0.959, 0.940], 
  [0.665, -0.575, -0.636]])


def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s))



for i in range(n_iterations):
  IN_01 = X.T
  OUT_01 = sigmoid((W_01 @ IN_01))
  IN_12 = OUT_01
  OUT_12 = sigmoid((W_12 @ IN_12))
  error =  Y.T - OUT_12
  print(np.sum(0.5 * (error) ** 2))
  # Backward
  grad_12 = OUT_12 * (1-OUT_12) * error
  grad_01 = OUT_01 * (1-OUT_01) * (W_12.T @ grad_12)
  
  W_01 = W_01 + eta * (grad_01 @ IN_01.T)
  W_12 = W_12 + eta * (grad_12 @ IN_12.T)



end = time.time()
print(end - start)

np.sum(0.5 * (Y.T -sigmoid((W_12 @ sigmoid((W_01 @ X.T))))) ** 2)

# 0.0005636044770111592
# 12.360254526138306



