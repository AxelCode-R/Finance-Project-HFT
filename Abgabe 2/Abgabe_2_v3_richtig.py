import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)



losses = []

eta = 0.03
n_iterations = 40000 

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
  
  losses.append(np.sum(0.5 * (error) ** 2))
  print(losses[-1])
  # Backward
  grad_12 = OUT_12 * (1-OUT_12) * error
  grad_01 = OUT_01 * (1-OUT_01) * (W_12.T @ grad_12)
  
  W_01 = W_01 + eta * (grad_01 @ IN_01.T)
  W_12 = W_12 + eta * (grad_12 @ IN_12.T)



print("Trainintes KNN hat einen Gesamtfehler von: \n", np.sum(0.5 * (Y.T - sigmoid((W_12 @ sigmoid((W_01 @ X.T))))) ** 2))


# Plot losses
x = list(range(len(losses)))
y = np.array(losses)
pyplot.figure(figsize=(6,4))
pyplot.plot(x, y, "g", linewidth=2)
pyplot.xlabel("x", fontsize = 16)
pyplot.ylim(0,1)
pyplot.show()


