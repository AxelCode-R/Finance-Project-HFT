import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)

n_input_neurons = 2
n_hidden_neurons = 2
n_output_neurons = 1

W_IH = np.array([[0.0,0.0,0.0],[-10.0,20.0,20.0],[30.0,-20.0,-20.0]])
W_HO = np.array([[0.0,0.0,0.0],[-30.0,20.0,20.0]])


data = np.array([[1.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,0.0]]) 

def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s.copy()))


def predict(data, W_IH, W_HO):
  network = pd.DataFrame(columns=["I", "W_IH", "H", "W_HO", "O", "y"])
  
  for i in range(len(data)):
    I = data[i]
    net = np.dot( W_IH, I )
    a = sigmoid(net)
    o = a
    H = np.array([net, a, o]).transpose().round(4)
    H[0] = np.ones(len(H[0]))
    
    net = np.dot( W_HO, H[:,len(H)-1] )
    a = sigmoid(net)
    o = a
    O = np.array([net, a, o]).transpose().round(4)
    O[0] = np.ones(len(O[0]))
    print("I: \n",I)
    print("W_IH: \n",W_IH)
    print("H: \n",H)
    print("W_HO: \n", W_HO)
    print("O: \n", O)
    print("Ausgabe des KNN: ",O[1,2])
    print("-----------------------------")
    network = network.append({"I":I, "W_IH":W_IH, "H":H, "W_HO":W_HO, "O":O, "y":O[1,2]}, 
                             ignore_index=True)
  return(network)
    
network = predict(data, W_IH, W_HO) 
    






# Richtiger Start
#https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7
#https://data-science-blog.com/blog/2019/01/29/fehler-ruckfuhrung-mit-der-backpropagation/
W_IH = np.array([[-0.251, 0.901, 0.464], [0.197, -0.688, -0.688], [-0.884, 0.732, 0.202]])
W_HO = np.array([[0.416, -0.959, 0.940], [0.665, -0.575, -0.636]])

Y = np.array([[1.0,0.0], [1.0,1.0], [1.0,1.0], [1.0,0.0]]) 
n_input_neurons=2
n_hidden_neurons=2
n_output_neurons=1
eta=0.03
n_iterations=40000 
data = np.array([[1.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,0.0]]) 



def predict(data, W_IH, W_HO):
  network = pd.DataFrame(columns=["I", "W_IH", "H", "W_HO", "O", "y", "d_f", "delta"])
  
  for i in range(len(data)):
    I = data[i]
    net1 = np.dot( W_IH, I )
    a1 = sigmoid(net1)
    o1 = a1
    H = np.array([net1, a1, o1]).transpose().round(4)
    H[0] = np.ones(len(H[0]))
    
    
    net2 = np.dot( W_HO, H[:,len(H)-1] )
    a2 = sigmoid(net2)
    o2 = a2
    O = np.array([net2, a2, o2]).transpose().round(4)
    O[0] = np.ones(len(O[0]))
    
    delta2 = o2 * (1-o2) * (Y[i]-o2)
    W_HO = W_HO + eta * np.dot(delta2,  o2.transpose())
    
    delta1 = o1 * (1-o1) * np.dot(delta2, W_IH)
    
    
    print("I: \n",I)
    print("W_IH: \n",W_IH)
    print("H: \n",H)
    print("W_HO: \n", W_HO)
    print("O: \n", O)
    print("Ausgabe des KNN: ",O[1,2])
    print("-----------------------------")
    network = network.append({"I":I, "W_IH":W_IH, "H":H, "W_HO":W_HO, "O":O, "y":O[1,2]}, 
                             ignore_index=True)
  return(network)







W = list()
W.append(np.array([[0.197, -0.688, -0.688], [-0.884, 0.732, 0.202]]))
W.append(np.array([[0.665, -0.575, -0.636]]))

dataX = np.array([[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]]) 
dataY = np.array([[0.0], [1.0], [1.0], [0.0]]) 

eta = 0.03

def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s.copy()))

def addB(v):
  return(np.append(np.array([1.0]), v))

save_OUT = list()
save_W = list()
save_ERROR = list()
for j in range(len(dataX)):
  # forward
  IN = [None]*len(W)
  OUT = [None]*len(W)
  for i in range(len(W)):
    IN[i] = addB(OUT[i-1]) if i!=0 else addB(dataX[j])
    OUT[i] = sigmoid(np.dot( W[i], IN[i] ))
  
  # backward
  delta = [None]*len(W)
  for i in range(-len(W)+1,1):
    delta[i] = OUT[i] * (1-OUT[i]) * np.dot(delta[i-1], W[i-1][0][1:3]) if i != (-len(W)+1) else OUT[i] * (1-OUT[i]) * (dataY[j] - OUT[i])
    
  deltaW = [None]*len(W)
  for i in range(len(W)):
    deltaW[i] = eta * np.outer(delta[i], addB(OUT[i]))











for j in range(len(dataX)):
  # forward
  IN = [None]*len(W)
  OUT = [None]*len(W)
  for i in range(len(W)):
    IN[i] = np.append(np.array([1.0]), OUT[i-1]) if i!=0 else dataX[j] 
    OUT[i] = sigmoid(np.dot( W[i], IN[i] ))[1:] if i!=len(W)-1 else sigmoid(np.dot( W[i], IN[i] ))
  
  # backward
  delta = [None]*len(W)
  for i in range(-len(W)+1,1):
    delta[i] = OUT[i] * (1-OUT[i]) * np.dot(W[i+1][:,1:len(W[i+1][1])], delta[i+1]) if i != (-len(W)+1) else OUT[i] * (1-OUT[i]) * (dataY[j] - OUT[i])
    
  deltaW = [None]*len(W)
  for i in range(len(W)):
    deltaW[i] = eta * np.outer(delta[i], np.append(np.array([1.0]), OUT[i]))













# komplett neuer start

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
  




