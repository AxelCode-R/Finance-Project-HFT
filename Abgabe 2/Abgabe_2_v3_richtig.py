import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
import time


############################################################################
# HELP Functions:
def print_dict(dict):
  for key, value in dict.items():
    print(key, ' : \n', value)
    
############################################################################
# KNN Functions:
def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s))


def forward(network):
  network["IN_01"] = network["X"].T
  network["OUT_01"] = sigmoid((network["W_01"] @ network["IN_01"]))
  network["IN_12"] = network["OUT_01"]
  network["OUT_12"] = sigmoid((network["W_12"] @ network["IN_12"]))
  network["error"] =  network["Y"].T - network["OUT_12"]
  return(network)


def backward(network, eta):
  network["grad_12"] = network["OUT_12"] * (1-network["OUT_12"]) * network["error"]
  network["grad_01"] = network["OUT_01"] * (1-network["OUT_01"]) * (network["W_12"].T @ network["grad_12"])
  
  network["new_W_01"] = network["W_01"] + eta * (network["grad_01"] @ network["IN_01"].T)
  network["new_W_12"] = network["W_12"] + eta * (network["grad_12"] @ network["IN_12"].T)
  return(network)


def fit_all(X, Y, W_01, W_12, eta = 0.03, n_iterations = 5000, print_network = False, print_error=False):
  # Init Values
  start_timer = time.time()
  losses = []
  network = {"X":[] ,"IN_01":[], "W_01":[], "OUT_01":[], "IN_12":[], "W_12":[], "Y":[], "error":[], "loss":[], "grad_01":[], "grad_12":[], "new_W_01":[], "new_W_12":[]}
  network.update({
  "new_W_01":W_01, 
  "new_W_12":W_12})
  
  for i in range(n_iterations):
    network.update({"X":X, "Y":Y, "W_01":network["new_W_01"], "W_12":network["new_W_12"]})
    network = forward(network)
     
    losses.append(np.sum(0.5 * (network["error"]) ** 2))
    if print_error:
      print(losses[-1])
     
    network = backward(network, eta)
     
    if print_network:
      print_dict(network)
     
  time_used = time.time() - start_timer
  return network, losses, time_used


def fit_one(X, Y, W_01, W_12, eta = 0.03, n_iterations = 5000, print_network = False, print_error=False):
  # Init Values
  start_timer = time.time()
  losses = []
  network = {"X":[] ,"IN_01":[], "W_01":[], "OUT_01":[], "IN_12":[], "W_12":[], "Y":[], "error":[], "loss":[], "grad_01":[], "grad_12":[], "new_W_01":[], "new_W_12":[]}
  network.update({
  "new_W_01":W_01, 
  "new_W_12":W_12})
  
  for i in range(n_iterations):
    temp_errors = []
    for k in range(len(X)):
      
      network.update({"X":X[[k]], "Y":Y[[k]], "W_01":network["new_W_01"], "W_12":network["new_W_12"]})
      network = forward(network)
     
      temp_errors.append( 0.5 * (network["error"]) ** 2 )
      
      network = backward(network, eta)
      
      if print_network:
        print_dict(network)
        
    losses.append(np.sum(temp_errors))
    if print_error:
      print(losses[-1])
    
  time_used = time.time() - start_timer
  return network, losses, time_used


############################################################################
# KNN DATA:
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
  
  
############################################################################
# CALL KNN_all (iterate over all rows in trainingdata at the same time)
network_all, losses_all, time_all = fit_all(
  X = X, Y = Y, W_01 = W_01, W_12 = W_12, eta = 0.03, n_iterations = 40000, print_network = False, print_error=False)    
  
print("Time network_all: ", time_all)
print("Error network_all: ", losses_all[-1])

############################################################################
# CALL KNN_one (iterate over each row in trainingdata
network_one, losses_one, time_one = fit_one(
  X = X, Y = Y, W_01 = W_01, W_12 = W_12, eta = 0.03, n_iterations = 40000, print_network = False, print_error=False)   

print("Time network_one: ", time_one)
print("Error network_one: ", losses_one[-1])



############################################################################
# CHARTS

def plot_error(errors, title):
  x = list(range(len(errors)))
  y = np.array(errors)
  pyplot.figure(figsize=(6,6))
  pyplot.plot(x, y, "g", linewidth=1)
  pyplot.xlabel("Iterations", fontsize = 16)
  pyplot.ylabel("Mean Square Error (all)", fontsize = 16)
  pyplot.title(title)
  pyplot.ylim(0,1)
  pyplot.show()



plot_error(losses_all, "fit_all\nlast error: "+str(round(losses_all[-1],6))+"\ntime: "+str(round(time_all,2)))

plot_error(losses_one, "fit_one\nlast error: "+str(round(losses_one[-1],6))+"\ntime: "+str(round(time_one,2)))



