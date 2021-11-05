#######https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

# https://www.kaggle.com/laotse/credit-risk-dataset
import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
import time
from sklearn import preprocessing




############################################################################
# HELP Functions:
def print_dict(dict):
  for key, value in dict.items():
    print(key, ' : \n', value)
    
############################################################################
# KNN Functions:
def sigmoid(s):
  return 1.0 / (1.0 + np.exp(-s))


def forward(network, print_details = False):
  network["IN_01"] = network["X"].T
  network["OUT_01"] = sigmoid((network["W_01"] @ network["IN_01"]))
  network["IN_12"] = network["OUT_01"]
  network["OUT_12"] = sigmoid((network["W_12"] @ network["IN_12"]))
  network["error"] =  network["Y"].T - network["OUT_12"]
  
  if print_details:
    for i in range(len(network["X"])):
      print(network["X"][i],  network["Y"][i][1], " -> ", network["OUT_12"].T[i][1])
  return(network)


def backward(network, eta):
  network["grad_12"] = network["OUT_12"] * (1-network["OUT_12"]) * network["error"]
  network["grad_01"] = network["OUT_01"] * (1-network["OUT_01"]) * (network["W_12"].T @ network["grad_12"])
  
  network["new_W_01"] = network["W_01"] + eta * (network["grad_01"] @ network["IN_01"].T)
  network["new_W_12"] = network["W_12"] + eta * (network["grad_12"] @ network["IN_12"].T)
  return(network)


def fit_all(X, Y, hidden_neurons, eta = 0.03, n_iterations = 5000, print_network = False, print_error=False, print_details=False):
  # Init Values
  start_timer = time.time()
  
  INPUT_N = X.shape[1]
  WGT_01_N = (hidden_neurons, INPUT_N)
  WGT_12_N = (Y.shape[1], hidden_neurons)
  OUTPUT_N = Y.shape[2]
  
  W_01 = np.zeros((hidden_neurons, INPUT_N)) + 0.5
  WGT_12

  losses = []
  network = {"X":[] ,"IN_01":[], "W_01":[], "OUT_01":[], "IN_12":[], "W_12":[], "OUT_12":[], "Y":[], "error":[], "loss":[], "grad_01":[], "grad_12":[], "new_W_01":[], "new_W_12":[]}
  network.update({
  "new_W_01":W_01, 
  "new_W_12":W_12})
  
  for i in range(n_iterations):
    network.update({"X":X, "Y":Y, "W_01":network["new_W_01"], "W_12":network["new_W_12"]})
    network = forward(network, print_details=print_details)
     
    losses.append(np.sum(0.5 * (network["error"]) ** 2))
    if print_error:
      print(losses[-1])
     
    network = backward(network, eta)
     
    if print_network:
      print_dict(network)
     
  time_used = time.time() - start_timer
  return network, losses, time_used



############################################################################
# KNN DATA:
  
data = pd.read_csv("Abgabe 3/credit_risk_dataset.csv")
data = data.replace({"Y": 1, "N":0})

scale_mapper = {'OWN':1, 'RENT':2, 'MORTGAGE':3, 'OTHER':4}
data["person_home_ownership"] = data["person_home_ownership"].replace(scale_mapper)
scale_mapper = {'PERSONAL':1, 'EDUCATION':2, 'MEDICAL':3, 'VENTURE':4, 'HOMEIMPROVEMENT':5,'DEBTCONSOLIDATION':6}
data["loan_intent"] = data["loan_intent"].replace(scale_mapper)
scale_mapper = {'D':4, 'B':2, 'C':3, 'A':1, 'E':5, 'F':6, 'G':7}
data["loan_grade"] = data["loan_grade"].replace(scale_mapper)


X_train = data.loc[0:599, data.columns != 'loan_status'].to_numpy()
Y_train = np.append(np.ones((600,1)), data.loc[0:599, data.columns == 'loan_status'].to_numpy(), axis=1)

X_test = data.loc[600:, data.columns != 'loan_status'].to_numpy()
Y_test = np.append(np.ones((data.shape[0]-600,1)), data.loc[600:, data.columns == 'loan_status'].to_numpy(), axis=1)
  
############################################################################
# CALL KNN_all (iterate over all rows in trainingdata at the same time)
network_all, losses_all, time_all = fit_all(
  X = X_train, Y = Y_train, 
  hidden_neurons = 3, 
  eta = 0.03, n_iterations = 2000, print_network = False, print_error=False, print_details=False)    
  
print("All that network_all info....:\n")
print_dict(network_all)
print("Time network_all: ", time_all)
print("Error network_all: ", losses_all[-1])



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
