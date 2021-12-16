
# etra example:
# https://github.com/llSourcell/how_to_do_math_for_deep_learning/blob/master/demo.py

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)
import math as ma


# Load Data
data = pd.read_csv("Abgabe 3/Kreditausfalldaten.csv").fillna(0)

# Categories to Numericals
data["Risk"] = data["Risk"].replace({'good':0, 'bad':1})

data["Sex"] = data["Sex"].replace({'male':1, 'female':2})

data["Purpose"] = data["Purpose"].replace({'radio/TV':1, 'education':6, 'furniture/equipment':4, 'car':5, 'business':8,
       'domestic appliances':7, 'repairs':3, 'vacation/others':2})

data["Housing"] = data["Housing"].replace({'own':3, 'free':1, 'rent':2})

data["Saving accounts"] = data["Saving accounts"].replace({'little':1, 'quite rich':3, 'rich':4, 'moderate':2})

data["Checking account"] = data["Checking account"].replace({'little':1, 'moderate':2,'rich':3})

# Remove Useless Columns
data = data.iloc[:,data.columns != "Unnamed: 0"]

# shuffle
#data = data.sample(frac=1, ).reset_index(drop=True)

# First 250 Defaults at the beginning
first_defaults = data.loc[data["Risk"]==0,:].index.values[:250]
data = pd.concat([data.loc[first_defaults],data.drop(first_defaults)], ignore_index=True)

# def NormalizeData(np_arr):
#   for i in range(np_arr.shape[1]):
#     np_arr[:,i] = (np_arr[:,i] - np.min(np_arr[:,i])) / (np.max(np_arr[:,i]) - np.min(np_arr[:,i]))
#   return(np_arr)

def NormalizeData(df):
  for i in range(len(df.columns)):
    df[df.columns[i]] = (df[df.columns[i]] - np.min(df[df.columns[i]])) / (np.max(df[df.columns[i]]) - np.min(df[df.columns[i]]))
  return(df)


data = NormalizeData(data)
training_n = 600
X_train = data.loc[0:(training_n-1), data.columns != 'Risk'].to_numpy(dtype=np.float64)
Y_train = data.loc[0:(training_n-1), data.columns == 'Risk'].to_numpy(dtype=np.float64)

X_test = data.loc[training_n:, data.columns != 'Risk'].to_numpy(dtype=np.float64)
Y_test = data.loc[training_n:, data.columns == 'Risk'].to_numpy(dtype=np.float64)


# X_train = NormalizeData(data.loc[0:(training_n-1), data.columns != 'Risk']).to_numpy(dtype=np.float64)
# Y_train = data.loc[0:(training_n-1), data.columns == 'Risk'].to_numpy(dtype=np.float64)
# 
# X_test = NormalizeData(data.loc[training_n:, data.columns != 'Risk']).to_numpy(dtype=np.float64)
# Y_test = data.loc[training_n:, data.columns == 'Risk'].to_numpy(dtype=np.float64)

# analyze
# data.loc[0:(training_n-1)].hist(bins=10,figsize=(8,8), range=(0,1))
# pyplot.tight_layout()
# pyplot.show()
# data.loc[training_n:].hist(bins=10,figsize=(8,8), range=(0,1))
# pyplot.tight_layout()
# pyplot.show()

def generate_weights(n_input, n_output, hidden_layer_neurons, default_W=None):
  W = []
  for i in range(len(hidden_layer_neurons)+1):
    if i == 0: # first layer
      if default_W == None:
        W.append(np.random.random((n_input+1, hidden_layer_neurons[i])))
      else:
        W.append(np.zeros((n_input+1, hidden_layer_neurons[i]))+default_W)
    elif i == len(hidden_layer_neurons): # last layer
      if default_W == None:
        W.append(np.random.random((hidden_layer_neurons[i-1]+1, n_output)))
      else:
        W.append(np.zeros((hidden_layer_neurons[i-1]+1, n_output))+default_W)
    else: # middle layers
      if default_W == None:
        W.append(np.random.random((hidden_layer_neurons[i-1]+1, hidden_layer_neurons[i])))
      else:
        W.append(np.zeros((hidden_layer_neurons[i-1]+1, hidden_layer_neurons[i]))+default_W)
  return(W)

def add_ones_to_input(x):
  return(np.append(x, np.array([np.ones(len(x))]).T, axis=1))



def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
  return x * (1 - x)


def forward(x, w):
  return( sigmoid(x @ w) )

def backward(IN, OUT, W, Y, grad, k):
  if k == len(grad)-1:
    grad[k] = deriv_sigmoid(OUT[k]) * (Y-OUT[k])
  else:
    grad[k] = deriv_sigmoid(OUT[k]) *(grad[k+1] @ W[k+1][0:len(W[k+1])-1].T)  ## hier @
  return(grad)

def generate_batches(batch_size, full_batch_size, random_batch_order=True):
  batches = np.arange(full_batch_size)
  if random_batch_order :
    np.random.shuffle(batches)
  return(np.array_split(batches, ma.ceil(full_batch_size/batch_size)))

def mean_square_error(error):
  return( 0.5 * np.sum(error ** 2) )

def train(X, Y, hidden_layer_neurons, alpha, epochs, alpha_decrese_to = None, default_W = None, batch_size = 1, random_batch_order = True):
  alpha_ = alpha
  n_input = len(X[0])
  n_output = len(Y[0])
  W = generate_weights(n_input, n_output, hidden_layer_neurons, default_W)
  ms_errors = []
  batches = generate_batches(batch_size, full_batch_size = len(X), random_batch_order = random_batch_order)
  for i in range(epochs):
    if alpha_decrese_to is not None:
      alpha_ = (alpha*(epochs-i)+alpha_decrese_to*i)/epochs
    error_temp = np.array([])
    for z in range(len(batches)):
      IN = []
      OUT = []
      grad = [None]*len(W)
      for k in range(len(W)):
        if k==0:
          IN.append(add_ones_to_input(X[batches[z],:]))
        else:
          IN.append(add_ones_to_input(OUT[k-1]))
        OUT.append(forward(x=IN[k], w=W[k]))
       
      error_temp = np.append(error_temp, Y[batches[z],:] - OUT[-1])
       
       
      for k in range(len(W)-1,-1, -1):
        grad = backward(IN, OUT, W, Y[batches[z],:], grad, k) 
       
      #print(grad)
      for k in range(len(W)):
        W[k] = W[k] + alpha_ * (IN[k].T @ grad[k])
    ms_errors.append(mean_square_error(error_temp))
    
  return W, ms_errors


# normalize all data and replace not with 0 start with 1
#np.random.seed(0)

#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [6,3], alpha = 0.1, alpha_decrese_to = 0.01, epochs = 5000, default_W = None, batch_size = 1, random_batch_order = True)
# W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [6,3], alpha = 0.01, epochs = 10000, default_W = None, batch_size = 1) # 19% und 28% wrong
# W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [6,2], alpha = 0.1, epochs = 5000, default_W = None, batch_size = 50)  # 15% und 31% sieht aber gut aus
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True) # 4.5% und 32% gut!
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.01, default_W = None, batch_size = 5, random_batch_order = True)
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [8,5], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.01, default_W = None, batch_size = 10, random_batch_order = True)

#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 30000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True) # 3.17 und 30.5

#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,9,6], alpha = 0.1,  epochs = 30000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True) # 2.67 30.5
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,9,6], alpha = 0.1,  epochs = 30000, alpha_decrese_to = 0.01, default_W = None, batch_size = 100, random_batch_order = True) # 1.5 36

#gut
W_train, ms_errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True)

#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.01,  epochs = 2000, alpha_decrese_to = 0.01, default_W = None, batch_size = 10, random_batch_order = False)
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 2000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = False) # good
#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.01,  epochs = 15000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = False)

#W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = [2], alpha = 0.01,  epochs = 2000, alpha_decrese_to = 0.01, default_W = 0.5, batch_size = 1, random_batch_order = False)


#ms_errors_train = np.array(list(map(mean_square_error, errors_train)))

def plot_error(errors, title):
  x = list(range(len(errors)))
  y = np.array(errors)
  pyplot.figure(figsize=(6,6))
  pyplot.plot(x, y, "g", linewidth=1)
  pyplot.xlabel("Iterations", fontsize = 16)
  pyplot.ylabel("Mean Square Error", fontsize = 16)
  pyplot.title(title)
  pyplot.ylim(0,max(errors)*1.1)
  pyplot.show()
  
plot_error(errors=ms_errors_train, title="MLP Credit Default")



def test(X, W):
  for i in range(len(W)):
    X = forward(add_ones_to_input(X), W[i])
  return(X)
  
  
def classify(Y_approx):
  return( np.round(Y_approx,0) )



# Test on Trainings-Data
print("")
print("Analyze Trainings-Data:")
result_train = test(X = X_train, W = W_train)
print("Mean Square error over all testdata: ", mean_square_error(Y_train - result_train))

classified_error_train = Y_train - classify(result_train)
print("Mean Square error over all classified testdata: ", mean_square_error(classified_error_train))

print("Probability of a wrong output: ", np.round(np.sum(np.abs(classified_error_train)) / len(classified_error_train) * 100, 2), "%" )
print("Probability of a right output: ", np.round((1 - np.sum(np.abs(classified_error_train)) / len(classified_error_train))*100,2),"%" )


confusion_matrix(Y_train, classify(result_train))


# Test on Test-Data
print("")
print("Analyze Test-Data:")
result_test = test(X = X_test, W = W_train)
print("Mean Square error over all testdata: ", mean_square_error(Y_test - result_test))


classified_error_test = Y_test - classify(result_test)
print("Mean Square error over all classified testdata: ", mean_square_error(classified_error_test))

print("Probability of a wrong output: ", np.round(np.sum(np.abs(classified_error_test)) / len(classified_error_test) * 100, 2), "%" )
print("Probability of a right output: ", np.round((1 - np.sum(np.abs(classified_error_test)) / len(classified_error_test))*100,2),"%" )


confusion_matrix(Y_test, classify(result_test))







import gc
gc.collect()

# Analyse Trainingphase
def train_analyse(X, Y, hidden_layer_neurons, alpha, epochs, alpha_decrese_to = None, default_W = None, batch_size = 1, random_batch_order = True, X_test = None, Y_test = None):
  alpha_ = alpha
  n_input = len(X[0])
  n_output = len(Y[0])
  W = generate_weights(n_input, n_output, hidden_layer_neurons, default_W)
  ms_errors_train = [None]*(epochs+1)
  ms_errors_test = [None]*(epochs+1)
  ms_errors_train[0] = mean_square_error(Y_train - test(X = X_train, W = W))
  ms_errors_test[0] = mean_square_error(Y_test - test(X = X_test, W = W))
  batches = generate_batches(batch_size, full_batch_size = len(X), random_batch_order = random_batch_order)
  for i in range(epochs):
    if alpha_decrese_to is not None:
      alpha_ = (alpha*(epochs-i)+alpha_decrese_to*i)/epochs
    error_temp = np.array([])
    for z in range(len(batches)):
      IN = []
      OUT = []
      grad = [None]*len(W)
      for k in range(len(W)):
        if k==0:
          IN.append(add_ones_to_input(X[batches[z],:]))
        else:
          IN.append(add_ones_to_input(OUT[k-1]))
        OUT.append(forward(x=IN[k], w=W[k]))
      error_temp = np.append(error_temp, Y[batches[z],:] - OUT[-1])
      for k in range(len(W)-1,-1, -1):
        grad = backward(IN, OUT, W, Y[batches[z],:], grad, k) 
      for k in range(len(W)):
        W[k] = W[k] + alpha_ * (IN[k].T @ grad[k])
    ms_errors_train[i+1] = mean_square_error(error_temp)
    ms_errors_test[i+1] = mean_square_error(Y_test - test(X = X_test, W = W))
  return W, ms_errors_train, ms_errors_test


#W_train, ms_errors_train, ms_errors_test = train_analyse(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.0001, default_W = None, batch_size = 50, random_batch_order = True, X_test=X_test, Y_test=Y_test)# gut
#W_train, ms_errors_train, ms_errors_test = train_analyse(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 10000, alpha_decrese_to = 0.0001, default_W = 0.5, batch_size = 50, random_batch_order = True, X_test=X_test, Y_test=Y_test) #4% 30%
W_train, ms_errors_train, ms_errors_test = train_analyse(X = X_train, Y = Y_train, hidden_layer_neurons = [2], alpha = 0.01,  epochs = 2000, alpha_decrese_to = 0.01, default_W = 0.5, batch_size = 50, random_batch_order = False, X_test=X_test, Y_test=Y_test)
#W_train, ms_errors_train, ms_errors_test = train_analyse(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 2000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True, X_test=X_test, Y_test=Y_test)

#W_train, ms_errors_train, ms_errors_test = train_analyse(X = X_train, Y = Y_train, hidden_layer_neurons = [9,6], alpha = 0.1,  epochs = 2000, alpha_decrese_to = 0.01, default_W = None, batch_size = 50, random_batch_order = True, X_test=X_test, Y_test=Y_test)

def plot_error_train_and_test(ms_errors_train, ms_errors_test, title):
  x = list(range(len(ms_errors_train)))
  y = np.array([np.array(ms_errors_train), np.array(ms_errors_test)]).T
  pyplot.figure(figsize=(6,6))
  pyplot.plot(x, y[:,0], "g", linewidth=1, label="train")
  pyplot.plot(x, y[:,1], "r", linewidth=1, label="test")
  pyplot.xlabel("Iterations", fontsize = 16)
  pyplot.ylabel("Mean Square Error", fontsize = 16)
  pyplot.title(title)
  pyplot.ylim(0,np.max(y)*1.1)
  #pyplot.xlim(-10,len(ms_errors_train)+10)
  pyplot.legend()
  pyplot.show()
  
plot_error_train_and_test(ms_errors_train, ms_errors_test, title="MLP Credit Default")

pyplot.close()
gc.collect()


# Test on Trainings-Data
print("")
print("Analyze Trainings-Data:")
result_train = test(X = X_train, W = W_train)
print("Mean Square error over all testdata: ", mean_square_error(Y_train - result_train))

classified_error_train = Y_train - classify(result_train)
print("Mean Square error over all classified testdata: ", mean_square_error(classified_error_train))

print("Probability of a wrong output: ", np.round(np.sum(np.abs(classified_error_train)) / len(classified_error_train) * 100, 2), "%" )
print("Probability of a right output: ", np.round((1 - np.sum(np.abs(classified_error_train)) / len(classified_error_train))*100,2),"%" )


confusion_matrix(Y_train, classify(result_train))


# Test on Test-Data
print("")
print("Analyze Test-Data:")
result_test = test(X = X_test, W = W_train)
print("Mean Square error over all testdata: ", mean_square_error(Y_test - result_test))


classified_error_test = Y_test - classify(result_test)
print("Mean Square error over all classified testdata: ", mean_square_error(classified_error_test))

print("Probability of a wrong output: ", np.round(np.sum(np.abs(classified_error_test)) / len(classified_error_test) * 100, 2), "%" )
print("Probability of a right output: ", np.round((1 - np.sum(np.abs(classified_error_test)) / len(classified_error_test))*100,2),"%" )


confusion_matrix(Y_test, classify(result_test))











#45 161
# Optimize Hyper-Parameters
# import itertools
# valid_hyper = [[3,4,5,6,7,8,9],[1,2,3,4,5,6],[1,0.5,0.1],[1,5,10,50,100,200]]
# combo_hyper = list(itertools.product(*valid_hyper))
# 
# res = pd.DataFrame(columns=['i','classified_error_train', 'classified_wgt_wrong_train', 'classified_error_test', 'classified_wgt_wrong_test']) #(8, 5)
# for i in range(len(combo_hyper)):
#   print(i)
#   np.random.seed(0)
#   W_train, errors_train = train(X = X_train, Y = Y_train, hidden_layer_neurons = list(combo_hyper[i][0:2]), alpha = combo_hyper[i][2],  epochs = 2000, alpha_decrese_to = 0.01, default_W = None, batch_size = combo_hyper[i][3], random_batch_order = True)
#   temp_train = Y_train - classify(test(X = X_train, W = W_train))
#   classified_error_train = mean_square_error(temp_train)
#   classified_wgt_wrong_train = np.round(np.sum(np.abs(temp_train)) / len(temp_train),6)
#   temp_test = Y_test - classify(test(X = X_test, W = W_train))
#   classified_error_test = mean_square_error(temp_test)
#   classified_wgt_wrong_test = np.round(np.sum(np.abs(temp_test)) / len(temp_test),6)
#   res.append({'i':i,'classified_error_train':classified_error_train, 'classified_wgt_wrong_train':classified_wgt_wrong_train, 'classified_error_test':classified_error_test, 'classified_wgt_wrong_test':classified_wgt_wrong_test}, ignore_index=True)
# 






# test ob 0.5 gewichte sinn machen
import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = np.zeros((3,4))+0.5
syn1 = np.zeros((4,1))+0.5
# syn0 = 2*np.random.random((3,4)) - 1
# syn1 = 2*np.random.random((4,1)) - 1

for j in range(100000):
      
    k0 = X
    k1 = nonlin(np.dot(k0,syn0))
    k2 = nonlin(np.dot(k1,syn1))
      
    k2_error = y - k2
      
    if (j% 10000) == 0: 
      print("Error:" + str(np.mean(np.abs(k2_error))))
      
    k2_delta = k2_error*nonlin(k2,deriv=True)
      
    k1_error = k2_delta.dot(syn1.T)
      
    k1_delta = k1_error * nonlin(k1,deriv=True)
      
    syn1 += k1.T.dot(k2_delta)
    syn0 += k0.T.dot(k1_delta)



syn1

syn0
