import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True)



train_data = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])
        
target = np.array(
    [[0],
     [1],
     [1],
     [0]])

lr = 0.2
num_epochs = 5000
num_input = 2
num_hidden = 2
num_output = 1

weights_01 = [np.random.uniform(size=(num_input, num_hidden))]
weights_12 = [np.random.uniform(size=(num_hidden, num_output))]

b01 = [np.random.uniform(size=(1,num_hidden))]
b12 = [np.random.uniform(size=(1,num_output))]

losses = []
hidden = []
hidden_out = []
output = []
output_final = []
grad01 = []
grad12 = []
error_term = []
# df_hyper = pd.DataFrame(columns = ["train_data", "target", "lr", "num_epochs", "num_input", "num_hidden", "num_output"])
# df_hyper = df_hyper.append({
#   "train_data": train_data, 
#   "target":target, 
#   "lr":lr, 
#   "num_epochs":num_epochs, 
#   "num_input":num_input, 
#   "num_hidden":num_hidden, 
#   "num_output":num_output}, 
#   ignore_index=True)
# #df_hyper["train_data"][0]
# 
# df = pd.DataFrame(columns=["i", "weights_01", "weights_12", "b01", "b12", "losses"])
# df = df.append({
#   "i": 0, 
#   "weights_01": weights_01, 
#   "weights_12": weights_12, 
#   "b01": b01, 
#   "b12": b12, 
#   "losses": losses}, 
#   ignore_index=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return x * (1 - x)
      
      
      
def forward(i):
    """
    A single forward pass through the network.
    Implementation of wX + b
    """

    hidden.append(np.dot(train_data, weights_01[i]) + b01[i])
    hidden_out.append(sigmoid(hidden[i]))

    output.append(np.dot(hidden_out[i], weights_12[i]) + b12[i])
    output_final.append(sigmoid(output[i]))



def update_weights(i):
    
    # Calculate the squared error
    loss = 0.5 * (target - output_final[i]) ** 2
    print(loss)
    losses.append(np.sum(loss))

    error_term.append((target - output_final[i]))

    # the gradient for the hidden layer weights
    grad01.append( train_data.T @ (((error_term[i] * deriv_sigmoid(output_final[i])) * weights_12[i].T) * deriv_sigmoid(hidden_out[i])) )

    # the gradient for the output layer weights
    grad12.append( hidden_out[i].T @ (error_term[i] * deriv_sigmoid(output_final[i])) )


    # updating the weights by the learning rate times their gradient
    weights_01.append( weights_01[i] + lr * grad01[i] )
    weights_12.append( weights_12[i] + lr * grad12[i] )

    # update the biases the same way
    b01.append( b01[i] + np.sum(lr * ((error_term[i] * deriv_sigmoid(output_final[i])) * weights_12[i].T) * deriv_sigmoid(hidden_out[i]), axis=0) )
    b12.append( b12[i] + np.sum(lr * error_term[i] * deriv_sigmoid(output_final[i]), axis=0) )


def train():
    """
    Train an MLP. Runs through the data num_epochs number of times.
    A forward pass is done first, followed by a backward pass (backpropagation)
    where the networks parameter's are updated.
    """
    for i in range(num_epochs):
        print(i)
        forward(i)
        update_weights(i)



def plot_error():
  x = list(range(num_epochs))
  y = np.array(losses)
  
  pyplot.figure(figsize=(6,4))
  pyplot.plot(x, y, "g", linewidth=2)
  pyplot.xlabel("x", fontsize = 16)
  pyplot.ylim(0,1)
  pyplot.show()

train()
plot_error()
