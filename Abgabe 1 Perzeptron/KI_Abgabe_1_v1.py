import numpy as np
import random as ra
import pandas as pd
import matplotlib.pyplot as pyplot
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def step(s):
    if( s >= 0 ):
        return(1)
    else:
        return(0)

def forward(w, x):
    return( step( np.dot(w.transpose(), x) ) )

def backward(w, x, correct_answer, approx_answer, alpha):
    return(w + alpha * (correct_answer - approx_answer) * x)


def fit(alpha, iterations, training_data_set, wgts):
    
    
    df = pd.DataFrame(columns=["iteration", "input_wgts", "approx_answer", "correct_answer", "error", "new_wgts"])
    
    
    for i in range(1, iterations+1):
        random_index = ra.randint(0, len(training_data_set)-1) 
        input_wgts = wgts
        input_x = training_data_set[random_index][0]
        correct_answer = training_data_set[random_index][1]

        approx_answer = forward(w = input_wgts, x = input_x)
        wgts = backward(w = input_wgts, 
                           x = input_x,
                           correct_answer = correct_answer,
                           approx_answer = approx_answer,
                           alpha = alpha)
        df = df.append({'iteration': i, 
                        'input_wgts': input_wgts, 
                        'approx_answer': approx_answer, 
                        'correct_answer': correct_answer, 
                        'error': correct_answer - approx_answer, 
                        'new_wgts': wgts}, 
                       ignore_index=True)
    return(df)
        
        
def test(wgts, test_data_set):
    
    df = pd.DataFrame(columns=["iteration", "dataset", "approx_answer", "correct_answer", "error"])
    
    
    for i in range(0, len(test_data_set)):

        x = test_data_set[i][0]
        correct_answer = test_data_set[i][1]

        approx_answer = step(skalarprodukt(w = wgts, 
                                               x = x))

        df = df.append({'iteration': i, 
                        "dataset": x,
                        'approx_answer': approx_answer, 
                        'correct_answer': correct_answer, 
                        'error': correct_answer - approx_answer}, 
                       ignore_index=True)
    return(df)
        
        
        
# INPUTS
alpha = 1
iterations = 30
biasVal = 1
training_data_set = [(np.array([1,0,0]), 0), 
                     (np.array([1,0,1]), 1), 
                     (np.array([1,1,0]), 1), 
                     (np.array([1,1,1]), 1)]

wgts = np.zeros(len(training_data_set[0][0]))
wgts[0] = -biasVal

result_fit = fit(alpha = alpha, iterations = iterations, training_data_set = training_data_set, wgts = wgts)   
print("Result of Fitting: \n", result_fit, "\n\n")
    
optimized_wgts = result_fit["new_wgts"][result_fit.last_valid_index()]
print("Optimized Wgts are: ", optimized_wgts, "\n")

result_test = test(wgts = optimized_wgts, test_data_set = training_data_set)   
print("Result of Test: \n", result_test, "\n\n")
     
print("Plot Trainingsphase und Testphase:")
fig, axs = pyplot.subplots(2)
axs[0].plot(result_fit["iteration"], result_fit["error"], "g", linewidth=2)
axs[0].set_title("Training")
axs[0].set_ylabel("Error")
axs[0].set_xlabel("Iteration")
axs[1].plot(result_test["iteration"], result_test["error"], "g", linewidth=2)
axs[1].set_title("Test")
axs[1].set_ylabel("Error")
axs[1].set_xlabel("Test Row")
axs[1].set_xticks(range(len(training_data_set)))
pyplot.tight_layout()
pyplot.show()



print("Plots der Trennlinien:")
data_trennlinie = fit(alpha = alpha, iterations = iterations, training_data_set = training_data_set, wgts = np.array([-0.28, 0.02, 0.05]))   


pic_rows = 10
pic_cols = 3
fig, axs = pyplot.subplots(pic_rows, pic_cols, figsize=(9,40))

for k in range(pic_rows):
    for i in range(pic_cols):
         #print(k*pic_cols+i)
         w = data_trennlinie["input_wgts"][k*pic_cols+i]
         x1 = np.arange(-0.5, 1.5, 0.01)
         x2 = np.zeros(len(np.arange(-0.5, 1.5, 0.01)))
         if w[2] != 0:
             x2 = - w[0] / w[2] - w[1] / w[2] * x1
         
         axs[k, i].plot(x1, x2, "g", linewidth=2)
         #axs[k, i].set_ylim((min(-2,np.min(x2)), max(6,np.max(x2))))
         axs[k, i].set_ylim(-2, 6)
         axs[k, i].plot(0,0,"ko")
         axs[k, i].plot([0,1,1],[1,0,1],"ro", )
         axs[k, i].set_title("Schritt " + str(k*pic_cols+i))
#pyplot.tight_layout()
pyplot.show()



