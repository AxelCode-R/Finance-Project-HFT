
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import math as ma
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
np.set_printoptions(suppress=True)


data_train = pd.read_excel("Abgabe 4/lendingclub_traindata.xlsx").fillna(0)
# data_train.hist(bins=10,figsize=(8,8), range=(0,1))
# pyplot.tight_layout()
# pyplot.show()


def calc_entropy(df, decision_on = "loan_status"):
  if len(df)==0:
    return(0)
  p = np.sum(df[decision_on] == 0)/len(df)
  if p == 0 or p == 1:
    return(0)
  result = -p * ma.log(p,2) - (1-p)*ma.log(1-p,2)
  return result

def calc_splitted_entropy(df, col, val, decision_on = "loan_status"):
  w = np.sum(df[col] > val)/len(df)
  result = w * calc_entropy(df.loc[df[col] > val], decision_on) + (1-w) * calc_entropy(df.loc[df[col] <= val], decision_on)
  return result

def find_minima(df, col, decision_on = "loan_status", round_at = 5):
  direction = 1
  step = (df[col].max()-df[col].min()) * 0.1
  val = df[col].min() + step
  best_entropy = 1
  stagnation = 0
  
  while stagnation <= 100:
    print(val)
    temp = calc_splitted_entropy(df, col, val)
    print(temp)
    if temp > best_entropy:
      print(1)
      direction = -direction
      step = 0.5 * step
      stagnation += 1
    elif round(temp,round_at) < round(best_entropy,round_at):
      print(2)
      stagnation = 0
    else:
      print(3)
      stagnation += 1
    best_entropy = temp
    val = val + direction * step
    
  return best_entropy, val


def find_minima2(df, col, decision_on = "loan_status", search_minima_intervalls = 1000):
  step = (df[col].max()-df[col].min())*(1/search_minima_intervalls)
  space = np.arange(df[col].min(),df[col].max()+step, step)
  entropys = [calc_splitted_entropy(df, col, x) for x in space]
  best_index = np.where(entropys==np.amin(entropys))[0][0]
  best_entropy = entropys[best_index]
  val = space[best_index]
  return best_entropy, val



def find_best_col(df, decision_on = "loan_status", search_minima_intervalls = 1000):
  cols = list(df.columns[df.columns != decision_on])
  entropys = np.ones(len(cols))
  vals = np.ones(len(cols))
  
  for i in range(len(cols)):
    entropys[i], vals[i] = find_minima2(df, col=cols[i], decision_on = "loan_status", search_minima_intervalls=search_minima_intervalls)
  
  best_i = int(np.where(entropys == min(entropys))[0][0])
  return cols[best_i], entropys[best_i], vals[best_i]




def make_node_and_leafs(df, decision_on = "loan_status", search_minima_intervalls = 1000, path = "I", condition = "", min_size = 1000, max_depth = 3, leafs = pd.DataFrame(columns=["path", "condition", "rows", "P_of_no_default", "entropy"])):
  if len(df) < min_size or (path.count("-")-1) >= max_depth or len(df.columns) <= 1:
    leafs = leafs.append({"path":path+"}", "condition":condition[0:(len(condition)-5)], "rows":len(df), "P_of_no_default":np.sum(df[decision_on] == 0)/len(df), "entropy":calc_entropy(df)}, ignore_index=True)
  else:
    col, entropy, val = find_best_col(df, decision_on, search_minima_intervalls)
    print("path:", path, "   entropy:", entropy, "  col:", col, "   val:", val, "  rows:", len(df))
    leafs = make_node_and_leafs( df.loc[df[col] > val, df.columns != col], decision_on, search_minima_intervalls, path+"-R", condition+col+" > "+str(float(round(val,5)))+" and ", min_size, max_depth, leafs)
    leafs = make_node_and_leafs( df.loc[df[col] <= val, df.columns != col], decision_on, search_minima_intervalls, path + "-L", condition+col+" <= "+str(float(round(val,5)))+" and ", min_size, max_depth, leafs)
  return(leafs)
  
  
  
#data_train["loan_status"] = (data_train["loan_status"]-1) * -1

leafs = make_node_and_leafs(df=data_train, decision_on = "loan_status", search_minima_intervalls = 1000, min_size = 1000, max_depth = 3)
leafs["entropy"] = (leafs["entropy"]*leafs["rows"])/len(data_train)

print("Entropy in data: ", calc_entropy(data_train))
print("Entropy in all leafs: ", np.sum(leafs["entropy"]))

  

data_temp = data_train.copy()
data_temp["ID"] = list(range(len(data_temp)))
conditions = "("+ ") | (".join(list(leafs.loc[leafs["P_of_no_default"] < 0.75, leafs.columns == "condition"]["condition"].replace("and","&")))+")"
data_temp = data_temp.query(conditions)
X = np.zeros(len(data_train))
X[list(data_temp["ID"])] = 1

Y = data_train.loc[:, data_train.columns == 'loan_status'].to_numpy()[:,0]

print("Wrong answers of the decission tree: ",np.sum(np.abs(Y-X))/len(Y) * 100, "%")
confusion_matrix(Y,X)
