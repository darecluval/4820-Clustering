# Project 2
# Clare DuVal
# March 4 2020

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.00001
num_iter = 400

#------------------------- FUNCTIONS ------------------------------#
# Calculate 1/(1+e^(-wTx))
def sigmoid(weights, X):
    temp = np.dot(pd.DataFrame(weights).T, pd.DataFrame(X))
    temp = sum(temp) 
    return (1.0/(1.0+np.exp(-temp)))

# Calculate the cost from the updated weights
def calCost(X, W, Y, m):
    totJ = 0
    for i in range(m):
        sig = np.sum(sigmoid(W,pd.DataFrame(X.loc[i])))
        totJ += Y[i] * np.log(sig) + (1 - Y[i]) * np.log(1-sig)
        
    return (-1/m) * totJ
    
# Predicted Yes, Actual No
def calFalsePositives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = Y.iloc[i]
        yPred = predY.iloc[i]
        if ((yVal == 0) & (yPred == 1)):
            count = count + 1
    return count 
    
# Predicted No, Actual Yes
def calFalseNegatives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = Y.iloc[i]
        yPred = predY.iloc[i]
        if ((yVal == 1) & (yPred == 0)):
            count = count + 1
    return count 
    
# Predicted Yes, Actual Yes
def calTruePositives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = Y.iloc[i]
        yPred = predY.iloc[i]
        if ((yVal == 1) & (yPred == 1)):
            count = count + 1
    return count 
  
# Predicted No, Actual No
def calTrueNegatives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = Y.iloc[i]
        yPred = predY.iloc[i]
        if ((yVal == 0) & (yPred == 0)):
            count = count + 1
    return count 
    
# Calculate ratio of those correctly predicted to all observations
def calAccuracy(TP, TN, FP, FN):
    return (TP + TN)/(TP + TN + FP + FN)
    
# Calculate ratio of correct positive observations to all observations
def calPrecision(TP, FP):
    return TP/(TP + FP)
   
# Calulate ratio of correct positive observations tp 
def calRecall(TP, FN):
    return TP/(TP + FN)

# Calculate weighted average of precision and recall
def calF1Score(recall, precision):
    return (2*recall*precision)/(recall+precision)

# Make file header with m n values
def makeHeader(M):
    list = [str(len(M)), str(len(M.columns)-1)]
    for col in range(len(M.columns)-2):
        list.append(None)
    return list
    

#---------------------------- MAIN ---------------------------------#
# Read in DivorceAll.txt that's separated by tabs into a dataframe
df = pd.read_csv('DivorceAll.txt', sep= '\t', skiprows=(1), header=None) 
df = df.sample(frac=1).reset_index(drop=True)


#--------- TRAIN ---------#
train_name = input('What is the name of the training set? ') + '.txt'
if (train_name == '.txt'):
    df_train = df.iloc[:math.ceil(.70*len(df))] 
    df_train.insert(0, "", 1.0)

    df_train.columns = makeHeader(df_train)
    df_train.to_csv('DuVal_Clare_Train.txt', header=True, index=None, sep='\t', mode='w')
else: 
    df_train = pd.read_csv(train_name, sep= '\t', skiprows=(1), header=None) 

df_train_X = df_train.iloc[:,:-1]
df_train_Y = df_train.iloc[:,-1]
n = len(df_train_X.columns)
m = len(df_train_X)


# Prepare the initial weights, cost arrays
best_w = []
for i in range(len(df_train_X.columns)):
    best_w.append(1)
best_w = np.transpose(best_w)
best_w = list(best_w)
js = []

# Run 400 iterations
for i in range(num_iter):
    
    tot = 0
    for j in range(m):     #row
        for k in range(n): #col
        
            tot += sigmoid(best_w, pd.DataFrame(df_train_X.loc[j]))-df_train_Y.iloc[j]
            best_w[k] = np.sum(best_w[k] - (learning_rate/m) * tot)
     
    cost = calCost(df_train_X, best_w, df_train_Y,m)
    js.append(cost)
            

#---------- TEST ---------#
test_name = input('What is the name of the testing set? ') + '.txt'
if (test_name == '.txt'):
    df_test = df.iloc[math.ceil(.70*len(df)):] 
    df_train.insert(0, "", 1.0)
    
    df_test.columns = makeHeader(df_test)
    df_test.to_csv('DuVal_Clare_Test.txt', header=True, index=None, sep='\t', mode='w')
else:
    df_test = pd.read_csv(test_name, sep= '\t', skiprows=(1), header=None) 
    
df_test_X = df_test.iloc[:,:-1].reset_index(drop=True)
df_test_Y = df_test.iloc[:,-1].reset_index(drop=True)


n = len(df_test_X.columns)
m = len(df_test_X)
df_pred_Y = []
    
# Test the weights to predict tests' values and compare the actual
for j in range(m):
    
    sig = sum(sigmoid(best_w, pd.DataFrame(df_test_X.loc[j])))
    print(sig)
    # If the sigmoid value is over .90 label as positive
    # else label negative
    if (sig > .90):
        df_pred_Y.append(1)
    else:
        df_pred_Y.append(0)
    

#Prepare data to find confusion matrix
df_pred_Y = pd.Series(df_pred_Y)


#------ PRINT STATS ------#
# Calculate confusion matrix
TP = calTruePositives(df_test_Y, df_pred_Y, m)
TN = calTrueNegatives(df_test_Y, df_pred_Y, m)
FP = calFalsePositives(df_test_Y, df_pred_Y, m)
FN = calFalseNegatives(df_test_Y, df_pred_Y, m)

# Calculate stats
acc = calAccuracy(TP, TN, FP, FN)
prec = calPrecision(TP, FP)
recall = calRecall(TP, FN)
F1 = calF1Score(recall, prec)

plt.plot(js)

# Print Outputs
print("TEST STATISTICS")
print("True Positives: " + str(TP))
print("True Negatives: " + str(TN))
print("False Positives: " + str(FP))
print("False Negatives: " + str(FN))
print("Accuracy: " + str(acc))
print("Precision: " + str(prec))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1))
    
#plt.plot(js)
    
    
    