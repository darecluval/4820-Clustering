# Project 3
# Clare DuVal
# March 12 2020

import pandas as pd
import re
import numpy as np

#------- FUNCTIONS -------#
# Make a library of all words found in a dataframe with a given spam/ham number
def makeLibrary(df, i):
    dict_x = {str(i): []}
    for r in df:
        for w in r.split():
            ap = re.sub(r'\W+', '', w)
            if ap != '':
                dict_x[str(i)].append(ap.lower().strip())  
    return dict_x

# Count the amount of times a word occurs in spam or ham
def countIn(w, dct, i):
    count = 0
    for c in dct[str(i)]:
        if w == c:
            count = count + 1
    return count

# Return the probability of a word in a given set
def prob(prob, tot):
    num = 1 + prob
    denom = 2 * 1 + tot
    return round(num/denom,6)

# Remove stop words, duplicates
def cleanDictionary(dict_x, stop, i,temp):
    for v in dict_x[str(i)]:
        if [v] not in stop and v not in temp: 
            temp.append(v)
    return temp

# P(A|B) where A and B can be substituted
def PAinB(dct, test, J):
    ret = 1
    for i in dct:
        if i in test:
            ret = ret * dct[i][J]
        else: 
            ret = ret * (1-dct[i][J])
    return ret
 
def testProb(dct, test, probH, probS):
    probHam = PAinB(dct, test, 0)
    probSpam = PAinB(dct, test, 1)
    temp = np.log(probHam*probH) - np.log(probSpam*probS)
    return int(round(1 / (1 + np.exp(temp))))

# Predicted Yes, Actual No
def calFalsePositives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = int(Y.iloc[i])
        yPred = int(predY[i])
        if ((yVal == 0) and (yPred == 1)):
            count = count + 1
    return count 
    
# Predicted No, Actual Yes
def calFalseNegatives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = int(Y.iloc[i])
        yPred = int(predY[i])
        if ((yVal == 1) & (yPred == 0)):
            count = count + 1
    return count 
    
# Predicted Yes, Actual Yes
def calTruePositives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = int(Y.iloc[i])
        yPred = int(predY[i])
        if ((yVal == 1) & (yPred == 1)):
            count = count + 1
    return count 
  
# Predicted No, Actual No
def calTrueNegatives(Y, predY, m):
    count = 0
    for i in range(m):
        yVal = int(Y.iloc[i])
        yPred = int(predY[i])
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


#--------- MAIN ---------#
# Read in words to ignore (common words)
stop_words = pd.read_csv('StopWords.txt', header=None) 
stop_words = stop_words.values.tolist()


#--------- TRAIN ---------#
train_name = input('What is the name of the training set? ') + '.txt'
if (train_name == '.txt'):
    df_train = pd.read_csv('SHTrain.txt', sep= '\t', header=None) 
else:
    df_train = pd.read_csv(train_name + '.txt', sep= '\t', header=None) 
    
# Clean columns
df_train.columns = ['Name']
df_train['Value'] = df_train['Name'].astype(str).str[0]
df_train['Name'] = df_train['Name'].astype(str).str[2:]

# Split into ham/spam\
df_train_ham = df_train[df_train.Value == '0']
num_ham = len(df_train_ham)
df_train_spam = df_train[df_train.Value == '1']
num_spam = len(df_train_spam)

# Make dictionary of spam and ham words
dict_ham = makeLibrary(df_train_ham['Name'],0)
all_words = cleanDictionary(dict_ham, stop_words,0,[])
dict_spam = makeLibrary(df_train_spam['Name'],1)
all_words = cleanDictionary(dict_spam, stop_words,1, all_words)


# Make dictionary of all words with values of how many times they appear
#      in spam and ham lists
dict_ret = {}
for w in all_words:
    temp = countIn(w, dict_ham, 0)
    dict_ret[str(w)] = [prob(temp, num_ham)]
    temp = countIn(w, dict_spam, 1)
    dict_ret[str(w)].append(prob(temp, num_spam))


#--------- TEST ---------#
test_name = input('What is the name of the testing set? ') + '.txt'
if (test_name == '.txt'):
    df_test = pd.read_csv('SHTest.txt', sep= '\t', header=None) 
else:
    df_test = pd.read_csv(test_name + '.txt', sep= '\t', header=None) 
num_test = len(df_test)

# Clean columns
df_test.columns = ['Name']
df_test['Value'] = df_test['Name'].astype(str).str[0]
df_test['Name'] = df_test['Name'].astype(str).str[2:]

# Split test's spam and ham
df_test_ham = df_test[df_test.Value == '0']
df_test_spam = df_test[df_test.Value == '1']
df_Y = df_test.Value

# Make predicted y array
y_pred = []
probH = round(num_ham/(num_ham + num_spam),6)
probS = round(num_spam/(num_ham + num_spam),6)
for i in df_test['Name']:
    i = i.lower()
    i = i.split(' ')
    newI = []
    for j in i:
        newI.append(re.sub(r'\W+', '', j))
    pred = testProb(dict_ret, newI, probH, probS)
    y_pred.append(pred)
    
    
#------ PRINT STATS ------#
# Calculate confusion matrix
tot = num_test
TP = calTruePositives(df_Y, y_pred, tot)
TN = calTrueNegatives(df_Y, y_pred, tot)
FP = calFalsePositives(df_Y, y_pred, tot)
FN = calFalseNegatives(df_Y, y_pred, tot)

# Calculate stats
acc = calAccuracy(TP, TN, FP, FN)
prec = calPrecision(TP, FP)
recall = calRecall(TP, FN)
F1 = calF1Score(recall, prec)

#plt.plot(js)
print("\nNumber of Ham emails in Test: " + str(len(df_test_ham)))
print("Number of Spam emails in Test: " + str(len(df_test_spam)))

# Print Outputs
print("\nTEST STATISTICS")
print("True Positives: " + str(TP))
print("True Negatives: " + str(TN))
print("False Positives: " + str(FP))
print("False Negatives: " + str(FN))
print("Accuracy: " + str(acc))
print("Precision: " + str(prec))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1))