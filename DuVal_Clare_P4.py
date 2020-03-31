# Project 4
# Clare DuVal
# March 31 2020

import pandas as pd
import re
import numpy as np
import math
import matplotlib.pyplot as plt

K = 2
Group_One = pd.DataFrame(columns=['one', 'two'])
Group_Two = pd.DataFrame(columns=['one', 'two'])

fig1 = plt.figure()
fig1 = fig1.add_subplot(111)
fig2 = plt.figure()
fig2 = fig2.add_subplot(111)


#------- FUNCTIONS -------#
# Find the squared distance of two coordinates
def getSquareDistance(a, b):
    return np.square(b['one']-a['one']) + np.square(b['two']-a['two'])

# Find the average centroid of a group
def averageCentroid(df):
    x = df['one'].mean()
    y = df['two'].mean()
    return pd.Series([x, y], index=['one', 'two'])


#------------------------------- MAIN ---------------------------------#
#--------- DATA ---------#
data_name = input('What is the name of the data file? ') 
if (data_name == ''):
    data = pd.read_csv('P4Data.txt', sep= '\t', header=None) 
else:
    data = pd.read_csv(data_name, sep= '\t', header=None)
    
#------ CENTROIDS -------#
centroids_name = input('What is the name of the centroids file? ') 
if (centroids_name == ''):
    centroids = pd.read_csv('P4Centroids.txt', sep= '\t')
else:
    centroids = pd.read_csv(centroids_name, sep= '\t')


# Prepare the data
centroids = centroids.reset_index()
centroids.columns = ['one', 'two']
data = data[1:]
data.columns = ['one', 'two']

# Print the coordinates of the two initial centroids
print("Initial Centroids:")
for i, j in centroids.iterrows():
    print(j['one'], j['two'])
print("\n")

xy1 = centroids.loc[0]
xy2 = centroids.loc[1]
    
# Plot the centroids
fig1.scatter(centroids['one'], centroids['two'])
fig1.scatter(data['one'], data['two'])

# Run K-means [K=2] to cluster the data into two groups
Size1 = 0
Size2 = 0

# Run until the two groups don't change
while(True):
    
    # Reset Groups
    Group_One = Group_One.iloc[0:0]
    Group_Two = Group_Two.iloc[0:0]
    
    # Split the data into two groups based on which centroid they're closest to
    for index, row in data.iterrows():
        distToOne = getSquareDistance(xy1, row)
        distToTwo = getSquareDistance(xy2, row)
        if (distToOne < distToTwo):
            Group_One = pd.concat([Group_One, pd.DataFrame(row).T])
        else:
            Group_Two = pd.concat([Group_Two, pd.DataFrame(row).T])
            
    # Break if the groups aren't changing
    if ((Size1 == len(Group_One)) and Size2 == len(Group_Two)):
        break;
    else:
        Size1 = len(Group_One)
        Size2 = len(Group_Two)
        
    # Create new Centroids
    xy1 = averageCentroid(Group_One)
    xy2 = averageCentroid(Group_Two)
    
    
# Make graph of the two groups
fig2.scatter(Group_One['one'], Group_One['two'])
fig2.scatter(Group_Two['one'], Group_Two['two'])
fig2.scatter(xy1['one'], xy1['two'], marker='^')
fig2.scatter(xy2['one'], xy2['two'], marker='^')

# Print final centroids
print("Final Centroids:")
print(str('x: ' + str(xy1['one']) + ',  y: ' + str(xy1['two'])))
print(str('x: ' + str(xy2['one']) + ',  y: ' + str(xy2['two'])))