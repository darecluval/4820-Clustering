# Project 4
# Clare DuVal
# March 31 2020

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

K = 2
Group_One = pd.DataFrame(columns=['one', 'two'])
Group_Two = pd.DataFrame(columns=['one', 'two'])


#------- FUNCTIONS -------#
def getSquareDistance(a, b):
    return np.square(b['one']-a['one']) + np.square(b['two']-a['two'])


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
plt.scatter(centroids['one'], centroids['two'])
plt.scatter(data['one'], data['two'])

# Run K-means [K=2] to cluster the data into two groups
Size1 = 0
Size2 = 0

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
    xy1 = average_Centroid(Group_One)
    xy2 = average_Centroid(Group_Two)