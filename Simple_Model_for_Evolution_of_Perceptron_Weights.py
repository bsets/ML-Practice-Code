
'''This code computes the evolution of weights for a simple one dimensional perceptron used for classifying flowers in the Iris 
data set over 100 iterations. The weights are updated for each example using the learning rule at each iteration and then each 
weight w0,w1 and w2 is averaged by taking the computed weights of the 100 examples. These new weights are then used in the next 
iteration for classification '''

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.font_manager import FontProperties


# Import Iris Data into Python

flower_data=pd.read_csv('H:/Self_Learning/Machine Learning Book Raschka/Chapter2/irisdata.csv',header=None) 
# This is from my local machine. Put 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' instead of 
# my machine location to get the data set from UC Irvine's repository

# Extract the output vector and give it labels -1 or 1 depending on the flower type
y = flower_data.iloc[0:100, 4].values   # Just the first 100 flowers have been extracted
y = np.where(y == 'Iris-setosa', -1, 1) # Iris setosa is assigned -1 and Iris-Versicolor is assigned 1

# Extract the first two dimensions of the flower - sepal length and petal length

X = flower_data.iloc[0:100, [0, 2]].values # Just two parameters have been extracted

k=np.ones((100,1)) # An column array of ones 

# Concatenate the column array of ones with the X array
X= np.concatenate((k,X),axis=1) 


# Define an array that stores the weights

mu, sigma = 0, 0.01
w = np.random.normal((mu, sigma, 3)) # The weight array that contains three numbers from a random distribution that has a mean of 0 and standard deviation of 0.01


eta=0.1 # Learning Rate for the perceptron
passes = 100 # Number of passes of the model over the data to refine the weights
c= np.empty([passes, 3]) # An empty array that stores the weights as they evolve with each pass
err= [] # A list to store the number of errors after each pass, i.e the instances where the predicted category is different from the actual category

# Loop over data to compute predicted classes, update weights and compute errors

for i in range (passes):

    y_hat = np.dot(X,w) # X.w
    y_predict=np.where(y_hat >= 0.0, 1, -1)
    update=y-y_predict
    err.append(np.count_nonzero (update))
    w+= [np.sum(update)*eta/100,np.dot(eta*update,X[:,1])/100,np.dot(eta*update,X[:,2])/100]
    c[i:,]=w
    
    
print (c)

print(update)

# Plot the evolution of weights

pylab.plot(range(10), label="Plot 1")
pylab.plot(range(10, 0, -1), label="Plot 2")
pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.plot(np.arange(passes), c[:,0])
plt.plot(np.arange(passes), c[:,1])
plt.plot(np.arange(passes), c[:,2])


plt.legend(['w0', 'w1', 'w2'], loc='upper left')

plt.show()

# Plot the evolution of error

plt.plot(np.arange(passes), err)

