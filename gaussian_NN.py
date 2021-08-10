#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


### Prepare the data ###
N = 500
cov = [[0.1,0],[0,0.1]]
x_L = np.random.multivariate_normal([-1,0], cov, N)
x_R = np.random.multivariate_normal([+1,0], cov, N)
y_L = np.array([[1,0]]*N)
y_R = np.array([[0,1]]*N)

plt.scatter(x_L[:,0],x_L[:,1], c='b')
plt.scatter(x_R[:,0],x_R[:,1], c='r')
plt.show()

x = np.concatenate((x_L, x_R))
y = np.concatenate((y_L, y_R)) 

ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = ratio, shuffle = True)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# In[ ]:


### Build the model ###
input_shape = (2,)
num_classes = 2

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(20, activation="sigmoid"),
        layers.Dense(20, activation="sigmoid"),
        layers.Dense(20, activation="sigmoid"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


# In[ ]:


### Train the model ###
batch_size = 128
epochs = 150

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[ ]:


### Evaluate the trained model ###
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print()

test = [[-1,0]]

#display the produced category array
print(model.predict(test))


# In[ ]:


### Scan parameter space and examine NN output ###

N = 100
x_1 = np.linspace(min(x[:,0]),max(x[:,0]), N)
x_2 = np.linspace(min(x[:,1]),max(x[:,1]), N)
xx, yy = np.meshgrid(x_1, x_2)
xx = xx.flatten()
yy = yy.flatten()
grid_data = np.array([xx,yy]).T

c = model.predict(grid_data)[:,1]

plt.scatter(grid_data[:,0], grid_data[:,1], c=c, cmap='rainbow')
plt.show()





