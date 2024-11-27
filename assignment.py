#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Step function (activation function)
def step_function(x, Y):
    return 1 if (x >= 0.5 and Y > 0.5) else 0


# In[2]:


# Perceptron class
class Perceptron:
    def _init_(self, learning_rate=0.01, epochs=100):
        # Initialize weight and bias
        self.weight = np.random.rand()  # Single weight for 1D output
        self.bias = np.random.rand()     # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs



# In[4]:


# Predict output for a given input
def predict(self, x):
        linear_output = x[0] * self.weight + x[1] * self.weight + self.bias
        return step_function(linear_output, linear_output)


# In[6]:


# Train the perceptron
def train(self, X, y):
     for epoch in range(self.epochs):
         for i in range(len(X)):
             prediction = self.predict(X[i])
             error = y[i] - prediction
             # Update weight and bias
             self.weight += self.learning_rate * error * X[i][0]
             self.bias += self.learning_rate * error


# In[7]:


# Training data (classification of number) (<0.5 and <0.5 as 0, >0.5 and >0.5 as 1, <0.5 and >0.5 as 0, >0.5 and <0.5 as 0)
X = np.array([[0.4, 0.3], [0.6, 0.8], [0.3, 0.9], [0.7, 0.2]])
y = np.array([0, 1, 0, 0])  # Corresponding labels



# In[11]:


# Create and train perceptron
perceptron = Perceptron()
perceptron.train(X, y)


# In[13]:


def user_input_classification():
    user_input1 = float(input("Enter first number between 0 and 1: "))
    user_input2 = float(input("Enter second number between 0 and 1: "))
    
    if 0 <= user_input1 <= 1 and 0 <= user_input2 <= 1:
        prediction = perceptron.predict(np.array([user_input1, user_input2]))
        print(f"Perceptron classification: {prediction}")
    else:
        print("Please enter numbers between 0 and 1.")

user_input_classification()


# In[ ]:




