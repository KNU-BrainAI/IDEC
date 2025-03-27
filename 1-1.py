
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt

def MSE_loss(y_predict, y):
    return ((y_predict-y)**2).sum()/len(y)

def function(x,w,b):
    return w*x+b

data = pd.read_csv('linear.csv')
x = data['X'].to_numpy().reshape(-1,1)
y = 5*x + 50*np.random.randn(100,1)


w = [0.5]
b = [0.5]

loss_history = []

#config
epochs = 100
learning_rate = 1e-4

for i in range(epochs):
    prediction = function(x,w[i],b[i])
    loss_history.append(MSE_loss(prediction,y))
    w.append(w[i]-(learning_rate/len(y))*((prediction-y)*x).sum())
    b.append(b[i]-learning_rate*(prediction-y).sum())
    
loss_history.append(MSE_loss(function(x,w[-1],b[-1]),y)) 

#초기 학습 plot
plt.plot(x,function(x,w[0],b[0]),c='r')
plt.scatter(x,y)
plt.text(20,400,"MSE Loss: {:.2f}".format(MSE_loss(function(x,w[0],b[0]),y)))
plt.show()

#내 학습 plot
plt.plot(x,function(x,w[-1],b[-1]),c='r')
plt.scatter(x,y)
plt.text(20,400,"MSE Loss: {:.2f}".format(MSE_loss(function(x,w[-1],b[-1]),y)))
plt.show()
