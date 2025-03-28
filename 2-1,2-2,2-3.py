#%% 실습 2-1


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('nonlinear.csv')
# plt.scatter(df['x'], df['y'])

nx, nh1, nh2, ny = 1, 6, 4, 1
U = np.random.randn(nx , nh1)
V = np.random.randn(nh1, nh2)
W = np.random.randn(nh2, ny)
learning_rate = 1.0

def sigmoid(v):
    return 1 / (1+np.exp(-v))

input = np.zeros(nx)

h1_out, h1_deriv = np.zeros(nh1), np.zeros(nh1) # 순전파시 계산 - 은닉계층 1
h1_delta = np.zeros(nh1)                        # 역전파시 계산
h2_out, h2_deriv = np.zeros(nh2), np.zeros(nh2) # 순전파시 계산 - 은닉계층 2
h2_delta = np.zeros(nh2)                        # 역전파시 계산
y_out, y_deriv = np.zeros(ny), np.zeros(ny)     # 순전파시 계산 - 출력계층
y_delta = np.zeros(ny)                          # 역전파시 계산


def forward(x):
    global input, h1_out, h1_deriv, h2_out, h2_deriv, y_out, y_deriv
    input = x
    h1_out = sigmoid ( U.T.dot(input) )          # 은닉계층 1로 전파
    h1_deriv = h1_out * (1 - h1_out)             # 은닉계층 1의 미분

    h2_out = sigmoid ( V.T.dot(h1_out) )         # 은닉계층 2로 전파
    h2_deriv = h2_out * (1 - h2_out)             # 은닉계층 2의 미분
  
    y_out = sigmoid( W.T.dot(h2_out) )           # 출력계층으로 전파
    y_deriv = y_out * (1 - y_out)                # 출력계층의 미분
    
def compute_error(target):
    return y_out - target

def backward(error):
    global y_delta, W, h2_delta, V, h1_delta, U
    y_delta = y_deriv * error                         # 출력 계층의 델타
    dW = - learning_rate * np.outer(h2_out, y_delta)  # W의 수정
    W = W + dW
    h2_delta = h2_deriv * W.dot(y_delta)              # 은닉 계층 2의 델타
    dV = - learning_rate * np.outer(h1_out, h2_delta) # V의 수정
    V = V + dV
    h1_delta = h1_deriv * V.dot(h2_delta)             # 은닉 계층 1의 델타
    dU = - learning_rate * np.outer(input, h1_delta)  # U의 수정
    U = U + dU

def train(x, target):
    forward(x)
    e = compute_error(target)
    backward(e)
    return e**2

loss = []
X = df['x'].to_numpy()
y_label = df['y'].to_numpy()
for i in range(100):
    e_accum = 0
    for x, y in zip(X, y_label):
        e_accum += train(x, y)
    loss.append(e_accum)
    
err_log = np.array(loss).flatten()
# plt.plot(err_log)
# plt.show()

def predict(X):
    y_hat = []
    for x in X:
        forward(x)
        y_hat.append(y_out)
    return y_hat

domain = np.linspace(0, 1, 100).reshape(-1,1) # 입력은 2차원 벡터로 변형
y_hat = predict(domain)
plt.scatter(df['x'], df['y'])
plt.scatter(domain, y_hat, color='r')


#%% 실습 2-2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = keras.models.Sequential( [
    keras.layers.Dense(32, activation= 'tanh'),
    keras.layers.Dense(16, activation= 'tanh'),
    keras.layers.Dense(8, activation= 'tanh'),
    keras.layers.Dense(4, activation= 'tanh'),
    keras.layers.Dense(1, activation= 'tanh'),
])


optimizer = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mse')


df = pd.read_csv('nonlinear.csv')
X = df['x'].to_numpy().reshape(-1,1)
y_label = df['y'].to_numpy().reshape(-1,1)


model.fit(X, y_label, epochs=100)

domain = np.linspace(0, 1, 100).reshape(-1,1) # 입력은 2차원 벡터로 변형
y_hat = model.predict(domain)
plt.scatter(df['x'], df['y'])
plt.scatter(domain, y_hat, color='r')


#%% 실습 2-3

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],
    random_state=0,test_size=0.30,stratify=iris_dataset['target'])
print('훈련용 데이터의 형태:', x_train.shape)
print('훈련용 데이터의 레이블 형태:',y_train.shape)
print('테스트용 데이터의 형태:', x_test.shape)
print('테스트용 데이터의 레이블 형태:',y_test.shape)
print('개별 훈련 데이터의 형태:', x_train[0].shape)


from tensorflow import keras

# 순차 모델을 생성하자
model = keras.models.Sequential( [
    keras.layers.Flatten(input_shape = (4,)),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dense(3, activation= 'softmax'),
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=30, batch_size=5, verbose=1)


print('신경망 모델의 학습 결과 :')
eval_loss, eval_acc = model.evaluate(x_test, y_test)
print('붓꽃 데이터의 분류 정확도 :', eval_acc)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
eval_loss, eval_acc = model.evaluate(x_test, y_test)
print('붓꽃 데이터의 분류 정확도 :', eval_acc)
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.xlabel('epoch')



model = keras.models.Sequential( [
    keras.layers.Flatten(input_shape = (4,)),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dense(32, activation= 'relu'),
    keras.layers.Dense(10, activation= 'relu'),
    keras.layers.Dense(3, activation= 'softmax'),
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size=5, verbose=1)




import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss value')

eval_loss, eval_acc = model.evaluate(x_test, y_test)
print('Iris 데이터의 분류 정확도 :', eval_acc)
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.xlabel('epoch')




from tensorflow import keras


model = keras.models.Sequential( [
    keras.layers.Flatten(input_shape = (4,)),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation= 'relu'),
    keras.layers.Dense(10, activation= 'relu'),
    keras.layers.Dense(3, activation= 'softmax'),
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size=5, verbose=1)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss value')
plt.legend()
eval_loss, eval_acc = model.evaluate(x_test, y_test)
print('Iris 데이터의 분류 정확도 :', eval_acc)
plt.plot(history.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.xlabel('epoch')


