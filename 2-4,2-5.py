#%% 실습 2-4


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 패션 MNIST 데이터는 keras의 데이터셋에 있으며, 이를 학습용, 테스트 데이터로 구분하자
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)  # 학습 이미지의 형태와 레이블을 출력한다
print(train_labels)
print(test_images.shape)

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.imshow(train_images[0],cmap='gray')       # 첫 번째 훈련용 데이터
ax2.imshow(train_images[1],cmap='gray')       # 두 번째 훈련용 데이터
ax3.imshow(train_images[2],cmap='gray')       # 세 번째 훈련용 데이터
plt.show()


# 0-255 구간의 픽셀을 정규화한다
train_images, test_images = train_images / 255, test_images / 255
# 순차 심층 신경망 모델을 만들자
model = keras.models.Sequential( [
   keras.layers.Flatten(input_shape=(28, 28)),
   keras.layers.Dropout(0.2),
   keras.layers.Dense(128, activation= 'relu'),
   keras.layers.Dense(32, activation= 'relu'),
   keras.layers.Dense(10, activation= 'softmax'),
])



model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=64,
                    epochs=10, validation_split=0.25)


plt.plot(history.history['loss'], 'b-',label='train')
plt.plot(history.history['val_loss'], 'r--',label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()


plt.plot(history.history['accuracy'], 'b-',label='training')
plt.plot(history.history['val_accuracy'], 'r--',label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test 정확도:', test_acc)


#%% 실습 2-5

randIdx = np.random.randint(0, 1000)
plt.imshow(test_images[0])


print(test_images[randIdx].shape)
new_image = test_images[randIdx][np.newaxis, :, :]
print(new_image.shape)

yhat = model.predict(new_image)
print(yhat.round(3))    # 소수점 아래 세 자리 정확도로 살펴보자

yhat = np.argmax( model.predict( test_images[randIdx][np.newaxis, :, :]) )
yhat

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_names[yhat])



def plot_images(images, labels, images_per_row=5):
  n_images = len(images)
  n_rows = (n_images-1) // images_per_row + 1
  fig, ax = plt.subplots(n_rows, images_per_row, 
                         figsize = (images_per_row * 2, n_rows * 2))
  for i in range(n_rows):
      for j in range(images_per_row):
          if i*images_per_row + j >= n_images: break
          img_idx = i*images_per_row + j
          a_image = images[img_idx].reshape(28,28)
          if n_rows>1: axis = ax[i, j]
          else: axis = ax[j]
          axis.get_xaxis().set_visible(False)
          axis.get_yaxis().set_visible(False)
          label = class_names[labels[img_idx]]
          axis.set_title(label)
          axis.imshow(a_image, interpolation='nearest',cmap='gray')

images = test_images[:25]
predictions = np.argmax(model.predict(images), axis=1)
print(predictions)
plot_images(images, predictions, images_per_row = 5)
