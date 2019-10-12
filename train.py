from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow, keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.seq2seq.python.ops import loss

cloth_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = cloth_data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#data preprocessing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
#normalise data
train_images = train_images / 255.0
test_images = test_images / 255.0

#check if it's going right
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1) #divide subplots
    plt.xticks([]) #disable xticks
    plt.yticks([]) #disable yticks
    plt.grid(False) #no grid
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#building model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'), #128 nodes
    keras.layers.Dense(10, activation='softmax') #10 ndoes
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model.fit(train_images, train_labels, epochs=10)
