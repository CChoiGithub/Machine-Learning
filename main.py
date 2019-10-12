from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow, keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.seq2seq.python.ops import loss
#import classes
from train import *
from prediction import *


#evaluate accuracy
test_lost, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc) #test accuracy(0.8761) compared to
                                     #training accuracy(0,9108), so overfitted
fig = plt.gcf()
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
fig.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
fig.show()


#plot a number of predictions to images in 5 by 3 array
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#Final prediction
img = test_images[1]
img = np.expand_dims(img, 0) #make it as a batch
predictions_single = model.predict(img)
print(np.argmax(predictions_single[0]))
