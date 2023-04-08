from tensorflow.keras.applications import VGG16
import numpy as np
import glob
import os
from tensorflow.keras.models import model_from_json
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from src.components.data_loader import load_labels, load_dataThreeChannel,load_dataAndLabels

img_rows, img_cols = 100,100
channels=3
num_classes=10
epochs = 2
model_arch = "artifacts/saved_model.json"
model_weights = "artifacts/model_weights.h5"

print('yahan tak theek hai')

#myDir ="train_data/*.png"
#myDir ="Data/Tobacco"
myDir ="Tobacco"

#labels = load_labels(myDir)
#data = load_dataThreeChannel(myDir,img_rows,img_cols)
data, labels = load_dataAndLabels(myDir,100,100)

print(data)
print(labels)


#Include_top=False, Does not load the last two fully connected layers which act as the classifier.
#We are just loading the convolutional layers. 
vgg_conv = VGG16(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable=False

model = Sequential() 
# Add the vgg convolutional base model
model.add(vgg_conv) 
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 # Show a summary of the model. Check the number of trainable parameters
model.summary()    



X_train = data.astype('float32')
X_train /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(labels, num_classes)

opt = Adam(learning_rate=0.01)
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])
# Train the model
history = model.fit(
      X_train,Y_train,
      epochs=epochs,
      verbose=1)


model_json = model.to_json()
open(model_arch,'w').write(model_json)
model.save_weights(model_weights,overwrite=True)

acc = history.history['acc']
loss = history.history['loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()