
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from src.components.data_loader import load_labels, load_dataThreeChannel,load_dataAndLabels
from tensorflow.keras.utils import to_categorical


img_rows, img_cols = 100,100
channels=3
num_classes=10
epochs = 10
model_arch = "artifacts/saved_model.json"
model_weights = "artifacts/model_weights.h5"


#myDir ="test_data/*.png"
#labels = load_labels(myDir)
#data = load_dataThreeChannel(myDir,img_rows,img_cols)

myDir ="Tobacco_Test"
data, labels = load_dataAndLabels(myDir,100,100)

model = model_from_json(open(model_arch).read())
model.load_weights(model_weights)

X_test = data.astype('float32')
X_test /= 255
print('x_test shape:', X_test.shape)
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_test = to_categorical(labels, num_classes)

opt = Adam(learning_rate=0.01)
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])
