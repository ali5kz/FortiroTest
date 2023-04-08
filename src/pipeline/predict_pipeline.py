
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from src.components.data_loader import load_labels, load_dataThreeChannel,load_dataAndLabels
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
label_list = ['Advertisement', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
#-----------------------------------------------------------------------------------------------------------------
def predict(img):
     
    img_rows=100
    img_cols=100
    # Resize image to target size
    img = img.resize((img_rows, img_cols))

    # Convert image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to 4D tensor with shape (1, height, width, channels)
    img = np.expand_dims(np.array(img), axis=0)
        

    model_arch = "artifacts/saved_model.json"
    model_weights = "artifacts/model_weights.h5"
   
    query = img

    model = model_from_json(open(model_arch).read())
    model.load_weights(model_weights)

    query = query.astype('float32')
    query /= 255

    opt = Adam(learning_rate=0.01)
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['acc'])

    predictions = model.predict(query)
    label = np.argmax(predictions, axis=1)
    print('Label:', label)
    
    index = int(label[0])
    

    return label_list[index]
#-----------------------------------------------------------------------------------------------------------------
