import numpy as np
import glob
import os
from PIL import Image

def load_labels(myDir):
    labels=[]
    fileList = glob.glob(myDir)
    for fname in fileList:
        fileName = os.path.basename(fname)
        curLabel = fileName.split("-")[0]
        labels.append(curLabel)
    return np.asarray(labels)
        

  

def load_dataThreeChannel(myDir,img_rows,img_cols):
    images=[]
    fileList = glob.glob(myDir)    
  #  x = np.array([np.array(Image.open(fname)).flatten() for fname in fileList])
  #  x = np.array([np.array(Image.open(fname)) for fname in fileList])
    for fname in fileList:
        #print(fname)
        img = Image.open(fname)
        output = np.array(img.resize((img_rows,img_cols), Image.ANTIALIAS))
        output = np.stack((output,)*3, -1)
        images.append(output)

         
    x=np.asarray(images)
    print(x.shape)
    return x



def load_dataAndLabels(data_dir,img_rows,img_cols):
    classes = sorted(os.listdir(data_dir)) # list subdirectories as class labels

    print(classes)
    images = []
    labels = []

    for i, class_label in enumerate(classes):
        class_dir = os.path.join(data_dir, class_label)
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".jpg"): # or any other image format
                img_path = os.path.join(class_dir, file_name)
                img = Image.open(img_path)
                output = np.array(img.resize((img_rows,img_cols), Image.ANTIALIAS))
                output = np.stack((output,)*3, -1)
                images.append(output)
                labels.append(i) # append index of class_label in sorted list of classes
    return np.asarray(images),np.asarray(labels)

