import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model

model = VGGFace(model='resnet50')
model = Model(inputs=model.input, outputs=model.layers[-2].output)
face = MTCNN()

def CaptureFaces(X,face):
    X = cv2.imread(X)
    output = face.detect_faces(X)
    x,y,w,h = output[0]['box']
    X = X[x:x+h, y:y+h]
    X = X.transpose(2,0,1)
    B = cv2.resize(X[0], (224,224))
    G = cv2.resize(X[1], (224,224))
    R = cv2.resize(X[2], (224,224))
    X = np.concatenate([[R],[G],[B]])
    return X.transpose(1,2,0)

def load_images(X,face):
    X = X + '/DataBase'
    lst = os.listdir(X)
    image_list = []
    for filename in lst:
        i = X + '/' + str(filename)
        img = CaptureFaces(i, face)
        image_list = image_list + [img]
    return np.array(image_list)

def Predict(X,model):
    X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
    X = model.predict(X)
    return X

def Vectors(X,model,face):
    X = load_images(X,face)
    L = X.shape[0]
    vec = []
    for i in range(0,L):
        X_i = X[i]
        vec = vec + [Predict(X_i,model)]
    vec = np.array(vec)
    return vec

path = '/home/deb/VggFace'
X = Vectors(path,model,face)
np.save(path + '/vec.npy', X)
