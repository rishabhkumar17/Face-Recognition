import cv2
import numpy as np
import os

def dist(v1,v2):

    return np.sqrt(sum((v1-v2)**2)) # Euclidean distance

# KNN
def knn(train,test,k=5):
    dist = []

    for i in range(train.shape[0]):

        ix = train[i, :-1]
        iy = train[i, :-1]

        d  = dist(test, ix)
        dist.append([d, iy])
        
    dk = sorted(dist, key=lambda x: x[0])[:k]
    
    labels = np.array(dk)[:, -1]
    
    output = np.unique(labels, return_counts = True)
    print(new_vals)
    
    index = np.argmax(output[1])
    return output[0][index]
################################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier("Datasets/haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = 'Datasets/'

face_data = []
labels = []

class_id = 0 # labels for the given file
names = {} #Mapping between id and name

# Data Preperation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate((face_data),axis=0)
face_labels = np.concatenate((labels),axis=0)

print(face_dataset.shape)
print(face_labels.shape)
