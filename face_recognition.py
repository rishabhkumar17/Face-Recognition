import numpy as np
import os
import cv2

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2)) # Euclidean distance

def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint, X[i])
        vals.append((d,Y[i]))
        
    vals = sorted(vals)
    
    vals = vals[:k]
    vals = np.array(vals)
    
    new_vals = np.unique(vals[:,1], return_counts = True)
    print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred