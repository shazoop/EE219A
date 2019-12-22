import torch
import numpy as np
from skimage.draw import random_shapes
import cv2
import random

#Function to generate a dimxdim image of a random shape of a size between given bounds
def shape_gen(dim, minimum,maximum):
    x = random_shapes((dim, dim), max_shapes=1, multichannel=False, intensity_range=(0,0),min_size = minimum, max_size=maximum)[0]
    x = (-1*x+255)/255
    return(x/np.linalg.norm(x))

#For a given collection of start images with (n,k,k), with n images of dimension k by k. Trans_range is one-sided,
#so trans_range = 4 allows translations of 4 up or down, 4 left or right. Same rules apply for rot_center, which how 
#much the center of rotation can shift to one-side both up/down. rot_angle is maximum angle in DEGREES
def rand_trans(data, trans_range, trans_mul = 1):  #rot_center=0.0, rot_angle=0.0
    n, row, col = data.shape
#     Index = [random.randint(0,1) for i in range(n)] #randomly choose either translation or rotation
    
    def M_gen(i):
#         if i == 0:
        hor = trans_mul*random.choice([-1,1]) 
        ver = trans_mul*random.choice([-1,1]) 
        x = np.float32([[1,0,hor],[0,1,ver]]) #create translation matrix
#         else:
#             center = (row/2+random.randint(-rot_center,rot_center),col/2+random.randint(-rot_center,rot_center))
#             angle = random.uniform(-rot_angle,rot_angle)
#             x = cv2.getRotationMatrix2D(center,angle,1) #create rotation matrix
        return(x)
    
    Z = np.array([M_gen(i) for i in range(n)])
    
    return(Z)

#Apply TRANSFORM to each image in DATA. DATA should be np.array of shape (n,k,k)
def image_gen(data, transform, movie_len):
    X = np.expand_dims(data,1)
    imgDim = X.shape[-1]
    for j in range(movie_len-1):
        Y = np.array([cv2.warpAffine(X[i,j],transform[i],(imgDim,imgDim)) for i in range(data.shape[0])])
        Y = np.expand_dims(Y,1) #add the second axis, which denotes time, to concatenate the images
        X = np.concatenate((X,Y),axis=1)
    return(X) #reshape so it's easier to work with
