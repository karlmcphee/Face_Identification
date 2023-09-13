#from sklearn import datasets
#from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import math
import sys
import os
import cv2
from skimage.feature.texture import local_binary_pattern as lbp
import Phase2Task1and2
from scipy import ndimage
from skimage.feature import hog, local_binary_pattern

images = []
name = []
for filename in os.listdir('./all'):
    name.append(filename)
    img = cv2.imread(os.path.join('./all',filename), 0)
    ff = filename.split('-')
    
    if img is not None:
        images.append(img)
print(len(images))
rng = np.random.RandomState(0)
l1 = [[0]*8]*8
l2 = [[0]*8]*8
l3 = [[0]*8]*8
l4 = [[0]*8]*8
l5 = [[0]*8]*8
l11 = []
l12 = []
l13 = []
hogs = []
lbps = []


from pymongo import MongoClient

try:
    conn = MongoClient('localhost', 27017)
    print("Connected successfully!!!")
except:  
    print("Could not connect to MongoDB")
db = conn.database
collection = db.my_feature_collection5
print(len(images))
nn = 0
nnn = 0
for img in images:
    img = img/255.0
    for i in range(0, 8):
        for j in range(0, 8):
            n = 0
            for k in range(0, 8):
                for l in range(0, 8):
                    n += img[i*8+k][j*8+l]
            n = n/64
            l1[i][j] = n
            m = 0
            skews = 0
            for k in range(0, 8):
                for l in range(0, 8):
                    m += (n-img[i*8+k][j*8+l])**2
                    skews += (n-img[i*8+k][j*8+l])**3
            m = math.sqrt(m/64)
            skews = (m/64)**(1.0/3)
            l1[i][j] = (n, m, skews)
            l11.append(n)
            l12.append(m)
            l13.append(skews)
    ll1 = (np.array(l1)).flatten()
    hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys')
    hog_image = hog_image.flatten()
    hogs.append(hog_image)
    var = ndimage.generic_filter(input=img, function=np.var, mode='constant', footprint=[[1,1,1],[1,0,1],[1,1,1]])
    uniform = lbp(img, P=8, R=1, method='uniform')
    # filtering uniform lbps to remove values which are non-uniform
    not_uniform = uniform==9
    uniform = np.where(not_uniform==True, 0, uniform)
    n5 = np.divide(uniform, var, out=np.zeros_like(uniform), where=var!=0)
    n5 = n5.flatten()
    lbps.append(n5)
#    emp_rec1 = {
#        "name": name[nnn],
#        "id": nn,
#        "color": ll1.tolist(),
#        "lbp": n5.tolist(),
#        "hist": hog_image.tolist()
#        }
#    nn+=1
#    nnn+=1
#    rec_id1 = collection.insert_one(emp_rec1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
scaler.fit_transform(np.array(l11).reshape(-1, 1))
scaler2.fit_transform(np.array(l12).reshape(-1, 1))
scaler3.fit_transform(np.array(l13).reshape(-1, 1))
print(l11)

cursor = collection.find()
for record in cursor:
    print(record)
    sys.exit()
