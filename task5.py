import Phase2Task1and2
import numpy as np
from pymongo import MongoClient
import math
import os
import sys
import cv2
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from scipy.stats import wasserstein_distance



if len(sys.argv) < 4:
    print("Please add the arguments file name, latent file, and number of matching images to the query (python task5 file-name latent-semanticsname 6)")
    sys.exit()

img = sys.argv[1]
sems = sys.argv[2]
k1 = sys.argv[3]

try:
    conn = MongoClient('localhost', 27017)
    print("Connected successfully!!!")
except:  
    print("Could not connect to MongoDB")
db = conn.database
collection = db.my_feature_collection4
cursor = collection.find()
x = 0
img2 = 'none'
name = "none"
record2 = ''
#first searching in database, then direct filepath
for record in cursor:
    x+=1
    if img == record['name']:
        name = record['name']
        break
if name == "none":
    for filename in os.listdir('./'):
        if filename == img:
            img2 = cv2.imread(img)
            name = filename
if name == "none":
    print("File not found, sorry")
    sys.exit()
sems2 = sems.split("-")
if len(sems2[1]) > 2 or sems2[1] == 'cc':
    atr = "type"
else:
    atr = "subj"
ref_data = []
ref_names = []
#dividing images by 255 because this is how they're currently put in the database
if img2 != 'none':
    img2 = img2/255.0
k = sems2[2]
sems2[3] = sems2[3].split('.')[0]
if sems2[0].upper() == 'CM':
    if img2 != 'none':
        ref_img = Phase2Task1and2.cm_analysis(img2)
    else:
        ref_img = record['color']
    for record in cursor:
        ref_data.append(record['color'])
        ref_names.append(record['name'])
elif sems2[0].upper() == 'ELBP':
    if img2 != 'none':
        ref_img = Phase2Task1and2.elbp_analysis(img2)
    else:
        ref_img = record['lbp']
    for record in cursor:
        ref_data.append(record['lbp'])
        ref_names.append(record['name'])
else:
    if img2 != 'none':
        ref_img = Phase2Task1and2.hog_analysis(img2)
    else:
        ref_img = record['hist']
    for record in cursor:
        ref_data.append(record['hist'])
        ref_names.append(record['name'])

if sems2[3].upper() == 'KMEANS':
    centroids = np.loadtxt(sems, dtype=float)
    ref_img = np.matmul(np.array(ref_img), centroids.transpose())
    dists = []
    for i in range(0, len(ref_data)):
        ref_data[i] = np.matmul(np.array(ref_data[i]), centroids.transpose())
        distx = np.linalg.norm(ref_data[i] - ref_img, ord=2)        
        dists.append((distx, ref_names[i]))
    dists.sort()
    print("Matching images: ")
    for i in range(0, min(len(dists), int(k1))):
        print(dists[i][1])

elif sems2[3].upper() == 'PCA':
    A = np.array(ref_img)
    pca = np.loadtxt(sems, dtype=float) 
    ref_img = np.matmul(np.array(ref_img), pca.transpose())
    dists = []
    for i in range(0, len(ref_data)):
        ref_data[i] = np.matmul(np.array(ref_data[i]), pca.transpose())
        distx = np.linalg.norm(ref_data[i] - ref_img, ord=2)        
        dists.append((distx, ref_names[i]))
    dists.sort()
    print("Matching images: ")
    for i in range(0, min(len(dists), int(k1))):
        print(dists[i][1])

elif sems2[3].upper() == 'SVD':
    A = np.array(ref_data)
    svd = np.loadtxt(sems, dtype=float) 
    ref_img = np.matmul(np.array(ref_img), svd.transpose())
    dists = []
    for i in range(0, len(ref_data)):
        ref_data[i] = np.matmul(np.array(ref_data[i]), svd.transpose())
        distx = np.linalg.norm(ref_data[i] - ref_img, ord=2)        
        dists.append((distx, ref_names[i]))
    dists.sort()
    print("Matching images: ")
    for i in range(0, min(len(dists), int(k1))):
        print(dists[i][1])
    
else:
    A = np.array(ref_data)
    lda = np.loadtxt(sems, dtype=float) 
    ref_img = np.matmul(np.array(ref_img), lda.transpose())
    dists = []
    for i in range(0, len(ref_data)):
        ref_data[i] = np.matmul(np.array(ref_data[i]), lda.transpose())
        distx = np.linalg.norm(ref_data[i] - ref_img, ord=2)        
        dists.append((distx, ref_names[i]))
    dists.sort()
    print("Matching images: ")
    for i in range(0, min(len(dists), int(k1))):
        print(dists[i][1])