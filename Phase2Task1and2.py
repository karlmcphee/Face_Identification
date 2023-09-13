from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature.texture import local_binary_pattern as lbp
from sklearn.preprocessing import StandardScaler
import os, sys
from collections import defaultdict
from scipy import ndimage
from skimage.feature import hog
from task3_4 import LDA, Kmeans
import pandas as pd

IMAGES_PATH = './all'

# -------------------------------------------------
# -------------- Utility functions ----------------
# -------------------------------------------------

# subject is the number of the subject
# returns a dictionary which maps type of image to image arrays
def getImagesOfSubject(subject):

    imagesOfSubject = defaultdict(list)
    
    for file in os.listdir(IMAGES_PATH):
        if '.png' in file:
            file = file.replace('.png', '')
        
        split = file.split('-')
        
        if int(split[2]) == subject:
            imageType = split[1]
            imArray = np.array(Image.open(IMAGES_PATH + '/' + file + '.png'))
            imagesOfSubject[imageType].append(imArray)   
    return imagesOfSubject

# type is the string of the type of the images
# returns a dictionary which maps subject in the image to image arrays of specified type
def getImagesOfType(imageType):

    imagesOfType = defaultdict(list)
    
    for file in os.listdir(IMAGES_PATH):
        if '.png' in file:
            file = file.replace('.png', '')
        
        split = file.split('-')
        
        if split[1] == imageType:
            subject = split[2]
            imArray = np.array(Image.open(IMAGES_PATH + '/' + file + '.png'))
            imagesOfType[int(subject)].append(imArray)

    return imagesOfType

# this function produces the average images from the results of getImagesOfType or 
# getImagesOfSubject for each type or subject and returns it in the form of a dictionary
# mapped from labels (subject or type) to (64*64) arrays representing 
# the respective average images
# 
# images is a dictionary of the form returned by getImagesOfType or getImagesOfSubject

def getAverageImages(images):
    # labels holds the subject IDs for task 1 and the image types for task 2
    labels = images.keys()
    
    # combined holds the result of the average of the pixel values all the images
    # of a type or subject
    combined = {}
    
    
    def combine(label):
        asNDArray = np.array(images[label])
        shape = asNDArray.shape
        combined[label] = asNDArray.reshape(shape[0], shape[1]*shape[2]).mean(axis=0).reshape(64, 64)
    
    for label in labels:
        combine(label)
    
    return combined


# --------------------------------------------------------
# -------------- Features (CM, ELBP, HOG) ----------------
# --------------------------------------------------------

# Color moments (mean, standard deviation, skewness)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def color_moments(imID):
    result = []
    # for splitting the image into 8x8 windows
    def split_8x8(pic):
        resultant = []

        for row in range(0, 57, 8):
            for column in range(0, 57, 8):
                resultant.append(pic[row:row+8, column:column+8])
        resultant = np.array(resultant)

        return resultant
    
    for block in split_8x8(imID):
        mean = np.mean(block)
        sd = np.std(block)
        skewness = skew(block.flatten())
        
        result.append((mean, sd, skewness))
    
    result = np.array(result, dtype=object)
    result.reshape(8, 3, 8)
    return result

def cm_analysis(data, shift_skewness=False):
    # get the color moments and flatten them
    for image in data:
        cm = color_moments(data[image])
        if shift_skewness:
            cm[:,2] += 10
        data[image] = cm
        data[image] = data[image].flatten()
    return data

# Extended Local Binary Patterns
def elbp(pic):
    var = ndimage.generic_filter(input=pic, function=np.var, mode='constant', footprint=[[1,1,1],[1,0,1],[1,1,1]])
    
    # uniform lbp
    uniform = lbp(pic, P=8, R=1, method='uniform')
    # filtering uniform lbps to remove values which are non-uniform
    not_uniform = uniform==9
    uniform = np.where(not_uniform==True, 0, uniform)
    
    return np.divide(uniform, var, out=np.zeros_like(uniform), where=var!=0)

def elbp_analysis(data):
    for image in data:
        data[image] = elbp(data[image])
        data[image] = data[image].flatten()
    return data

# Histogram of gradients
def hog_fd(image):
    hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys')
    
    return hog_image

def hog_analysis(data):
    for image in data:
        data[image] = hog_fd(data[image])
        data[image] = data[image].flatten()
    return data
    

# ------------------------------------------------------------
# -------------- Dimensionality Reduction --------------------
# --------- Implementations (PCA, SVD, k-means) --------------
# ------------------------------------------------------------

# Implelmentation of PCA
class PCA:
    
    def __init__(self, A, k):
        self.original_data = A
        self.std_data = StandardScaler().fit_transform(self.original_data)
        self.num_PrincipalComponents = k
        self.PrincipalComponents = None
    
    # A is matrix such that each sample is 1 row and each feature is 1 column
    def findLatentFeatures(self):
        # first we calculate the covariance matrix on the standardised data
        cov = np.cov(self.std_data.T)
        
        # then we calculate the eigvecs and eigvals of the cov matrix
        eigVals, eigVecs = np.linalg.eig(cov)
        
        # transpose the eigenvectors matrix to have each row be 1 eigVector
        eigVecs = eigVecs.T
        
        # now we sort according to the eignevalues in a non-increasing order
        sorted_indices = np.argsort(eigVals)[::-1]
        eigVals, eigVecs = eigVals[sorted_indices], eigVecs[sorted_indices]
        
        # transpose the eigVecs matrix again to make the each PC into a column
        # and store the top-k principal components
        self.PrincipalComponents = eigVecs[0:self.num_PrincipalComponents].T
        
    # Now we use the principal components that were calculated to calculate the projection
    # of the original data onto the new reduced dimension space
    def project(self):
        return np.dot(self.std_data, self.PrincipalComponents)

# Implementation of SVD
def SVD_decomposition(A, k):
    # Standard scaling the data
    A = StandardScaler().fit_transform(A)
    A = A.transpose()
    # eigvals and eigvecs of data transpose mult data
    s, v = np.linalg.eig(A.T @ A)
    
    # sorting s and v according to descending order of s
    sorted_indices = np.argsort(s)[::-1]
    s, V = s[sorted_indices], v[:,sorted_indices]
    s = np.sqrt(s)
    
    # S is diagonal array with diagonals as values in s
    S = np.diag(s)
    
    # returning dimensionally reduced matrix
    return A @ V[:,:k]

'''
def color_moments(imID):
    result = []
    
    # for splitting the image into 8x8 windows
    def split_8x8(pic):
        resultant = []

        for row in range(0, 57, 8):
            for column in range(0, 57, 8):
                resultant.append(pic[row:row+8, column:column+8])
        resultant = np.array(resultant)

        return resultant
    
    for block in split_8x8(imID):
        mean = np.mean(block)
        sd = np.std(block)
        skewness = skew(block.flatten())
        
        result.append((mean, sd, skewness))
    
    result = np.array(result, dtype=object)
    result.reshape(8, 3, 8)
    return result
'''

# -------------------------------------------------
# -------------- Driver Functions -----------------
# -------------------------------------------------

def getTopKLatentSemanticsPairs(k, latentFeatures):
    # semanticRanks is a dictionary mapping from the latent semantic (L1, L2, etc) to
    # the ordered subject-weight or type-weight tuples
    semanticsRanks = defaultdict(list)
    
    for latentSemantic_idx in range(0, k):
        sortedByCurrSemantic = sorted(latentFeatures.items(),
                                      key=lambda x: x[1][latentSemantic_idx], 
                                      reverse=True)
        
        for labelFeaturePair in sortedByCurrSemantic:
            label = labelFeaturePair[0]
            labelWeight = labelFeaturePair[1][latentSemantic_idx]
            
            semanticsRanks['L' + str(latentSemantic_idx + 1)].append((label, labelWeight))
    
    return semanticsRanks

# featureModel is one of 'CM', 'ELBP' or 'HOG'
# task is either 1 or 2, signifying task 1 or 2 from the phase 2 requirements doc
# imageType is X (image type) 
# subjectID is Y (subject ID)
# reduction_technique is one of 'PCA', 'SVD', 'LDA', 'K-MEANS'
# k is the number of top latent semantics returned

def topKLatentSemantics(featureModel, task, imageType, subjectID, reduction_technique, k):
    data = None
    if task == 1:
        # get the images of type imageType
        data = getImagesOfType(imageType)

    
    elif task == 2:
        # get the images of the specified subject
        data = getImagesOfSubject(subjectID)
    
    # get the average images
    data = getAverageImages(data)
    if featureModel == 'CM':
        # Remove negative values from skewness if using LDA for reduction
        if reduction_technique == 'LDA':
            data = cm_analysis(data, shift_skewness=True)
        else:
            data = cm_analysis(data)
  
    elif featureModel == 'ELBP':
        data = elbp_analysis(data)
    elif featureModel == 'HOG':
        data = hog_analysis(data)
    # dictionary mapped from subject/type keys to latent features values
    latentFeatures = None
    
    if reduction_technique == 'PCA':
        # A is the data array to be passed to PCA
        A = np.array(list(data.values()))
        
        pca = PCA(A=A, k=k)
        pca.findLatentFeatures()
        latentFeatures = pca.project()
        i = 0
        result = {}
        
        for image in data: 
            result[image] = latentFeatures[i]
            i += 1
        
        latentFeatures = result
        comps = pca.PrincipalComponents*1000
        
    elif reduction_technique == 'SVD':
        A = np.array(list(data.values()), dtype=float)
        latentFeatures = SVD_decomposition(A, k)
        comps = latentFeatures
        i = 0
        result = {}
        
        for image in data: 
            result[image] = latentFeatures[i]
            i += 1
        
        latentFeatures = result
    
    else:
        A = np.array(list(data.values()), dtype=float)
        A = A.transpose()
        columns = list(data.keys())
        A = pd.DataFrame(A, columns=columns)
        semanticPairs = {}
        
        if reduction_technique == 'LDA':
            enums, model, lats = LDA(A, k)
            for index, topic in enumerate(enums):
                semanticPairs['L' + str(index + 1)] = topic
        if reduction_technique == 'K-MEANS':
            A = A.transpose()
            enums, n, lats = Kmeans(A, k)
            for index, topic in enumerate(enums):
                semanticPairs['L' + str(index + 1)] = topic
        return (semanticPairs, np.array(lats)*100)
    return (getTopKLatentSemanticsPairs(k, latentFeatures), comps)