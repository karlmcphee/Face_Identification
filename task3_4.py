import sys
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import stats
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from math import log2

IMAGE_PATH = './all/'

# Color moments
def CM(image):
    moments = np.empty((0))
    len_window = 8
    for i in range(0, len(image), len_window):
        for j in range(0, len(image), len_window):
            sub_image = image[i:i+len_window, j:j+len_window]
            mean = sub_image.mean()
            std = sub_image.std()
            skew = stats.skew(sub_image.flatten())
            moments = np.append(moments, [mean, std, skew])
    return moments

# Extended Local binary patterns
def ELBP(image):
    points = 24
    radius = 3
    elbp = local_binary_pattern(image, points, radius, 'uniform')
    return elbp

# Histograms of oriented gradients
def HOG(image):
    orientation_bins = 9
    cell_size = (8, 8)
    block_size = (2, 2)
    hog_image = hog(image, orientations=orientation_bins, pixels_per_cell=cell_size,
                    cells_per_block=block_size, block_norm='L2-Hys')
    return hog_image

# Load all PNG images in the path
def loadImages(path):
    file_names = os.listdir(path)
    images = {}
    for name in file_names:
        if name.endswith('.png'):
            image = loadImage(path, name)
            images[name] = image
    return images

# load a image given its name
def loadImage(path, id):
    image = plt.imread(path + id)
    return image

# Given the model type, extract features from a image
def featureDescriptors(image, model):
    feature = []
    if model == 'cm8x8':
        feature = CM(image)
    elif model == 'elbp':
        feature = ELBP(image)
    elif model == 'hog':
        feature = HOG(image)
    return feature

# Euclidean distance
def euclideanSimilarity(a, b):
    return 1 / (1 + np.linalg.norm(a - b, ord=2))

def euclideanDistance(a, b):
    return np.linalg.norm(a - b, ord=2)

# Cosine Similarity
def cosineSimilarity(a, b):
    return np.inner(a, b) / (norm(a) * norm(b))

# Create Similarity Matrix
def createSimilarityMatrix(images, model, type):
    matrix = {}
    index_list = set()

    # Extract features from each image
    for key, value in images.items():
        feature = featureDescriptors(value , model)
        name = key.replace('.png', '').split('-')
        image_type = name[1]
        subject_id = int(name[2])
        sample_id = int(name[3])
        # Store features according to the type of the similarity matrix
        if type == 'type':
            if image_type not in matrix.keys():
                matrix[image_type] = {}
            matrix[image_type][(subject_id, sample_id)] = value
            index_list.add(subject_id)
        else:
            if subject_id not in matrix.keys():
                matrix[subject_id] = {}
            matrix[subject_id][(image_type, sample_id)] = value
            index_list.add(image_type)
    index_list = sorted(index_list)

    # Create Similarity Matrix
    column_list = sorted(list(matrix.keys()))
    similarity_matrix = pd.DataFrame(columns = column_list)
    for i in column_list:
        row = {}
        target = matrix[i]
        for j in column_list:
            if i == j:
                row[j] = 1
                continue
            object = matrix[j]
            similarity = computeSimilarity(target, object, index_list)
            row[j] = similarity
        similarity_matrix = similarity_matrix.append(row, ignore_index=True)
    similarity_matrix.index = column_list
    return similarity_matrix

# Compute Similarity by Euclidean distance
def computeSimilarity(a, b, index):
    distance = 0
    count = 0
    samples = range(1, 11)
    for i in index:
        for sample in samples:
            indices = (i, sample)
            if indices not in a.keys() or indices not in b.keys():
                continue
            f1 = a[indices]
            f2 = b[indices]
            distance += euclideanSimilarity(f1, f2)
            count += 1
    distance /= count
    return distance

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
    # eigvals and eigvecs of data transpose mult data
    s, v = np.linalg.eig(A.T @ A)

    # sorting s and v according to descending order of s
    sorted_indices = np.argsort(s)[::-1]
    s, V = s[sorted_indices], v[:,sorted_indices]
    s = np.sqrt(s)

    # S is diagonal array with diagonals as values in s
    S = np.diag(s)

    # U is the product of A with V
    U = A @ V[:,:k] / s[:k]

    # returning the resconstructed-from-decomposition matrix
    return U, S[:k,:k], V.T[:k,:]

# Latent Dirichlet Allocation
def LDA(matrix, k):
    # LDA
    model = LatentDirichletAllocation(n_components=k)
    model = model.fit(matrix)
    transformation = model.transform(matrix).T
    components = model.components_

    # Sort subjects by weight
    subject_weight = []
    for index, row in enumerate(transformation):
        pairs = zip(matrix.columns, row)
        subject_weight.append(sorted(pairs, key=lambda x:x[1], reverse=True))
    # Print latent semantics
    for index, pairs in enumerate(subject_weight):
        print('Latent semantic', index+1)
        for pair in pairs:
            print(pair)

    return subject_weight, model, transformation


# K-means++
def Kmeans(matrix, k):
    points = matrix.values.tolist()
    centroids = []
    # Initial with random centroids
    rand = random.randint(0, len(points)-1)
    centroids.append(points[rand])

    # Find k centroids
    for x in range(1, k):
        ranking = []
        # Calculate the distances between each node and centroids
        for index, point in enumerate(points):
            # Skip the point which is a centroid
            if point in centroids:
                continue
            distance = 0
            for centroid in centroids:
                distance += euclideanDistance(np.array(centroid), np.array(point))
            ranking.append((index, distance))
        # Choose the farthest node as a centroid
        ranking = sorted(ranking, key=lambda x:x[1])
        if len(ranking) == 0:
            for i in range(0, k-1):
                ranking.append((i, i))
        index = ranking[0][0]
        centroids.append(points[index])

    # Recalculate the centroids
    previous_clusters = None
    clusters = {c: [] for c in range(k)}
    # Until the same points are assigned to each cluster in consecutive rounds
    while not previous_clusters or not np.array_equal(clusters, previous_clusters):
        previous_clusters = clusters.copy()
        clusters = {c: [] for c in range(k)}
        # Assign points to the nearst centroids
        for point in points:
            distances = []
            for index, centroid in enumerate(centroids):
                distance = euclideanDistance(np.array(point), np.array(centroid))
                distances.append((index, distance))
            distances = sorted(distances, key=lambda x:x[1], reverse=False)
            index = distances[0][0]
            clusters[index].append(point)

        # Recalculate centroids
        for key, value in clusters.items():
            if not value:
                continue
            centroids[key] = np.mean(value, axis=0)

    # Transform the original matrix
    new_space = np.array(centroids).T
    u, s, vt = SVD_decomposition(new_space, len(new_space))
    transformation = matrix.dot(np.real(u)).T

    # Sort subjects by weight
    subject_weight = []
    for index, row in transformation.iterrows():
        pairs = zip(transformation.columns, row)
        subject_weight.append(sorted(pairs, key=lambda x:x[1], reverse=True))
    # Print latent semantics
    for index, pairs in enumerate(subject_weight):
        print('Latent semantic', index+1)
        for pair in pairs:
            print(pair)
    return u, subject_weight, centroids

if __name__ == "__main__":
    # Create similarity matrix
    images = loadImages(IMAGE_PATH)
    similarity_matrix = createSimilarityMatrix(images, 'cm8x8', 'subject') # 2nd argument: [cm8x8, elbp, hog], 3rd argument: [type, subject]
    similarity_matrix.to_csv('cm8x8_subject.csv')
    # Read a similarity matrix
    similarity_matrix = pd.read_csv('cm8x8_subject.csv', index_col = 0)
    # Do Kmeans or LDA on the similarity matrix
    pairs, u = Kmeans(similarity_matrix, 5)
    pairs, model = LDA(similarity_matrix, 5)

    #pca = PCA(similarity_matrix, 5)
    #pca.findLatentFeatures()
    #print(pca.project())
    #pca.findLatentFeatures()
    #latentFeatures = pca.project()