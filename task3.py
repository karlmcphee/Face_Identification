import task3_4
import sys
import Phase2Task1and2
import numpy as np

if len(sys.argv) < 4:
	print("Please add the arguments feature model, semantic number and dimensionality technique (ex: CM 5 LDA)")
	sys.exit()

model = sys.argv[1].upper()
k = int(sys.argv[2])
technique = sys.argv[3].upper()

if model != "ELBP" and model != "CM" and model != "HOG":
    print("Please enter a correct type model")
    sys.exit()

images = task3_4.loadImages('./all/')
category = 'type'
similarity_matrix = task3_4.createSimilarityMatrix(images, model, category)
file_name = model + '_' + category + '.csv'
similarity_matrix.to_csv(file_name)

if model == 'KMEANS':
    pairs, u, z = task3_4.Kmeans(similarity_matrix, k)
    fname = str(k) + '_' + model + '_' + category + '_kmeans.txt'
    np.savetxt(fname, z, fmt='%1.3f')
elif model == 'LDA':
    pairs, model, z = task3_4.LDA(similarity_matrix, k)
    fname = str(k) + '_' + model + '_' + category + '_lda.txt'
    np.savetxt(fname, np.array(z)*100, fmt='%1.3f')
elif model == 'SVD':
    n = Phase2Task1and2.SVD_decomposition(similarity_matrix, k)
    n = n.transpose()
    fname = str(k) + '_' + model + '_' + category + '_svd.txt'
    np.savetxt(fname, n, fmt='%1.3f')
else:
    pca = Phase2Task1and2.PCA(similarity_matrix, k)
    pca.findLatentFeatures()
    latentFeatures = pca.project()
    fname = str(k) + '_' + model + '_' + category + '_pca.txt'
    np.savetxt(fname, np.array(pca.PrincipalComponents*1000), fmt='%1.3f')
