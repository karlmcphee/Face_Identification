import Phase2Task1and2
import task3_4
import sys
import string
import numpy as np


if len(sys.argv) < 4:
    print("Please add the arguments feature model, type, semantic number, \
    k, and dimensionality technique (ex: python task1 elbp cc 4 svd)")
    sys.exit()

model= sys.argv[1].upper()
imageType = sys.argv[2]

if model != "ELBP" and model != "CM" and model != "HOG":
    print("Please enter a correct type model (ex: python task1 elbp cc 4 svd)")
    sys.exit()
validImageTypes = [
    "cc",
    "con",
    "detail",
    "emboss",
    "jitter",
    "neg",
    "noise01",
    "noise02",
    "original",
    "poster"
    "rot",
    "smooth",
    "stipple"
    ]

if imageType not in validImageTypes:
    print("Please enter the right image imageType")
    sys.exit()

k = sys.argv[3]
reductionMethod = sys.argv[4].upper()

#file names for future tasks
fname = model + "-" + imageType + "-" + k + '-' + reductionMethod + ".txt"
wname = 'weights-' + model + "-" + imageType + '-' + k +"-" + reductionMethod + ".txt"
f = open(wname, "w")
if reductionMethod == 'PCA':
    #n1 is the weight products from task1, n2 is the dimensionality-reducing latent semantics
    #n1 will be saved as weight files for analysis, but are not needed for the other activities
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 1, imageType, '0', reductionMethod.upper(), int(k))
    np.savetxt(fname, np.array(np.real(n2)).transpose(), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'SVD':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 1, imageType, '0', reductionMethod.upper(), int(k))
    np.savetxt(fname, np.array(np.real(n2)).transpose(), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'KMEANS':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 1, imageType, '0', 'K-MEANS', int(k))
    np.savetxt(fname, np.real(n2), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'LDA':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 1, imageType, '0', 'LDA', int(k))
    np.savetxt(fname, np.array(np.real(n2)), fmt='%d')
    f.write(str(n1))
    f.close()     
else:
   print("Please enter the correct decomposition technique (ex: python task1 elbp cc 4 svd)")