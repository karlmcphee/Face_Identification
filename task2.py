import Phase2Task1and2
import task3_4
import sys
import numpy as np


if len(sys.argv) < 4:
    print("Please add the arguments feature model, subject number, semantic number, \
    k, and dimensionality technique (ex: python task1 elbp 1 4 svd)")
    sys.exit()

model= sys.argv[1].upper()
try:
    int(sys.argv[2])
except ValueError:
    print("Please enter a subject number for argument 3")
    sys.exit()
subj = int(sys.argv[2])
if model != "ELBP" and model != "CM" and model != "HOG":
    print("Please enter a correct type model (ex: python task1 elbp 1 4 svd)")
    sys.exit()

k = sys.argv[3]

reductionMethod = sys.argv[4].upper()
fname = model + "-" + str(subj) + "-" + k + '-' + reductionMethod + ".txt"
wname = 'weights-' + model + "-" + str(subj) + "-" + k + reductionMethod + ".txt"

#Two files are being written to: one with the ordered weights and one with the dimensionality reduction matrices

f = open(wname, "w")
if reductionMethod == 'PCA':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 2, "type", subj, reductionMethod.upper(), int(k))
    np.savetxt(fname, np.array(np.real(n2)).transpose(), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'SVD':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 2, "type", subj, reductionMethod.upper(), int(k))
    np.savetxt(fname, np.array(np.real(n2)).transpose(), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'KMEANS':
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 2, "type", subj, 'K-MEANS', int(k))
    np.savetxt(fname, np.array(np.real(n2)), fmt='%d')
    f.write(str(n1))
    f.close()     
elif reductionMethod == 'LDA':    
    n1, n2 = Phase2Task1and2.topKLatentSemantics(model, 2, "type", subj, 'LDA', int(k))
    np.savetxt(fname, np.array(np.real(n2)), fmt='%d')
    f.write(str(n1))
    f.close()     
else:
   print("Please enter the correct decomposition technique (ex: python task1 elbp cc 4 svd)")