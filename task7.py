
    checker2 = sorted(checker, key=checker.get, reverse=True)
    print(checker2[0])

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
    print("Matching subject:")
    checker = {}
    for i in range(0, min(len(dists), int(k1))):
        nn = dists[i][1].split('-')[2]
        if nn not in checker:
            checker[nn] = 1
        else:
            checker[nn]+= 1
    checker2 = sorted(checker, key=checker.get, reverse=True)
    print(checker2[0])

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
    print("Matching subject:")
    checker = {}
    for i in range(0, min(len(dists), int(k1))):
        nn = dists[i][1].split('-')[2]
        if nn not in checker:
            checker[nn] = 1
        else:
            checker[nn]+= 1
    checker2 = list(checker)
    checker2 = sorted(checker, key=checker.get, reverse=True)
    print(checker2[0])

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
    print("Matching subject:")
    checker = {}
    for i in range(0, min(len(dists), int(k1))):
        nn = dists[i][1].split('-')[2]
        if nn not in checker:
            checker[nn] = 1
        else:
            checker[nn]+= 1
    checker2 = list(checker)
    checker2 = sorted(checker, key=checker.get, reverse=True)
    print(checker2[0])