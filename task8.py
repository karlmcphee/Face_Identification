from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import sys

def creates_similarity_graph(adjacency_matrix, n):

    num = len(adjacency_matrix[1])

    #sort values in each row, store in orderindex ndarray
    orderindex = np.zeros((num, num))
    for i in range(num):
        orderindex[i] = np.argsort(adjacency_matrix[i])
    
    mat = copy.deepcopy(adjacency_matrix)
    #remain the n most similar subjects, set others to 0
    for i in range(num):
        for j in range(num):
            hasValue = False
            for k in range(2, n+2):
                if j == orderindex[i][num - k]:
                    hasValue = True
                    break
                else:
                    hasValue = False

            if hasValue == False:
                mat[i][j] = 0.0

    #print(adjacency_matrix)
    G = nx.from_numpy_matrix(mat)  
    nx.draw(G, with_labels=True)
    plt.show()

    return G


def ascos(G, c=0.9, max_iter=100, remove_self=False):
  
  #find neighbor nodes
  node_ids = G.nodes()
  node_id_find = { }
  for i, n in enumerate(node_ids):
    node_id_find[n] = i

  nb_ids = [G.neighbors(n) for n in node_ids]
  nbs = [ ]
  
  # store neighbor nodes ID
  for nb_id in nb_ids:
    nbs.append([node_id_find[n] for n in nb_id])
  del(node_id_find)

  n = G.number_of_nodes()
  sim = np.eye(n)
  sim_old = np.zeros(shape = (n, n))

  for iter_ctr in range(max_iter):

    #if converge, stop operation
    if _is_converge(sim, sim_old, n, n):
      break
    sim_old = copy.deepcopy(sim)

    #calculate ascos++ values
    for i in range(n):
      dg = G.degree(weight='weight')
      degree_values = [v for k, v in dg]
      w_i = degree_values[i]

      for j in range(n):
        if i == j:
            continue
        
        s_ij = 0.0
        for n_i in nbs[i]:
          node1 = list(node_ids)
          node2 = list(node_ids)
          w_ik = G[node1[i]][node2[n_i]]['weight'] if 'weight' in G[node1[i]][node2[n_i]] else 1
          s_ij += float(w_ik) * (1 - math.exp(-w_ik)) * sim_old[n_i, j]
        sim[i, j] = c * s_ij / w_i if w_i > 0 else 0 

  #remove itself in matrix
  if remove_self:
    for i in range(n):
      sim[i,i] = 0

  return sim


def _is_converge(sim, sim_old, row, col, eps=1e-4):
  for i in range(row):
    for j in range(col):
      if abs(sim[i,j] - sim_old[i,j]) >= eps:
        return False
  return True


def find_significant_subjects(matrix, m):

    num = len(matrix[1])

    #sort values in each row, store in orderindex ndarray
    subject_sum = np.zeros(num)
    for i in range(num):
      temp = 0
      for j in range(num):
        temp += matrix[i][j]
      subject_sum[i] = temp

    orderindex = np.zeros(num)
    for i in range(num):
      orderindex = np.argsort(subject_sum)
    
    output = [""] * m
    for i in range(m):
      output[i] = "subject " + str(orderindex[i] + 1)
    print("The most significant " + str(m) + " subjects:")
    print(output)


if __name__ == "__main__":

    if len(sys.argv) < 4:
      print("[Task8] Please enter the filename of similarity, n, m")
      sys.exit()

    filename = sys.argv[1]
    if len(filename) > 0:
      #get data
      mydata = genfromtxt(filename, delimiter=',')
      adjacency = mydata[1:,1:]
      Graph = creates_similarity_graph(adjacency, int(sys.argv[2]))

      sim = ascos(Graph, remove_self=True)
      find_significant_subjects(sim, int(sys.argv[3]))