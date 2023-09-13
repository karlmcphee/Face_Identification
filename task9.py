#!/usr/bin/env python
# coding: utf-8

# In[61]:


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

IMAGE_PATH = './all/'

def read_file(file_name):
    print('read file',file_name)
    try:
        df = pd.read_csv('./' + file_name, header=None)
        df = df.loc[1:,1:].astype(float).to_numpy()
        return df
    except:
        print('read file failed, plz typing the write file name')
        return []


def createSimilarityGraph(similarity_matrix, n):
    print('[Task9] generate createSimilarityGraph...')
    rank_similarity = []
    columns = len(similarity_matrix[0])
    if n > columns:
        return 
    
#     print('similarity_matrix', similarity_matrix)
    for i in range(len(similarity_matrix)):
        temp = similarity_matrix[i]
        sorted_temp = sorted(temp, reverse=True)
        target = sorted_temp[n]
        for j in range(len(similarity_matrix[i])):
            x = similarity_matrix[i][j]
            if x - target>= 0 and i != j:
                rank_similarity.append((i+1, j+1))
#         rank_similarity.append(list(map(lambda x: x if x-target >= 0 else 0 , temp)))
    g = nx.DiGraph()
    g.add_edges_from(rank_similarity)
    nx.draw(g, with_labels=True)
    plt.savefig('ppr_network.png', dpi=300, bbox_inches='tight')
    image_PIL = Image.open(r'./ppr_network.png')
    image_PIL.show()
    print('[Task9] createSimilarityGraph success')
    # plt.show()

def personalized_page_rank(similarity_matrix, m, sID_1, sID_2, sID_3):
    print('[Task9] calculate personalized page rank...')
    subjects = [sID_1 - 1, sID_2 - 1, sID_3 - 1]
    columns = len(similarity_matrix[0])
    if m > columns:
        return 
    # find the connection in input subject IDs
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if i in subjects or j in subjects:
                continue
            elif i!=j:
                similarity_matrix[i][j] = 0
            else:
                continue
    # create weights
    weight_similarity = similarity_matrix.mean(axis=1)
    norm = np.linalg.norm(weight_similarity) 
    weight_similarity = weight_similarity / norm
    weights = {}
    for index in range(len(weight_similarity)):
        weight = weight_similarity[index]
        weights[index] = weight
    # create graph
    G = nx.from_numpy_matrix(np.array(similarity_matrix))
    # calculate pagerank
    ppr = PageRank(G, True, weights)
    ppr.rank()
    ppr_ranks = ppr.ranks
    ppr_ranks = sorted(ppr_ranks.items(), key=lambda item: item[1], reverse=True)
    for i in range(len(ppr_ranks)):
        index, value = ppr_ranks[i]
        ppr_ranks[i] = (index + 1, value)
    
    text_file = open("task9_ppr_output.txt", "wt")
    res = ''
    for obj in ppr_ranks[3:3 + m]:
        res+="subjectID: {} : ppr value: {} \n".format(obj[0], obj[1])
    n = text_file.write(res)
    text_file.close()
    print('[Task9] personalized_page_rank m most significant nodes of input subjects')
    print(res)
    print('[Task9] calculate personalized page rank success')
    return res

# calculate value from package
#     ppr1 = nx.pagerank(G, personalization=weights)
#     ppr1 = dict(sorted(ppr1.items(), key=lambda item: item[1], reverse=True))
#     print('ppr1', ppr1)
#     print('ppr1', max(ppr1.values()))
    
    pass
    
class PageRank:
    def __init__(self, graph, directed, weights=None):
        self.graph = graph
        self.V = len(self.graph)
        self.d = 0.85
        self.directed = directed
        self.ranks = dict()
        self.weights = weights
    
    def rank(self):
        # initial node value
        for key, node in self.graph.nodes(data=True):
            if self.directed:
                # ppr
                if self.weights != None:
                    self.ranks[key] = self.weights[key]
                else:
                #  Assign uniform personalization vector if not given
                    self.ranks[key] = 1/float(self.V)
            else:
                self.ranks[key] = node.get('rank')
        # iterate and update ppr
        for _ in range(10):
            for key, node in self.graph.nodes(data=True):
                rank_sum = 0
                if self.directed:
                    neighbors = self.graph.edges(key)
                    for n in neighbors:
                        outlinks = len(self.graph.edges(n[1]))
                        if outlinks > 0:
                            rank_sum += (1 / float(outlinks)) * self.ranks[n[1]]
                else: 
                    neighbors = self.graph[key]
                    for n in neighbors:
                        if self.ranks[n] is not None:
                            outlinks = len(self.graph.neighbors(n))
                            rank_sum += (1 / float(outlinks)) * self.ranks[n]
            
                # actual page rank compution
                self.ranks[key] = ((1 - float(self.d)) * (1/float(self.V))) + self.d*rank_sum

        return self.ranks

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("[Task9] Please add the correct arguments for file_name of similarity, n, m, subjectID1, subjectID2, subjectID3")
        sys.exit()
    df = read_file(sys.argv[1])
    if len(df) > 0:
        createSimilarityGraph(df, int(sys.argv[2]))
        personalized_page_rank(df, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))