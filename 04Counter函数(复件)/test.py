# import math
# import collections
# import numpy as np
# node_index =['a','b','c','d','e','f','g','h','i','k']
# Sra_ind = list(np.random.permutation(10))
# print(Sra_ind)
# shu_node_ind = node_index[Sra_ind]
# print(shu_node_ind)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
 
G = nx.Graph()
Matrix = np.array(
    [
        [0, 1, 1, 1, 1, 1, 0, 0],  # 0
        [0, 0, 1, 0, 1, 0, 0, 0],  # 1
        [0, 0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 1, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0, 1, 0, 0],  # 4
        [0, 0, 1, 0, 0, 0, 1, 1],  # 5
        [0, 0, 0, 0, 0, 1, 0, 1],  # 6
        [0, 0, 0, 0, 0, 1, 1, 0]   # 7
    ]
)
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if Matrix[i][j] == 1:
            G.add_edge(i, j)
# for i in range(3):
#     for j in range(1):
#         G.add_edge(i, j)
nx.draw(G)
plt.show()
Sra_ind = list(np.random.permutation(8))#随即生成八个序列
print(Sra_ind)
shu_node_ind = Matrix[Sra_ind]#按照序列将原来的邻接矩阵重组
print(shu_node_ind)