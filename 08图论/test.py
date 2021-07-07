import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
Matrix = np.array(
    [
        [ 1, 0, 0, 1, 0],  # e
        [0, 1, 0, 0, 1],  # a
        [1, 0, 1, 0, 0],  # b
        [0, 1, 0, 1, 0],  # c
        [0,  0, 1, 0, 1],  # d
    ]
)
Matrix = np.array(random.randint((2),size=(9,9)))
G=[]
# #20是生成的图的个数
# for k in range (20):
#     G.append(nx.Graph())
#     for i in range(len(Matrix)):
#         for j in range(len(Matrix)):
#             if(Matrix[i][j] == 1):
#                 G[-1].add_edge(i, j)
#     nx.draw(G[-1])
#     plt.savefig(str(k)+'.pdf')
#     # 最后一行不能删
#     plt.show()

G.append(nx.Graph())
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
            G[-1].add_edge(i, j)
nx.draw(G[-1])
plt.show()