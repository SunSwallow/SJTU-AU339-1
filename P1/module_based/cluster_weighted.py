import community as community_louvain
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from networkx.drawing.nx_pydot import to_pydot
from matplotlib.font_manager import *
import matplotlib


# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus']=False

# 定义图的节点和边

nodes = [str(i) for i in range(1,43)]
edges = [('1', '2', 1026), ('4', '1', 402), ('2', '3', 1026), ('2', '7', 398),
             ('3', '11', 450), ('12', '4', 181), ('4', '5', 254), ('10', '9', 289),
             ('10', '11', 235), ('10', '16', 163), ('5', '6', 414), ('5', '13', 148),
             ('6', '20', 382), ('6', '7', 253), ('7', '8', 252), ('7', '14', 219),
             ('11', '17', 163), ('8', '9', 289), ('8', '15', 245), ('9', '23', 396),
             ('12', '13', 306), ('12', '18', 272), ('13', '19', 199), ('16', '17', 199),
             ('17', '25', 235), ('15', '14', 360), ('15', '22', 217), ('14', '21', 219),
             ('19', '20', 344), ('19', '18', 332), ('18', '26', 144), ('20', '21', 308),
             ('23', '24', 432), ('23', '28', 198), ('23', '22', 108), ('22', '21', 414),
             ('25', '42', 810), ('25', '24', 109), ('24', '29', 284), ('21', '32', 432),
             ('27', '26', 326), ('27', '31', 316), ('26', '30', 271), ('28', '29', 414),
             ('28', '33', 235), ('29', '34', 181), ('34', '33', 418), ('30', '39', 432),
             ('30', '31', 342), ('31', '32', 648), ('31', '36', 253), ('33', '32', 486),
             ('33', '38', 252), ('32', '37', 235), ('35', '36', 180), ('35', '40', 180),
             ('36', '37', 648), ('37', '38', 468), ('37', '41', 184), ('42', '41', 1063),
             ('40', '41', 792), ('40', '39', 198)]

# 定义有向图
G = nx.MultiGraph()

# 添加节点和边
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges)

# best partition
partition = community_louvain.best_partition(G)
partition_array = np.array(list(partition.values()))
num_clusters = len(np.unique(partition_array))

print("Number of Clusters: %d" % (len(np.unique(partition_array))))
np.save('cluster_weighted', list(partition.values()))