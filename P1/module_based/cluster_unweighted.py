import numpy as np

def Q(array, cluster):
	# 总边数
	m = sum(sum(array)) / 2
	k1 = np.sum(array, axis=1)
	k2 = k1.reshape(k1.shape[0], 1)
	# 节点度数积
	k1k2 = k1 * k2
	# 任意两点连接边数的期望值
	Eij = k1k2 / (2 * m)
	# 节点v和w的实际边数与随机网络下边数期望之差
	B = array - Eij
	# 获取节点、社区矩阵
	node_cluster = np.dot(cluster, np.transpose(cluster))
	results = np.dot(B, node_cluster)
	# 求和
	sum_results = np.trace(results)
	# 模块度计算
	Q = sum_results / (2 * m)
	print("Q:", Q)
	return Q


if __name__ == '__main__':
	adj_matrix = np.load('adj_matrix.npy')

	# 节点类别分别是1,2,2
	cluster = np.eye(42)

	Q_value_save = Q(adj_matrix, cluster)

	time = 1
	while time > 0:
		time = 0
		for i in range(42):
			this_cluster = cluster[i]
			linked_index = np.where(adj_matrix[i] == 1)

			for near_node in linked_index:
				if len(np.where((this_cluster == cluster[near_node]) == False)[0]) != 0:
					# 分类不同
					cluster_copy = cluster.copy()
					cluster_copy[near_node] = cluster_copy[i]

					Q_value = Q(adj_matrix, cluster_copy)
					if Q_value > Q_value_save:
						Q_value_save = Q_value
						cluster = cluster_copy
						time += 1
					else:
						# 不大于
						pass
	cluster_find = cluster
	cluster_buffer = []
	cluster_flag = []
	for cluster in range(len(cluster_find)):
		in_flag = False
		for cluster_saved in range(len(cluster_buffer)):
			if len(np.where((cluster_buffer[cluster_saved] == cluster_find[cluster]) == False)[0]) == 0:
				# 已经在里面了
				in_flag = True
				cluster_flag.append(cluster_saved)
				break
		if in_flag == False:
			cluster_buffer.append(cluster_find[cluster])
			cluster_flag.append(len(cluster_buffer) - 1)
	np.save('cluster_unweighted', cluster_flag)
