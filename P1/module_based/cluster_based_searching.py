import numpy as np
import copy

cluster_find = np.load('cluster_weighted.npy')
cost = np.load('cost.npy')
path = np.load('path.npy')

searching_list = []
cluster_based_paths = []
num_clusters = len(np.unique(np.array(cluster_find)))
for i in range(num_clusters):
	cluster_based_paths.append(np.where(np.array(cluster_find) == i)[0].tolist())

# algorithm parameters
length_min = np.inf
path_min = None
cluster_based_paths_min = copy.deepcopy(cluster_based_paths)
T = 10000.0
np.random.seed(17)

# cluster swap parameters
cluster_index1 = 0
cluster_index2 = 0
cluster_swap = False

# in-cluster swap parameters
flag_3 = False
loc1 = 0
loc2 = 0
loc3 = 0


for i in range(1000000):
	length_sum = 0

	searching_list = []
	for cluster_path in cluster_based_paths:
		searching_list.extend(cluster_path)

	start_point = searching_list[0]
	iter_list = searching_list.copy()
	# 删除中间的点
	while True:
		if len(iter_list) == 0:
			break
		else:
			if len(iter_list) == 1:
				length_sum += cost[start_point][searching_list[0]]
				iter_list = []
			else:
				length_sum += cost[iter_list[0]][iter_list[1]]
				start_point = iter_list[1]
				for each in path[iter_list[0]][iter_list[1]]:
					if (int(each) - 1 in iter_list) and ((int(each) - 1) != iter_list[1]):
						iter_list.remove(int(each) - 1)

	# 如果更新了，print 一下
	if length_sum < length_min:
		length_min = length_sum
		path_min = searching_list.copy()
		cluster_based_paths_min = copy.deepcopy(cluster_based_paths)

		print(length_min)
		print(path_min)
	# 没更新的话，模拟退火地看是不是要采用
	else:
		if np.random.rand() < np.exp(-(length_sum - length_min) / T):
			length_min = length_sum
			path_min = searching_list.copy()
			cluster_based_paths_min = copy.deepcopy(cluster_based_paths)
		else:
			# 模拟退火不接受，swap 回去
			# 如果 cluster 交换了
			if cluster_swap:
				tmp1 = cluster_based_paths[cluster_index1].copy()
				tmp2 = cluster_based_paths[cluster_index2].copy()

				cluster_based_paths[cluster_index1] = tmp2
				cluster_based_paths[cluster_index2] = tmp1
			# 如果是组内交换了
			else:
				if not flag_3:
					tmp1 = cluster_based_paths[cluster_index1][loc1]
					tmp2 = cluster_based_paths[cluster_index1][loc2]

					cluster_based_paths[cluster_index1][loc1] = tmp2
					cluster_based_paths[cluster_index1][loc2] = tmp1

				else:
					tmp2 = cluster_based_paths[cluster_index1][loc1]
					tmp3 = cluster_based_paths[cluster_index1][loc2]
					tmp1 = cluster_based_paths[cluster_index1][loc3]

					cluster_based_paths[cluster_index1][loc1] = tmp1
					cluster_based_paths[cluster_index1][loc2] = tmp2
					cluster_based_paths[cluster_index1][loc3] = tmp3

	T = T * 0.999995

	# 以较小的概率进行 cluster 交换
	if np.random.rand() < 0.02:
		# 换 cluster，第 0 个不能换
		cluster_swap = True
		cluster_index1 = int(np.clip(np.random.rand() * len(cluster_based_paths), 1, num_clusters - 1))
		cluster_index2 = int(np.clip(np.random.rand() * len(cluster_based_paths), 1, num_clusters - 1))

		while cluster_index1 == cluster_index2:
			cluster_index2 = int(np.clip(np.random.rand() * len(cluster_based_paths), 1, num_clusters - 1))

		tmp1 = cluster_based_paths[cluster_index1].copy()
		tmp2 = cluster_based_paths[cluster_index2].copy()

		cluster_based_paths[cluster_index1] = tmp2
		cluster_based_paths[cluster_index2] = tmp1
		# 组交换和组内交换只进行一个，所以直接跳出
		continue
	else:
		cluster_swap = False

	# cluster 内交换的话先确定一个 cluster
	cluster_index1 = int(np.clip(np.random.rand() * len(cluster_based_paths), 0, num_clusters - 1))
	len_cluster = len(cluster_based_paths[cluster_index1])

	if np.random.rand() < 0.5:
		# 双交换
		if cluster_index1 == 0:
			loc1 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
			loc2 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
			while loc1 == loc2:
				loc2 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
		else:
			loc1 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))
			loc2 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))
			while loc1 == loc2:
				loc2 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))


		tmp1 = cluster_based_paths[cluster_index1][loc1]
		tmp2 = cluster_based_paths[cluster_index1][loc2]

		cluster_based_paths[cluster_index1][loc1] = tmp2
		cluster_based_paths[cluster_index1][loc2] = tmp1
		flag_3 = False

	else:
		# 三交换
		if cluster_index1 == 0:

			loc1 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
			loc2 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
			loc3 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))

			while loc1 == loc2 :
				loc2 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
			while (loc3 == loc1) or (loc3 == loc2) :
				loc3 = int(np.clip(np.random.rand() * len_cluster, 1, len_cluster - 1))
		else:
			loc1 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))
			loc2 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))
			loc3 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))

			while loc1 == loc2:
				loc2 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))
			while (loc3 == loc1) or (loc3 == loc2):
				loc3 = int(np.clip(np.random.rand() * len_cluster, 0, len_cluster - 1))

		tmp1 = cluster_based_paths[cluster_index1][loc1]
		tmp2 = cluster_based_paths[cluster_index1][loc2]
		tmp3 = cluster_based_paths[cluster_index1][loc3]

		cluster_based_paths[cluster_index1][loc1] = tmp2
		cluster_based_paths[cluster_index1][loc2] = tmp3
		cluster_based_paths[cluster_index1][loc3] = tmp1

		flag_3 = True

print(length_min)
print(path_min)
