import numpy as np

cost = np.load('cost.npy')
path = np.load('path.npy')

searching_list = [i for i in range(0,42)]

length_min = np.inf
path_min = None
T = 100000.0

np.random.seed(17)

flag_3 = False

loc1 = 0
loc2 = 0
loc3 = 0

for i in range(5000000):
	length_sum = 0
	# for j in range(41):
	# 	length_sum += cost[searching_list[j]][searching_list[j+1]]
	# 	assert cost[searching_list[j]][searching_list[j+1]] == cost[searching_list[j+1]][searching_list[j]]

	start_point = searching_list[0]
	iter_list = searching_list.copy()
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
					if (int(each)-1 in iter_list) and ((int(each)-1) != iter_list[1]):
						iter_list.remove(int(each)-1)

	if length_sum < length_min:
		length_min = float(length_sum)
		path_min = searching_list
		print(length_min)
		print(path_min)
	# 模拟退火
	else:
		if np.random.rand() < np.exp(-(length_sum - length_min) / T):
			length_min = length_sum
			path_min = searching_list
		else:
			if not flag_3:
				tmp1 = searching_list[loc1]
				tmp2 = searching_list[loc2]

				searching_list[loc1] = tmp2
				searching_list[loc2] = tmp1

			else:
				tmp2 = searching_list[loc1]
				tmp3 = searching_list[loc2]
				tmp1 = searching_list[loc3]

				searching_list[loc1] = tmp1
				searching_list[loc2] = tmp2
				searching_list[loc3] = tmp3



	T = T * 0.999995

	if np.random.rand() < 0.5:
		# 双交换
		loc1 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))
		loc2 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))

		while loc1 == loc2:
			loc2 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))

		tmp1 = searching_list[loc1]
		tmp2 = searching_list[loc2]

		searching_list[loc1] = tmp2
		searching_list[loc2] = tmp1
		flag_3 = False

	else:
		# 三交换
		loc1 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))
		loc2 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))
		loc3 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))

		while loc1 == loc2:
			loc2 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))
		while (loc3 == loc1) or (loc3 == loc2):
			loc3 = int(np.clip(np.random.rand() * len(searching_list), 1, len(searching_list) - 1))

		tmp1 = searching_list[loc1]
		tmp2 = searching_list[loc2]
		tmp3 = searching_list[loc3]

		searching_list[loc1] = tmp2
		searching_list[loc2] = tmp3
		searching_list[loc3] = tmp1

		flag_3 = True

print(length_min)
print(path_min)