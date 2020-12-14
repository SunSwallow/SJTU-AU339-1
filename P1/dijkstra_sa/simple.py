import numpy as np

cost = np.load('cost.npy')

searching_list = [i for i in range(0,42)]

length_min = np.inf
path_min = None
T = 100000.0

np.random.seed(25)

flag_3 = False

loc1 = 0
loc2 = 0
loc3 = 0

file = open("log.txt", "w")

for i in range(2500000):
	length_sum = 0
	for j in range(41):
		length_sum += cost[searching_list[j]][searching_list[j+1]]

	if length_sum < length_min:
		length_min = float(length_sum)
		path_min = searching_list
		print(length_min)
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

	if i % 10000 == 0:
		file.write(str(length_min)+',')

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

file.close()
print(length_min)
print(path_min)