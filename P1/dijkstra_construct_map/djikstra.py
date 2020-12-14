import pandas as pd
import numpy as np


class Dijkstra(object):
	def __init__(self, node_map):
		# 点不重复
		self.map = node_map
		self.visited_nodes_link = []
		self.avail_nodes = {}
		self.visited_nodes = []

	def find_map(self, begin_node, end_node):

		if int(begin_node) == int(end_node):
			return [begin_node], 0

		self.avail_nodes = {}
		for link in self.map:
			if begin_node in link:
				if link[0] == begin_node:
					self.avail_nodes[link[1]] = [begin_node, link[2]]
				elif link[1] == begin_node:
					self.avail_nodes[link[0]] = [begin_node, link[2]]
				else:
					raise NotImplementedError
		self.visited_nodes.append(begin_node)
		min_name = begin_node

		while True:
			# 每次推进一个点
			last_node = None
			min_cost = np.inf
			min_name = None
			for avail_name in self.avail_nodes.keys():
				if self.avail_nodes[avail_name][1] < min_cost:
					min_name = avail_name
					min_cost = self.avail_nodes[avail_name][1]
					last_node = self.avail_nodes[avail_name][0]

			# 找到了下一个要访问的点
			self.avail_nodes.pop(min_name)
			self.visited_nodes.append(min_name)
			self.visited_nodes_link.append([last_node, min_name])

			for link in self.map:
				if min_name in link:
					if link[0] == min_name:
						add_name = link[1]
					elif link[1] == min_name:
						add_name = link[0]
					else:
						raise NotImplementedError
					if add_name in self.visited_nodes:
						continue
					if add_name in self.avail_nodes.keys():
						if (link[2] + min_cost) < self.avail_nodes[add_name][1]:
							self.avail_nodes[add_name] = [min_name, link[2] + min_cost]
					else:
						self.avail_nodes[add_name] = [min_name, link[2] + min_cost]

			# 找到了，就爬
			if min_name == end_node:
				path =[min_name]
				path_curr = end_node
				while True:
					for link in self.visited_nodes_link:
						if link[1] == path_curr:
							path_curr = link[0]
							path.append(path_curr)
							break
					if path_curr == begin_node:
						break
				return path, min_cost
				# print(path)
				# print(total_cost)
				# break


if __name__ == "__main__":
	node_list = [('1', '2', 1026), ('4', '1', 402), ('2', '3', 1026), ('2', '7', 398),
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

	Dij = Dijkstra(node_list)
	cost_list = []
	path_list = []

	for i in range(1, 43):
		cost_list.append([])
		path_list.append([])
		for j in range(1, 43):
			Dij = Dijkstra(node_list)
			path, cost = Dij.find_map(str(i), str(j))
			cost_list[i-1].append(cost)
			path_list[i-1].append(path)

	cost_array = np.array(cost_list)
	path_array = np.array(path_list)
	print(path_array)

	np.save('cost', cost_array)
	np.save('path', path_array)
