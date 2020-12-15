# 导入所需库-----------------------------------------------------------
import numpy as np
import collections
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random 
import copy

# 超参数定义----------------------------------------------------------
NUM_OF_SENSORS = 100
NUM_OF_CLIENTS = 100
SERVERS_CONNECT_ABILITY = 5
CLIENT_CONNECT_NEED = 3
SENSOR_DISTANCE = 50
LENGTH_OF_SQUARE = 400

maxiter = 100
SIZE = 10

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.001
'''
工具函数介绍


    init_nodes_and_distance_matrix(seed=15):
        接受seed作为随机数种子，返回sensors, clients, distances矩阵
    

    get_maximum_theoretically(distances):
        接收distances矩阵，返回一个理论最高可连接client数量的估计上界


    draw_picture(sensors, clients, distances, connection_matrix):
        绘图程序
        接收以上参数，其中connects_matrix[i][j]=1 代表i号sensor决定连接j号client

    
    greedy_direct(sensors, clients, distances):
        直接贪心
        接收以上参数，返回connection_matrix和成功client数量

    
    greedy_most_near(sensors, clients, distances):
        最近邻贪心
        接收以上参数，返回connection_matrix和成功client数量

    
    GA(sensors, clients, distances):
        元启发式算法：遗传算法
        接收以上参数，返回connection_matrix、成功client数量、迭代记录


    max_flow(sensors, clients, distances):
        最大流算法：
        接收以上参数，返回connection_matrix和成功client数量

'''
# 生成节点和距离矩阵------------------------------------------------------
def init_nodes_and_distance_matrix(seed=0):
    '''
    初始化传感器sensors, 被感知节点clients 和 距离矩阵 distances
    sensors: array, (100,2)
    clients: array, (100,2)
    distances: array, (100,100), 大于等于50的距离直接置为inf
    '''
    sensors = []
    clients = []
    distances = np.ones((NUM_OF_SENSORS,NUM_OF_CLIENTS))
    np.random.seed(seed)
    for i in range(NUM_OF_SENSORS):
        sensors.append([np.random.uniform(0,LENGTH_OF_SQUARE),np.random.uniform(0,LENGTH_OF_SQUARE)])
    for i in range(NUM_OF_CLIENTS):
        clients.append([np.random.uniform(0,LENGTH_OF_SQUARE),np.random.uniform(0,LENGTH_OF_SQUARE)])
        
    sensors = np.array(sensors)
    clients = np.array(clients)
    for i in range(NUM_OF_SENSORS):
        for j in range(NUM_OF_CLIENTS):
            dis = np.sqrt(np.sum(np.square(sensors[i] - clients[j])))
            distances[i][j] = dis if dis<=SENSOR_DISTANCE else float('inf')

    return sensors, clients, distances


# 返回最高连接数-----------------------------------------------------------
def get_maximum_theoretically(distances):
    tmp = np.zeros_like(distances)
    tmp[distances<=SENSOR_DISTANCE] = 1
    tmp = np.sum(tmp, axis=0)
    return np.sum(tmp>=CLIENT_CONNECT_NEED)


# 绘图程序-----------------------------------------------------------------
def draw_picture(sensors, clients, distances, connection_matrix):
    f = plt.figure(figsize=(40,40))
    ax = plt.subplot(1,1,1)
    ax.scatter(sensors[:,0],sensors[:,1], c = 'r',marker='*')
    ax.scatter(clients[:,0],clients[:,1], c = 'b',marker='o')

    for i in range(NUM_OF_SENSORS): # 绘制所有在检测范围内的线
        for j in range(NUM_OF_CLIENTS):
            if distances[i][j] <= SENSOR_DISTANCE:
                ax.plot((sensors[i][0],clients[j][0]), (sensors[i][1],clients[j][1]), 'g-.')

    for i in range(NUM_OF_SENSORS): # 对于每一个sensor，绘制其检测范围内以最大连接能力可以连接的最近的点
        for j in range(NUM_OF_CLIENTS):
            if connection_matrix[i][j] == 1:
                ax.plot((sensors[i][0],clients[j][0]), (sensors[i][1],clients[j][1]), 'r-')
    
    for j in range(NUM_OF_CLIENTS): #绘制检测成功的client
        if np.sum(connection_matrix[:,j])>=CLIENT_CONNECT_NEED:
            ax.scatter(clients[j][0],clients[j][1], c = 'k',marker='o',s = 160)

    
    ax.set_ylim(0,LENGTH_OF_SQUARE)
    ax.set_xlim(0,LENGTH_OF_SQUARE)

# 直接贪心-----------------------------------------------------------------------------------------------
def greedy_direct(sensors, clients, distances):
    sensor_connecttion = np.zeros((NUM_OF_SENSORS,1))
    client_connecttion = np.zeros((NUM_OF_CLIENTS,1))
    connection_matrix = np.zeros((NUM_OF_SENSORS, NUM_OF_CLIENTS)) # connects_matrix[i][j]=1 代表i号sensor决定连接j号client
    # 这个函数用来储存每个sensor连接其检测能力下的最近的n个client, 也可以说是决定连接的client,不过没有必要最终使得每个决定连接的client都有足够多（CLIENT_CONNECT_NEED）个server连接
                
    for i in range(NUM_OF_SENSORS):
        for j in range(NUM_OF_CLIENTS):
            if (np.sum(np.square(sensors[i] - clients[j])) <= SENSOR_DISTANCE ** 2) and (sensor_connecttion[i][0] < 5) :
                sensor_connecttion[i][0] += 1
                client_connecttion[j][0] += 1
                assert connection_matrix[i][j] == 0
                connection_matrix[i][j] = 1


    num_of_success_clients = len(client_connecttion[client_connecttion>=CLIENT_CONNECT_NEED])
    
    return connection_matrix, num_of_success_clients


# 最近邻贪心------------------------------------------------------------------------------------------
def greedy_most_near(sensors, clients, distances):
    sensor_connecttion = np.zeros((NUM_OF_SENSORS,1))
    client_connecttion = np.zeros((NUM_OF_CLIENTS,1))
    connection_matrix = np.zeros((NUM_OF_SENSORS, NUM_OF_CLIENTS)) # connects_matrix[i][j]=1 代表i号sensor决定连接j号client
    # 这个函数用来储存每个sensor连接其检测能力下的最近的n个client, 也可以说是决定连接的client,不过没有必要最终使得每个决定连接的client都有足够多（CLIENT_CONNECT_NEED）个server连接
                
    for i in range(NUM_OF_SENSORS):
        sortdist = np.sort(distances[i])
        for j in np.argsort(distances[i])[:min(SERVERS_CONNECT_ABILITY,len(sortdist[sortdist <= SENSOR_DISTANCE]))]:
            sensor_connecttion[i][0] += 1
            client_connecttion[j][0] += 1
            assert connection_matrix[i][j] == 0
            connection_matrix[i][j] = 1

    num_of_success_clients = len(client_connecttion[client_connecttion>=CLIENT_CONNECT_NEED])
    return connection_matrix, num_of_success_clients

# 元启发式算法：遗传算法
def GA(sensors, clients, distances):
    potentialConnect = {}
    potentialConnectDist = {}
    potentialConnectFlag = {}
    for i in range(NUM_OF_SENSORS):
        length = len(np.argwhere(distances[i,:] <= SENSOR_DISTANCE).reshape(1,-1)[0])
        potentialConnect[i] = (distances[i,:]).argsort()[:length] 
        potentialConnectDist[i] = distances[i,:][(distances[i,:]).argsort()][:length] 
        potentialConnectFlag[i] = np.zeros((length,1))

    connetion = np.zeros((NUM_OF_SENSORS,NUM_OF_CLIENTS))
    Gene = []
    PotentialGene = []
    for key in potentialConnect.keys():
        if len(potentialConnect[key]) <= SERVERS_CONNECT_ABILITY:
            for i in potentialConnect[key]:
                connetion[key][i] = 1
        else:
            Gene.append(key)
            PotentialGene.append(potentialConnect[key])

    def measureConnection(connetion):
        return np.sum(np.sum(connetion,axis=0)>=CLIENT_CONNECT_NEED)

    def measureGene(sole):
        tmpConnect = copy.deepcopy(connetion)
        for i in range(len(Gene)):
            for j in sole[i]:
                tmpConnect[Gene[i]][j] = 1
        return measureConnection(tmpConnect)

    def crossover_and_mutation(pop, CROSSOVER_RATE = 0.5):
        new_pop = []
        for father in pop:  
            child = father  
            if np.random.rand() < CROSSOVER_RATE: 
                mother = pop[np.random.randint(SIZE)]  
                cross_points = np.random.randint(0, len(child))  
                child[cross_points:] = mother[cross_points:] 
            mutation(child)  
            new_pop.append(child)
        return new_pop

    def mutation(child, MUTATION_RATE= 0.1):
        if np.random.rand() < MUTATION_RATE:  
            mutate_point = np.random.randint(0, len(child))  
            child[mutate_point] =  np.random.choice(PotentialGene[mutate_point],SERVERS_CONNECT_ABILITY, replace=False)

    def get_fitness(ranks): 
        return ((np.array(ranks) - min(ranks) )* 2) ** 2 + 1e-3 


    def select(pop, fitness):    # nature selection wrt pop's fitness
        idx = np.random.choice(np.arange(SIZE), size=SIZE, replace=True,
                               p=(fitness)/(fitness.sum()) )
        return [pop[i] for i in idx]
    Pop = []
    Rank = []
    
    for leng in range(SIZE):
        sole = []
        tmpConnect = copy.deepcopy(connetion)
        for i in range(len(Gene)):
            snrs = np.random.choice(PotentialGene[i],SERVERS_CONNECT_ABILITY, replace=False)
            sole.append(snrs)
            for j in snrs:
                tmpConnect[Gene[i]][j] = 1
        Pop.append(sole)
        Rank.append(measureConnection(tmpConnect))

    maxitem = Pop[Rank.index(max(Rank))]
    maxium = max(Rank)
    RankRec = []

    for _ in range(maxiter):
        Pop = crossover_and_mutation(Pop)
        ranks = [0 for i in range(SIZE)]
        for i in range(SIZE):
            ranks[i] = measureGene(Pop[i])

        if max(ranks) >= maxium:
            maxium = max(ranks)
            maxitem = copy.deepcopy(Pop[ranks.index(max(ranks))])
        Pop[ranks.index(min(ranks))] = copy.deepcopy(maxitem) 
        ranks[ranks.index(min(ranks))] = maxium
        if _ == maxiter:
            break

        fitness = get_fitness(ranks)

        Pop = select(Pop, fitness)
        RankRec.append(max(ranks))

    
    sole = maxitem
    
    tmpConnect = copy.deepcopy(connetion)
    for i in range(len(Gene)):
        for j in sole[i]:
            tmpConnect[Gene[i]][j] = 1


        
    return tmpConnect, np.sum(np.sum(tmpConnect,axis=0)>=CLIENT_CONNECT_NEED), RankRec


# 最大流算法------------------------------------------------------------------------------
class Graph: 
    def __init__(self,graph): 
        self.graph = graph 
        self.ROW = len(graph) 
        self.Origingraph = copy.deepcopy(graph) 
        self.TMPgraph = copy.deepcopy(graph) 
        self.iter = 0
        #初始温度，停止温度与降温系数
        self.maxiter = 10000
        self.tmp = 1e5
        self.tmp_min = 1e-3
        self.alpha = 0.98

    def bfs(self, s, t, parent):
        visited = [False]*self.ROW
        queue = [s] 
        visited[s] = True

        while queue: 
            u = queue.pop(0) 
            for ind, val in enumerate(self.graph[u]): 
                if visited[ind] == False and val > 0 : 
                    queue.append(ind) 
                    visited[ind] = True
                    parent[ind] = u 
        return True if visited[t] else False
    
    def FordFulkerson(self, source, sink): 
        parent = [-1]*self.ROW 
  
        max_flow = 0 
        
        while self.bfs(source, sink, parent): #判断增广路径 
            path_flow, s = float("Inf"), sink
            while s != source: 
                #计算增广路径的最小流量，通过最小流量计算残差网络
                path_flow = min(path_flow, self.graph[parent[s]][s]) 
                s = parent[s] 

            max_flow += path_flow 
            v = sink 
            while v != source: #计算残差网络
                u = parent[v] 
                self.graph[u][v] -= path_flow 
                self.graph[v][u] += path_flow 
                v = parent[v] 

        return max_flow 
    

def max_flow(sensors, clients, distances):
    tmp = np.zeros_like(distances)
    tmp[distances<=SENSOR_DISTANCE] = 1

    total_graph = np.zeros((NUM_OF_SENSORS+NUM_OF_CLIENTS+2,NUM_OF_SENSORS+NUM_OF_CLIENTS+2))
    total_graph[0, 1:1+NUM_OF_SENSORS]=SERVERS_CONNECT_ABILITY
    total_graph[1+NUM_OF_SENSORS:1 + NUM_OF_SENSORS+NUM_OF_CLIENTS,1 + NUM_OF_SENSORS+NUM_OF_CLIENTS]=CLIENT_CONNECT_NEED
    total_graph[1:1+NUM_OF_SENSORS,1+NUM_OF_SENSORS:1 + NUM_OF_SENSORS+NUM_OF_CLIENTS]=tmp

    g = Graph(total_graph) 
    source = 0; sink = 1 + NUM_OF_SENSORS+NUM_OF_CLIENTS
    g.FordFulkerson(source,sink)
    connection_matrix = np.transpose(np.array(g.graph[1+NUM_OF_SENSORS:1 + NUM_OF_SENSORS+NUM_OF_CLIENTS, 1:1+NUM_OF_SENSORS]))

    # 询问部分, 连接数为2的client去询问其他连接数不足的client，是否能够通过交换连接达到增加的目的
    flag = True
    while flag:
        flag, tmp = query(connection_matrix, distances)
        if flag:
            connection_matrix = tmp
            # print(np.count_nonzero(np.sum(connection_matrix, axis=0)>=CLIENT_CONNECT_NEED))

    return connection_matrix, np.count_nonzero(np.sum(connection_matrix, axis = 0)>=CLIENT_CONNECT_NEED)


def query(connection_matrix, distances):

    tmp = np.zeros_like(distances)
    tmp[distances<=SENSOR_DISTANCE] = 1

    clients_with_2_server = list(np.where(np.sum(connection_matrix, axis = 0)==2)[0])
    clients_with_1_server = list(np.where(np.sum(connection_matrix, axis = 0)==1)[0])
    clients_not_enough = clients_with_2_server + clients_with_1_server
    for i in clients_with_2_server:
        for j in clients_not_enough:
            if i==j:
                continue
            same_servers = np.where(tmp[:,i]*connection_matrix[:,j]*(1-connection_matrix[:,i])==1)[0]
            if len(same_servers)==0:
                continue
            for server in same_servers:
                if np.sum(connection_matrix[server,:])<SERVERS_CONNECT_ABILITY: # 未达到最大限制，说明没有必要交换
                    continue
                else:
                    # print("交换连接")
                    connection_matrix[server,i]=1
                    connection_matrix[server,j]=0
                    return True,connection_matrix
    return False,0