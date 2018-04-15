from collections import deque
import maxflow

class Graph:
    """
    @type numberOfVertices:int
    @type adjacencyList:list[dict[int,int]]
    """

    def __init__(self, numberOfVertices, edgeList, undirected=True):
        self.numberOfVertices = numberOfVertices
        self.undirected = undirected
        self.adjacencyList = [{} for _ in range(self.numberOfVertices)]
        for u, v, weight in edgeList:
            self.adjacencyList[u][v] = weight
            # self.adjacencyList[v][u] = weight

    def copy(self):
        """
        @rtype:Graph
        """
        newGraph = Graph(self.numberOfVertices,[],self.undirected)
        for u in range(self.numberOfVertices):
            for v in self.adjacencyList[u].keys():
                newGraph.adjacencyList[u][v] = self.adjacencyList[u][v]
        return newGraph

    def DirectedVersion(self):

        newGraph = self.copy()
        for u in range(self.numberOfVertices):
            for v in self.adjacencyList[u].keys():
                newGraph.adjacencyList[v][u] = newGraph.adjacencyList[u][v]
        return newGraph


def getBfsTree(graph, u, v):
    """
    @type graph:Graph
    @type u:int
    @type v:int
    """
    bfsQueue = deque(maxlen=graph.numberOfVertices)
    bfsQueue.append(u)
    # parents for each node in bfs search tree
    parents = [None] * graph.numberOfVertices
    # starting node has parent as -1
    parents[u] = -1
    while len(bfsQueue) > 0:
        i = bfsQueue.popleft()
        if i == v:
            break
        for j in graph.adjacencyList[i].keys():
            # j will not enter queue twice
            if parents[j] is not None:
                continue
            # check if edge (i,j) has some cpacity left
            if graph.adjacencyList[i][j] == 0:
                continue
            bfsQueue.append(j)
            parents[j] = i
    return parents


def Maxflow(graph, u, v):
    """
    @type graph:Graph
    @type u:int
    @type v:int
    """
    residualGraph = graph.copy()
    maximumFlow = 0
    while True:

        parents = getBfsTree(residualGraph, u, v)
        # check if vertex v is reachable
        if parents[v] is None:
            break
        # bottleNeck will be the Maxflowmaxflow for the current path
        bottleNeck = residualGraph.adjacencyList[parents[v]][v]
        maximumFlow = maximumFlow+bottleNeck
        i = v
        while i != u:
            bottleNeck = min(
                residualGraph.adjacencyList[parents[i]][i], bottleNeck)
            i = parents[i]
        # update the residual graph accordingly
        i = v
        while i != u:
            residualGraph.adjacencyList[parents[i]
                                        ][i] = residualGraph.adjacencyList[parents[i]][i]-bottleNeck
            residualGraph.adjacencyList[i][parents[i]
                                           ] = residualGraph.adjacencyList[i][parents[i]]+bottleNeck
            i = parents[i]
    partition = [v]*residualGraph.numberOfVertices
    parents = getBfsTree(residualGraph, u, v)
    for vertex, isVisited in enumerate(parents):
        if isVisited is not None:
            partition[vertex] = u
    return maximumFlow, partition


def makeMove(graph, EnergyFunction, alphaLabel, assignment):
    """
    @type graph: Graph
    @type u: int
    @type alphaLabel: int
    """
    # new vertex numbers
    numberOfVertices = graph.numberOfVertices
    # introduce a new vertex alpha
    alpha = numberOfVertices
    numberOfVertices = numberOfVertices+1
    # introduce a new vertex alpha bar
    alphaBar = numberOfVertices
    numberOfVertices = numberOfVertices+1
    edgeList = []
    newGraph = maxflow.GraphFloat()
    newGraph.add_nodes(graph.numberOfVertices)
    for vertex in range(graph.numberOfVertices):
        for u in graph.adjacencyList[vertex].keys():

            if(assignment[vertex] == assignment[u]):
                e = EnergyFunction(
                    vertex, assignment[vertex], u, assignment[u])
                newGraph.add_edge(vertex,u,e,e)
                # edgeList.append((vertex, u, EnergyFunction(
                #     vertex, assignment[vertex], u, assignment[u])))
            else:
                newNode = newGraph.add_node()
                # numberOfVertices = numberOfVertices+1
                e = EnergyFunction(
                    vertex, assignment[vertex], u, alphaLabel)
                # edgeList.append((vertex, newNode, EnergyFunction(
                #     vertex, assignment[vertex], u, alphaLabel)))
                newGraph.add_edge(vertex,newNode,e,e)
                e=EnergyFunction(
                    vertex, alphaLabel, u, assignment[u])
                # edgeList.append((newNode, u, EnergyFunction(
                #     vertex, alphaLabel, u, assignment[u])))
                newGraph.add_edge(newNode,u,e,e)
                e=EnergyFunction(
                    vertex, assignment[vertex], u, assignment[u])
                # edgeList.append((newNode, alphaBar, EnergyFunction(
                #     vertex, assignment[vertex], u, assignment[u])))
                newGraph.add_tedge(newNode,0,e)
        if(assignment[vertex] != alphaLabel):
            e=EnergyFunction(vertex, assignment[vertex])
            # edgeList.append(
            #     (vertex, alphaBar, EnergyFunction(vertex, assignment[vertex])))
        else:
            e = 1000000000
            # edgeList.append(
                # (vertex, alphaBar, 1000000000))
        newGraph.add_tedge(vertex,EnergyFunction(vertex,alphaLabel),e)
        # edgeList.append(
        #     (vertex, alpha, EnergyFunction(vertex, alphaLabel))
        # )

    # for e
    # newGraph = Graph(numberOfVertices=numberOfVertices, edgeList=edgeList)
    # del edgeList[:]
    # del edgeList
    # newGraph = newGraph.DirectedVersion()

    print "line 148"
    k = newGraph.maxflow()
    print k
    # now the partition returns if a vertex is reachable by alpha or alpha bar
    # print newGraph.get_segments()
    # since the cut defines the assignment hence if i is reachable by alpha then (i,alpha) is on the cut and vice versa. So if partition[i]=alphaBar then the assignment is alpha
    for i in range(graph.numberOfVertices):
        if newGraph.get_segment(i):
            assignment[i] = alphaLabel


    return assignment


def alpha_expansion(graph, EnergyFunction, numberOfLabels, iterations):
    """
    @type graph: Graph
    @type numberOfLabels: int
    @type iterations: int
    """
    # initializing all the labels as 0
    assignment = [0 for _ in range(graph.numberOfVertices)]
    curLabel = 0
    for iteration in range(iterations):
        curLabel = (curLabel+1) % numberOfLabels
        assignment = makeMove(graph=graph, EnergyFunction=EnergyFunction,
                              alphaLabel=curLabel, assignment=assignment)
    return assignment


# edgel = [
#     (0, 1, 2),
#     (0, 2, 1),
#     (1, 3, 1),
#     (2, 3, 1)
# ]
# g = Graph(4, edgel)
# print Maxflowmaxflow(g, 0, 3)
