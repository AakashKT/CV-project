from collections import deque


class Graph:
    """
    @type numberOfVertices:int
    @type adjacencyList:list[dict[int,int]]
    """

    def __init__(self, numberOfVertices, edgeList):
        self.numberOfVertices = numberOfVertices
        self.__edgeList__ = edgeList
        self.adjacencyList = [{} for _ in range(self.numberOfVertices)]
        for u, v, weight in edgeList:
            self.adjacencyList[u][v] = weight

    def copy(self):
        """
        @rtype:Graph
        """
        newGraph = Graph(numberOfVertices=self.numberOfVertices,
                         edgeList=self.__edgeList__)
        return newGraph


def getDfsTree(graph, u, v):
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


def maxflow(graph, u, v):
    """
    @type graph:Graph
    @type u:int
    @type v:int
    """
    residualGraph = graph.copy()
    maximumFlow = 0
    while True:

        parents = getDfsTree(residualGraph, u, v)
        # check if vertex v is reachable
        if parents[v] is None:
            break
        # bottleNeck will be the maxflow for the current path
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
    parents = getDfsTree(residualGraph, u, v)
    for vertex, isVisited in enumerate(parents):
        if isVisited is not None:
            partition[vertex] = u
    return maximumFlow, partition


edgel = [
    (0, 1, 1), (1, 0, 0),
    (0, 2, 1), (2, 0, 0),
    (1, 3, 1), (3, 1, 0),
    (2, 3, 1), (3, 2, 0)
]
g = Graph(4, edgel)
print maxflow(g, 0, 3)
