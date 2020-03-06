import numpy as np
import sys



class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.adj = [[0 for column in range(vertices)]
                      for row in range(vertices)]
        self.edgelist = []
        self.MSTedges = []
        self.MST_adj = []

    # A utility function to print the constructed MST stored in parent[]

    def printMST(self, parent):
        print("Edge \t Weight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.adj[i][parent[i]])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initilaize min value
        min = float('inf')

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):
        # Key values used to pick minimum weight edge in cut
        key = [float('inf')] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

        for count in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                # adj[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if adj[u][v] is smaller than key[v]
                if self.adj[u][v] > 0 and mstSet[v] == False and key[v] > self.adj[u][v]:
                    key[v] = self.adj[u][v]
                    parent[v] = u

        edgelist = []
        for i in range(1, self.V):
            edgelist.append([parent[i], i])
            # self.printMST(parent)
        self.MSTedges = edgelist
        return edgelist

    def edge2adj_directed(self, edgelist):
        # nodes must be numbers in a sequential range starting at 0 - so this is the number of nodes.
        E = edgelist
        size = len(set([n for e in E for n in e]))
        # make an empty adjacency list
        adjacency = [[0] * size for _ in range(size)]
        # populate the list for each edge
        for sink, source in E:
            adjacency[sink][source] = 1
        return adjacency

    def edge2adj(self, edgelist):
        # nodes must be numbers in a sequential range starting at 0 - so this is the number of nodes.
        E = edgelist
        size = len(set([n for e in E for n in e]))
        # make an empty adjacency list
        adjacency = [[0] * size for _ in range(size)]
        # populate the list for each edge
        for sink, source in E:
            adjacency[sink][source] = 1
            adjacency[source][sink] = 1
        return adjacency

    def adj2edge(self, adj):
        edgelist = []
        I, J = np.where(np.asarray(adj) > 0)
        for i in np.arange(len(I)):
            edgelist.append([I[i], J[i]])
        return edgelist


'''
### example usage
 
g = Graph(5)
g.graph = [[0, 2, 0, 6, 0],
           [2, 0, 3, 8, 5],
           [0, 3, 0, 0, 7],
           [6, 8, 0, 0, 9],
           [0, 5, 7, 9, 0]]

g.primMST();

Output:
Edge   Weight
0 - 1    2
1 - 2    3
0 - 3    6
1 - 4    5

'''