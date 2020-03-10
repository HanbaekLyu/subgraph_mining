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
        self.MST_parent = [] # ith entry = parent of node i in the spanning tree
        self.MST_ordering = []

    # A utility function to print the constructed MST stored in parent[]

    def printMST(self, parent):
        print("Edge \t Weight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.adj[i][parent[i]])

    def depth_1st_order_MST(self, edgelist):
        ### Computes depth-first ordering of nodes in the MST
        B = self.edge2adj(edgelist)
        # print('B', B)
        key = [float('inf')] * self.V
        key[0] = 0 # first node is the root
        ordered = [False] * self.V
        ordered[0] = True
        ordering = [0]

        while False in ordered:
            for v in range(self.V):
            # v = next(x for x in range(self.V) if ordered[x])
            # v = ordered.index(False) # smallest index of True in list ordered
            # v = v - 1  # largest index of True in the first block of True's in the list ordered
                if ordered[v]:
                    for u in range(self.V):
                        if B[u][v] == 1 and not ordered[u]:
                            ordering.append(u)
                            ordered[u] = True
        return ordering

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initilaize min value
        min = float('inf')
        min_index = 0
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):
        '''
        1) Create a set mstSet that keeps track of vertices already included in MST.
        2) Assign a key value to all vertices in the input graph. Initialize all key values as INFINITE.
        Assign key value as 0 for the first vertex so that it is picked first.
        3) While mstSet doesn’t include all vertices:
            a) Pick a vertex u which is not there in mstSet and has minimum key value.
            b) Include u to mstSet.
            c) Update key value of all adjacent vertices of u. To update the key values, iterate through all adjacent
            vertices. For every adjacent vertex v, if weight of edge u-v is less than the previous key value of v,
            update the key value as weight of u-v
        '''

        # Key values used to pick minimum weight edge in cut
        ### Make sure g.adj is not all zeros -- it will return minKey error

        key = [float('inf')] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V
        parent[0] = -1  # First node is always the root of the tree

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

            # print('parent', parent)
        edgelist = []
        for i in range(1, self.V):
            edgelist.append([parent[i], i])
            # self.printMST(parent)
        self.MSTedges = edgelist
        self.MST_parent = parent
        self.MST_ordering = self.depth_1st_order_MST(edgelist)
        self.MST_adj = self.edge2adj(edgelist)
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
        V = set([n for e in E for n in e])
        size = len(V)
        min_node = min(V)
        # make an empty adjacency list
        adjacency = [[0] * size for _ in range(size)]
        # populate the list for each edge
        for sink, source in E:
            adjacency[sink-min_node][source-min_node] = 1
            adjacency[source-min_node][sink-min_node] = 1
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