"""
Degree distributions for graphs.
"""

def make_complete_graph(num_nodes):
    """
    Return the complete directed graph with 
    the specified number of nodes.
    """
    if num_nodes <= 0:
        return dict()
        
    graph = {}
    for node in range(num_nodes):
        nodes = range(num_nodes)
        nodes.remove(node)
        graph[node] = set(nodes)
    return graph
    
def compute_in_degrees(digraph):
    """
    Computes the in-degrees for the nodes in the graph.
    """
    indegrees = {node: 0 for node in digraph}
    for head in digraph:
        for tail in digraph[head]:
            indegrees[tail] += 1
    return indegrees
    
def in_degree_distribution(digraph):
    """
    Computes the unnormalized distribution of the in-degrees of the graph.
    """
    distr = {}
    indegs = compute_in_degrees(digraph)
    for node in indegs:
        deg = indegs[node]
        if deg not in distr:
            distr[deg] = 1
        else:
            distr[deg] += 1
    return distr
    
EX_GRAPH0 = {0: set([1, 2]),
             1: set([]),
             2: set([])}
              
EX_GRAPH1 = {0: set([1, 4, 5]),  
             1: set([2, 6]),
             2: set([3]),
             3: set([0]),
             4: set([1]),
             5: set([2]),
             6: set([])}
             
EX_GRAPH2 = {0: set([1, 4, 5]),  
             1: set([2, 6]),
             2: set([3, 7]),
             3: set([7]),
             4: set([1]),
             5: set([2]),
             6: set([]),
             7: set([3]),
             8: set([1, 2]),
             9: set([0, 3, 4, 5, 6, 7])}