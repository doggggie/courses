
# general imports
import random
import numpy as np
#import urllib2
import time
import math

# Desktop imports
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

"""
Queue class
"""

class Queue:
    """
    A simple implementation of a FIFO queue.
    """

    def __init__(self):
        """ 
        Initialize the queue.
        """
        self._items = []

    def __len__(self):
        """
        Return the number of items in the queue.
        """
        return len(self._items)
    
    def __iter__(self):
        """
        Create an iterator for the queue.
        """
        for item in self._items:
            yield item

    def __str__(self):
        """
        Return a string representation of the queue.
        """
        return str(self._items)

    def enqueue(self, item):
        """
        Add item to the queue.
        """        
        self._items.append(item)

    def dequeue(self):
        """
        Remove and return the least recently inserted item.
        """
        return self._items.pop(0)

    def clear(self):
        """
        Remove all items from the queue.
        """
        self._items = []
        



"""
Provided code for application portion of module 2

Helper class for implementing efficient version
of UPA algorithm
"""
class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm
    
    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that each node number
        appears in correct ratio
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors


"""
Provided code for Application portion of Module 2
"""
############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order
    
def fast_targeted_order(ugraph):
    new_graph = copy_graph(ugraph)
    num_nodes = len(ugraph)
    
    degree_sets = {}    
    for i in range(num_nodes):
        degree_sets[i] = set()
    
    for node in new_graph:
        degree = len(new_graph[node])
        degree_sets[degree].add(node)
    
    order = []
    for deg in range(num_nodes - 1, -1, -1):
        while degree_sets[deg]:
            node = degree_sets[deg].pop()
            
            for neighbor in new_graph[node]:
                neighbor_degree = len(new_graph[neighbor])
                degree_sets[neighbor_degree].remove(neighbor)
                degree_sets[neighbor_degree - 1].add(neighbor)
                new_graph[neighbor].remove(node) #
            order.append(node)
            #delete_node(new_graph, node)
            new_graph.pop(node)
    return order

##########################################################
# Code for loading computer network graph

#NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_filename):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
#    graph_file = urllib2.urlopen(graph_url)
    graph_file = open(graph_filename, 'r')
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


    
    
    
def make_er_ugraph(num_nodes, probability):
    """
    Return an undirected graph generated by ER algorithm.
    """
    if num_nodes <= 0:
        return dict()
        
    ugraph = {}
    for node in range(num_nodes):
        ugraph[node] = set()

    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
            randnum = random.random()
            if randnum < probability:
                # adding edge (node1, node2)
                ugraph[node1].add(node2)
                ugraph[node2].add(node1)
    return ugraph


def make_complete_ugraph(num_nodes):
    """
    Return the complete undirected graph with 
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

def make_upa_graph(num_nodes, num_conn):
    upa = UPATrial(num_conn)
    graph = make_complete_ugraph(num_conn)
    for node in range(num_conn, num_nodes):
        new_node_neighbors = upa.run_trial(num_conn)
        graph[node] = new_node_neighbors
        for neighbor in new_node_neighbors:
            graph[neighbor].add(node)
    return graph   

def bfs_visited(ugraph, start_node):
    """
    returns the set consisting of all nodes that are visited by a breadth-first search that starts at start_node.
    """
    node_q = Queue()
    visited = set([start_node])
    node_q.enqueue(start_node)
    while len(node_q) > 0:
        node = node_q.dequeue()
        for neighbor in ugraph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                node_q.enqueue(neighbor)
    return visited

def cc_visited(ugraph):
    """
    Takes the undirected graph ugraph and returns a list of sets, where each set consists of all the nodes (and nothing else) in a connected component, and there is exactly one set in the list for each connected component in ugraph and nothing else.
    """
    remaining_nodes = set(ugraph.keys())
    cc_list = []
    while remaining_nodes:
        node = remaining_nodes.pop()
        visited = bfs_visited(ugraph, node)
        cc_list.append(visited)
        for visited_node in visited:
            remaining_nodes.discard(visited_node)
    return cc_list

def largest_cc_size(ugraph):
    """
    Returns the size of the largest connected component in ugraph.
    """
    cc_list = cc_visited(ugraph)
    largest_size = 0
    for connected_component in cc_list:
        size = len(connected_component)
        if size > largest_size:
            largest_size = size
    return largest_size
    
def compute_resilience(ugraph, attack_order):
    """
    Takes the undirected graph ugraph, a list of 
    nodes attack_order and iterates through the 
    nodes in attack_order. For each node in the 
    list, the function removes the given node and 
    its edges from the graph and then computes 
    the size of the largest connected component 
    for the resulting graph.
    """
    resilience = [largest_cc_size(ugraph)]
    #new_graph = ugraph.copy()
    new_graph = {}
    for node in ugraph:
        new_graph[node] = ugraph[node]
    for attack_node in attack_order:
        try:
            new_graph.pop(attack_node)
        except KeyError:
            continue
        for node in new_graph:
            if attack_node in new_graph[node]:
                new_graph[node].discard(attack_node)
        
        resilience.append(largest_cc_size(new_graph))
    
    return resilience


def count_edge(ugraph):
    return sum(map(lambda x:len(x), ugraph.values())) / 2

def random_order(graph):
    return list(np.random.permutation(graph.keys()))


q1 = False
q3 = False
q4 = True


if q1 or q4:

    graph_fn = "alg_rf7.txt"

    num_node = 1239
    num_edge = 3047

    num_conn = int(0.5 + 0.5 * ((2 * num_node - 1) - 
                math.sqrt((2*num_node-1)**2 - 8 * num_edge)))
    probability_er = num_edge / float(num_node * (num_node - 1) / 2.0)


    graphs = []
    labels = []
    graphs.append(load_graph(graph_fn))
    labels.append('Computer network graph')
    graphs.append(make_upa_graph(num_node, num_conn))
    labels.append('UPA graph with m = ' + str(num_conn))
    graphs.append(make_er_ugraph(num_node, probability_er))
    labels.append('ER graph with P = ' + str(round(probability_er, 6)))
        
    plt.figure()
    plt.hold(True)
    for i in range(len(graphs)):
        graph = graphs[i]
        label = labels[i]
        if q1:
            attack_order = random_order(graph)
        elif q4:
            attack_order = fast_targeted_order(graph)
                
        resilience = compute_resilience(graph, attack_order)
        plt.plot(range(len(resilience)), resilience, label=label)
        
        num_node = len(graph)
        removal_num = int(num_node * 0.2 + 0.5)
        remaining_num = num_node - removal_num
        largest_cc = resilience[removal_num]
        print label, "   largest cc after", removal_num, \
            "nodes are removed:", largest_cc, \
            "  range [", int(remaining_num * 0.75 + 0.5), ",", \
            int(remaining_num * 1.25 + 0.5), "]"
        
    #plt.subplot(1,1,1)
    plt.title("Graph resilience")
    plt.xlabel("Number of nodes removed in an order based on connectivity")
    plt.ylabel("Size of largest connect component after node removal")
    plt.legend()
    
    plt.show()
        


if q3:
    num_conn = 5
    
    slow_times = []
    fast_times = []
    
    largest_n = 4000
    num_nodes = range(10, largest_n, 10)
    
    for num_node in num_nodes:
        print "num_node", num_node
        upa_graph = make_upa_graph(num_node, num_conn)
    
        time_start = time.time()
        order1 = targeted_order(upa_graph)
        time_end = time.time()
        slow_times.append(time_end - time_start)
        
        time_start = time.time()
        order2 = fast_targeted_order(upa_graph)
        time_end = time.time()
        fast_times.append(time_end - time_start)
    
    plt.figure()
    plt.hold(True)
    
    plt.plot(num_nodes, slow_times, label="targeted_order")
    plt.plot(num_nodes, fast_times, label="fast_targeted_order")
        
    #plt.subplot(1,1,1)
    plt.title("Targeted order running time comparison")
    plt.xlabel("Number of nodes in DPA graph")
    plt.ylabel("Running time in seconds (desktop Python)")
    plt.legend()
    
    plt.figure()
    plt.hold(True)
    
    plt.plot(np.log(num_nodes[160:]), np.log(np.array(slow_times[160:])+0.001), label="targeted_order")
    plt.plot(np.log(num_nodes[160:]), np.log(np.array(fast_times[160:])+0.001), label="fast_targeted_order")
        
    plt.title("Targeted order running time comparison (log-log plot)")
    plt.xlabel("log(Number of nodes in DPA graph)")
    plt.ylabel("log(Running time in seconds) (desktop Python)")
    plt.legend()

    plt.show()    