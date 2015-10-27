"""
Cluster class for Module 3
"""

import math
import random
import matplotlib.pyplot as plt
#import timeit
import time

# URLs for cancer risk data tables of various sizes
# Numbers indicate number of counties in data table

DIRECTORY = "http://commondatastorage.googleapis.com/codeskulptor-assets/"
DATA_3108_URL = DIRECTORY + "data_clustering/unifiedCancerData_3108.csv"
DATA_896_URL = DIRECTORY + "data_clustering/unifiedCancerData_896.csv"
DATA_290_URL = DIRECTORY + "data_clustering/unifiedCancerData_290.csv"
DATA_111_URL = DIRECTORY + "data_clustering/unifiedCancerData_111.csv"

DATA_3108_FILE = "unifiedCancerData_3108.csv"
DATA_896_FILE = "unifiedCancerData_896.csv"
DATA_290_FILE = "unifiedCancerData_290.csv"
DATA_111_FILE = "unifiedCancerData_111.csv"

DATA_FILE = DATA_896_FILE

NUM_CLUSTERS = 9
NUM_ITERATIONS = 5

MAP_URL = DIRECTORY + "data_clustering/USA_Counties.png"
map_filename = "USA_Counties.png"

# Define colors for clusters.  Display a max of 16 clusters.
COLORS = ['Aqua', 'Yellow', 'Blue', 'Fuchsia', 
          'Black', 'Green', 'Lime', 'Maroon', 
          'Navy', 'Olive', 'Orange', 'Purple', 
          'Red', 'Brown', 'Teal']



class Cluster:
    """
    Class for creating and merging clusters of counties
    """
    
    def __init__(self, fips_codes, horiz_pos, vert_pos, population, risk):
        """
        Create a cluster based the models a set of counties' data
        """
        self._fips_codes = fips_codes
        self._horiz_center = horiz_pos
        self._vert_center = vert_pos
        self._total_population = population
        self._averaged_risk = risk
        
        
    def __repr__(self):
        """
        String representation assuming the module is "alg_cluster".
        """
        rep = "alg_cluster.Cluster("
        rep += str(self._fips_codes) + ", "
        rep += str(self._horiz_center) + ", "
        rep += str(self._vert_center) + ", "
        rep += str(self._total_population) + ", "
        rep += str(self._averaged_risk) + ")"
        return rep


    def fips_codes(self):
        """
        Get the cluster's set of FIPS codes
        """
        return self._fips_codes
    
    def horiz_center(self):
        """
        Get the averged horizontal center of cluster
        """
        return self._horiz_center
    
    def vert_center(self):
        """
        Get the averaged vertical center of the cluster
        """
        return self._vert_center
    
    def total_population(self):
        """
        Get the total population for the cluster
        """
        return self._total_population
    
    def averaged_risk(self):
        """
        Get the averaged risk for the cluster
        """
        return self._averaged_risk
   
        
    def copy(self):
        """
        Return a copy of a cluster
        """
        copy_cluster = Cluster(set(self._fips_codes), 
                               self._horiz_center, self._vert_center,
                               self._total_population, self._averaged_risk)
        return copy_cluster


    def distance(self, other_cluster):
        """
        Compute the Euclidean distance between two clusters
        """
        vert_dist = self._vert_center - other_cluster.vert_center()
        horiz_dist = self._horiz_center - other_cluster.horiz_center()
        return math.sqrt(vert_dist ** 2 + horiz_dist ** 2)
        
    def merge_clusters(self, other_cluster):
        """
        Merge one cluster into another
        The merge uses the relatively populations of each
        cluster in computing a new center and risk
        
        Note that this method mutates self
        """
        if len(other_cluster.fips_codes()) == 0:
            return self
        else:
            self._fips_codes.update(set(other_cluster.fips_codes()))
 
            # compute weights for averaging
            self_weight = float(self._total_population)                        
            other_weight = float(other_cluster.total_population())
            self._total_population = self._total_population + \
                                     other_cluster.total_population()
            self_weight /= self._total_population
            other_weight /= self._total_population
                    
            # update center and risk using weights
            self._vert_center = self_weight * self._vert_center + \
                                other_weight * other_cluster.vert_center()
            self._horiz_center = self_weight * self._horiz_center + \
                                 other_weight * other_cluster.horiz_center()
            self._averaged_risk = self_weight * self._averaged_risk + \
                                  other_weight * other_cluster.averaged_risk()
            return self

    def cluster_error(self, data_table):
        """
        Input: data_table is the original table of cancer data used in creating the cluster.
        
        Output: The error as the sum of the square of the distance from each county
        in the cluster to the cluster center (weighted by its population)
        """
        # Build hash table to accelerate error computation
        fips_to_line = {}
        for line_idx in range(len(data_table)):
            line = data_table[line_idx]
            fips_to_line[line[0]] = line_idx
        
        # compute error as weighted squared distance from counties to cluster center
        total_error = 0
        counties = self.fips_codes()
        for county in counties:
            line = data_table[fips_to_line[county]]
            singleton_cluster = Cluster(set([line[0]]), line[1], line[2], 
                                        line[3], line[4])
            singleton_distance = self.distance(singleton_cluster)
            total_error += (singleton_distance ** 2) * singleton_cluster.total_population()
        return total_error
            
        
            

        
    
"""
Example code for creating and visualizing
cluster of county-based cancer risk data

Note that you must download the file
http://www.codeskulptor.org/#alg_clusters_matplotlib.py
to use the matplotlib version of this code
"""




###################################################
# Code to load data tables



def load_data_table(filename):
    """
    Import a table of county-based cancer risk data
    from a csv format file
    """
#    data_file = urllib2.urlopen(data_url)
    data_file = open(filename, "r")
    data = data_file.read()
    data_lines = data.split('\n')
    print "Loaded", len(data_lines), "data points"
    data_tokens = [line.split(',') for line in data_lines]
    return [[tokens[0], float(tokens[1]), float(tokens[2]), 
              int(tokens[3]), float(tokens[4])] 
            for tokens in data_tokens]


############################################################
# Code to create sequential clustering
# Create alphabetical clusters for county data

def sequential_clustering(singleton_list, num_clusters):
    """
    Take a data table and create a list of clusters
    by partitioning the table into clusters based on its ordering
    
    Note that method may return num_clusters or num_clusters + 1 final clusters
    """
    
    cluster_list = []
    cluster_idx = 0
    total_clusters = len(singleton_list)
    cluster_size = float(total_clusters)  / num_clusters
    
    for cluster_idx in range(len(singleton_list)):
        new_cluster = singleton_list[cluster_idx]
        if math.floor(cluster_idx / cluster_size) != \
           math.floor((cluster_idx - 1) / cluster_size):
            cluster_list.append(new_cluster)
        else:
            cluster_list[-1] = cluster_list[-1].merge_clusters(new_cluster)
            
    return cluster_list
                

"""
Student template code for Project 3
Student will implement five functions:

slow_closest_pair(cluster_list)
fast_closest_pair(cluster_list)
closest_pair_strip(cluster_list, horiz_center, half_width)
hierarchical_clustering(cluster_list, num_clusters)
kmeans_clustering(cluster_list, num_clusters, num_iterations)

where cluster_list is a 2D list of clusters in the plane
"""

######################################################
# Code for closest pairs of clusters

def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2 are integer indices for two clusters
    
    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))

def pair_distances(cluster_list, idx1, idx2s):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2s are integer and integer lists indices for two clusters
    
    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    min_dist, min_idx1, min_idx2 = float("inf"), -1, -1
    for i in idx2s:
        dist, idx1, idx2 = pair_distance(cluster_list, idx1, i)
        if dist < min_dist:
            min_dist, min_idx1, min_idx2 = dist, idx1, idx2
    return min_dist, min_idx1, min_idx2
    
def slow_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (slow)

    Input: cluster_list is the list of clusters
    
    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.       
    """
    min_dist = float("inf")
    min_idx1 = -1
    min_idx2 = -1
    for idx1 in range(len(cluster_list) - 1):
        for idx2 in range(idx1 + 1, len(cluster_list)):
            dist, idx1, idx2 = pair_distance(cluster_list, idx1, idx2)
            if dist < min_dist:
                min_dist, min_idx1, min_idx2 = dist, idx1, idx2
                
    return (min_dist, min_idx1, min_idx2)



def fast_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (fast)

    Input: cluster_list is list of clusters SORTED such that horizontal positions of their
    centers are in ascending order
    
    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.       
    """
    ncluster = len(cluster_list)
    if ncluster <= 3:
        return slow_closest_pair(cluster_list)
    
    cluster_list.sort(key = lambda x: x.horiz_center())
    
    mididx = ncluster / 2
    
    dist, idx1, idx2 = float("inf"), -1, -1
    
    subcluster = []
    for idx in range(mididx):
        subcluster.append(cluster_list[idx])
    subdist, subidx1, subidx2 = fast_closest_pair(subcluster)
        
    if subdist < dist:
        dist, idx1, idx2 = subdist, subidx1, subidx2
        
    subcluster = []
    for idx in range(mididx, ncluster):
        subcluster.append(cluster_list[idx])
    subdist, subidx1, subidx2 = fast_closest_pair(subcluster)
    subidx1 += mididx
    subidx2 += mididx
    if subdist < dist:
        dist, idx1, idx2 = subdist, subidx1, subidx2
                
    horiz_ctr = (cluster_list[mididx-1].horiz_center() + 
                 cluster_list[mididx].horiz_center()) / 2.0
    
    subdist, subidx1, subidx2 = closest_pair_strip(cluster_list, horiz_ctr, dist)
    #print "strip", subdist, subidx1, subidx2
    if subdist < dist:
        dist, idx1, idx2 = subdist, subidx1, subidx2

    return dist, idx1, idx2


def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Helper function to compute the closest pair of clusters in a vertical strip
    
    Input: cluster_list is a list of clusters produced by fast_closest_pair
    horiz_center is the horizontal position of the strip's vertical center line
    half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.       
    """
    idxset = filter(lambda i: abs(cluster_list[i].horiz_center() - horiz_center) < half_width,
                    range(len(cluster_list)))
    idxset.sort(key = lambda i: cluster_list[i].vert_center())
    nidxset = len(idxset)
    min_dist, min_idx1, min_idx2 = float("inf"), -1, -1
    for idxu in range(nidxset - 1):
        for idxv in range(idxu + 1, min(idxu + 3, nidxset - 1) + 1):
            dist, idx1, idx2 = pair_distance(cluster_list, idxset[idxu], idxset[idxv])
            if dist < min_dist:
                min_dist, min_idx1, min_idx2 = dist, idx1, idx2
#    if len(cluster_list) == 8:
#        print "closest strip", min_dist, min_idx1, min_idx2, idxset, horiz_center, half_width
#        for i in range(len(cluster_list)):
#            print "  ", i, cluster_list[i].horiz_center(), cluster_list[i].vert_center()
                
    return min_dist, min_idx1, min_idx2
            
 
    
######################################################################
# Code for hierarchical clustering


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function mutates cluster_list
    
    Input: List of clusters, integer number of clusters
    Output: List of clusters whose length is num_clusters
    """
    num_cluster_curr = len(cluster_list)
    while num_cluster_curr > num_clusters:
        #if num_cluster_curr % 100 == 0:
        #    print "current # clusters", num_cluster_curr

        min_dist, min_idx1, min_idx2 = fast_closest_pair(cluster_list)
#        min_dist1, min_idx11, min_idx21 = slow_closest_pair(cluster_list)
        
#        if min_idx1 != min_idx11 or min_idx2 != min_idx21:
#            print "NOT EQUAL! len: ", num_cluster_curr, min_idx1, min_idx2, min_dist, min_idx11, min_idx21, min_dist1
        
        #print "\nmin_dist/idx1/idx2", min_dist, min_idx1, min_idx2
        cluster_list[min_idx1].merge_clusters(cluster_list[min_idx2])
        cluster_list.pop(min_idx2)        
        num_cluster_curr = len(cluster_list)
    return cluster_list


######################################################################
# Code for k-means clustering

    
def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    Note: the function mutates cluster_list
    
    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """

    # position initial clusters at the location of clusters with largest populations
    clusters = list(cluster_list)
    clusters.sort(key=lambda x: -x.total_population())
    
    ctrs = []
    for idx in range(num_clusters):
        ctrs.append((clusters[idx].horiz_center(),
                     clusters[idx].vert_center()))
    
    for dummyidx in range(num_iterations):
        
        print "iteration #", dummyidx
            
        clusters = []
        for jctr in range(num_clusters):
            clusters.append(Cluster(set(), #fips_codes, 
                                    ctrs[jctr][0], #horiz_pos, 
                                    ctrs[jctr][1], #vert_pos, 
                                    0, #population, 
                                    0)) #risk))
    
        for jlist in range(len(cluster_list)):
            
            min_dist, min_idx = float("inf"), -1
            for kctr in range(num_clusters):
                dist = math.sqrt((ctrs[kctr][0] - cluster_list[jlist].horiz_center()) ** 2 + 
                                 (ctrs[kctr][1] - cluster_list[jlist].vert_center()) ** 2)
                if dist < min_dist:
                    min_dist, min_idx = dist, kctr
            clusters[min_idx].merge_clusters(cluster_list[jlist])
            
        for jctr in range(num_clusters):
            ctrs[jctr] = (clusters[jctr].horiz_center(), 
                          clusters[jctr].vert_center())
            
    return clusters


def compute_distortion(cluster_list, data_table):
    distortion = 0
    for cluster in cluster_list:
        distortion += cluster.cluster_error(data_table)
    return distortion

# Applications - Question 1

def gen_random_clusters(num_clusters):
    clusters = []
    for i in range(num_clusters):
        x = 2 * random.random() - 1
        y = 2 * random.random() - 1
        clusters.append(Cluster([], x, y, 1.0, 0.0))
    return clusters
    
def timing_closest_pairs():
    slow_timings = []
    fast_timings = []
    num_clusters = []
    for ncluster in range(2, 201):
#        print ncluster
        
        clusters = gen_random_clusters(ncluster)
        
        timestart = time.clock() 
        sdist, sidx1, sidx2 = slow_closest_pair(clusters)
        timeend = time.clock()
        slow_timings.append(timeend - timestart)

        timestart = time.clock()
        fdist, fidx1, fidx2 = fast_closest_pair(clusters)
        timeend = time.clock()
        fast_timings.append(timeend - timestart)        
        
        num_clusters.append(ncluster)
#        stime = timeit.timeit('slow_closest_pair(clusters)')
#        ftime = timeit.timeit('fast_closest_pair(clusters)')
#        slow_timings.append(stime)
#        fast_timings.append(ftime)

    plt.figure()
    plt.plot(num_clusters, slow_timings, 'r', label='slow_closest_pairs')
    plt.hold(True)
    plt.plot(num_clusters, fast_timings, 'g', label='fast_closest_pairs')
    plt.title("Running times of slow and fast_closest_pairs (DESKTOP Python)")    
    plt.xlabel("Number of clusters")
    plt.ylabel("Running time in seconds")
    plt.legend(loc=2)
    plt.show()

timing_closest_pairs()


# Helper functions

def circle_area(pop):
    """
    Compute area of circle proportional to population
    """
    return math.pi * pop / (200.0 ** 2)


def plot_clusters(data_table, cluster_list, draw_centers = False):
    """
    Create a plot of clusters of counties
    """

    fips_to_line = {}
    for line_idx in range(len(data_table)):
        fips_to_line[data_table[line_idx][0]] = line_idx
     
    # Load map image
    #map_file = urllib2.urlopen(MAP_URL)
    map_img = plt.imread(map_filename)

    # Scale plot to get size similar to CodeSkulptor version
    ypixels, xpixels, bands = map_img.shape
    DPI = 60.0                  # adjust this constant to resize your plot
    xinch = xpixels / DPI
    yinch = ypixels / DPI
    plt.figure(figsize=(xinch,yinch))
    implot = plt.imshow(map_img)
   
    # draw the counties colored by cluster on the map
    if not draw_centers:
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_line[fips_code]]
                plt.scatter(x = [line[1]], y = [line[2]], s =  circle_area(line[3]), lw = 1,
                            facecolors = cluster_color, edgecolors = cluster_color)

    # add cluster centers and lines from center to counties            
    else:
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_line[fips_code]]
                plt.scatter(x = [line[1]], y = [line[2]], s =  circle_area(line[3]), lw = 1,
                            facecolors = cluster_color, edgecolors = cluster_color, zorder = 1)
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            cluster_center = (cluster.horiz_center(), cluster.vert_center())
            for fips_code in cluster.fips_codes():
                line = data_table[fips_to_line[fips_code]]
                plt.plot( [cluster_center[0], line[1]],[cluster_center[1], line[2]], cluster_color, lw=1, zorder = 2)
        for cluster_idx in range(len(cluster_list)):
            cluster = cluster_list[cluster_idx]
            cluster_color = COLORS[cluster_idx % len(COLORS)]
            cluster_center = (cluster.horiz_center(), cluster.vert_center())
            cluster_pop = cluster.total_population()
            plt.scatter(x = [cluster_center[0]], y = [cluster_center[1]], s =  circle_area(cluster_pop), lw = 2,
                        facecolors = "none", edgecolors = "black", zorder = 3)


    plt.show()



#####################################################################
# Code to load cancer data, compute a clustering and 
# visualize the results


def run_example():
    """
    Load a data table, compute a list of clusters and 
    plot a list of clusters

    Set DESKTOP = True/False to use either matplotlib or simplegui
    """
    data_table = load_data_table(DATA_FILE)
    
    singleton_list = []
    for line in data_table:
        singleton_list.append(Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
    #cluster_list = sequential_clustering(singleton_list, NUM_CLUSTERS)	
    #print "Displaying", len(cluster_list), "sequential clusters"



    # Applications - Question 2
    #timestart = time.clock()
    #cluster_list = hierarchical_clustering(singleton_list, NUM_CLUSTERS)
    #distortion = compute_distortion(cluster_list, data_table)
    #print "hierarchical distortion", distortion, "len(cluster)", len(cluster_list)
    #timeend = time.clock()
    #print "hierarchical clustering of", len(singleton_list), \
    #       "takes", timeend - timestart, "seconds" 
    #print "Displaying", len(cluster_list), "hierarchical clusters"

    # Applications - Question 3
    #timestart = time.clock()        
    #cluster_list = kmeans_clustering(singleton_list, NUM_CLUSTERS, NUM_ITERATIONS)
    #print "kmeans distortion", distortion
    #timeend = time.clock()
    #print "kmeans clustering of", len(singleton_list), \
    #       "takes", timeend - timestart, "seconds" 
    #print "Displaying", len(cluster_list), "k-means clusters"
            
    # draw the clusters using matplotlib or simplegui
    #plot_clusters(data_table, cluster_list, True)

    
    if False:
        kmeans = []
        hierarchy = []
        nclusters = []
        for ncluster in range(6, 21):
            print "ncluster", ncluster
            cluster_list = kmeans_clustering(singleton_list, ncluster, NUM_ITERATIONS)
            distortion = compute_distortion(cluster_list, data_table)
            kmeans.append(distortion)
            nclusters.append(ncluster)
    
        cluster_list = list(singleton_list)
        for ncluster in range(20, 5, -1):
            print "ncluster", ncluster
            cluster_list = hierarchical_clustering(cluster_list, ncluster)
            distortion = compute_distortion(cluster_list, data_table)
            hierarchy.append(distortion)
        hierarchy.reverse()
            
        plt.figure()
        plt.plot(nclusters, kmeans, 'r', label='k-means')
        plt.hold(True)
        plt.plot(nclusters, hierarchy, 'g', label='hierarchical')
        plt.title("Distortion values for different number\n of clusters on 896 county data")    
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        plt.legend(loc=1)
        plt.show()
    
#if __name__ == "__main__":            
#    run_example()





    





  
        






        




    
            