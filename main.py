##############################################################################
# This skeleton was created by Efehan Guner (efehanguner21@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True

# For debugging. Checks number of any's in an equivalence class
def num_anys(ec_gen):
    result = 0
    for gen in ec_gen:
        if( gen.data == "Any" ):
            result += 1
            
    return result


# For debugging. Checks number of any's in an equivalence class
class TreeArrayWrapper:
    def __init__(self, objects):
        self.objects = objects

    def __str__(self):
        return f"{', '.join(map(str, self.objects))}"
    
# For debugging. Prints data in ec and then prints generalization.
def print_difference(ec, ec_gen, threshold = 2):
    #threshold = 2
    if( num_anys(ec_gen) > len(ec_gen) - threshold ):
        return
    
    gen_i = 0
    keys = list(ec.keys())
    l = len(ec_gen) 
    for i in range(len(ec)):
        cat = keys[i]
        print(f"{cat}: \t{ec[cat]}")
        
        # Assumes categories in ec and ec_gen in the same order
        if(cat == ec_gen[gen_i].category):
            print(f"Generalization: {ec_gen[gen_i].data}")
            gen_i = (gen_i + 1) % l



class Queue:
    def __init__(self, max_size = 128):
        self.max_size = max_size
        self.queue = np.empty(max_size, dtype=object)
        self.size = 0
        self.front = 0

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.max_size

    def enqueue(self, item):
        if self.is_full():
            raise ValueError("Queue is full")
        self.queue[(self.front + self.size) % self.max_size] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
        front_item = self.queue[self.front]
        self.front = (self.front + 1) % self.max_size
        self.size -= 1
        return front_item       

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None
        self.category = None
    
    def set_category(self, category):
        self.category = category
        
        if (self.children == []):
            return
        
        for child in self.children:
            child.set_category(category)
        
    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

    def print_subtree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_subtree()

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def get_leaves(self):
        root = self
        
        leaves = []
        queue = Queue()
        queue.enqueue(root)
        
        if ( root.children == [] ):
            leaves.append(root)
            return leaves
        
        while ( not queue.is_empty() ):
            current = queue.dequeue()
            
            if ( current.children == [] ):
                leaves.append(current)
            
            else:
                for child in current.children:
                    queue.enqueue(child)
        
        return leaves
    
    def get_ancestors(self):
        node = self
        ancestors = [node]
        
        while (node.parent != None):
            node = node.parent
            ancestors.append(node)
            
        return ancestors
    
    def get_path_from_root(self):
        
        ancestors = self.get_ancestors()
        ancestors.reverse()
        
        return ancestors
    
    def find_leaf(self, val):
        leaves = self.get_leaves()
        
        for leaf in leaves:
            if( leaf.data == val ):
                return leaf
            
    def find_ancestor(self, ancestor_value):
        path_to_self = self.get_path_from_root()

        ancestor_node = None
        # Check all values in the path. 

        for j in range(len(path_to_self)):
            # Node is j'th node.
            node = path_to_self[j]

            # Leaves the last node on the path
            # equal to the ancestor node. Equals
            # to the ancestor node because it is in
            # the path.
            if(ancestor_value == node.data):
                ancestor_node = node

        return ancestor_node

    def search(self, val):
    
        queue = Queue()
        queue.enqueue(self)

        while (not queue.is_empty()):
            current = queue.dequeue()

            for child in current.children:
                if (child.data == val):
                    return child
                queue.enqueue(child)

        return None
    
    def __str__(self):
        return f"{self.data}"
    
# Input is readlines of the DGH
# Output is the root of the DGH,
# Ie the node with data Any
# Algorithm uses stack to do BFS
# To populate children arrays.
def build_tree(result):
    #result is f.readlines() of the dgh.

    last_level = 0

    #Starts with Root, ie "Any"
    root = TreeNode(result[0].strip())
    stack = [root]

    for i in range(1, len(result)):
        current_level = result[i].rstrip().count("\t")
        node = TreeNode(result[i].strip())


        #Get children
        if(current_level > last_level):
            stack[-1].add_child(node)
            stack.append(node)

        #Get sibling
        elif(current_level == last_level):
            stack.pop()
            stack[-1].add_child(node)
            stack.append(node)

        #Go back to the parent
        else:
            while( current_level <= last_level ):
                last_level -= 1
                stack.pop()
            #stack.pop()

            stack[-1].add_child(node)
            stack.append(node)

        #print(result[i], count)
        last_level = result[i].rstrip().count("\t")

    return root


# Input is a DGH and values from an equivalence class as array
# Output is the least common ancestor of the values.
# Algorithm finds the first divergence of paths from the root.
def find_least_generalized(node, values):
    leaves = node.get_leaves()
    nodes = []
    # Get a list of values as nodes
    for val in values:
        
        for leaf in leaves:
            
            if( val == leaf.data ):
                nodes.append(leaf)
                break
                
    # Check the first divergence in the paths from the root
    
    paths = []
    for node in nodes:
        paths.append(node.get_path_from_root())
    
    if (len(paths) == 0):
        raise ValueError("Paths empty")
    
    # Find shortest path
    min = 999
    for path in paths:
        if( min > len(path) ):
            min = len(path)
    
    for j in range(min):
        
        last_common_node = paths[0][j]

        for i in range(len(paths)):

            if( (last_common_node.data != paths[i][j].data)):
                #print(f"Found divergence")
                return last_common_node.parent

            last_common_node = paths[i][j]
    
    return last_common_node

# Assumes not all categories have DGH's, ie income or index.
# Returns the categories with a DGH with the order of categories
def get_generalizable_categories(DGHs, categories):
    dgh_cat = list(DGHs.keys())
    
    result = []
    for cat in categories:
        if cat in dgh_cat:
            result.append(cat)
    
    return result

# Input is an equivalence class and
# generalization of the elements.
# Outputs generalized ec.
def generate_ec(ec, ec_gen):
    categories = list(ec.keys())
    new_ec = {key: [] for key in categories}
    gen_i = 0
    l = len(ec_gen)
    ec_size = len(ec[categories[0]])
    for i in range(len(ec)):
        cat = categories[i]
        
        if(cat == ec_gen[gen_i].category):
            #print(f"matching category:\n\tdata: {ec[cat]}\n\tgen: {ec_gen[gen_i].data}")   
            new_ec[cat] = [ec_gen[gen_i].data]*ec_size
            
            gen_i = (gen_i + 1) % l
            
        else:
            new_ec[cat] = ec[cat].copy()
        
    return new_ec

# Input is DGHs and an equivalence class
# Output is array of TreeNode's equaling
# least common ancestor of the categories
def calculate_ec(DGHs, ec):
    categories = list(ec.keys())
    
    categories = get_generalizable_categories(DGHs, categories)
    
    if ( ec[categories[0]] == [] ):
        raise ValueError("ec empty")
        
    generalization = []
    
    for cat in categories:
        vals = ec[cat]
        root = DGHs[cat]
        
        generalization.append(find_least_generalized(root, vals))
        
    #print(f"gen: {TreeArrayWrapper(generalization)}")
    return generalization


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    with open(DGH_file, 'r') as f:
        lines = f.readlines()
    root = build_tree(lines)

    # Format: /folder/category.txt
    root.set_category(DGH_file.split("/")[-1].split(".")[0]) 

    return root


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs

# After Part 3
def create_ec(dataset, closest_k):
    
    categories = list(dataset[0].keys())
    ec = {key: [] for key in categories}
    
    for (index,_) in closest_k:
        data = dataset[index]
        for cat in categories:
            ec[cat].append(data[cat])
    return ec

def calculate_lm(rec, DGHs):
    
    categories = list(rec.keys())
    gen_categories = get_generalizable_categories(DGHs, categories)
    cost = 0
    
    M = 1/len(categories)
    w = {'cat': 1/M}
    
    for cat in gen_categories:
        root = DGHs[cat]
        val = rec[cat]


        if( val == 'Any'):
            cost += w['cat'] * 1#1
            continue


        val_node = root.search(val)

        cost += w['cat'] * (len(val_node.get_leaves()) - 1) / (len(root.get_leaves()) - 1)
    return cost

def generalize(r1, r2, DGHs):
    
    
    categories = list(r1.keys())
    
    categories = get_generalizable_categories(DGHs, categories)
    
    generalization = {}
    
    for cat in categories:
        vals = [r1[cat], r2[cat]]
        root = DGHs[cat]
        
        generalization[cat] = find_least_generalized(root, vals).data

        
    return generalization

def calculate_dist(r1, r2, DGHs):
    
    # generalize r1, r2
    r = generalize(r1, r2, DGHs)
    
    cost = calculate_lm(r, DGHs)
    
    return cost

# For optimization. Can be used for finding K-Min
# elements in an array. Can do search and sort log(len)
class KMin:
    def __init__(self, len):
        self.array = [(0, float("inf"))]*len
        self.ID = 0
        self.DIST = 1

    def get_min(self):
        return self.array[0]
    
    def get_max(self):
        return self.array[len(self.array) - 1]

    def add(self, elem):
        dist = elem[self.DIST]
        
        if( self.get_max()[self.DIST] < dist):
            return
        
        add_index = self.binary_search(elem)

        self.array.insert(add_index, elem)
        self.array.pop()

    
    def binary_search(self, elem):
        low, high = 0, len(self.array) - 1

        while low <= high:
            mid = (low + high) // 2
            mid_value = self.array[mid][self.DIST]

            if mid_value < elem[self.DIST]:
                low = mid + 1
            else:
                high = mid - 1

        return low

# Part 4

class SpecializationNode():
    def __init__(self, DGHs, dataset):
        self.DGHs = DGHs
        self.records = dataset
        self.num_records = len(dataset)
        self.current_state = list(self.DGHs.values())
        self.children = []
        
    def split(self, category):
        
        root = self.DGHs[category]
        
        split_root = [[child] for  child in root.children]
            
        for child in split_root:
            new_DGHs = self.DGHs.copy()
            new_DGHs[category] = child
            new_dataset = []

            leaves_data = [leaf.data for leaf in child.get_leaves()]
            
            count = 0
            for record in self.records:
                #value = record[category]
                if( record[category] in leaves_data ):
                    new_dataset.append(record)
                    count += 1
            
            if( count < k ):
                self.children = []
                return -1
            
            new_node = SpecializationNode(new_DGHs, new_dataset)
            self.children.append(new_node)
        
        return self.calculate_cost()
    
    def calculate_cost(self):
        num_children = len(self.children)
        
        if num_children == 0:
            return -1
        
        uniform = self.num_records / num_children
        cost = 0
        for child in self.children:
            cost += abs( (child.num_records / self.num_records) - uniform)
        return cost
    
    def create_children(self, k):
        categories = list(self.DGHs.values())
        
        min_children = KMin(len(categories))
        
        
        for root in categories:
            min_children.add((root, len(root.children)))
        
        bests = KMin(len(categories))
        for index in len(categories):
            (root, num_child) = min_children.array[index]
            
            node = self.copy()
            
            cost = node.split(root.category)
            
            if cost == -1:
                continue
            
            
            
            bests.add( ( (num_child, cost) , node ) )
                
        best_categories = []
        min_num = min_children.get_min()
        for (root, num_child) in min_children.array:
            if num_child == min_num:
                best_categories.append(root)
                
        
        best_node = self.split_decision(self, best_categories)
        self.children = best_node.children
        return 
    
        
    def copy(self):
        return SpecializationNode(self.DGHs, self.dataset)
        
    # Categories is a list of TreeNodes, ie DGH trees.
    def split_decision(self, categories):
        
        costs = KMin(len(categories))
        for cat in categories:
            node = self.copy()
            
            costs.add((node, node.split(cat)))
        
        return costs.array
            
    def build_tree(self, k):
        root = self
        
        leaves = []
        queue = Queue()
        queue.enqueue(root)
        
        if ( root.children == [] ):
            leaves.append(root)
            return leaves
        
        while ( not queue.is_empty() ):
            current = queue.dequeue()
            
            current.create_children()
            
            if ( current.children == [] ):
                
                leaves.append(current)
            
            else:
                for child in current.children:
                    # Check length
                    #if child.num_records < k:
                        # Failure
                    
                    queue.enqueue(child)
        
        return leaves
    
    def create_ec(self):
    
        categories = list(self.records[0].keys())
        ec = {key: [] for key in categories}

        for record in self.records:
            for cat in categories:
                ec[cat].append(record[cat])
        return ec

##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    categories = list(raw_dataset[0].keys())
    gen_categories = get_generalizable_categories(DGHs, categories)

    cost = 0

    for i in range(len(raw_dataset)):
        raw = raw_dataset[i]
        anon = anonymized_dataset[i]

        for cat in gen_categories:
            root = DGHs[cat]

            raw_val = raw[cat]
            anon_val = anon[cat]

            # Find the raw elem in the leaves.
            raw_node = root.find_leaf(raw_val)

            # Find the path to raw elem. Then anon elem
            # is in the path. 
            path_to_raw = raw_node.get_path_from_root()
            
            anon_node = raw_node.find_ancestor(anon_val)

            # We add one because we start counting at zero
            dist_anon = path_to_raw.index(anon_node) + 1
            dist_raw = len(path_to_raw)

            # print(f"data {i}: {raw}, raw, anon: ({dist_raw},{dist_anon}")

            cost += dist_raw - dist_anon
    return cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    categories = list(raw_dataset[0].keys())
    gen_categories = get_generalizable_categories(DGHs, categories)

    cost = 0
    M = len(raw_dataset[0]) - 1
    w = {'cat':1/M}
    
    for i in range(len(raw_dataset)):
        raw = raw_dataset[i]
        anon = anonymized_dataset[i]
        
        for cat in gen_categories:
            root = DGHs[cat]

            raw_val = raw[cat]
            anon_val = anon[cat]

            if( anon_val == 'Any'):
                cost += w['cat'] * 1#1
                continue

            # Find the raw elem in the leaves.
            raw_node = root.find_leaf(raw_val)
            
            anon_node = raw_node.find_ancestor(anon_val)
            
            
            #lm_cost_raw = w['cat'] * (len(raw_node.get_leaves()) - 1) / (len(root.get_leaves()) - 1)
            lm_cost_anon = w['cat'] * (len(anon_node.get_leaves()) - 1) / (len(root.get_leaves()) - 1)

            # print(f"anon: {anon_node}\traw: {raw_node}")

            cost += lm_cost_anon #- lm_cost_raw
      
    return cost


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...
    
    categories = list(raw_dataset[0].keys())
    ec = {key: [] for key in categories}

    for i in range(D):
        ec_index = (i+1) #% k
        
        last_ec = (ec_index > D - (D%k) - k )
        
        for (category, elem) in list(raw_dataset[i].items()):
            ec[category].append( elem )
            
        # If equivalence class is full, we process it and we reset
        # the equivalence class.
        # The logic is: If we are at the last equivalence class,
        # then we should continue when we have k entries. 
        # We can process when the index is D, ie the last entry.
        if( ((not last_ec) and (ec_index % k == 0)) or (ec_index == D) ):
        
            # Calculate and build the generalized equivalence class 
            ec_gen = calculate_ec(DGHs, ec)

            generalized_ec = generate_ec(ec, ec_gen)
            
            # Debugging
            #print_class(ec)
            #print_difference(ec, ec_gen, 1) # 1 is number of non-any's
            
            # Add the generalized equivalence class to the cluster
            
            if( last_ec ):
                ec_len = len(generalized_ec[categories[0]])
            else:
                ec_len = k
            
            cluster = [{} for _ in range(ec_len)]
            
            for cat in categories:
                vals = generalized_ec[cat]
                
                for index in range(ec_len):
                    cluster[index][cat] = vals[index]
            
            clusters.append(cluster)
            
            # Reset ec
            ec = {key: [] for key in categories}

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D
    
    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    #return anonymized_dataset
    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)[:5000]
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.
    
    anonymized_dataset = [0]*len(raw_dataset)
    
    used = np.zeros(len(raw_dataset), dtype = bool)
    
    unused_indexes = np.where(used == False)[0]
    num_unused = len(unused_indexes)
    rec = unused_indexes[0]
    
    categories = list(raw_dataset[0].keys())
    
    while( num_unused >= 2*k):
        
        # CREATE CLUSTER
        #closest_k = [(0,float('inf'))]*(k-1)

        closest_k = KMin(k-1)
        # clustering loop
        for index_entry in range(num_unused):
            if unused_indexes[index_entry] == rec:
                continue
        
            current_entry = raw_dataset[unused_indexes[index_entry]]

            # calculate distance
            current_dist = calculate_dist(raw_dataset[rec], current_entry, DGHs)

            #if(current_dist == 0):
                #print(f" zero_distance: index_entry: {index_entry}\t rec: {rec}\n")
                #print(f"\tentry: {current_entry}\t rec: {raw_dataset[rec]}\n")

            # Add to kmin array
            closest_k.add((index_entry, current_dist))
            

            
        closest_k.array.append( (rec, 0) )
        
        #print(closest_k)
        # CREATE EQUIVALENCE CLASS
        ec = create_ec(raw_dataset, closest_k.array)
        
        # Calculate and build the generalized equivalence class 
        ec_gen = calculate_ec(DGHs, ec)

        generalized_ec = generate_ec(ec, ec_gen)

        # Add the generalized equivalence class to the cluster

        cluster = [{} for _ in range(k)]

        for cat in categories:
            vals = generalized_ec[cat]

            for index in range(k):
                cluster[index][cat] = vals[index]
        
        # UPDATE USED ARRAY
        for (index, _) in closest_k.array:
            used[index] = True
        
        unused_indexes = np.where(used == False)[0]
        num_unused = len(unused_indexes)
        rec = unused_indexes[0]
        
        #print(f"unused: {num_unused}\t new_rec: {rec}\n")
        
        #print(f"\t{np.where(used == True)[0]}")
        
        # APPEND TO THE ANONYMIZED DATASET
        
        for append_index in range(k):
            #anonymized_dataset.append(cluster[append_index])
            (dataset_index, _) = closest_k.array[append_index]
            anonymized_dataset[dataset_index] = cluster[append_index]
    
    # Handle the last equivalence class
    closest_k = [(index, 0) for index in unused_indexes]
    k = len(closest_k)
    
    for index in unused_indexes:
        ec = create_ec(raw_dataset, closest_k)

        # Calculate and build the generalized equivalence class 
        ec_gen = calculate_ec(DGHs, ec)

        generalized_ec = generate_ec(ec, ec_gen)

        # Add the generalized equivalence class to the cluster

        cluster = [{} for _ in range(k)]

        for cat in categories:
            vals = generalized_ec[cat]

            for index in range(k):
                cluster[index][cat] = vals[index]
    
    for append_index in range(k):
        (dataset_index, _) = closest_k[append_index]
        anonymized_dataset[dataset_index] = cluster[append_index]
    
    write_dataset(anonymized_dataset, output_file)


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    #TODO: complete this function.
    len_data = len(raw_dataset)

    for i in range(len_data): 
        raw_dataset[i]['index'] = i
    categories = list(raw_dataset[0].keys())

    # Add indexes
    

    spec_tree = SpecializationNode(DGHs, raw_dataset)
    leaves = spec_tree.build_tree(k)

    clusters = []

    for leaf in leaves:
        if( leaf.num_records == 0):
            continue
        ec = leaf.create_ec()

        ec_gen = calculate_ec(DGHs, ec)

        generalized_ec = generate_ec(ec, ec_gen)

        # Add the generalized equivalence class to the cluster

        cluster = [{} for _ in range(len(ec))]

        for cat in categories:
            vals = generalized_ec[cat]

            for index in range(len(ec)):
                cluster[index][cat] = vals[index]

    anonymized_dataset = [None] * len_data
    
    for cluster in clusters:
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300