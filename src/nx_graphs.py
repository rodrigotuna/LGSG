import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import torch
import heapq

SIZES_THS = {
    1: {
        'CiteSeer' :  [3266, 2516, 2045],
        'Cora' : [7082, 5581, 4495],
        'Facebook': [8962, 5636, 3579],
        'Wiki' : [6960, 5373, 4313]
    },
    2: {'CiteSeer': [3235, 2358, 1847], 'Cora': [6439, 4737, 3602], 'Facebook': [6290, 3973, 2592], 'Wiki': [8294, 6280, 4779]}, 3: {'CiteSeer': [1543, 1094, 873], 'Cora': [6466, 4836, 3697], 'Facebook': [4918, 2900, 1867], 'Wiki': [7374, 5637, 4505]}, 4: {'CiteSeer': [1982, 1440, 1179], 'Cora': [6784, 5118, 3929], 'Facebook': [9745, 6369, 4011], 'Wiki': [8118, 5992, 4640]}, 5: {'CiteSeer': [2046, 1474, 1187], 'Cora': [5515, 4189, 3360], 'Facebook': [7091, 4233, 2596], 'Wiki': [8437, 6212, 4588]}
} 

DATASETS=['CiteSeer', 'Cora', 'Facebook', 'Wiki']
SIZES = [5000, 6000, 7000, 8000, 9000, 10000]
THS = [0.35,0.375,0.40]
NUM_RUNS = 5

def read_graph(file):
    G = pickle.load(open(file, 'rb'))
    return G

##NODE AGGREGATION
def node_aggregation(samples, sizes, dataset, run):
    num_nodes = 0
    goes_to = np.zeros(len(samples)+1, dtype=int)

    size = []

    for id,sample in enumerate(samples):
        embeddings = sample[0]
        adj = sample[1]
        assert (adj == adj.T).all()

        keep_nodes = ~(adj == 0).all(axis = 1)
        embeddings = embeddings[keep_nodes]
        adj = adj[keep_nodes]
        adj = adj[:, keep_nodes]
        
        samples[id][0] = embeddings
        samples[id][1] = adj

        if id != 0:
            goes_to[id] = num_nodes
        num_nodes += len(embeddings)

    goes_to[len(samples)] = num_nodes

    total_nodes = np.zeros((num_nodes, 128))
    dsu = np.arange(num_nodes, dtype=int)

    for id in range(len(samples)):
        total_nodes[goes_to[id]:goes_to[id+1]] = samples[id][0]

    pairs_of_nodes = np.zeros(num_nodes*num_nodes)
    pairs_x = np.zeros(num_nodes*num_nodes, dtype=int)
    pairs_y = np.zeros(num_nodes*num_nodes, dtype=int)

    size = np.ones(num_nodes)

    pair_it = 0

    for id,sample in enumerate(samples):
        print(f"{id}/{len(samples)}")
        for ide, embed in enumerate(sample[0]):
            pairs_of_nodes[pair_it:pair_it + goes_to[id]] = np.linalg.norm(total_nodes[0:goes_to[id]] - embed.numpy(), axis=1)
            pairs_x[pair_it:pair_it + goes_to[id]] = np.arange(goes_to[id]) 
            if id != 0:
                pairs_y[pair_it:pair_it + goes_to[id]] = goes_to[id] + ide
            pair_it += goes_to[id]

    pairs_of_nodes = pairs_of_nodes[:pair_it]
    pairs_x = pairs_x[:pair_it]
    pairs_y = pairs_y[:pair_it]        

    def find(i):
        while(dsu[i] != i):
            i = dsu[i]
        return i

    def same(a,b):
        return find(a) == find(b)


    def unite(a,b):
        a = find(a)
        b = find(b)
        if (size[a] < size[b]): a,b = b,a
        size[a] += size[b]
        dsu[b] = a



    index = np.argsort(pairs_of_nodes)

    for i in index:
        if num_nodes <= min(sizes):
            break
        x = pairs_x[i]
        y = pairs_y[i]

        if not same(x,y):
            num_nodes -= 1
            print(num_nodes)
            unite(x,y)

        if num_nodes in sizes:
            G  =nx.Graph()
            for id,sample in enumerate(samples):
                embeddings = sample[0]
                adj = sample[1]

                for idx in range(len(adj)):
                    for idy in range(len(adj)):
                        if idx == idy:
                            continue
                        if adj[idx][idy] == 1:
                            G.add_edge(find(goes_to[id] + idx), find(goes_to[id] + idy))

            pickle.dump(G, open(f'../eval/gen/{dataset}_nodes_{num_nodes}_{run}.pickle', 'wb'))

for dataset in DATASETS:
        for run in  range(2,NUM_RUNS+1):
            samples = pickle.load(open(f'../eval/gen/sample_{dataset}{run}.pickle', 'rb'))
            node_aggregation(samples, SIZES_THS[run][dataset], dataset, run)

def threshold_method(samples, threshold, dataset): 
    print("Generating graph with threshold: ", threshold)
    nodes = np.zeros((0, 128))
    G = nx.Graph()
    for id,sample in enumerate(samples):
        print(f"{id}/1000")
        embeddings = sample[0]
        adj = sample[1]
        nodeids = []
        for embedding in embeddings: 
            id = None
            vec = nodes - embedding.numpy()
            if vec.shape[0] != 0:
                vec = np.linalg.norm(vec, axis=1)
                idx = np.argmin(vec)
                if  vec[idx] < threshold:
                        nodeids.append(idx)
                        continue
                
            id = nodes.shape[0]
            nodes = np.append(nodes, [embedding.numpy()], axis=0)
            nodeids.append(id)


        for i in range(len(adj)):
            for j in range(i+1, len(adj)):
                if adj[i][j] == 1:
                    G.add_edge(nodeids[i], nodeids[j])
        
    print("Saving Graph")
    pickle.dump(G, open(f'../eval/gen/{dataset}_ths_{threshold}_{run}.pickle', 'wb'))
    return G.number_of_nodes()

##THRESHOLD
copy = {}
for run in range(1, NUM_RUNS+1):
    copy[run] = {}
    for dataset in DATASETS:
        copy[run][dataset] = []
        for threshold in THS:
            samples = pickle.load(open(f'../eval/gen/sample_{dataset}{run}.pickle', 'rb'))
            copy[run][dataset].append(threshold_method(samples, threshold, dataset))

print(copy)