import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import torch
import heapq

REAL_GRAPH = 'cora.pickle'
GRAPHS = ['1000_ts_1000_gs.pickle', '1000_ts_2000_gs.pickle', '2000_ts_1000_gs.pickle', '2000_ts_2000_gs.pickle']
FOCAL = ['1000_ts_1000_gs_focal.pickle', '1000_ts_2000_gs_focal.pickle']
FIFTY = ['1000(50)_ts_1000_gs.pickle', '1000(50)_ts_2000_gs.pickle', '500(50)_ts_1000_gs.pickle', '500(50)_ts_2000_gs.pickle']
TWENTY = ['1000(20)_ts_1000_gs.pickle', '1000(20)_ts_2000_gs.pickle', '1000(20)_ts_1000_gs_fair.pickle', '1000(20)_ts_2000_gs_fair.pickle']

SIZES_THS = {
    'CiteSeer' :  [3266, 2516, 2045],
    'Cora' : [7082, 5581, 4495],
    'Facebook': [8962, 5636, 3579],
    'Wiki' : [6960, 5373, 4313]
} 

def read_graph(file):
    G = pickle.load(open(file, 'rb'))
    return G

def create_graph(file, num_samples):
    G = nx.Graph()
    with open(file, "r") as f:
        for i in range(num_samples):
            s = f.readline()
            N = int(s[2:])
            s = f.readline()
            s = f.readline()
            nodes = s.split(' ')[:-1]
            nodes = [int(v) for v in nodes]
            s = f.readline()
            for i in range(N):
                l = f.readline().split(' ')[:-1]
                for j in range(N):
                    if(l[j]) == '1':
                        G.add_edge(nodes[i], nodes[j])

            s = f.readline()
    pickle.dump(G, open(f'1000(20)_ts_{num_samples}_gs_fair.pickle', 'wb'))
    return G

def eval(G):
    degrees = list(dict(G.degree()).values())
    avg_deg = sum(degrees)/len(degrees)
    triangles = nx.triangles(G)
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    rel_edge_dist = -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())
    plaw = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    gini = avg_deg_diff / (np.mean(degrees)*2)
    print(f"| |{G.number_of_nodes()}|{G.number_of_edges()}|{round(avg_deg,2)}|{int(sum(triangles.values())/3)}|{max(scc_size)}|{round(rel_edge_dist,4)}|{round(plaw.alpha, 4)}|{round(gini,4)}| |")



def IoU(G_gen, G_real):
    union = G_real.number_of_edges()
    intersection = 0
    for u,v in G_gen.edges():
        if G_real.has_edge(u,v):
            intersection += 1
        else:
            union += 1
    return intersection/union

# create_graph("generated_subgraphs/generated_samples19.txt", 1000)
# create_graph("generated_subgraphs/generated_samples19.txt", 2000)

# G = read_graph('latent4.pickle')
# eval(G)
#2 35 | |7817|16876|4.32|6566|7043|0.9442|1.8864|0.4335| | not sure what is this
# 2 35 |6493|15265|4.7|8615|6086|0.9258|1.8776|0.4812| | 35 seems to be best for this

import sys

samples = pickle.load(open(f'../eval/gen/sample_{sys.argv[1]}.pickle', 'rb'))
print(len(samples))


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
    if num_nodes <= min(SIZES_THS[sys.argv[1]]):
        break
    x = pairs_x[i]
    y = pairs_y[i]

    if not same(x,y):
        num_nodes -= 1
        print(num_nodes)
        unite(x,y)

    if num_nodes in SIZES_THS[sys.argv[1]]:
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

        pickle.dump(G, open(f'../eval/gen/{sys.argv[1]}_nodes_{num_nodes}.pickle', 'wb'))

exit()
#Reportar as duas formas de criar grafos.
samples = pickle.load(open(f'../eval/gen/sample_{sys.argv[1]}.pickle', 'rb'))


for threshold in [0.35, 0.40, 0.45, 0.50]: 
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
    pickle.dump(G, open(f'../eval/gen/{sys.argv[1]}_ths_{threshold}.pickle', 'wb'))
        






# G_real = read_graph(REAL_GRAPH)
# for graph in FIFTY + TWENTY:
#     print(graph)
#     G = read_graph(graph)
#     eval(G)
#     print(round(IoU(G, G_real),4))



# from torch_geometric.datasets import Planetoid
# import networkx as nx
# import numpy as np

# graph = Planetoid("../data","Cora").get(0)
# sensitive_attribute = graph.y.detach().clone()

# print(graph.y.unique(return_counts=True))

# from torch_geometric.utils import to_networkx

# G = to_networkx(graph, to_undirected=True)

# x = nx.eigenvector_centrality(G)
# x_by_class = np.zeros(7)

# for i in range(2708):
#     x_by_class[sensitive_attribute[i]] += x[i]**2

# print(x_by_class)


## 17 - size 50 500 subgraphs
## 18 - size 20 1000 subgraphs
## 19 - size 20 1000 subgraphs fair
## 20 - size 20 2000 subgraphs 500 epochs