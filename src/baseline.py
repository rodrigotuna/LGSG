import networkx as nx
import sys
import pickle
import numpy as np
from scipy.linalg import kron

init_matrix = {
    'Cora': np.array([[0.6577, 0.7199],[0.7199, 0.07658]]),
    'CiteSeer': np.array([[0.5634, 0.764],[0.764, 0.05786]]),
    'Facebook': np.array([[0.9999,0.6834], [0.6834, 0.3155]]),
    'Wiki': np.array([[0.9999, 0.6824],[0.5578, 0.1126]])
}

SIZES_THS = {
    1: {
        'CiteSeer' :  [3266, 2516, 2045],
        'Cora' : [7082, 5581, 4495],
        'Facebook': [8962, 5636, 3579],
        'Wiki' : [6960, 5373, 4313]
    },
    2: {'CiteSeer': [3235, 2358, 1847], 'Cora': [6439, 4737, 3602], 'Facebook': [6290, 3973, 2592], 'Wiki': [8294, 6280, 4779]}, 3: {'CiteSeer': [1543, 1094, 873], 'Cora': [6466, 4836, 3697], 'Facebook': [4918, 2900, 1867], 'Wiki': [7374, 5637, 4505]}, 4: {'CiteSeer': [1982, 1440, 1179], 'Cora': [6784, 5118, 3929], 'Facebook': [9745, 6369, 4011], 'Wiki': [8118, 5992, 4640]}, 5: {'CiteSeer': [2046, 1474, 1187], 'Cora': [5515, 4189, 3360], 'Facebook': [7091, 4233, 2596], 'Wiki': [8437, 6212, 4588]}
} 


SIZES = [5000, 6000, 7000, 8000, 9000, 10000]



def get_graph(model, dataset):
    if model == 'Kron':
        return init_matrix[dataset]
    else:
        return pickle.load(open(f"../eval/real/{dataset}.pickle" , "rb"))
    
def fit(G, nodes, model):
    if model == "ER":
        n = G.number_of_nodes()
        p = G.number_of_edges() / (n*(n-1)/2)
        return nx.fast_gnp_random_graph(nodes, p)
    if model == "BA":
        n = G.number_of_nodes()
        e = G.number_of_edges()
        m = round(e/n)
        return nx.barabasi_albert_graph(nodes, m)
    if model == "Kron":
        adj = G
        init = G
        print(nodes)
        while adj.shape[0] < nodes:
            adj = kron(adj, init)
            print(adj.shape)

        sampled_nodes = np.arange(adj.shape[0])
        sampled_nodes = np.random.choice(sampled_nodes, size=nodes, replace=False)

        adj = adj[sampled_nodes, sampled_nodes]
        random_adj = np.random.uniform(size=(nodes, nodes))
        adj = (adj >= random_adj)
        print(adj.shape)
        return nx.from_numpy_matrix(adj)

    

def main():
    for run in range(1,6):
        for dataset in ['Wiki', 'Cora', 'Facebook', 'CiteSeer']:
            for model in ['ER', 'BA', 'Kron']:
                G = get_graph(model, dataset)
                for num_nodes in SIZES + SIZES_THS[run][dataset]:
                    G_pred = fit(G, num_nodes, model)
                    pickle.dump(G_pred, open(f"../eval/gen/{dataset}_{model}_{num_nodes}_{run}.pickle" , "wb"))


if __name__ == "__main__":
    main()
