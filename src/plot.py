import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import os

DATASETS = ['CiteSeer', 'Cora', 'Facebook', 'Wiki']
MODELS = ['ths','nodes', 'ER', 'BA', 'Kron']
SIZES = [5000, 6000, 7000, 8000, 9000, 10000]

PRESENT_NAME = {
    'ths': 'LGSG TM',
    'nodes': "LGSG NA", 
    'ER':"ER",
    'BA': "BA", 
    'Kron': 'Kronecker'
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

THS = [0.35,0.375,0.40]

def read(file):
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
            if G.number_of_edges() >= 5278:
                break
    pickle.dump(G, open(f'sagess_real_for_real.pickle', 'wb'))
    return G

def eval(G):
    degrees = list(dict(G.degree()).values())
    #max_deg = max(degrees)
    avg_deg = sum(degrees)/len(degrees)
    triangles = nx.triangles(G)
    scc = nx.connected_components(G)
    #scc_size = [len(c) for c in scc]
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    degrees_np[degrees_np == 0] = np.finfo(np.float32).eps
    rel_edge_dist = -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())
    #plaw = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    gini = avg_deg_diff / (np.mean(degrees)*2)
    print(f" & {G.number_of_nodes()} & {G.number_of_edges()} & {round(avg_deg,4)} & {round(rel_edge_dist,4)}  & {round(gini,4)} & {round(clusterCoeff(G),4)} & {round(assortativity(G),4)} \\\\")
    return [avg_deg, rel_edge_dist, gini, clusterCoeff(G), assortativity(G)]


def max_deg(G):
    degrees = list(dict(G.degree()).values())
    return max(degrees)

def avg_deg(G):
    degrees = list(dict(G.degree()).values())
    return sum(degrees)/len(degrees)

def triangles(G):
    triangles = nx.triangles(G)
    return int(sum(triangles.values())/3)


def max_scc(G):
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    return max(scc_size)

def edge_dist_ent(G):
    degrees = list(dict(G.degree()).values())
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    degrees_np[degrees_np == 0] = np.finfo(np.float32).eps
    return -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())

def plaw(G):
    degrees = list(dict(G.degree()).values())
    pl = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    return pl.alpha

def gini(G):
    degrees = list(dict(G.degree()).values())
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    return avg_deg_diff / (np.mean(degrees)*2)

def clusterCoeff(G):
    return nx.average_clustering(G)

def assortativity(G):
    return nx.degree_assortativity_coefficient(G)

def gather_metrics():
    for dataset in DATASETS:
        metric = []
        for run in range(1,6):
            metric_r = []
            metric_s = []
            for ths in THS:
                path = "../eval/gen"
                file = f"{dataset}_ths_{ths}"
                if run != 1:
                    file += f"_{run}"
                G = read(f"{path}/{file}.pickle")
                metric_s.append(eval(G))
            metric_r.append(metric_s)
            for model in MODELS[1:]:
                metric_s = []
                for size in SIZES_THS[run][dataset]:
                    path = "../eval/gen"
                    file = f"{dataset}_{model}_{size}"
                    if run != 1:
                        file += f"_{run}"
                    G = read(f"{path}/{file}.pickle")
                    metric_s.append(eval(G))
                metric_r.append(metric_s)
            metric.append(metric_r)
        metric = np.array(metric)
        np.save(open(f"../results/m_{dataset}_ths_run.npy", "wb"), metric)


    for dataset in DATASETS:
        metric = []
        for run in range(1,6):
            metric_r = []
            for size in SIZES:
                metric_s = []
                for model in MODELS[1:]:
                    path = "../eval/gen"
                    file = f"{dataset}_{model}_{size}"
                    if run != 1:
                        file += f"_{run}"
                    if not os.path.exists(f"{path}/{file}.pickle"):
                        break
                    G = read(f"{path}/{file}.pickle")
                    metric_s.append(eval(G))
                metric_r.append(metric_s)
            metric.append(metric_r)
        metric = np.array(metric)
        np.save(open(f"../results/m_{dataset}_run.npy", "wb"), metric)


real = np.load(open("../results/real.npy", "rb"))
citeseer = np.load(open("../results/m_CiteSeer_run.npy", "rb"), allow_pickle=True)
new_array = []
for run in range(5):
    new_array_line = []
    for size in range(4):
        new_array_line.append(citeseer[run][size])
    new_array.append(new_array_line)
citeseer = np.array(new_array)
citeseer_mean = np.mean(citeseer, axis=(0,1))



cora = np.load(open("../results/m_Cora_run.npy", "rb"))
cora_mean = np.mean(cora, axis=(0,1))

facebook = np.load(open("../results/m_Facebook_run.npy", "rb"))
facebook_mean = np.mean(facebook, axis=(0,1))

wiki = np.load(open("../results/m_Wiki_run.npy", "rb"))
wiki_mean = np.mean(wiki, axis=(0,1))
metric = np.stack([citeseer_mean, cora_mean, facebook_mean, wiki_mean], axis=0)
metric = np.transpose(metric, axes=(1,0,2))
metric = np.abs((metric - real)/real)

metric = np.transpose(metric, (2, 0, 1))


MEASURE = ["Avg. Degree", "Edge Distribution Entropy", "Gini", "Cluster Coeff.", "Assortivity"]
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(16,8))
spec = gridspec.GridSpec(ncols=17, nrows=2, figure=fig)
for i in range(5):
    ni = i%3
    nj = (i>2)
    ax = fig.add_subplot(spec[nj, ni*6 + 3*nj:ni*6+5 + 3*nj])
    ax.set_title(MEASURE[i])
    w = 0.10
    ind = np.arange(4)
    for j in range(metric[i].shape[0]):
        ax.bar(ind + j * w, metric[i][j],w, label=PRESENT_NAME[MODELS[j]])
    
    ax.set_xticks(ind+3*w)
    ax.set_xticklabels(DATASETS)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=7)
plt.savefig(f"../plots/runs_latent_nodes.pdf")

metric = [citeseer, cora, facebook, wiki]
# 5 X 6 X 4 X 5 ive runs, 5modeles, 3 sizes, 5 metrics
# after: metrics, models, sizes,
for k, dataset in enumerate(DATASETS):
    metric[k] = np.transpose(metric[k], (3,2,1,0))
    MEASURE = ["Avg. Degree", "Edge Distribution Entropy", "Gini", "Cluster Coeff.", "Assortativity"]
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16,8))
    spec = gridspec.GridSpec(ncols=17, nrows=2, figure=fig)
    for i in range(5): #metrics
        ni = i%3
        nj = (i>2)
        ax = fig.add_subplot(spec[nj, ni*6 + 3*nj:ni*6+5 + 3*nj])
        ax.set_title(MEASURE[i])
        w = 0.10
        ind = np.arange(5)
        ax.axhline(real[k][i], linestyle='--')
        for j in range(metric[k][i].shape[0]): #models
            mat = metric[k][i][j] #size * runs
            lbound = np.mean(mat, axis=1) - np.std(mat, axis=1)
            ubound = np.mean(mat, axis=1) + np.std(mat, axis=1)
            ax.plot(SIZES[:mat.shape[0]], np.mean(mat, axis=1), label=PRESENT_NAME[MODELS[j]])
            ax.fill_between(SIZES[:mat.shape[0]], lbound, ubound, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=7)
    plt.savefig(f"../plots/runs_latent_nodes_influence_{dataset}_runs.pdf")



def draw_graphs():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    G_real = read("../eval/real/Cora.pickle")
    G = read("../eval/gen/Cora_ths_0.4.pickle")
    G1 = read("../eval/gen/Cora_nodes_4495.pickle")
    G_kron = read("../eval/gen/Cora_BA_4495.pickle")
    nx.draw(G_real, node_size=0.5, width=0.5,ax=ax1)
    ax1.set_title("Cora")
    ax1.text(1.1, -1.1, "a)", 
                size=15, weight='bold')
    nx.draw(G1, node_size=0.5, width=0.5, ax=ax2)
    ax2.set_title("LGSG NA")
    ax2.text(1.1, -1.1, "b)", 
                size=15, weight='bold')
    nx.draw(G, node_size=0.5, width=0.5, ax=ax3)
    ax3.set_title("LGSG TM")
    ax3.text(1.1, -1.1, "c)", 
                size=15, weight='bold')
    nx.draw(G_kron, node_size=0.5, width=0.5, ax=ax4)
    ax4.set_title("Barabási-Albert")
    ax4.text(1.1, -1.1, "d)", 
                size=15, weight='bold')
    plt.savefig("runs_graph_comparison_2x2.pdf")



draw_graphs()