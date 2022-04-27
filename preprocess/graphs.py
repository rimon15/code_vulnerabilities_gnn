import networkx as nx
import pickle
import gensim
import multiprocessing as mp
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
import torch
from .tokens import tokenizer
import numpy as np

NUM_CORES = max(1, int(mp.cpu_count() / 4))

def get_graph_indices_names(graphs):
    '''
    Gets the method name corresponding to the index of the graph in the list
            Parameters:
                graphs: the list of ASTs or CFGs

            Return:
                a map of function name -> index in graphs list
    '''
    to_ret = {}
    for i in range(len(graphs)):
        G = graphs[i]
        pos = [n for n,d in G.in_degree() if d==0]
        try:
            name_full = G.nodes[pos[0]]['label']
            key = name_full[name_full.find(",") + 1:-2]
            to_ret[key] = i
        except IndexError:
            print("Graph at index " + str(i) + " has no nodes with indegree 0, skipping.")
    return to_ret

def find_graph(graph_names, graphs, file_name, func_name):
    '''
    Gets the label and graph given a function name
            Parameters:
                graph_names: the dict of function names to indices
                graphs: the list of graphs
                file_name: the filename to determine whether there is a vulnerability in the function
                func_name: the function name

            Return:
                A tuple containing (label, graph), or empty tuple if the function name is not found
    '''
    label = 0
    if "vuln" in file_name:
        label = 1
    try:
        idx = graph_names[func_name.strip()]
        return (label, graphs[idx])
    except KeyError:
        print("Function name ''" + func_name + "' not found, skipping.")
        return ()

# def find_graph_by_func(params):
#     '''
#     Gets the corresponding graph by the function name, labels it as 1 if it is vulnerable, 0 otherwise

#             Paremeters:
#                 params: tuple containing the graphs to parse, the file name to check for the label, and function name
#     '''
#     graphs, file_name, func_name = params
#     label = 0
#     if "vuln" in file_name:
#         label = 1
#     for G in graphs:
#         pos = [n for n,d in G.in_degree() if d==0]
#         if func_name.strip() in G.nodes[pos[0]]['label']:
#             return (label, G)

def pkl_relevant_graphs_with_labels(out_path, graphs, file_names, func_names):
    '''
    Writes the relevant graphs with their label (<label>, <nx MultiGraph>) to a pickle

            Parameters:
                out_path: the path to write the final pickle tuples to
                graphs: the graph to search in
                file_names: the file names for the functions
                func_names: the function names 

    '''
    graph_names = get_graph_indices_names(graphs)
    tuples = []
    for i in range(len(file_names)):
        t = find_graph(graph_names, graphs, file_names[i], func_names[i])
        tuples.append(t)

    print("Writing pickle for " + out_path)
    pickle.dump(tuples, open(out_path, "wb"))

def get_all_tokens_from_graph(t):
    '''
    '''
    _, G = t
    tokens = []
    for n in G.nodes:
        l = G.nodes[n]['label']
        tokens.append(tokenizer(l[l.find(",") + 1:-2]))
    return tokens

def graph_to_torch_data(params):
    '''
    '''
    label, G, wv = params
    newG = nx.Graph()
    for n in G.nodes:
        l = G.nodes[n]['label']
        cur_toks = tokenizer(l[l.find(",") + 1:-2])
        try:
            print(wv[cur_toks])
            newG.add_nodes_from([(int(n), {'x': np.array(wv[cur_toks])})])
        except KeyError: #add empty vector to node
            newG.add_nodes_from([(int(n), {'x': np.zeros(wv.vector_size)})])
    
    for e in G.edges:
        in_node, out_node, _ = e
        newG.add_edge(int(in_node), int(out_node))
    dG = from_networkx(newG)
    dG.label = label
    dG.y = label
    xs = []
    for cur_node in dG.x:
        to_add = np.mean(np.array(cur_node), axis = 0)
        if to_add.shape != (100,): # pad empty nodes
            to_add = np.zeros(wv.vector_size)
        xs.append(to_add)
    
    dG.x = torch.Tensor(xs)
    return dG

def write_torch_data(in_path, out_path):
    '''
    '''
    print("Loading data from " + in_path)
    tuples = pickle.load(open(in_path, "rb"))
    # Remove empty tuples
    tuples = [t for t in tuples if t]
    w2v = gensim.models.Word2Vec()
    pool = mp.Pool(NUM_CORES)

    all_tokens = list(tqdm(pool.imap(get_all_tokens_from_graph, tuples), total=len(tuples)))
    flat_all_tokens = [item for sublist in all_tokens for item in sublist]
    w2v.build_vocab(sentences=flat_all_tokens)
    print("Training w2v model...")
    w2v.train(flat_all_tokens, total_examples=w2v.corpus_count, epochs=1)

    args_iterable = [(tuples[i][0], tuples[i][1], w2v.wv) for i in range(len(tuples))]
    torch_data = list(tqdm(map(graph_to_torch_data, args_iterable), total=len(args_iterable)))
    print("Writing torch geometric data...")
    pickle.dump(torch_data, open(out_path, "wb"))