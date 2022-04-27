"Note: single core performance will take a long time to write all of the files, it is advisable to have a multi-core machine"

import h5py
import os
from typing import List, Dict, Tuple
import multiprocessing as mp
import sys
import subprocess
import networkx as nx
import pickle
from tqdm import tqdm

# Set the number of threads to use for preprocessing dataset tasks
NUM_CORES = max(1, int(mp.cpu_count() / 4))

'''
This section contains the parallel processing functions
'''

def label_vulns(cwe119: bool, cwe120: bool, cwe469: bool, cwe_other: bool):
    '''
    Checks if the conditions of the function contain an overall vulnerability
    (due to pickling, we can't use this as a lambda)

            Parameters:
                The values corresponding to the extracted functions

            Return:
                Whether the function has a vulnerability based on the provided flags
    '''
    return cwe119 | cwe120 | cwe469 | cwe_other

def create_vdisc_file(path: str, name: str, func: str, vuln: bool):
    '''
    Creates a file for each C++ function encountered in the VDISC dataset

            Parameters:
                path: the folder path to write the new file in
                names: the names of each file
                funcs: the functions to write
                vulns: the vulnerability labels for each function

    '''
    f = open(path + name + ".cc", "x")
    f.write("void " + func) # For some reason the dataset doesn't include the return type, so we pad with void so that Joern works properly

# def get_dot_file(params):
#     '''
#     Gets the dot file corresponding to the function name

#             Parameters:
#                 path: the path to the dot files directory
#                 file_name: the file name to rename the file to
#                 func_name: the function name to search for to rename

#              Returns: 
#                 a tuple mapping the file name to the networkx graph
#     '''
#     path, file_name, func_name = params
#     func_name = func_name.strip()
#     #"grep -rnw '" + path + "' -e 'digraph \"" + func_name + "\"'"
#     proc = subprocess.Popen("grep -rnw '" + path + "' -e 'digraph \"" + func_name + "\"'", stdout=subprocess.PIPE, shell=True)
#     res = proc.stdout.readline().decode('UTF-8')

#     try:
#         dot_file = open(res[0:res.find(':')], "r")
#     except IOError:
#         return ()

def graph_from_dot(fpath):
    '''
    Creates the networkx graph from the dot file

            Parameters:
                fpath: the path to the dot file

            Returns:
                a networkx MultiGraph of the parsed dot file
    '''
    return nx.drawing.nx_pydot.read_dot(fpath)

#     graph = nx.drawing.nx_pydot.read_dot(dot_file)
#     return (file_name, graph)

# def parse_export_joern(path: str, name: str, graph_type: str):
#     '''
#     Parses and exports the function in the specified folder into the dot format

#             Parameters:
#                 path: the filepath to the function folder to parse
#                 name: the function name
#                 graph_type: the type of graph to extract (cfg, ast, ...)
#     '''
#     os.system("joern-parse -o " + path + name + "/cpg.bin " + path + name + "/" + name + ".cc")
#     os.system("joern-export " + path + name + "/cpg.bin --repr " + graph_type + " --out " + path + name + "/" + graph_type)


# def export_joern(in_path: str, graph_type: str, out_path: str):
#     '''
#     Exports the parsed source files in dot format

#             Parameters:
#                 in_path: the folder path to get the joern binary
#                 graph_type: the type to export (cpg, cfg, ...)
#                 out_path: the path to write the dot files to
#     '''
#     #os.system("pwd")
#     os.system("joern-export " + out_path + ".bin " + "--repr " + graph_type + " --out " + out_path + "_" + graph_type)

'''
This section contains the actual execution of the data generation processes
'''

def process_vdisc_data(in_path: str, out_path: str, write: bool) -> Tuple[List[str], List[str]]:
    '''
    Retrieves the Draper VDISC vulnerability dataset from the filesystem, processes the data,
    and outputs the functions with a vulernability tag comment to the data folder. This data then
    needs to be processed by Joern to retrieve the CPGs

            Parameters:
                in_path: the filepath for the hdf5 VDISC file to process
                out_path: the filepath to write the resulting functions to
                write: if true, writes the source code files

            Return:
                a list of the function folder names
    '''
    file = h5py.File(in_path, "r")
    pool = mp.Pool(NUM_CORES)
    print("Using " + str(NUM_CORES) + " core(s).\n")
    keys = list(file.keys())
    num_keys = len(keys)

    print("Processing HDF5 files...\n")
    funcs = file[keys[num_keys - 1]][:]
    cwe119 = file[keys[0]][:]
    cwe120 = file[keys[1]][:]
    cwe469 = file[keys[2]][:]
    cwe_other = file[keys[3]][:]
    #args_iterable = [(cwe119[i], cwe120[i], cwe469[i], cwe_other[i]) for i in range(0, len(cwe119))]
    #is_vuln = list(pool.starmap(label_vulns, args_iterable))
    is_vuln = list(map(label_vulns, cwe119, cwe120, cwe469, cwe_other))
    # The names of the files will include _vuln if there is vulnerability in the function
    func_file_names = ["func_" + str(i) + ("_vuln" if is_vuln[i] else "") for i in range(len(funcs))]
    func_names = [f[0:f.decode('UTF-8').find('(')].decode('UTF-8') for f in funcs]

    if not os.path.isdir(out_path):
        sys.exit("The file path '" + out_path + "' to write out to does not exist. Please add it.")

    args_iterable = [(out_path, func_file_names[i], funcs[i].decode('UTF-8'), is_vuln[i]) for i in range(len(funcs))]
    if write:
        print("Writing VDISC source function files (" + str(len(func_file_names)) + ")...\n")
        pool.starmap(create_vdisc_file, args_iterable)
    else:
        print("Write is False so NOT writing files")
    pool.close()
    pool.join()

    return func_file_names, func_names

def run_joern(in_path: str, out_path: str):
    '''
    Runs Joern on the specified path, and collects the output which will then be used 
    by graphs.py to convert into networkx from dot
            Parameters:
            in_path: the file path to the data folders to parse
    '''
    "In order to view the garphs, type joern <out file>"

    graph_types = ['ast', 'cfg', 'ddg', 'pdg']
    print("Running " + "joern-parse -o " + out_path + " " + in_path)
    os.system("joern-parse -o " + out_path + "/cpg.bin " + in_path)
    graph_types = ['ast', 'cfg', 'ddg', 'pdg']
    for g in graph_types:
        print("Running joern-export on joern binary file for " + g + " ...")
        os.system("joern-export " + out_path + "/cpg.bin " + "--repr " + g + " --out " + out_path + g)

def load_graphs(in_path: str, out_path: str): #, file_names: List[str], func_names: List[str]
    '''
    FInds the corresponding dot files for each graph, and loads it to a pickle
            Parameters:
                in_path: the file path for the data folder to parse
                out_path: the file path to write out the pickle to
    '''
    pool = mp.Pool(NUM_CORES)

    print("Loading graphs from dot files from (" + in_path + ")...")
    #args_iterable = [(in_path, file_names[i], func_names[i]) for i in range(len(file_names))]
    args_iterable = os.listdir(in_path)
    args_iterable = [in_path + a for a in args_iterable]

    graphs = list(tqdm(pool.imap(graph_from_dot, args_iterable), total=len(args_iterable)))
    #graphs = process_map(get_dot_file, args_iterable, max_workers=NUM_CORES, chunksize=10)
    #graphs = pool.starmap(get_dot_file, args_iterable)
    pool.close()
    pool.join()

    print("Writing to pickle (" + out_path + ")...")
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)
    
def get_graphs(in_path: str):
    '''
    Gets the list of graphs from the specified path (as a pickle)

            Parameters:
            in_path: the path to parse the pickle from
    '''
    return pickle.load(open(in_path, "rb"))