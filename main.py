import preprocess.datasets as data
import preprocess.graphs as graphs
import models.graph_models as models
import pickle

def main():
    '''
    Central execution for all project tasks
    '''
    #fnames, func_names = data.process_vdisc_data("../vdisc_dataset/VDISC_test.hdf5", "./data/VDISC/test/code/", False)
    #data.run_joern("data/VDISC/test/code/", "data/VDISC/test/joern/")
    #data.load_graphs("data/VDISC/test/joern/ast/", "data/VDISC/test/graphs/ast.pkl")
    #data.load_graphs("data/VDISC/test/joern/cfg/", "data/VDISC/test/graphs/cfg.pkl")
    #data.load_graphs("data/VDISC/test/joern/ddg/", "data/VDISC/test/graphs/ddg.pkl")
    #data.load_graphs("data/VDISC/test/joern/pdg/", "data/VDISC/test/graphs/pdg.pkl")

    #all_asts = pickle.load(open("data/VDISC/test/graphs/ast.pkl", "rb"))
    #all_cfg = pickle.load(open("data/VDISC/test/graphs/cfg.pkl", "rb"))
    #all_ddg = pickle.load(open("data/VDISC/test/graphs/ddg.pkl", "rb"))
    #all_pdg = pickle.load(open("data/VDISC/test/graphs/pdg.pkl", "rb"))
    #graphs.pkl_relevant_graphs_with_labels("data/VDISC/test/tuples/ast_tuples.pkl", all_asts, fnames, func_names)
    #graphs.pkl_relevant_graphs_with_labels("data/VDISC/test/tuples/cfg_tuples.pkl", all_cfg, fnames, func_names)
    #graphs.pkl_relevant_graphs_with_labels("data/VDISC/test/tuples/ddg_tuples.pkl", all_ddg, fnames, func_names)
    #graphs.pkl_relevant_graphs_with_labels("data/VDISC/test/tuples/pdg_tuples.pkl", all_pdg, fnames, func_names)

    # use ast and cfg
    #graphs.write_torch_data("data/VDISC/test/tuples/ast_tuples.pkl", "models/data_lists/ast_data.pkl")
    #graphs.write_torch_data("data/VDISC/test/tuples/cfg_tuples.pkl", "models/data_lists/cfg_data.pkl")

    # Run the models
    ast_torch_data = pickle.load(open("models/data_lists/torch_data_ast.pkl", "rb"))
    #cfg_torch_data = pickle.load(open("models/data_lists/torch_data_cfg.pkl", "rb"))
    # classifier_gcn = models.GraphClassifier("GraphGCN",c_hidden=50,
    #                                    layer_name="GCN",
    #                                    num_layers=6,
    #                                    dp_rate_linear=0.5,
    #                                    dp_rate=0.1)
    # classifier_gat = models.GraphClassifier("GraphGAT",c_hidden=256,
    #                                    layer_name="GAT",
    #                                    num_layers=3,
    #                                    dp_rate_linear=0.5,
    #                                    dp_rate=0.0)
    classifier_ggcn = models.GraphClassifier("GraphGGCN",c_hidden=150,
                                       layer_name="GGCN",
                                       num_layers=6,
                                       dp_rate_linear=0.5,
                                       dp_rate=0.1)
    #print(classifier_gcn.run_model(ast_torch_data))
    #print(classifier_gat.run_model(ast_torch_data))
    print(classifier_ggcn.run_model(ast_torch_data))

    
    

if __name__ == "__main__":
    main()