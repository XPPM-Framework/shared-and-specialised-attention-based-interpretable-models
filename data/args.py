import json
import os
import pickle
from pathlib import Path
from typing import Mapping


def save_args(args, path: Path):
    def paths_to_str(d):
        if isinstance(d, Mapping):
            return {k: paths_to_str(v) for k, v in d.items()}
        elif isinstance(d, Path):
            return str(d)
        else:
            return d

    json.dump(paths_to_str(args), open(path, "w"), indent=4)
    print(f"Saved args to {path}")


def get_args(experiment_dir: Path, MILESTONE_DIR: str, MILESTONE: str, EXPERIMENT: str, n_size:int, *, args: dict = None):
    """
    Gets default arguments for the given milestone directory, etc. and overrides them with the provided arguments.
    Args:
        MILESTONE_DIR:
        MY_WORKSPACE_DIR:
        MILESTONE:
        EXPERIMENT:
        args:

    Returns:

    """
    default_args = {
        "milestone_dir": MILESTONE_DIR,
        "folder": os.path.join(MILESTONE_DIR, "output_files"),
        "lstm_act": None,
        "dense_act": None,
        "optim": "Adam",  # "Adagrad", "Adam"
        "norm_method": "lognorm",  # "max", "lognorm"
        "model_type": "shared_cat",  # "specialized", "concatenated", "shared_cat", "joint", "shared"
        "l_size": 50,  # LSTM layer sizes
        "weights": experiment_dir/"weights.pickle",
        "milestone": MILESTONE,  # "All" or activity name
        "experiment": EXPERIMENT,  # "OHE", "No_loops"
        "prefix_length": "fixed",  # "variable", "fixed"
        "n_size": n_size,  # (Explanation) Prefix size
        "experiment_dir": experiment_dir,
        "batch_size": 128,
        "epochs": 256,
    }
    if args is not None:
        default_args.update(args)
    return default_args


def _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT, N_SIZE):

    parameters = dict()

    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")
    #       Specific model training parameters
    parameters['lstm_act'] = None # optimization function see keras doc
    parameters['dense_act'] = None # optimization function see keras doc
    parameters['optim'] = 'Adam' #'Adagrad' # optimization function see keras doc
    parameters['norm_method'] = 'lognorm' # max, lognorm
    # Model types --> specialized, concatenated, shared_cat, joint, shared
    parameters['model_type'] = 'shared_cat'
    parameters['l_size'] = 50 # LSTM layer sizes
    parameters['n_size'] = N_SIZE
    #    Generation parameters

    parameters['file_name'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2012_Prefixes.csv') 
    parameters['file_name_all'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2012_Prefixes_all.csv') 
    parameters['processed_file_name'] = os.path.join(MILESTONE_DIR, 'BPIC_2012_Processed.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR,'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['processed_val_vec'] = os.path.join(MILESTONE_DIR, 'vec_val.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR ,'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p') 
    parameters['args'] = os.path.join(MILESTONE_DIR,'args.p')
    parameters['milestone']=MILESTONE
    parameters['experiment'] = EXPERIMENT
    parameters['prefix_length']='fixed' #'variable'

    parameters['log_name'] = 'bpic12'

    return parameters

def _params_bpic17(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT, N_SIZE):

    parameters = dict()

    parameters['folder'] =  os.path.join(MILESTONE_DIR, "output_files")
    #       Specific model training parameters
    parameters['lstm_act'] = None # optimization function see keras doc
    parameters['dense_act'] = None # optimization function see keras doc
    parameters['optim'] = 'Adam' #'Adagrad' # optimization function see keras doc
    parameters['norm_method'] = 'lognorm' # max, lognorm
    # Model types --> specialized, concatenated, shared_cat, joint, shared
    parameters['model_type'] = 'shared_cat'
    parameters['l_size'] = 50 # LSTM layer sizes
    parameters['n_size'] = N_SIZE
    #    Generation parameters

    parameters['file_name'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2017_Prefixes.csv') 
    parameters['file_name_all'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2017_Prefixes_all.csv') 
    parameters['processed_file_name'] = os.path.join(MILESTONE_DIR, 'BPIC_2017_Processed.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR,'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['processed_val_vec'] = os.path.join(MILESTONE_DIR, 'vec_val.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR ,'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p') 
    parameters['args'] = os.path.join(MILESTONE_DIR,'args.p')
    parameters['milestone']=MILESTONE
    parameters['experiment'] = EXPERIMENT
    parameters['prefix_length']='fixed' #'variable'

    parameters['log_name'] = 'bpic17'

    return parameters

def get_parameters(dataset, MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT,N_SIZE):
    
    if dataset == 'bpic12':
        return _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT,N_SIZE)
    elif dataset == 'bpic17':
        return _params_bpic17(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT,N_SIZE)
    else:
        raise  ValueError("Please specific dataset 'bpic12' or 'bpic17'")

def saver(args, vec_train, vec_test, ac_weights, rl_weights, ne_index, index_ac, index_rl, index_ne):

    # saving the processed tensor
    with open(args['processed_training_vec'], 'wb') as fp:
        pickle.dump(vec_train, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args['processed_test_vec'], 'wb') as fp:
        pickle.dump(vec_test, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # converting the weights into a dictionary and saving
    weights = {'ac_weights':ac_weights, 'rl_weights':rl_weights, 'next_activity':len(ne_index)}
    with open(args['weights'], 'wb') as fp:
        pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # converting the weights into a dictionary and saving
    indexes = {'index_ac':index_ac, 'index_rl':index_rl,'index_ne':index_ne}
    with open(args['indexes'], 'wb') as fp:
        pickle.dump(indexes, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #saving the arguements (args)
    with open(args['args'], 'wb') as fp:
        pickle.dump(args, fp, protocol=pickle.HIGHEST_PROTOCOL)

def loader(MILESTONE_DIR):
    with open(os.path.join(MILESTONE_DIR,'args.p'), 'rb') as fp:
        args = pickle.load(fp)
    
    with open(args['processed_training_vec'], 'rb') as fp:
        vec_train = pickle.load(fp)
    with open(args['processed_test_vec'], 'rb') as fp:
        vec_test = pickle.load(fp)
        
    with open(args['weights'], 'rb') as fp:
        weights = pickle.load(fp)

    with open(args['indexes'], 'rb') as fp:
        indexes = pickle.load(fp)
    
    return args, vec_train, vec_test, weights, indexes