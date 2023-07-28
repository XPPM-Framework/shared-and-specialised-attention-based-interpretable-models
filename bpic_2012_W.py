from pathlib import Path

from data.processor import *
from data.args import *

#################################
# "Run Experiment" dependencies #
#################################

import os
import pandas as pd
from pandas import DataFrame

import tensorflow as tf
from keras import callbacks, Model
from keras.utils import to_categorical
from keras.models import load_model, save_model

from models.shared import shared_model, shared_model_fit, generate_inputs_shared
from models.specialised import specialised_model, specialised_model_fit, plot_specialised
from models.explain import shared_explain_local, shared_explain_global, explain_local, results_df


# Experiment parameters
milestone = 'All'  # 'A_PREACCEPTED' # 'W_Nabellen offertes', 'All'
subset = "W"
experiment = 'OHE'  # 'Standard'#'OHE', 'No_loops'
n_size = 5
max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50

batch_size = 128  # 32, 64, 128, 256

# Local environment
METHOD_DIR = (Path(os.getenv("METHODS_DIR", Path("../../methods"))) / "Wickramanayake2022").resolve()
MY_WORKSPACE_DIR = str(METHOD_DIR / 'BPIC12')
MILESTONE_DIR = os.path.join(os.path.join(MY_WORKSPACE_DIR, milestone), experiment)


def _pre_processing_bpic_2012_w(dataset: Path | str, args):
    print("Preprocessing...")

    # This code will be specific for all next activity prediction only, since we save the models and vectors by prefix length groups
    # if milestone == 'All':
    #   args['indexes'] = MILESTONE_DIR+'indexes_'+str(max_size)+'.p'
    #   args['pre_index'] = MILESTONE_DIR+'pre_index_'+str(max_size)+'.p'
    #   args['processed_test_vec'] = MILESTONE_DIR+'vec_test_'+str(max_size)+'.p'
    #   args['processed_training_vec'] = MILESTONE_DIR+'vec_training_'+str(max_size)+'.p'
    #   args['weights'] = MILESTONE_DIR+'weights_'+str(max_size)+'.p'

    # Preprocessing
    log_df = pd.read_csv(dataset)
    log_df = log_df.reset_index(drop=True)

    log_df = log_df.fillna('none')

    if experiment == 'No_Loops':
        log_df = reduce_loops(log_df)

    ###################################
    # Data Encoding and Vectorization #
    ###################################

    # Index creation for activity

    ac_index = create_index(log_df, 'task')
    index_ac = {v: k for k, v in ac_index.items()}

    # Index creation for unit

    rl_index = create_index(log_df, 'role')
    index_rl = {v: k for k, v in rl_index.items()}

    # Index creation for next activity

    ne_index = create_index(log_df, 'next_activity')

    index_ne = {v: k for k, v in ne_index.items()}

    # mapping the dictionary values as columns in the dataframe
    log_df['ac_index'] = log_df['task'].map(ac_index)
    log_df['rl_index'] = log_df['role'].map(rl_index)
    log_df['ne_index'] = log_df['next_activity'].map(ne_index)

    # Split train/test set
    log_df_train, log_df_test = split_train_test(log_df, 0.3)  # 70%/30%

    # Normalize numerical features
    numerical_features = ['timelapsed']
    log_df_train = normalize_events(log_df_train, args, numerical_features)
    log_df_test = normalize_events(log_df_test, args, numerical_features)

    training_traces = len(log_df_train['prefix_id'].unique())
    test_traces = len(log_df_test['prefix_id'].unique())

    # Reformat events: converting the dataframe into a dictionary to aid vectorization
    log_train = reformat_events(log_df_train, ac_index, rl_index, ne_index)
    log_test = reformat_events(log_df_test, ac_index, rl_index, ne_index)

    # Reformat events: converting the dataframe into a dictionary to aid vectorization
    log_train = reformat_events(log_df_train, ac_index, rl_index, ne_index)
    log_test = reformat_events(log_df_test, ac_index, rl_index, ne_index)

    # Obtain the maximum trc_len and cases for each set

    trc_len_train, cases_train = lengths(log_train)
    trc_len_test, cases_test = lengths(log_test)
    # trc_len_val, cases_val = lengths(log_val)

    trc_len = max([trc_len_train, trc_len_test])

    # Converting the training log into a tensor
    vec_train = vectorization(log_train, ac_index, rl_index, ne_index, trc_len, cases_train)
    vec_test = vectorization(log_test, ac_index, rl_index, ne_index, trc_len, cases_test)
    # vec_val = vectorization(log_val,ac_index, rl_index, ne_index,trc_len,cases_val)

    #########################################################
    # Generating Initial Embedding Weights for shared Model #
    #########################################################
    ac_weights = to_categorical(sorted(index_ac.keys()), num_classes=len(ac_index))
    ac_weights[0] = 0  # embedding weights for label none = 0

    rl_weights = to_categorical(sorted(index_rl.keys()), num_classes=len(rl_index))
    rl_weights[0] = 0  # embedding weights for label none = 0

    # converting the weights
    weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index)}

    # converting the weights into a dictionary and saving
    indexes = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne}

    # converting the weights into a dictionary and saving
    pre_index = {'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

    print("Preprocessing done")

    return {
        "vec_train": vec_train,
        "vec_test": vec_test,
        "weights": weights,
        "index_ne": index_ne,
        "indexes": indexes,
        "pre_index": pre_index,
    }, log_df_train, log_df_test


def _evaluate(model, args, *, index_ne, indexes, vec_test, batch_size, **kwargs):
    ##################
    ##################
    # Run Experiment #
    ##################
    ##################
    x_test, y_test = generate_inputs_shared(vec_test, args, indexes)
    # Evaluate on test data
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred = model.predict(x_test)
    m1_y_test = y_test.argmax(axis=1)  # The true labels
    m1_y_pred = y_pred.argmax(axis=1)  # The predicted labels
    target_names = [index_ne[i] for i in range(len(index_ne))]
    # Commented out because dependency was broken
    # print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))

    return {
        "results": results,
        "y_pred": y_pred,
        "m1_y_test": m1_y_test,
        "m1_y_pred": m1_y_pred,
        "target_names": target_names,
    }


def _train_model(params, indexes, pre_index, vec_train, weights, batch_size: int, epochs: int, **kwargs):
    # Only using shared model because it provided the best performance in the reported results
    shared = shared_model(vec_train, weights, indexes, pre_index, params)
    shared.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    shared_history = shared_model_fit(vec_train, shared, indexes, pre_index, MY_WORKSPACE_DIR, batch_size, epochs, params)

    return shared, shared_history


def _load_model(model_path: Path | str):
    #shared_trained_model = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    loaded_model = load_model(str(model_path), safe_mode=False)
    return loaded_model


def train(dataset_path: Path, params, model_path: Path | str = None):
    print(f"Training model for {dataset_path.name}")
    model_input, df_train, df_test = _pre_processing_bpic_2012_w(dataset=dataset_path, args=params)

    epochs = 250

    # Consumes: args, index_ne, indexes, pre_index, subset, vec_test, vec_train, weights
    model, training_history = _train_model(params, batch_size=batch_size, epochs=epochs, **model_input)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    save_model(model, model_path)
    print(f"Saved trained model to {model_path.absolute()}")


def evaluate(dataset_path: Path, params, model_path: Path | str = None):
    print(f"Training model for {dataset_path.name}")
    model_input, df_train, df_test = _pre_processing_bpic_2012_w(dataset=dataset_path, args=params)

    model = _load_model(model_path)
    print(f"Loaded model from {model_path.absolute()}")

    results = _evaluate(model, params, batch_size=batch_size, **model_input)

    compile_results(df_train, df_test, model_input, results)

    # TODO: Pickle results to file

    print(results)


def compile_results(df_train, df_test, model_input, results) -> DataFrame:
    vec_test = []
    df_test
    pass


def main():
    load_path = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    args = get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)
    train(args)


if __name__ == '__main__':
    main()

