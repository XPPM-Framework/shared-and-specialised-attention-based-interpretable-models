from pathlib import Path

from data.processor import *
from data.args import *

#################################
# "Run Experiment" dependencies #
#################################

import os
import pandas as pd

import tensorflow as tf
from keras import callbacks, Model
from keras.utils import to_categorical
from keras.models import load_model

from models.shared import shared_model, shared_model_fit, generate_inputs_shared
from models.specialised import specialised_model, specialised_model_fit, plot_specialised
from models.explain import shared_explain_local, shared_explain_global, explain_local, results_df



"""
# Some debug tensorflow prints
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
print(tf.version.VERSION)
"""

# Experiment parameters
milestone = 'All'  # 'A_PREACCEPTED' # 'W_Nabellen offertes', 'All'
subset = "W"
experiment = 'OHE'  # 'Standard'#'OHE', 'No_loops'
n_size = 5
max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50

# Local environment
METHOD_DIR = (Path(os.getenv("METHODS_DIR", Path("../../methods"))) / "Wickramanayake2022").resolve()
MY_WORKSPACE_DIR = str(METHOD_DIR / 'BPIC12')
MILESTONE_DIR = os.path.join(os.path.join(MY_WORKSPACE_DIR, milestone), experiment)


def pre_processing_bpic_2012_w():
    # Define parameters
    args = get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)

    # This code will be specific for all next activity prediction only, since we save the models and vectors by prefix length groups
    # if milestone == 'All':
    #   args['indexes'] = MILESTONE_DIR+'indexes_'+str(max_size)+'.p'
    #   args['pre_index'] = MILESTONE_DIR+'pre_index_'+str(max_size)+'.p'
    #   args['processed_test_vec'] = MILESTONE_DIR+'vec_test_'+str(max_size)+'.p'
    #   args['processed_training_vec'] = MILESTONE_DIR+'vec_training_'+str(max_size)+'.p'
    #   args['weights'] = MILESTONE_DIR+'weights_'+str(max_size)+'.p'

    args['file_name_A_ex'] = os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                          "bpic12_translated_completed_A_ex.csv")
    args['file_name_O_ex'] = os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                          "bpic12_translated_completed_O_ex.csv")
    args['file_name_A_all'] = os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                           "bpic12_translated_completed_A_all.csv")
    args['file_name_O_all'] = os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                           "bpic12_translated_completed_O_all.csv")
    args['file_name_W_all'] = os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                           "bpic12_translated_completed_W_all.csv")

    # Preprocessing
    log_df = pd.read_csv(args['file_name_W_all'])
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
    rl_weights[0] = 0  # embeddig weights for label none = 0

    # converting the weights
    weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index)}

    # converting the weights into a dictionary and saving
    indexes = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne}

    # converting the weights into a dictionary and saving
    pre_index = {'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

    run_experiment(args, index_ne, indexes, pre_index, subset, vec_test, vec_train,
                   weights)



def run_experiment(args, index_ne, indexes, pre_index, subset, vec_test, vec_train,
                   weights):
    ##################
    ##################
    # Run Experiment #
    ##################
    ##################
    print("Running experiment")
    # Only using shared model because it provided the best performance in the reported results
    shared = shared_model(vec_train, weights, indexes, pre_index, args)
    shared.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    batch_size = 128  # 32, 64, 128, 256
    epochs = 250
    shared_history = shared_model_fit(vec_train, shared, indexes, pre_index, MY_WORKSPACE_DIR, batch_size, epochs, args)
    trained_model = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    shared.save(trained_model)
    shared_trained_model = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    loaded_shared_model = load_model(shared_trained_model)
    x_test, y_test = generate_inputs_shared(vec_test, args, indexes)
    # Evaluate on test data
    results = loaded_shared_model.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred_shared = loaded_shared_model.predict(x_test)
    m1_y_test = y_test.argmax(axis=1)
    m1_y_pred = y_pred_shared.argmax(axis=1)
    target_names = [index_ne[i] for i in range(len(index_ne))]
    # Commented out because dependency was broken
    # print(classification_report(y_test.argmax(axis=1), y_pred_shared.argmax(axis=1), target_names=target_names))


    # TODO: Get the resulting data
    return {"Hier könnte ihr Ergebnis stehen": "..."}


if __name__ == '__main__':
    result = pre_processing_bpic_2012_w()

    print(result)

