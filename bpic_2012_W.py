from pathlib import Path

from numpy import ndarray

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


def _pre_processing_bpic_2012_w(dataset: DataFrame, args):
    print("Preprocessing...")

    # This code will be specific for all next activity prediction only, since we save the models and vectors by prefix length groups
    # if milestone == 'All':
    #   args['indexes'] = MILESTONE_DIR+'indexes_'+str(max_size)+'.p'
    #   args['pre_index'] = MILESTONE_DIR+'pre_index_'+str(max_size)+'.p'
    #   args['processed_test_vec'] = MILESTONE_DIR+'vec_test_'+str(max_size)+'.p'
    #   args['processed_training_vec'] = MILESTONE_DIR+'vec_training_'+str(max_size)+'.p'
    #   args['weights'] = MILESTONE_DIR+'weights_'+str(max_size)+'.p'

    # Preprocessing
    log_df = dataset.reset_index(drop=True)

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
    #log_df_train, log_df_test = split_train_test(log_df, 0.3)  # 70%/30%

    # Normalize numerical features
    numerical_features = ['timelapsed']
    log_df = normalize_events(log_df, args, numerical_features)

    case_ids = len(log_df['prefix_id'].unique())

    # Reformat events: converting the dataframe into a dictionary to aid vectorization
    log_df = reformat_events(log_df, ac_index, rl_index, ne_index)

    # Obtain the maximum trc_len and cases for each set

    trc_len, cases = lengths(log_df)

    trc_len = max([trc_len])

    # Converting the training log into a tensor
    vec = vectorization(log_df, ac_index, rl_index, ne_index, trc_len, cases)

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
        "vec": vec,  # Vectorized data
        "weights": weights,
        "index_ne": index_ne,
        "indexes": indexes,
        "pre_index": pre_index,
    }, log_df


def _evaluate_model(model, params, *, index_ne, indexes, vec, **kwargs):
    x_test, y_test = generate_inputs_shared(vec, params, indexes)
    # Evaluate on test data
    loss, acc = model.explain(x_test, y_test, batch_size=batch_size)
    y_pred = model.predict(x_test)
    target_names = [index_ne[i] for i in range(len(index_ne))]
    # Commented out because dependency was broken
    # print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))

    return y_pred, y_test, loss, acc


def _train_model(params, indexes, pre_index, vec, weights, batch_size: int, epochs: int, **kwargs):
    # Only using shared model because it provided the best performance in the reported results
    model = shared_model(vec, weights, indexes, pre_index, params)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    train_history = shared_model_fit(vec, model, indexes, pre_index, MY_WORKSPACE_DIR, batch_size, epochs, params)

    return model, train_history


def train(dataset_path: Path, params, model_path: Path | str = None):
    print(f"Training model for {dataset_path.name}")
    log_df = pd.read_csv(dataset_path)
    model_input, df_train = _pre_processing_bpic_2012_w(log_df, args=params)
    vec = model_input["vec"]
    indexes = model_input["indexes"]

    epochs = 250

    # Consumes: args, index_ne, indexes, pre_index, subset, vec_test, vec_train, weights
    model, training_history = _train_model(params, batch_size=batch_size, epochs=epochs, **model_input)

    # Ensure model file path exists
    if Path(model_path).parent:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)
    print(f"Saved trained model to {model_path.absolute()}")
    x_train, y_train = generate_inputs_shared(vec, params, indexes)
    loss, acc = model.explain(x_train, y_train, batch_size=batch_size)

    print(f"Training Loss: {loss}\nTraining Accuracy: {acc}")

    return loss, acc


def evaluate(dataset_path: Path, params, model_path: Path | str = None):
    log_df = pd.read_csv(dataset_path)
    model = load_model(str(model_path), safe_mode=False)
    print(f"Loaded model from {model_path.absolute()}")

    model_input, modified_data = _pre_processing_bpic_2012_w(log_df, args=params)

    y_pred, y_test, loss, acc = _evaluate_model(model, params, batch_size=batch_size, **model_input)

    print(f"Evaluation Results")
    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")

    true_labels = y_test.argmax(axis=1)
    pred_labels = y_pred.argmax(axis=1)

    result_df = compile_results(modified_data, true_labels, pred_labels, model_input)

    # Predictions and are written to a files in the same directory as the model with respective suffixes
    result_df.to_csv(model_path.parent / f"{model_path.stem}_predictions.csv", index=False)

    return result_df

    # TODO: Pickle results to file


def compile_results(modified_dataset: list[dict], true_labels: ndarray, pred_labels: ndarray, model_inputs: dict) -> DataFrame:
    """
    Combine the true and predicted labels into a single dataframe which associates it with each prefix_id
    :param modified_dataset: The dataset modified by _pre_processing_bpic_2012_w
    :param true_labels:
    :param pred_labels:
    :param model_inputs:
    :return:
    """
    # Cut everything to the smallest length
    true_labels = true_labels[:len(modified_dataset)]
    pred_labels = pred_labels[:len(modified_dataset)]

    activity_indexes = model_inputs["indexes"]["index_ac"]
    true_labels = list(map(lambda x: activity_indexes[x], true_labels))
    pred_labels = list(map(lambda x: activity_indexes[x], pred_labels))

    # add true_labels column to dataframe
    result_df = pd.DataFrame({"prefix_id": map(lambda x: x["caseid"], modified_dataset), "true_labels": true_labels, "pred_labels": pred_labels})

    return result_df


def main():
    load_path = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    args = get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)
    train(args)


if __name__ == '__main__':
    main()

