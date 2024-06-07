import math
import os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from keras import Model
from keras.src.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np

from data.processor import reduce_loops, create_index, split_train_test, normalize_events, reformat_events, lengths, \
    vectorization
from data.args import get_parameters

import pandas as pd

from models.explain import shared_explain_local, explanation_to_row, explain_local
from models.shared import shared_model, shared_model_fit, generate_inputs_shared
from models.specialised import specialised_model, specialised_model_fit, generate_inputs_specialised


def preprocess(log_df: pd.DataFrame, min_size: int, max_size: int, milestone: str, experiment: str):
    """
    Adds relevant computed columns to the log_df and filters the log_df based on the given parameters.
    Args:
        log_df:
        min_size:
        max_size:
        milestone:
        experiment:

    Returns:

    """
    log_df = log_df.reset_index(drop=True)

    if 'next_activity' not in log_df.columns:
        for case, case_df in log_df.groupby('caseid'):
            case_df = case_df.sort_values(by='end_timestamp', ascending=True)
            case_df['next_activity'] = case_df['task'].shift(-1)
            # Replace nan with 'none' because of indexing
            # case_df['next_activity'] = case_df['next_activity'].fillna('none')
            log_df.loc[case_df.index, 'next_activity'] = case_df['next_activity']

        # Drop events with next_activity = None, so mostly only the last events
        log_df = log_df[~log_df["next_activity"].isna()]

    log_df = log_df.fillna('none')

    if experiment == 'No_Loops':
        log_df = reduce_loops(log_df)

    if 'trace_start' not in log_df.columns:
        log_df["trace_start"] = log_df.groupby('caseid')['end_timestamp'].transform('min')
    # Add timelapsed column from end_timestamp and trace_start
    if 'timelapsed' not in log_df.columns:
        log_df['end_timestamp'] = pd.to_datetime(log_df['end_timestamp'])
        log_df['trace_start'] = pd.to_datetime(log_df['trace_start'])
        log_df['timelapsed'] = ((log_df['end_timestamp'] - log_df['trace_start']).dt.total_seconds() / 3600).apply(
            math.ceil)

    # Add prefix_id if not already exists
    if 'prefix_id' not in log_df.columns:
        prefix_ids = {}
        prefix_lengths = {}
        for case, case_df in log_df.groupby('caseid'):
            case_df.sort_values(by='end_timestamp', ascending=True)
            i = 1
            for event in case_df.iterrows():
                # One based prefix_ids
                prefix_ids[event[0]] = str(case) + '_' + f"{i:02d}"
                prefix_lengths[event[0]] = i
                i += 1
        log_df['prefix_id'] = pd.Series(prefix_ids)
        log_df['prefix_length'] = pd.Series(prefix_lengths, dtype=int)

        # We also need to duplicate the all previous events for every prefix_id
        # This is done in a separate loop for convenience
        for case, case_df in log_df.groupby('caseid'):
            case_df.sort_values(by='end_timestamp', ascending=True)
            previous_events = []
            additional_rows = []
            for event in case_df.iterrows():
                event = event[1].to_dict()
                current_prefix_id = event["prefix_id"]
                new_events = []
                for row in previous_events:
                    new_row = deepcopy(row)
                    new_row["prefix_id"] = current_prefix_id
                    new_events.append(new_row)
                additional_rows.extend(new_events)
                previous_events.append(event)
            additional_row_df = pd.DataFrame(additional_rows, columns=log_df.columns)
            log_df = pd.concat([log_df, additional_row_df], ignore_index=True)

    if milestone != 'All':
        log_df = log_df[log_df['milestone'] == milestone]
    else:
        log_df = log_df[(log_df['prefix_length'] > min_size) & (log_df['prefix_length'] <= max_size)]

    return log_df


def train_shared(vec_train, weights, args, batch_size: int = 128, epochs: int = 250):
    print("Training Shared Model")
    shared = shared_model(vec_train, weights, args)
    shared.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    shared_history = shared_model_fit(vec_train, shared, batch_size, epochs, args)
    return shared


def evaluate_shared(model_shared, vec_test, args, batch_size: int):
    print("Evaluate on test data")
    x_test, y_test = generate_inputs_shared(vec_test, args)
    # results = model_shared.evaluate(x_test, y_test, batch_size=batch_size)
    results = (0, 0)
    y_pred_shared = model_shared.predict(x_test)

    m1_y_test = y_test.argmax(axis=1)
    m1_y_pred = y_pred_shared.argmax(axis=1)

    print("test loss, test acc:", results)
    return {'loss': results[0], 'accuracy': results[1]}


def explain_shared(model, vec_test, indices: dict[str, dict], args, batch_size: int):
    """

    Args:
        model:
        vec_test:
        indices: Index mappings for activity (ac), role (rl), and next activity (ne). E.g., index_ac, ac_index, ...
        batch_size:
        args:

    Returns:

    """
    # Shared Model Explainability
    x_test, y_test = generate_inputs_shared(vec_test, args)

    # results = model.evaluate(x_test, y_test, batch_size=batch_size)
    #y_pred_shared = model.predict(x_test)
    # print("test loss, test acc:", results)

    shared_model_attn_weights = Model(inputs=model.input,
                                      outputs=[model.output, model.get_layer('timestep_attention').output,
                                               model.get_layer('feature_importance').output])

    # Final general nan replacement with warning
    if (nan_value_count := np.isnan(x_test).sum()) > 0:
         print(f"Replacing {nan_value_count} nan values in x_test with 0")
         for i in range(len(x_test)):
             x_test[i] = np.nan_to_num(x_test[i], nan=0.0)

    shared_output_with_attention = shared_model_attn_weights.predict(x_test)

    # TODO: Make this output explanations
    explanation_rows = []
    for i in tqdm(range(0, len(x_test[0])), desc="Explaining", unit="trace"):
        #print(f"Case {i}: Ground Truth: {y_test[i].argmax()}, Prediction: {y_pred_shared[i].argmax()}")
        ground_truth, predicted, trace_explanation = shared_explain_local(shared_output_with_attention, x_test, y_test,
                                                                          indices, -1, i, visualize=False)
        explanation_row = explanation_to_row(ground_truth, predicted, trace_explanation)
        explanation_rows.append(explanation_row)
    df_explanations = pd.concat(explanation_rows, ignore_index=True)
    return df_explanations


def train_specialised(vec_train, weights, indices, args, batch_size: int = 256, epochs: int = 200):
    print("Training Specialised Model")
    specialised = specialised_model(vec_train, weights, indices, args)
    specialised.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    specialised_history = specialised_model_fit(vec_train, specialised, indices, batch_size, epochs, args)
    # specialised_path = os.path.join(os.path.join(args["milestone_dir"], 'trained_models'),
    #                                 'specialised_model_' + subset + '.h5')
    # specialised.save(specialised_path)
    return specialised


def evaluate_specialised(model_specialised, vec_test, indices, args, batch_size: int):
    print("Evaluate on test data")
    x_test, y_test = generate_inputs_specialised(vec_test, args, indices)

    # Evaluate the model on the test data using `evaluate`

    results = model_specialised.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred = model_specialised.predict(x_test)
    print("test loss, test acc:", results)
    return results


def explain_specialised(model, vec_test, indices: dict[str, dict], args, batch_size: int):
    """

    Args:
        model:
        vec_test:
        indices: Index mappings for activity (ac), role (rl), and next activity (ne). E.g., index_ac, ac_index, ...
        batch_size:
        args:

    Returns:

    """
    # Shared Model Explainability
    x_test, y_test = generate_inputs_specialised(vec_test, args, indices)

    model_attn_weights = Model(inputs=model.input,
                               outputs=[model.output,
                                        model.get_layer('timestep_attention').output,
                                        model.get_layer('ac_importance').output,
                                        model.get_layer('rl_importance').output,
                                        model.get_layer('t_importance').output])

    output_with_attention = model_attn_weights.predict(x_test)

    results = model.evaluate(x_test, y_test, batch_size=batch_size)

    # explain_local(output_with_attention, x_test, y_test, indices, -1, 53)
    #y_pred_shared = model.predict(x_test)
    print("test loss, test acc:", results)

    explanation_rows = []
    for i in tqdm(range(0, len(x_test[0])), desc="Explaining", unit="trace"):
        #print(f"Case {i}: Ground Truth: {y_test[i].argmax()}, Prediction: {y_pred_shared[i].argmax()}")
        ground_truth, predicted, trace_explanation = explain_local(output_with_attention,
                                                                   x_test, y_test, indices, -1, i, visualize=False)
        explanation_row = explanation_to_row(ground_truth, predicted, trace_explanation)
        explanation_rows.append(explanation_row)
    df_explanations = pd.concat(explanation_rows, ignore_index=True)
    return df_explanations


def encode(log_df: pd.DataFrame, indices: dict[str, dict] = None):
    """
    Create indices and apply them to the event log.
    """
    if indices is None:
        # Index creation for activity
        ac_index = create_index(log_df, 'task')
        index_ac = {v: k for k, v in ac_index.items()}

        # Index creation for unit

        rl_index = create_index(log_df, 'role')
        index_rl = {v: k for k, v in rl_index.items()}

        # Index creation for next activity
        ne_index = create_index(log_df, 'next_activity')
        # Hackery because if not all activities are in the next activity the index will be missing
        if len(ne_index) < len(ac_index) - 1:
            ne_index = ac_index

            # ne_index = {}
            # i = 0
            # for ac in ac_index:
            #     if ac == "none":
            #         continue
            #     ne_index[ac] = i
            #     i += 1

        index_ne = {v: k for k, v in ne_index.items()}

        indices = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne,
                   'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

    else:
        ac_index = indices['ac_index']
        rl_index = indices['rl_index']
        ne_index = indices['ne_index']
        index_ac = indices['index_ac']
        index_rl = indices['index_rl']
        index_ne = indices['index_ne']

        # Ensure that rows are strings
        log_df['task'] = log_df['task'].astype(str)
        log_df['role'] = log_df['role'].astype(str)
        log_df['next_activity'] = log_df['next_activity'].astype(str)

    # Mapping the dictionary values as columns in the dataframe
    log_df['ac_index'] = log_df['task'].map(ac_index)
    log_df['rl_index'] = log_df['role'].map(rl_index)
    log_df['ne_index'] = log_df['next_activity'].map(ne_index)

    # Add 'task_index' to log if not already exists
    if 'task_index' not in log_df.columns:
        log_df['task_index'] = log_df['task'].map(ac_index)

    return log_df, indices


def get_weights(ac_index, index_ac, index_rl, ne_index, rl_index):
    ac_weights = to_categorical(sorted(index_ac.keys()), num_classes=len(ac_index))
    ac_weights[0] = 0  # embedding weights for label none = 0
    rl_weights = to_categorical(sorted(index_rl.keys()), num_classes=len(rl_index))
    rl_weights[0] = 0  # embedding weights for label none = 0
    # converting the weights
    weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index)}
    return weights


def df_log_from_list(log_list: list[dict], indices: dict[str, dict], log_config: dict[str, str]) -> pd.DataFrame:
    """
    Reverts index mappings on activities, etc. to obtain a DataFrame of the log which matches the explanation DataFrame.
    Also fixes case_id to have a two-digit prefix_id and the associated sorting
    Args:
        log_list: The event log as a list of traces
        indices: Dict of index mappings for activity, role, and next activity
        log_config: Config of the event log. Only "case_id_key" is required.
    """

    def fix_case_id(case_id: str) -> str:
        split = case_id.split("_")
        if len(split) < 2 or len(split[1]) != 1:
            return case_id
        # Single digit prefix_id
        return f"{split[0]}_{int(split[1]):02d}"

    log_list_mapped = []
    for trace in log_list:
        trace_mapped = {
            log_config["case_id_key"]: "".join(trace["caseid"].split("_")[:-1]),
            "prefix_id": fix_case_id(trace["caseid"]),
            "ac_prefix": [indices["index_ac"].get(ac, "none") for ac in trace["ac_order"]],
            "rl_prefix": [indices["index_rl"].get(rl, "none") for rl in trace["rl_order"]],
            "tbtw_prefix": trace["tbtw"],
            "next_activity": indices["index_ne"][trace["next_activity"]],
        }
        log_list_mapped.append(trace_mapped)
    return pd.DataFrame(log_list_mapped)


def apply_log_config(log_df: pd.DataFrame, log_config: dict) -> pd.DataFrame:
    """
    If given try to rename the columns of the log to match the expected column names.
    Expected column names:
    Args:
        log_df: The event log to apply the configuration to.
        log_config: Contains keys: case_id_key, activity_key, timestamp_key, timeformat, resource_key

    Returns: A copy of the event log dataframe with the columns renamed appropriately.

    """
    rename_dict = {
        log_config["case_id_key"]: "caseid",
        log_config["activity_key"]: "task",
        log_config["resource_key"]: "role",
        log_config["timestamp_key"]: "end_timestamp",
    }

    return log_df.rename(rename_dict, axis=1)


def restore_indices_int(indices: dict[str, dict]):
    for index in indices:
        try:
            indices[index] = {int(k): v for k, v in indices[index].items()}
        except ValueError:
            pass
        try:
            indices[index] = {k: int(v) for k, v in indices[index].items()}
        except ValueError:
            pass
    return indices
