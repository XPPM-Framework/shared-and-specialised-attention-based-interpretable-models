import os
from pathlib import Path
from tqdm import tqdm

from keras import Model
from keras.src.utils import to_categorical
from tensorflow.keras.models import load_model

from data.processor import reduce_loops, create_index, split_train_test, normalize_events, reformat_events, lengths, \
    vectorization
from data.args import get_parameters

import pandas as pd

from models.explain import shared_explain_local, explanation_to_row
from models.shared import shared_model, shared_model_fit, generate_inputs_shared
from models.specialised import specialised_model, specialised_model_fit, generate_inputs


def preprocess(log_df: pd.DataFrame, min_size: int, max_size: int, milestone: str, experiment: str):
    log_df = log_df.reset_index(drop=True)
    if milestone != 'All':
        log_df = log_df[log_df['milestone'] == milestone]
    else:
        log_df = log_df[(log_df['prefix_length'] > min_size) & (log_df['prefix_length'] <= max_size)]

    log_df = log_df.fillna('none')

    if experiment == 'No_Loops':
        log_df = reduce_loops(log_df)

    return log_df


def train_shared(vec_train, weights, args):
    print("Training Shared Model")
    shared = shared_model(vec_train, weights, args)
    shared.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    batch_size = 128  # 32, 64, 128, 256
    epochs = 250
    shared_history = shared_model_fit(vec_train, shared, MY_WORKSPACE_DIR, batch_size, epochs, args)

    # Save to original path, will only be used internally to the method
    shared_path = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'), 'shared_model_' + subset + '.h5')
    shared.save(shared_path)
    return shared


def evaluate_shared(model_shared, vec_test, args, batch_size: int):
    print("Evaluate on test data")
    x_test, y_test = generate_inputs_shared(vec_test, args)
    results = model_shared.evaluate(x_test, y_test, batch_size=batch_size)
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

    print("Evaluate on test data")

    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred_shared = model.predict(x_test)
    print("test loss, test acc:", results)

    shared_model_attn_weights = Model(inputs=model.input,
                                      outputs=[model.output, model.get_layer('timestep_attention').output,
                                               model.get_layer('feature_importance').output])

    shared_output_with_attention = shared_model_attn_weights.predict(x_test)

    # TODO: Make this output explanations
    explanation_rows = []
    for i in tqdm(range(0, len(x_test[0])), desc="Explaining", unit="trace"):
        #print(f"Case {i}: Ground Truth: {y_test[i].argmax()}, Prediction: {y_pred_shared[i].argmax()}")
        ground_truth, predicted, trace_explanation = shared_explain_local(shared_output_with_attention, x_test, y_test, indices, -1, i, visualize=False)
        explanation_row = explanation_to_row(ground_truth, predicted, trace_explanation)
        explanation_rows.append(explanation_row)
    df_explanations = pd.concat(explanation_rows, ignore_index=True)
    return df_explanations


def train_specialized(args):
    ...


def evaluate_specialized(args):
    ...


def encode(log_df: pd.DataFrame):
    # Index creation for activity
    ac_index = create_index(log_df, 'task')
    index_ac = {v: k for k, v in ac_index.items()}

    # Index creation for unit

    rl_index = create_index(log_df, 'role')
    index_rl = {v: k for k, v in rl_index.items()}

    # Index creation for next activity
    ne_index = create_index(log_df, 'next_activity')

    index_ne = {v: k for k, v in ne_index.items()}

    # Mapping the dictionary values as columns in the dataframe
    log_df['ac_index'] = log_df['task'].map(ac_index)
    log_df['rl_index'] = log_df['role'].map(rl_index)
    log_df['ne_index'] = log_df['next_activity'].map(ne_index)

    indices = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne,
               'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}
    return log_df, indices


def main(experiment_dir: Path):
    n_size = 5
    max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
    min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50

    args = get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)

    log_df = pd.read_csv(args['file_name_W_all'])

    log_df = preprocess(log_df, min_size, max_size)

    log_df, indices = encode(log_df)
    index_ac = indices['index_ac']
    index_rl = indices['index_rl']
    index_ne = indices['index_ne']
    ac_index = indices['ac_index']
    rl_index = indices['rl_index']
    ne_index = indices['ne_index']

    # converting the weights into a dictionary and saving
    indexes = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne}
    # converting the weights into a dictionary and saving
    pre_index = {'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

    log_df_train, log_df_test = split_train_test(log_df, 0.3)  # 70%/30%

    numerical_features = ['timelapsed']
    log_df_train = normalize_events(log_df_train, args, numerical_features)
    log_df_test = normalize_events(log_df_test, args, numerical_features)
    training_traces = len(log_df_train['prefix_id'].unique())
    test_traces = len(log_df_test['prefix_id'].unique())
    log_train = reformat_events(log_df_train, ac_index, rl_index, ne_index)
    log_test = reformat_events(log_df_test, ac_index, rl_index, ne_index)
    trc_len_train, cases_train = lengths(log_train)
    trc_len_test, cases_test = lengths(log_test)
    trc_len = max([trc_len_train, trc_len_test])

    vec_train = vectorization(log_train, ac_index, rl_index, ne_index, trc_len, cases_train)
    vec_test = vectorization(log_test, ac_index, rl_index, ne_index, trc_len, cases_test)

    ac_weights = to_categorical(sorted(index_ac.keys()), num_classes=len(ac_index))
    ac_weights[0] = 0  # embedding weights for label none = 0

    rl_weights = to_categorical(sorted(index_rl.keys()), num_classes=len(rl_index))
    rl_weights[0] = 0  # embedding weights for label none = 0
    # converting the weights
    weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index)}

    ##################
    # Train - Shared #
    ##################
    x_test, y_test = generate_inputs_shared(vec_test, args)

    batch_size = 128  # 32, 64, 128, 256
    epochs = 250
    model_shared = train_shared(vec_train, weights, args)

    shared_path = experiment_dir / 'shared_model.h5'
    model_shared.save(shared_path)

    print("Evaluate on test data")

    results = model_shared.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred_shared = model_shared.predict(x_test)
    print("test loss, test acc:", results)

    m1_y_test = y_test.argmax(axis=1)
    m1_y_pred = y_pred_shared.argmax(axis=1)

    ###################
    # Train - Special #
    ###################
    print("Specialized")
    specialised = specialised_model(vec_train, weights, indexes, pre_index, args)
    specialised.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    batch_size = 256  # 32, 64, 128, 256
    epochs = 200
    specialised_history = specialised_model_fit(vec_train, specialised, indexes, pre_index, MY_WORKSPACE_DIR,
                                                batch_size, epochs, args)
    specialised_path = os.path.join(os.path.join(MILESTONE_DIR, 'trained_models'),
                                    'specialised_model_' + subset + '.h5')
    specialised.save(specialised_path)

    x_test, y_test = generate_inputs(vec_test, args, indexes, experiment)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")

    results = specialised.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred = specialised.predict(x_test)
    print("test loss, test acc:", results)


def main_shared(experiment_dir: Path, data_path: Path, model_path: Path, retrain: bool = False):
    n_size = 8
    max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
    min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50

    args = get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)
    # args["prefix_length"] = "dynamic"
    log_df = pd.read_csv(data_path)

    log_df = preprocess(log_df, min_size, max_size, milestone, experiment)

    log_df_encoded, indices = encode(log_df)
    index_ac = indices['index_ac']
    index_rl = indices['index_rl']
    index_ne = indices['index_ne']
    ac_index = indices['ac_index']
    rl_index = indices['rl_index']
    ne_index = indices['ne_index']

    # converting the weights into a dictionary and saving
    indexes = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne}
    # converting the weights into a dictionary and saving
    pre_index = {'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

    log_df_train, log_df_test = split_train_test(log_df_encoded, 0.3)  # 70%/30%

    numerical_features = ['timelapsed']
    log_df_train = normalize_events(log_df_train, args, numerical_features)
    log_df_test = normalize_events(log_df_test, args, numerical_features)
    training_traces = len(log_df_train['prefix_id'].unique())
    test_traces = len(log_df_test['prefix_id'].unique())
    log_train = reformat_events(log_df_train, ac_index, rl_index, ne_index)
    log_test = reformat_events(log_df_test, ac_index, rl_index, ne_index)
    trc_len_train, cases_train = lengths(log_train)
    trc_len_test, cases_test = lengths(log_test)
    trc_len = max([trc_len_train, trc_len_test])

    vec_train = vectorization(log_train, ac_index, rl_index, ne_index, trc_len, cases_train)
    vec_test = vectorization(log_test, ac_index, rl_index, ne_index, trc_len, cases_test)

    ac_weights = to_categorical(sorted(index_ac.keys()), num_classes=len(ac_index))
    ac_weights[0] = 0  # embedding weights for label none = 0

    rl_weights = to_categorical(sorted(index_rl.keys()), num_classes=len(rl_index))
    rl_weights[0] = 0  # embedding weights for label none = 0
    # converting the weights
    weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index)}

    ##########
    # Shared #
    ##########
    print("Shared Model")
    batch_size = 128  # 32, 64, 128, 256
    epochs = 250
    if not retrain and model_path.exists():
        model_shared = load_model(model_path)
    else:
        model_shared = train_shared(vec_train, weights, args)
        model_shared.save(model_path)

    # Test
    eval_results = evaluate_shared(model_shared, vec_test, args, batch_size)

    print("Explain")
    df_explanation = explain_shared(model_shared, vec_test, indices, args, batch_size)
    df_log_test = df_log_from_list(log_test, indices)
    df_complete = pd.merge(df_log_test, df_explanation, left_index=True, right_index=True)
    df_complete.to_csv(experiment_dir / "explanations.csv", index=False)


def df_log_from_list(log_list: list[dict], indices: dict[str, dict]) -> pd.DataFrame:
    """
    Reverts index mappings on activities, etc. to obtain a DataFrame of the log which matches the explanation DataFrame.
    Also fixes case_id to have a two-digit prefix_id and the associated sorting
    Args:
        log_list: The event log as a list of traces
        indices: Dict of index mappings for activity, role, and next activity
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
            "caseid": fix_case_id(trace["caseid"]),
            "ac_prefix": [indices["index_ac"][ac] for ac in trace["ac_order"]],
            "rl_prefix": [indices["index_rl"][rl] for rl in trace["rl_order"]],
            "tbtw_prefix": trace["tbtw"],
            "next_activity": indices["index_ne"][trace["next_activity"]],
        }
        log_list_mapped.append(trace_mapped)
    return pd.DataFrame(log_list_mapped)


if __name__ == '__main__':
    milestone = 'All'  # 'A_PREACCEPTED' # 'W_Nabellen offertes', 'All'
    subset = "W"
    experiment = 'OHE'  # 'Standard'#'OHE', 'No_loops'
    MY_WORKSPACE_DIR = os.path.join(os.getcwd(), 'BPIC12')
    MILESTONE_DIR = os.path.join(os.path.join(MY_WORKSPACE_DIR, milestone), experiment)

    experiment_dir = Path("experiment")
    params = {
        "data_path": Path(os.path.join(os.path.join(MY_WORKSPACE_DIR, "Translated_dataset"),
                                       "bpic12_translated_completed_W_all.csv")),
        "model_path": experiment_dir / "shared_model.h5",
        "result_dir": Path(),
    }

    main_shared(experiment_dir, params["data_path"], params["model_path"], retrain=False)
