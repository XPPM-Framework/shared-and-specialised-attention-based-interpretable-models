import json
import os
import pickle
from pathlib import Path
from typing import Annotated

from keras.src.saving.saving_api import load_model

from data.args import get_parameters, get_args, save_args

import typer
import pandas as pd
from keras.src.utils import to_categorical

from data.processor import split_train_test, normalize_events, reformat_events, lengths, vectorization
from processing import preprocess, encode, get_weights, train_shared, train_specialised, evaluate_shared, \
    explain_shared, df_log_from_list, evaluate_specialised, explain_specialised, apply_log_config

app = typer.Typer(pretty_exceptions_enable=False)


@app.command("train")
def train(model_type: str, dataset: Path, model_path: Path, *,
          log_config: dict = typer.Option(default=None, help="Mapping of log columns", parser=json.loads),
          experiment_dir: Path = None,
          milestone: str = "All", experiment: str = "OHE",
          n_size: int = typer.Option(default=5, help="(Explanation) Prefix size"),
          parameters: Annotated[dict, typer.Argument(parser=json.loads)] = None):
    """

    :param model_type: The type of model to train.Literal["shared", "specialised"]\n
    :param dataset: The dataset to train on. Needs the columns, "caseid", "task", "role", "end_timestamp", "trace_start"
    :param model_path: The path to save the model to.\n
    :param log_config: Mapping of log columns.\n
    :param experiment_dir: The directory to save the relevant experiment files to. Defaults to model_path.parent.\n
    :param milestone: "All" or activities in the dataset.\n
    :param experiment: "OHE" or "No_loops", which filters out loops from the traces.\n
    :param n_size: Prefix size (for explanations).\n
    :param parameters: Dictionary of parameters for the algorithm.\n
    """
    typer.echo(f"Running experiment with method Wickramanayake2022 on dataset {dataset}")

    if experiment_dir is None:
        experiment_dir = Path(model_path.parent)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    model_path.parent.mkdir(exist_ok=True, parents=True)

    MY_WORKSPACE_DIR = os.path.join(os.getcwd(), )
    MILESTONE_DIR = os.path.join(os.path.join(MY_WORKSPACE_DIR, milestone), experiment)

    # Get the arguments to save next to the model after training
    args = get_args(experiment_dir, MILESTONE_DIR, milestone, experiment, n_size, args=parameters)
    log_df = pd.read_csv(dataset)
    if log_config:
        log_df = apply_log_config(log_df, log_config)

    max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
    min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50
    # TODO: Add prefix_id to the log_df (caseid_index) if not already exists
    log_df = preprocess(log_df, min_size, max_size, milestone, experiment)
    log_df_encoded, indices = encode(log_df)

    args["indices"] = indices
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

    numerical_features = ['timelapsed']
    log_df_encoded = normalize_events(log_df_encoded, args, numerical_features)
    training_traces = len(log_df_encoded['prefix_id'].unique())
    log_train = reformat_events(log_df_encoded, ac_index, rl_index, ne_index)
    trc_len, cases_train = lengths(log_train)
    args["trc_len"] = trc_len
    vec_train = vectorization(log_train, ac_index, rl_index, ne_index, trc_len, cases_train)

    weights = get_weights(ac_index, index_ac, index_rl, ne_index, rl_index)
    args["weights"] = experiment_dir / f"{model_path.stem}-weights.pickle"
    pickle.dump(weights, open(args["weights"], "wb"))

    batch_size = args["batch_size"]
    epochs = args["epochs"]
    args["model_type"] = model_type
    if model_type == "shared":
        model_shared = train_shared(vec_train, weights, args, batch_size, epochs)
        model_shared.save(model_path)
    elif model_type == "specialised":
        model_shared = train_specialised(vec_train, weights, indices, args, batch_size, epochs)
        model_shared.save(model_path)
    else:
        raise Exception("Model type not recognized.")

    args_path = model_path.with_stem(f"{model_path.stem}-args").with_suffix(".json")
    save_args(args, args_path)


@app.command("explain")
def explain(dataset: Path, model_path: Path,  *,
            log_config: dict = typer.Option(default=None, help="Mapping of log columns", parser=json.loads),
            experiment_dir: Path = None, args_path: Path = None,
            parameters: Annotated[dict, typer.Argument(parser=json.loads)] = None):
    if experiment_dir is None:
        experiment_dir = model_path.parent
    if args_path is None:
        args_path = model_path.with_stem(f"{model_path.stem}-args").with_suffix(".json")
    # TODO: Load model and settings from model_path
    model = load_model(model_path)
    args = json.load(open(args_path))
    if parameters:
        args.update(parameters)

    indices = args["indices"]
    index_ac = indices['index_ac']
    index_rl = indices['index_rl']
    index_ne = indices['index_ne']
    ac_index = indices['ac_index']
    rl_index = indices['rl_index']
    ne_index = indices['ne_index']

    log_df = pd.read_csv(dataset)
    if log_config:
        log_df = apply_log_config(log_df, log_config)
    max_size = 1000  # 3, 5, 10, 15, 20, 30, 50, 95
    min_size = 0  # 0, 3, 5, 10, 15, 20, 30, 50
    log_df = preprocess(log_df, min_size, max_size, args["milestone"], args["experiment"])
    log_df_encoded, indices = encode(log_df)

    numerical_features = ['timelapsed']
    log_df_test = normalize_events(log_df_encoded, args, numerical_features)
    test_traces = len(log_df_test['prefix_id'].unique())
    log_test = reformat_events(log_df_test, ac_index, rl_index, ne_index)

    # We do not consider the trc_len of the test variable as we have used the trc_len in the args for training
    trc_len = args["trc_len"]
    _, cases_test = lengths(log_test)
    vec_test = vectorization(log_test, ac_index, rl_index, ne_index, trc_len, cases_test)

    batch_size = args["batch_size"]
    epochs = args["epochs"]
    model_type = args["model_type"]
    print(f"Explain {model_type} model")
    if model_type == "shared":
        eval_results = evaluate_shared(model, vec_test, args, batch_size)
        df_explanation = explain_shared(model, vec_test, indices, args, batch_size)

    elif model_type == "specialised":
        eval_results = evaluate_specialised(model, vec_test, indices, args, batch_size)
        df_explanation = explain_specialised(model, vec_test, indices, args, batch_size)
    else:
        raise Exception(f"Model type '{model_type}' not recognized.")

    print("eval loss, accuracy", eval_results)
    df_log_test = df_log_from_list(log_test, indices)
    df_complete = pd.merge(df_log_test, df_explanation, left_index=True, right_index=True)
    # Reverse "ac_prefix", "rl_prefix", "tbtw_prefix" columns as they are written in reverse order
    df_complete["ac_prefix"] = df_complete["ac_prefix"].apply(lambda x: list(reversed(x)))
    df_complete["rl_prefix"] = df_complete["rl_prefix"].apply(lambda x: list(reversed(x)))
    df_complete["tbtw_prefix"] = df_complete["tbtw_prefix"].apply(lambda x: list(reversed(x)))
    output_path = experiment_dir / f"{model_path.stem}-explanations.csv"
    df_complete.to_csv(output_path, index=False)
    print(f"Explanations saved to {output_path}")


if __name__ == '__main__':
    app(standalone_mode=False)
