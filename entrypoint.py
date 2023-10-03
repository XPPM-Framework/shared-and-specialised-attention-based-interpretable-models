import os
import sys
from pathlib import Path
from typing import Optional, TextIO

import bpic_2012_W
from data.args import get_parameters

import typer

app = typer.Typer()


@app.command("train")
def train(dataset: Path, configuration: str, model_path: Path, *, log_file: Optional[Path] = None):
    """

    :param dataset: The dataset to train on.\n
    :param configuration: The method configuration to load.\n
    :param model_path: The path to save the model to.\n
    :param log_file: Overwrite for the log_file. If not specified, logs to stdout. Not properly implemented yet.\n
    """
    typer.echo(f"Running experiment with method Wickramanayake2022 on dataset {dataset}")

    # Add proper file extension to model_file if not given
    if not model_path.suffix:
        model_path = model_path.with_suffix(".keras")

    config = get_configuration(configuration)

    typer.echo(f"Writing results to {log_file if log_file is not None else 'stdout'}")
    log_file: TextIO = open(str(log_file), "w") if log_file is None else sys.stdout
    # TODO: Switch to proper logging
    # typer.echo(f"Logging to {output if output is not None else 'stdout'}")
    # output: TextIO = open(str(log), "w") if output is None else sys.stdout

    loss, acc = bpic_2012_W.train(params=config, dataset_path=dataset, model_path=model_path)

    log_file.close()


@app.command("evaluate")
def evaluate(dataset: Path, configuration: str, model_path: Path, *, log_file: Optional[Path] = None):
    # Add proper file extension to model_file if not given
    if not model_path.suffix:
        model_path = model_path.with_suffix(".keras")

    typer.echo(f"Evaluating model saved in {model_path} method Wickramanayake2022 on dataset {dataset}")

    config = get_configuration(configuration)

    typer.echo(f"Writing results to {log_file if log_file is not None else 'stdout'}")
    log_file: TextIO = open(str(log_file), "w") if log_file is None else sys.stdout
    # TODO: Switch to proper logging
    # typer.echo(f"Logging to {output if output is not None else 'stdout'}")
    # output: TextIO = open(str(log), "w") if output is None else sys.stdout

    bpic_2012_W.evaluate(params=config, dataset_path=dataset, model_path=model_path)

    log_file.close()


def get_configuration(configuration: str) -> dict:
    METHOD_DIR = (Path(os.getenv("METHODS_DIR", Path("../../methods"))) / "Wickramanayake2022").resolve()
    MY_WORKSPACE_DIR = str(METHOD_DIR / 'BPIC12')
    match configuration:
        case "bpic12_W":
            milestone = 'All'  # 'A_PREACCEPTED' # 'W_Nabellen offertes', 'All'
            experiment = 'OHE'  # 'Standard'#'OHE', 'No_loops'
            MILESTONE_DIR = os.path.join(os.path.join(MY_WORKSPACE_DIR, milestone), experiment)
            n_size = 5

            return get_parameters('bpic12', MILESTONE_DIR, MY_WORKSPACE_DIR, milestone, experiment, n_size)

    """
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
    """


if __name__ == '__main__':
    app(standalone_mode=False)
