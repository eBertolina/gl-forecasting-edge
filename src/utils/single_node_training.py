from pathlib import Path
from typing import Callable

import keras
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from keras.callbacks import ModelCheckpoint

from gossiplearning.config import Config
from gossiplearning.models import LabelledData
from utils.data import load_npz_data
from utils.evaluation import plot_node_history

def aggregate_datasets(datasets: list[LabelledData]) -> LabelledData:
    X = np.concatenate([X for X, Y in datasets], axis=0)
    Y = np.concatenate([Y for Y, Y in datasets], axis=0)
    return shuffle(X, Y)


def train_single_node(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], nn.Module],
    node: int,
    verbose: int = 0,
) -> None:
    train_datasets = []
    val_datasets = []
    test_datasets = []

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz_data(
        str(datasets_folder / f"node_{node}.npz")
    )

    train_datasets.append((X_train, Y_train))
    val_datasets.append((X_val, Y_val))
    test_datasets.append((X_test, Y_test))

    train = aggregate_datasets(train_datasets)
    val = aggregate_datasets(val_datasets)

    model = model_creator()
    with torch.no_grad():
            model.load_state_dict(model.state_dict())

    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=config.training.patience,
    #     min_delta=config.training.min_delta,
    #     restore_best_weights=True,
    # )

    plots_folder = datasets_folder / "plots" / "training"
    plots_folder.mkdir(exist_ok=True, parents=True)

    model_checkpoint = ModelCheckpoint(
        filepath=str(output_folder / f"{node}_single.h5"),
        #torch.save(model.state_dict(), path / "centralized.pt")
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    history = model.train_single_node(
        train_data=train,
        validation_data=val,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_batch,
        epochs=50,
        path=output_folder,
        id=node
    )

    for fn in range(config.training.n_output_vars):
        plot_node_history(
            history=history,
            file=plots_folder / f"node{node}_single.png",
        )


def train_single_nodes(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], nn.Module],
    verbose: int = 0,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for i in range(config.n_nodes):
        train_single_node(
            config,
            datasets_folder,
            output_folder,
            model_creator,
            i,
            verbose,
        )




def old_train_single_node(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], keras.Model],
    node: int,
    verbose: int = 0,
) -> None:
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz_data(
        str(datasets_folder / f"node_{node}.npz")
    )

    model = model_creator()

    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=config.training.patience,
    #     min_delta=config.training.min_delta,
    #     restore_best_weights=True,
    # )

    plots_folder = datasets_folder / "plots" / "training"
    plots_folder.mkdir(exist_ok=True, parents=True)

    model_checkpoint = ModelCheckpoint(
        filepath=str(output_folder / f"{node}_single.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    history = model.fit(
        X_train,
        [Y_train[:, fn] for fn in range(config.training.n_output_vars)],
        validation_data=(
            X_val,
            [Y_val[:, fn] for fn in range(config.training.n_output_vars)],
        ),
        validation_batch_size=config.training.batch_size,
        verbose=verbose,
        callbacks=[
            # early_stopping,
            model_checkpoint,
        ],
        epochs=100,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_batch,
        #use_multiprocessing=False,
    ).history

    for fn in range(config.training.n_output_vars):
        plot_node_history(
            history=history,
            file=plots_folder / f"node{node}_single.svg",
        )


def old_train_single_nodes(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], keras.Model],
    verbose: int = 0,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for i in range(config.n_nodes):
        train_single_node(
            config,
            datasets_folder,
            output_folder,
            model_creator,
            i,
            verbose,
        )
