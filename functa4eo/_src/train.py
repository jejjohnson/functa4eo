import os

os.environ["KERAS_BACKEND"] = "torch"

from typing import Optional
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
import hydra_zen
from hydra_zen import (
    MISSING,
    builds,
    make_custom_builds_fn,
    ZenStore,
    hydrated_dataclass,
    store,
)
import xarray as xr
from pathlib import Path
from dataclasses import dataclass
from operator import methodcaller
import keras
from knerf._src.layers.siren import SirenLayer
from loguru import logger

from knerf._src.layers.siren import SirenLayer
from functa4eo._src.data import MEDTemperatureDataset, MEDSSHDataset
from functa4eo._src.domain import Period
from functa4eo._src.preprocess import Coarsen
from sklearn.utils import shuffle
from functa4eo._src.transforms import MinMaxDF, StandardScalerDF
from functa4eo._src.callbacks import DVCLiveCallback
from functa4eo._src.losses import psnr

###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


VARIABLE_DS_NAME = {"ssh": "zos", "sst": "thetao", "u": "uo", "v": "vo"}

VARIABLE_FILE_NAME = {
    "ssh": "ssh",
    "sst": "temperature",
    "u": "currents",
    "v": "currents",
}


# DATA STORE
@dataclass
class DataDirectories:
    base_path: str = "/home/juanjohn/data/ocean/reanalysis"


DATADIR = sbuilds(DataDirectories)
store(DATADIR, group="paths", name="stream")

BASE_PATH = "/home/juanjohn/data/ocean/reanalysis"


MED_TEMPERATURE = sbuilds(MEDTemperatureDataset)
MED_SSH = sbuilds(MEDSSHDataset)
store(MED_TEMPERATURE, group="data", name="temperature")
store(MED_SSH, group="data", name="ssh")

###############
# PREPROCESSING
###############

# PERIOD
YEARLONG = sbuilds(Period, time_min="2010-01-01", time_max="2011-01-01")
store(YEARLONG, group="preprocess/period", name="year")

# COARSEN
HALFRES = sbuilds(Coarsen, latitude=2, longitude=2, boundary="trim")
THIRDRES = sbuilds(Coarsen, latitude=3, longitude=3, boundary="trim")
store(HALFRES, group="preprocess/coarsen", name="halfres")
store(THIRDRES, group="preprocess/coarsen", name="thirdres")


##############
# MODEL
##############
from functa4eo._src.model import init_siren_model, init_siren_multihead_model

SIREN_MH = sbuilds(init_siren_multihead_model)
SIREN = sbuilds(init_siren_model)


def load_pretrained(model_path):
    return keras.saving.load_model(model_path)


PRETRAINED_MODEL_PATH = "/home/juanjohn/projects/nerf4ssh/content/tutorials/saved_models/final_model_siren_mh_ssh_st_res1_2010-01-01_2010-01-07.keras"
PRETRAINED = sbuilds(load_pretrained, model_path=PRETRAINED_MODEL_PATH)

store(SIREN, group="model", name="siren")
store(SIREN_MH, group="model", name="siren_mh")
store(PRETRAINED, group="model", name="pretrained")


@dataclass
class TrainParams:
    random_state: int = 42
    num_epochs: int = 1_000
    patience: int = 20
    batch_size: int = 8_000
    validation_split: float = 0.9


from functa4eo._src.optimizer import init_cosine_decay_lr

LRCOSINE = pbuilds(init_cosine_decay_lr, alpha=0.1, warmup_target=1e-2)
store(LRCOSINE, group="training/learning_rate", name="cosine")

MSE_LOSS = sbuilds(
    keras.losses.MeanSquaredError,
    reduction="sum_over_batch_size",
)

# OPTIMIZERS
ADAM = pbuilds(keras.optimizers.Adam)


PARAMS = sbuilds(TrainParams)
store(PARAMS, group="training/params", name="default")
store(ADAM, group="training/optimizer", name="adam")
store(MSE_LOSS, group="training/loss", name="mse")


def train_fn(paths, data, preprocess, model, training):
    logger.info(f"Starting")
    # open data
    ds = data.open_dataset(paths.base_path)
    # subset period
    ds = preprocess["period"](ds)
    ds = preprocess["coarsen"](ds)

    from ocn_tools._src.geoprocessing.temporal import time_rescale

    ds = time_rescale(ds, freq_dt=1, freq_unit="seconds", t0=None)

    # change to dataframe structure
    df = ds[data.variable_name].to_dataframe().reset_index().dropna()

    df = shuffle(df, random_state=42)

    # Coordinate Transformation
    coord_cols = ["time", "longitude", "latitude"]

    coords_scaler = MinMaxDF(columns=coord_cols)
    obs_scaler = StandardScalerDF(columns=[data.variable_name])

    x = coords_scaler.fit_transform(df).values
    t, x = x[..., 0], x[..., 1:]

    y = obs_scaler.fit_transform(df).values

    logger.info(f"Shapes: x={x.shape} | t={t.shape} | y={y.shape}")

    total_steps = int(
        (len(x) / training["params"].batch_size) * training["params"].num_epochs
    )
    lr = training["learning_rate"](total_steps=total_steps)
    loss = training["loss"]
    optimizer = training["optimizer"](learning_rate=lr)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/model_siren_spatial_at_epoch_{epoch}.keras",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="loss", patience=50),
        DVCLiveCallback(save_dvc_exp=True),
    ]

    logger.info(f"Compiling Model...")
    # 3 - train model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            psnr,
        ],
    )

    logger.info(f"Starting training...")
    history_siren = model.fit(
        x=[t, x],
        y=y,
        validation_split=training["params"].validation_split,
        shuffle=True,
        batch_size=training["params"].batch_size,
        epochs=training["params"].num_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return None


store(
    train_fn,
    hydra_defaults=[
        "_self_",
        {"paths": "stream"},
        {"data": "temperature"},
        {"preprocess/period": "year"},
        {"preprocess/coarsen": "halfres"},
        {"model": "siren_mh"},
        {"training/params": "default"},
        {"training/learning_rate": "cosine"},
        {"training/loss": "mse"},
        {"training/optimizer": "adam"},
    ],
)


if __name__ == "__main__":
    from hydra_zen import zen

    store.add_to_hydra_store()

    # Generate the CLI For train_fn
    z = zen(train_fn)

    z.hydra_main(
        config_name="train_fn",
        config_path=None,
        version_base="1.3",
    )
