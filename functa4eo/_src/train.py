import os

os.environ["KERAS_BACKEND"] = "torch"

from typing import Optional
from hydra_zen import instantiate
from loguru import logger
import typer
from functa4eo._src.transforms import MinMaxDF, StandardScalerDF
from sklearn.utils import shuffle
import keras_core as keras
from functa4eo._src.losses import psnr
from functa4eo._src.callbacks import DVCLiveCallback

VARIABLE_DS_NAME = {"ssh": "zos", "sst": "thetao", "u": "uo", "v": "vo"}

VARIABLE_FILE_NAME = {
    "ssh": "ssh",
    "sst": "temperature",
    "u": "currents",
    "v": "currents",
}


def validate_units(ds):
    var_names = list(ds.data_vars.keys())

    if "zos" in var_names:
        ds["zos"].attrs["units"] = "meters"
    if "thetao" in var_names:
        ds["thetao"].attrs["units"] = "degC"
    if "bottomT" in var_names:
        ds["bottomT"].attrs["units"] = "degC"
    if "uo" in var_names:
        ds["uo"].attrs["units"] = "meter / second"
    if "vo" in var_names:
        ds["vo"].attrs["units"] = "meter / second"
    return ds


def main(
    stage: str = "train",
    variable: str = "ssh",
    res: int = 1,
    validation_split: float = 0.9,
    batch_size: int = 10_000,
    epochs: int = 100,
    t0: str = "2010-01-01",
    t1: str = "2010-02-01",
    model: Optional[str] = None,
    learning_rate: float = 1e-4,
):
    logger.info(f"TRAINING for {variable}")
    variable_name = VARIABLE_DS_NAME[variable]
    variable_file_name = VARIABLE_FILE_NAME[variable]

    # logger.info("initializing")
    # data = instantiate(config.MED_CURRENTS, variable=variable_file_name)
    # data = validate_units(data)

    # # run light preprocessing
    # logger.info("preprocessing!")
    # logger.info(f"Training period: {t0}--{t1}")
    # data = data.sel(time=slice(t0, t1))
    # # data = instantiate(config.period)(data, time_min=t0, time_max=t1)
    # data = instantiate(config.coarsen, longitude=res, latitude=res)(data)
    # data = instantiate(config.trescale)(data)

    # # change to dataframe structure
    # df = data[variable_name].to_dataframe().reset_index().dropna()

    # df = shuffle(df, random_state=42)

    # # Coordinate Transformation

    # coord_cols = ["time", "longitude", "latitude"]

    # scaler = MinMaxDF(columns=coord_cols)
    # obs_scaler = StandardScalerDF(columns=[variable_name])

    # x = scaler.fit_transform(df).values

    # # take coordinates
    # t, x = x[..., 0], x[..., 1:]
    # y = obs_scaler.fit_transform(df).values

    # logger.info(f"Shapes: x={x.shape} | t={t.shape} | y={y.shape}")

    # logger.info(f"Starting Training Script...")

    # if model is not None:
    #     logger.info(f"Using model: {model}")
    #     model = keras.saving.load_model(model)
    # else:
    #     logger.info(f"Initializing SIREN")
    #     model = instantiate(config.SIREN)

    # logger.info(f"Model: {model.name}")

    # lr = instantiate(config.COSINE_DECAY)
    # loss = instantiate(config.MSE_LOSS)
    # optimizer = instantiate(config.ADAM)(learning_rate=learning_rate)

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         filepath="./checkpoints/model_siren_spatial_at_epoch_{epoch}.keras",
    #         save_best_only=True,
    #     ),
    #     keras.callbacks.EarlyStopping(monitor="val_loss", patience=50),
    #     DVCLiveCallback(save_dvc_exp=True),
    # ]

    # logger.info(f"Compiling Model...")
    # # 3 - train model
    # model.compile(
    #     loss=loss,
    #     optimizer=optimizer,
    #     metrics=[
    #         psnr,
    #     ],
    # )

    # logger.info(f"Starting training...")
    # history_siren = model.fit(
    #     x=[t, x],
    #     y=y,
    #     validation_split=validation_split,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     callbacks=callbacks,
    #     verbose=1,
    # )

    # score = model.evaluate(x=[t, x], y=y, batch_size=100_000, verbose=1)
    # predictions = model.predict(x=[t, x], batch_size=100_000)


if __name__ == "__main__":
    typer.run(main)
